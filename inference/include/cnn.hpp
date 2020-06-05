// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _CNN_H
#define _CNN_H

#include <string>
#include <cstring>
#include <tuple>
#include <typeinfo>
#include "model.hpp"
#include "model_tools.h"
#include "tensor.hpp"
#include "operator.hpp"
#include "tensor_desc.h"
#include "factory.hpp"
#include "cpu/factory_cpu.hpp"
#ifdef _USE_MALI
#include "ocl/factory_ocl.hpp"
#endif

class CNN: public Model {
public:

    /**
     * @param name
     */
    CNN() {}

    explicit CNN(Arch arch, DataType dt, std::string name) : Model(arch, dt, name) { }
    virtual ~CNN() = default;
    /**
     * @param op
     * @param in
     * @param out
     */

    void sort_operators_sequential(const ModelSpec* ms)
    {
        int opNum = ms->num_operator_specs;
        for (int i = 0; i < opNum; i++) {
            std::string opName = ms->ops[i].name;
            if (opName.compare("data") == 0) {
                continue;
            }
            this->sortedOps.push_back(opName);
        }
    }

    void initialize_ops(const ModelSpec* ms)
    {
        int opNum = ms->num_operator_specs;

        Vec<std::string> modelInputTensorNames;
        for (int i = 0; i < ms->num_inputs; i++) {
            modelInputTensorNames.push_back(ms->input_names[i]);
        }
        for (int i = 0; i < ms->num_outputs; i++) {
            this->modelOutputTensorNames.push_back(ms->output_names[i]);
        }
        this->modelInputTensorNames = modelInputTensorNames;

        U32 operatorIndex = 0;
        HashMap<std::string, U32> operatorIndexMap;
        for (int i = 0; i < opNum; i++) {
            OperatorSpec curOps = ms->ops[i];
            std::string opName = curOps.name;
            if (opName.compare("data") == 0) {
                continue;
            }
            operatorIndexMap[opName] = operatorIndex++;
        }

        for (int i = 0; i < opNum; i++) {
            OperatorSpec curOps = ms->ops[i];
            std::string opName = curOps.name;
            if (opName.compare("data") == 0) {
                continue;
            }
            Vec<std::string> inputTensorsName;
            Vec<std::string> outputTensorsName;
            int inputTensorsNum = curOps.num_inputs;
            for (int j = 0; j < inputTensorsNum; j++) {
                inputTensorsName.push_back(curOps.input_tensors_name[j]);
            }
            
            int outputTensorsNum = curOps.num_outputs;
            for (int j = 0; j < outputTensorsNum; j++) {
                outputTensorsName.push_back(curOps.output_tensors_name[j]);
            }

            int numTensors = inputTensorsNum + outputTensorsNum;
            Vec<I32> tensorPositions(numTensors);
            memcpy(tensorPositions.data(), curOps.tensor_positions, numTensors * bytesOf(DT_I32));

            // create op object
            std::shared_ptr<Factory> factory;
#ifdef _USE_MALI
            if(this->schedule == MALI) {
                auto factory_ocl = (Factory*)(new FactoryOCL());
                factory = std::shared_ptr<Factory>(factory_ocl);
            } else {
#endif            
                auto factory_cpu = (Factory*)(new FactoryCPU());
                factory = std::shared_ptr<Factory>(factory_cpu);
#ifdef _USE_MALI
            }
#endif           
            std::shared_ptr<Operator> op = factory->createOperators(curOps, this->dt, operatorIndexMap, inputTensorsName);
            op->set_op_name(opName);
#ifdef _USE_MALI
            if(this->schedule == MALI) CHECK_STATUS(op->set_mali_handle(this->handle));
#endif
            op->set_op_schedule(this->schedule);
            op->set_tensor_positions(tensorPositions);
            op->init_feature_scale(curOps.num_quant_feature, curOps.feature_scale);
            this->ops.push_back(op);

            // setup operatorMap, tensorMap, operatorTensorMap
            this->add(op, inputTensorsName, outputTensorsName);
        }

        // setup WeightSpec ptr in WeightOperator
        for (int i = 0; i < ms->num_weight_specs; i++) {
            WeightSpec curOpWs = ms->ws[i];
            std::string opName = curOpWs.op_name;
            auto op = this->operatorMap[opName];
            auto weightOp = dynamic_cast<WeightOperator*>(op.get());
            weightOp->set_weightspec_ptr(curOpWs);
            if (curOpWs.bytes_of_vec != 0) {
                CHECK_REQUIREMENT(curOpWs.vec != nullptr);
                weightOp->set_hasBias(true);
            }
            // These two pointers will be managed by engine via shared_ptr, so mt_destroy_model should not free them
            ms->ws[i].weight = nullptr;
            ms->ws[i].vec = nullptr;
        }
    }

    void ready(HashMap<std::string, TensorDesc> inputDescMap) override
    {
#ifdef _DEBUG
        const char* funcStr = "[DEBUG] ready()";
        std::cout << "[INFO] schedule: " << this->schedule << std::endl;
        std::cout << funcStr << " Model input num: " << this->modelInputTensorNames.size() << std::endl;
        for (auto item: this->modelInputTensorNames) {
            std::cout << "    input: " << item << std::endl;
        }
#endif

        this->infer_output_tensors_size(inputDescMap);
        // handle the weight ops
        for (auto op : this->ops) {
#ifdef _DEBUG
            std::cout << funcStr << " op: " << op->get_name() << std::endl;
#endif
            if (op->is_weight()) {
                if (op->get_op_type() == OT_Conv) {
                    auto convOpPtr = dynamic_cast<Convolution*>(op.get());
                    CHECK_STATUS(convOpPtr->init_weight_bias_from_model(nullptr));
                    CHECK_STATUS(convOpPtr->infer_forward_algorithm(this->algorithmMap));
                } else if (op->get_op_type() == OT_Deconvolution) {
                    auto convOpPtr = dynamic_cast<Deconvolution*>(op.get());
                    CHECK_STATUS(convOpPtr->init_weight_bias_from_model(nullptr));
                    CHECK_STATUS(convOpPtr->infer_forward_algorithm());
                } else if (op->get_op_type() == OT_FC) {
                    auto fcOpPtr = dynamic_cast<FullyConnected*>(op.get());
                    CHECK_STATUS(fcOpPtr->init_weight_bias_from_model(nullptr));
                    CHECK_STATUS(fcOpPtr->infer_forward_algorithm(this->algorithmMap));
                } else if (op->get_op_type() == OT_Embedding) {
                    auto embeddingOpPtr = dynamic_cast<Embedding*>(op.get());
                    CHECK_STATUS(embeddingOpPtr->init_weight_bias_from_model(nullptr));
                } else if (op->get_op_type() == OT_LayerNorm) {
                    auto layernormOpPtr = dynamic_cast<LayerNorm*>(op.get());
                    CHECK_STATUS(layernormOpPtr->init_weight_bias_from_model(nullptr));
                } else if (op->get_op_type() == OT_Scale) {
                    auto scaleOpPtr = dynamic_cast<Scale*>(op.get());
                    CHECK_STATUS(scaleOpPtr->init_weight_bias_from_model(nullptr));
                } else if (op->get_op_type() == OT_LSTM) {
                    auto lstmOpPtr = dynamic_cast<LSTMCell*>(op.get());
                    CHECK_STATUS(lstmOpPtr->init_weight_bias_from_model(nullptr));
                } else if (op->get_op_type() == OT_SharedWeight) {
                    auto weightOpPtr = dynamic_cast<SharedWeight*>(op.get());
                    CHECK_STATUS(weightOpPtr->init_weight_bias_from_model(nullptr));
                    std::string weightOpOutputName = (std::get<1>(this->operatorTensorMap[op->get_name()]))[0];
                    Tensor weightTensor = weightOpPtr->weightTensors[0];
                    this->tensorMap[weightOpOutputName]->set_shared_ptr(weightTensor.get_shared_ptr());
                    this->weightOpOutputNames.insert(weightOpOutputName);
                }
            }
            if(op->get_op_type() == OT_MatMul) {
                auto matmulOpPtr = dynamic_cast<MatMul*>(op.get());
                CHECK_STATUS(matmulOpPtr->infer_forward_algorithm(this->algorithmMap));
            }
        }
        
#ifdef _USE_MALI
        if(this->schedule == MALI) this->infer_gclmem_descs(inputDescMap);
#endif        
        this->infer_tmp_memory_size();
        this->assign_tmp_tensor();
        //transform filter
        for (auto op : this->ops) {
            if (op->is_weight()) {
                if (op->get_op_type() == OT_Conv) {
                    auto convOpPtr = dynamic_cast<Convolution*>(op.get());
                    CHECK_STATUS(convOpPtr->transform_filter());
                } else if (op->get_op_type() == OT_Deconvolution) {
                    auto convOpPtr = dynamic_cast<Deconvolution*>(op.get());
                    CHECK_STATUS(convOpPtr->transform_filter());
                } else if (op->get_op_type() == OT_FC) {
                    auto fcOpPtr = dynamic_cast<FullyConnected*>(op.get());
                    CHECK_STATUS(fcOpPtr->transform_filter());
                } else if (op->get_op_type() == OT_LSTM) {
                    auto lstmOpPtr = dynamic_cast<LSTMCell*>(op.get());
                    CHECK_STATUS(lstmOpPtr->transform_filter());
                }    
            }
        }
        this->infer_tmp_memory_size();
        temp->alloc(this->maxTmpElements);
        this->assign_output_tensor();
#ifdef _USE_MALI
        if(this->schedule == MALI) CHECK_STATUS(gcl_finish(handle.get()));
#endif
    }

    void reready(HashMap<std::string, TensorDesc> inputDescMap)
    {
        this->infer_output_tensors_size(inputDescMap);
#ifdef _USE_MALI
        if(this->schedule == MALI) this->infer_gclmem_descs(inputDescMap);
#endif        
        this->infer_tmp_memory_size();
        temp->alloc(this->maxTmpElements);
#ifdef _USE_MALI
        if(this->schedule == MALI) {
            for(auto it : outputTensors) {
                auto outTensor = it.second;
                std::shared_ptr<GCLMem> oclMem = outTensor->get_shared_ptr();
                U32 orgSize = oclMem->desc.byteSize;
                U32 size = orgSize * 2;
                oclMem->desc.byteSize = size;
                oclMem->desc.use_map = true;
                oclMem->desc.flags  = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
            }
            CHECK_STATUS(gcl_clean_kernelVec(this->handle.get()));
            Model::run_mali_prepare(true);
            CHECK_STATUS(gcl_finish(this->handle.get())); 
        }
#endif
    }

    /**
     * @param inputTensorsName
     * @param outputTensorsName
     */
    EE mark_input_output(const ModelSpec* ms)
    {
        inputTensors.clear();
        for (I32 i = 0; i < ms->num_inputs; i++) {
            std::string str = ms->input_names[i];
            auto it = tensorMap.find(str);
            if (tensorMap.end() != it) {
#ifdef _USE_MALI
                if(this->schedule == MALI) {
                    it->second->alloc();//alloc ocl gpu memory for inputTensor
                    std::shared_ptr<Tensor> tmpTensorCPU(new Tensor());
                    tmpTensorCPU->set_desc(ms->input_dims[i]);
                    tmpTensorCPU->alloc();
                    auto p = std::pair<std::string, std::shared_ptr<Tensor>>(str, tmpTensorCPU);
                    inputTensorsHost.insert(p);
                }
#endif 
                (*(it->second)).set_desc(ms->input_dims[i]);
                inputTensors.insert(*it);
            } else {
                return NOT_MATCH;
            }
        }
        outputTensors.clear();
        for (I32 i = 0; i < ms->num_outputs; i++) {
            std::string str = ms->output_names[i];
            auto it = tensorMap.find(str);
            
            if (tensorMap.end() != it) {
#ifdef _USE_MALI
                if(this->schedule == MALI) {//resize outputTensors to map 
                    auto outTensor = it->second;
                    std::shared_ptr<GCLMem> oclMem = outTensor->get_shared_ptr();
                    U32 orgSize = oclMem->desc.byteSize;
                    U32 size = orgSize * 2;
                    oclMem->desc.use_map = true;
                    oclMem->desc.flags  = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
                    outTensor->get_memory()->alloc(size);
                }
#endif                
                outputTensors.insert(*it);
            } else {
                return NOT_MATCH;
            }
        }

#ifdef _USE_MALI
        if(this->schedule == MALI) {
           U32 tmpBufSize = 0;
           for(auto it  = inputTensors.begin(); it != inputTensors.end(); it++) {
               Tensor* inputTensor = it->second.get();
               TensorDesc desc = inputTensor->get_desc();
               GCLMem_t   mem  = inputTensor->get_val();
               U32        size = 0;
               tensor_computing_set_input_infer_tmpBuf_size(mem, desc, &size, MALI); 
               tmpBufSize = (tmpBufSize < size) ? size : tmpBufSize;
           }

           if(tmpBufSize > maxTmpElements) {
               maxTmpElements = tmpBufSize;
               temp->alloc(maxTmpElements);
           }
        }
#endif
        return SUCCESS;
    }

#ifdef _USE_MALI
    void mali_prepare() {
        Model::run_mali_prepare(false);
        CHECK_STATUS(gcl_finish(this->handle.get())); 
    }
#endif

    void copy_to_named_input(std::string inputName, U8* data) {
        if(inputTensors.find(inputName) == inputTensors.end()) CHECK_STATUS(NOT_MATCH); 
        auto tensorPtr = this->inputTensors[inputName];
        TensorDesc desc = tensorPtr->get_desc();
#ifdef _USE_MALI
        if(this->schedule == MALI) {
            OclMemory* mem = (OclMemory*) tensorPtr->get_memory();
            auto tempMem = std::static_pointer_cast<GCLMem>(temp->get_shared_ptr());
            mem->set_tmpBuf(tempMem);
        }
#endif        
        tensorPtr->set_val_by_copy(desc, data);
    }

    void set_input_tensors_value(HashMap<std::string, std::shared_ptr<U8>> modelTensorsInput) {
        for(auto &modelTensorInput : modelTensorsInput) {
            std::string inputName    = modelTensorInput.first;
            std::shared_ptr<U8> data = modelTensorInput.second;
            if(inputTensors.find(inputName) == inputTensors.end()) CHECK_STATUS(NOT_MATCH); 
            auto tensorPtr = this->inputTensors[inputName];
#ifdef _USE_MALI
            if(this->schedule == MALI) {
                TensorDesc desc = tensorPtr->get_desc();
                auto mem = (OclMemory*) tensorPtr->get_memory();
                auto tempMem = std::static_pointer_cast<GCLMem>(temp->get_shared_ptr());
                mem->set_tmpBuf(tempMem);
                tensorPtr->set_val_by_copy(desc, data.get());
            } else {
#endif        
                tensorPtr->set_shared_ptr(data);
#ifdef _USE_MALI
            }
#endif        
        }
    }

    HashMap<std::string, std::shared_ptr<Tensor>> get_inputs() {
#ifdef _USE_MALI
        if(this->schedule == MALI) return this->inputTensorsHost;
#endif
        return this->inputTensors;
    }

    HashMap<std::string, std::shared_ptr<Tensor>> get_outputs()
    {
#ifdef _USE_MALI
        if(this->schedule == MALI) {
            for(auto it  = outputTensors.begin(); it != outputTensors.end(); it++) {
                auto outputTensor = it->second;
                auto host_desc = outputTensor->get_desc();
                auto mem = (OclMemory*)outputTensor->get_memory();
                mem->get_val_to_hostptr(host_desc, NULL, CL_TRUE);
            }
            CHECK_STATUS(gcl_finish(handle.get()));
        }
#endif
        return this->outputTensors;
    }
    Tensor get_tensor_by_name(std::string tensorName) {
        if (this->tensorMap.find(tensorName) != this->tensorMap.end()) {
#ifdef _USE_MALI
            if(this->schedule == MALI) {
                if(this->outputTensors.find(tensorName) != this->outputTensors.end()) {
                    auto outputTensor = outputTensors[tensorName];
                    auto host_desc = outputTensor->get_desc();
                    auto mem = (OclMemory*)outputTensor->get_memory();
                    mem->get_val_to_hostptr(host_desc, NULL, CL_TRUE);
                } else {
                    CHECK_STATUS(NOT_SUPPORTED);
                }
                CHECK_STATUS(gcl_finish(handle.get()));
            }
#endif
            return *(this->tensorMap[tensorName].get());
        } else {
            std::shared_ptr<Tensor> tensor(new Tensor());
            TensorDesc desc;
            desc.dt = this->dt;
            desc.nDims = 0;
            tensor->set_desc(desc);
            return *tensor.get();
        }
    }

    void set_modelInputTensorNames(Vec<std::string> modelInputTensorNames) {
        this->modelInputTensorNames = modelInputTensorNames;
    }

    Vec<std::string> get_model_input_tensor_names() {
        return  this->modelInputTensorNames;
    }

    Vec<std::string> get_model_output_tensor_names() {
        return this->modelOutputTensorNames;
    }

    EE infer_output_tensors_size(HashMap<std::string, TensorDesc> inputDescMap) override
    {
        bool reassignMemory = false;
        this->set_input_tensors_desc(inputDescMap);
#ifdef _DEBUG
        const char* funcStr = "[DEBUG] infer_output_tensors_size()";
        std::cout << funcStr << std::endl;
        for (auto iter: inputDescMap) {
            std::cout << funcStr << " input: " << iter.first << " " << tensorDesc2Str(iter.second) << std::endl;
        }
#endif
        int opsNum = this->sortedOps.size();
        for (int i = 0; i < opsNum; i++) {
            std::string opName = sortedOps[i];

            auto op = this->operatorMap[opName];
#ifdef _DEBUG
            std::cout << funcStr << " op: " << opName << " type: " << OperatorTypeName()[op->get_op_type()] << std::endl;
#endif
            Vec<std::string> curOpInputTensorName = std::get<0>(this->operatorTensorMap[opName]);
            Vec<std::string> curOpOutputTensorName = std::get<1>(this->operatorTensorMap[opName]);
            int curOpInNum = curOpInputTensorName.size();
            int curOpOutNum = curOpOutputTensorName.size();
            Vec<TensorDesc> inTensorDescs;
            Vec<TensorDesc> outTensorDescs;

            for (int j = 0; j < curOpOutNum; j++) {
                TensorDesc dummyTensorDesc;
                outTensorDescs.push_back(dummyTensorDesc);
            }

            Vec<Tensor> inTensors, outTensors;
            for (std::string inputTensorName: curOpInputTensorName) {
#ifdef _DEBUG
                std::cout << "    inputTensorName: " << inputTensorName << " ";
#endif
                inTensorDescs.push_back(this->tensorMap[inputTensorName]->get_desc());

#ifdef _DEBUG
                std::cout << tensorDesc2Str(this->tensorMap[inputTensorName]->get_desc());
                std::cout << std::endl;
#endif
            }
#ifdef _USE_MALI
            if(this->schedule != MALI) {
#endif
                for (int k = 0; k < curOpInNum; k++) {
                    U32 size = tensorNumBytes(inTensorDescs[k]);
                    I32 slot = op->tensorPos[k];
                    if (slot == -1) {  //These tensors will be standalone
                        continue;
                    }
                    if (slot >= (I32)this->storageSizes.size()) {
                        this->storageSizes.resize(slot+1, 0);
                    }
                    if (size > this->storageSizes[slot]) {
                        this->storageSizes[slot] = size;
                        reassignMemory = this->memoryAssigned;
                    }
                }
#ifdef _USE_MALI
            }
#endif
            CHECK_STATUS(op->infer_output_tensors_size(inTensorDescs, &outTensorDescs));
            for (std::string inputTensorName: curOpInputTensorName) inTensors.push_back(*this->tensorMap[inputTensorName].get());

            for (int k = 0; k < curOpOutNum; k++) {
                std::string outputTensorName = curOpOutputTensorName[k];
#ifdef _DEBUG
                std::cout << "    outputTensorName: " << outputTensorName << " ";
#endif
                TensorDesc outputTensorDesc = outTensorDescs[k];
#ifdef _DEBUG
                std::cout << tensorDesc2Str(outputTensorDesc);
                std::cout << std::endl;
#endif
#ifdef _USE_MALI
                if(this->schedule != MALI) {
#endif 
                    U32 size = tensorNumBytes(outputTensorDesc);
                    I32 slot = op->tensorPos[curOpInNum + k];
                    if (slot != -1) {
                        if (slot >= (I32)this->storageSizes.size()) {
                            this->storageSizes.resize(slot+1, 0);
                        }
                        if (size > this->storageSizes[slot]) {
                            this->storageSizes[slot] = size;
                            reassignMemory = this->memoryAssigned;
                        }
                    } else {
                        if (this->memoryAssigned && size > tensorNumBytes(this->tensorMap[outputTensorName]->get_desc()))
                            reassignMemory = true;
                    }
#ifdef _USE_MALI                
                }
#endif          
                this->tensorMap[outputTensorName]->set_desc(outputTensorDesc);
                outTensors.push_back(*this->tensorMap[outputTensorName].get());
            }
            op->set_input_output_tensors(inTensors, outTensors);
        }
#ifdef _DEBUG
        U32 originalSize = 0;
        U32 standaloneSize = 0;
        for (auto tensor : this->tensorMap) {
            originalSize += tensorNumBytes(tensor.second->get_desc());
            if (weightOpOutputNames.find(tensor.first) != weightOpOutputNames.end()) {
                standaloneSize += tensorNumBytes(tensor.second->get_desc());
            }
        }
        std::cout << "Originally " << this->tensorMap.size() << " tensors, taking " << originalSize << " bytes.\n";

        std::cout << "Storage reduced to " << storageSizes.size() << " reuse slots: \n";
        U32 totalSize = 0;
        for (U32 size : storageSizes) {
            std::cout << size << " bytes, ";
            totalSize += size;
        }
        std::cout << "\nIn total " << totalSize << " bytes.\n";

        if (0 != standaloneSize) {
            std::cout << "Another " << standaloneSize << " bytes are reserved for standalone tensors (e.g. loop topology).\n";
        }

        std::cout << "Reuse ratio is " << (F32)originalSize / (totalSize+standaloneSize) << std::endl;
#endif
        if (reassignMemory) {
            this->assign_output_tensor();
        }
        return SUCCESS;
    }

#ifdef _USE_MALI
    EE infer_gclmem_descs(HashMap<std::string, TensorDesc> inputDescMap) override
    {
#ifdef _DEBUG
        const char* funcStr = "[DEBUG] infer_gclmem_desc()";
        std::cout << funcStr << std::endl;
#endif

        for (auto iter: inputDescMap) {
            U32 stride[3] = {0, 0, 0};
            U32 offset[3] = {0, 0, 0};
            GCLMemDesc gclTmpDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
            auto mem = (OclMemory*)(this->tensorMap[iter.first]->get_memory());
            mem->set_mem_desc(gclTmpDesc);
        }

        int opsNum = this->sortedOps.size();
        for (int i = 0; i < opsNum; i++) {
            std::string opName = sortedOps[i];
            auto op = this->operatorMap[opName];
            Vec<std::string> curOpInputTensorName = std::get<0>(this->operatorTensorMap[opName]);
            Vec<std::string> curOpOutputTensorName = std::get<1>(this->operatorTensorMap[opName]);
            Vec<GCLMemDesc> inGCLMemDescs;
            Vec<GCLMemDesc> outGCLMemDescs;
            U32 j;

            for (j = 0; j < curOpOutputTensorName.size(); j++) {
                U32 stride[3] = {0, 0, 0};
                U32 offset[3] = {0, 0, 0};
                GCLMemDesc gclTmpDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
                outGCLMemDescs.push_back(gclTmpDesc);
            }        
            
            for (std::string inputTensorName: curOpInputTensorName) {
                auto mem =(OclMemory*)(this->tensorMap[inputTensorName]->get_memory());
                auto desc = mem->get_mem_desc();
                inGCLMemDescs.push_back(desc);
            }
            CHECK_STATUS(op->infer_gclmem_desc(&inGCLMemDescs, &outGCLMemDescs));

            j = 0;
            for (std::string inputTensorName: curOpInputTensorName) {
                auto tensorTmp = this->tensorMap[inputTensorName];
                auto mem = (OclMemory*)(tensorTmp->get_memory());
                mem->set_mem_desc(inGCLMemDescs[j]);
                j++;
            }
            
            j = 0;
            for (std::string outputTensorName: curOpOutputTensorName) {
                auto tensorTmp = this->tensorMap[outputTensorName];
                auto mem = (OclMemory*)(tensorTmp->get_memory());
                mem->set_mem_desc(outGCLMemDescs[j]);
                j++;
            }
        }

        for (int i = 0; i < opsNum; i++) {
            std::string opName = sortedOps[i];
            auto op = this->operatorMap[opName];
#ifdef _DEBUG
            std::cout << funcStr << " op: " << opName << " type " << op->get_op_type() << std::endl;
#endif
            Vec<std::string> curOpInputTensorName = std::get<0>(this->operatorTensorMap[opName]);
            Vec<std::string> curOpOutputTensorName = std::get<1>(this->operatorTensorMap[opName]);
            Vec<Tensor> inTensors, outTensors;
            for (std::string inputTensorName: curOpInputTensorName) {
                auto tensorTmp = this->tensorMap[inputTensorName];
                inTensors.push_back(*tensorTmp.get());
#ifdef _DEBUG
                auto mem = (OclMemory*)(tensorTmp->get_memory());
                auto desc = mem->get_mem_desc();
                std::cout << "    inputTensorName:  " << inputTensorName << " ";
                std::cout << gclMemDesc2Str(desc) << std::endl;
#endif
            }

            for (std::string outputTensorName: curOpOutputTensorName) {
                auto tensorTmp = this->tensorMap[outputTensorName];
                outTensors.push_back(*tensorTmp.get());
#ifdef _DEBUG
                auto mem = (OclMemory*)(tensorTmp->get_memory());
                auto desc = mem->get_mem_desc();
                std::cout << "    outputTensorName: " << outputTensorName << " ";
                std::cout << gclMemDesc2Str(desc) << std::endl;
#endif
            }
            op->set_input_output_tensors(inTensors, outTensors);
        }    
        return SUCCESS;
    }
#endif    

    void assign_output_tensor() override
    {
#ifdef _DEBUG
        const char* funcStr = "[DEBUG] assign_output_tensor()";
        std::cout << funcStr << std::endl;
#endif

        Vec<std::shared_ptr<U8>> storages;
#ifdef _USE_MALI
        if(this->schedule != MALI) {
#endif        
            for (U32 i = 0; i < storageSizes.size(); i++) {
                storages.push_back(std::shared_ptr<U8>((U8*)operator new(storageSizes[i])));
            }
#ifdef _USE_MALI
        }
#endif        

        for (std::string opName: sortedOps) {
#ifdef _DEBUG
            std::cout << funcStr << " op: " << opName << "\n    input  tensor names: ";
#endif
            U32 tensorIter = 0;
            std::shared_ptr<Operator> op = this->operatorMap[opName];
            Vec<Tensor> inTensors, outTensors;
            Vec<std::string> inTensorNames = std::get<0>(this->operatorTensorMap[opName]);
            Vec<std::string> outTensorNames = std::get<1>(this->operatorTensorMap[opName]);
            for (std::string inName: inTensorNames) {
#ifdef _DEBUG
                std::cout << inName << " to Slot " << op->tensorPos[tensorIter] << ", ";
#endif
#ifdef _USE_MALI
                if(this->schedule != MALI) {
#endif                
                    if (op->tensorPos[tensorIter] != -1) {
                        this->tensorMap[inName].get()->set_shared_ptr(storages[op->tensorPos[tensorIter]]);
                    } else {
                        if (this->weightOpOutputNames.find(inName) == this->weightOpOutputNames.end()) {
                            this->tensorMap[inName].get()->alloc();
			}
                    }
                    tensorIter++;
#ifdef _USE_MALI                    
                }
#endif
                inTensors.push_back(*(this->tensorMap[inName].get()));
            }
#ifdef _DEBUG
            std::cout << "\n    output tensor names: ";
#endif

            for (std::string outName: outTensorNames) {
#ifdef _DEBUG
                std::cout << outName << " to Slot " << op->tensorPos[tensorIter] << ", ";
#endif
#ifdef _USE_MALI
                if(this->schedule == MALI) {
                    this->tensorMap[outName].get()->alloc();
                } else {
#endif                
                    if (this->weightOpOutputNames.find(outName) == this->weightOpOutputNames.end()) {
                        if (op->tensorPos[tensorIter] != -1) {
                            this->tensorMap[outName].get()->set_shared_ptr(storages[op->tensorPos[tensorIter]]);
                        } else {
                            this->tensorMap[outName].get()->alloc();
                        }
                    }
                    tensorIter++;
#ifdef _USE_MALI                
                }
#endif                
                outTensors.push_back(*(this->tensorMap[outName].get()));
            }
#ifdef _DEBUG
            std::cout << std::endl;
#endif
            op->set_input_output_tensors(inTensors, outTensors);
        }
        this->memoryAssigned = true;
    }

private: 
    void add(std::shared_ptr<Operator> op, Vec<std::string> inputTensorsName, Vec<std::string> outputTensorsName)
    {
        std::string name = op->get_name();
        this->operatorMap[name] = op;

        std::tuple<Vec<std::string>, Vec<std::string>> in_outTensors(std::make_tuple(inputTensorsName, outputTensorsName));
        if (this->operatorTensorMap.find(name) == this->operatorTensorMap.end()) {
            this->operatorTensorMap[name] = in_outTensors;
        }
        else {
            std::cout << "[ERROR] duplicate tensor name: " << name << std::endl;
            exit(1);
        }
        this->operatorTensorMap[name] = in_outTensors;

        for (std::string input : inputTensorsName) {
	    std::shared_ptr<Tensor> tmp;
#ifdef _USE_MALI
	    if(this->schedule == MALI) {
                tmp = std::shared_ptr<Tensor>(new Tensor(this->handle));
	    } else {
#endif
                tmp = std::shared_ptr<Tensor>(new Tensor());
#ifdef _USE_MALI
	    }
#endif
            auto p = std::pair<std::string, std::shared_ptr<Tensor>>(input, tmp);
            this->tensorMap.insert(p);
        }

        for (std::string output : outputTensorsName) {
	    std::shared_ptr<Tensor> tmp;
#ifdef _USE_MALI
	    if(this->schedule == MALI) {
                tmp = std::shared_ptr<Tensor>(new Tensor(this->handle));
	    } else {
#endif
                tmp = std::shared_ptr<Tensor>(new Tensor());
#ifdef _USE_MALI
	    }
#endif
            auto p = std::pair<std::string, std::shared_ptr<Tensor>>(output, tmp);
            this->tensorMap.insert(p);
        }
    }

    void set_input_tensors_desc(HashMap<std::string, TensorDesc> inputDescMap) {
        for (auto iter: inputDescMap) {
            TensorDesc desc = iter.second;
#ifdef _USE_MALI
            if(this->schedule == MALI) {
                if(desc.df == DF_NCHW)  desc.df = DF_NCHW_ORG_MALI;
            }
#endif
            (this->tensorMap[iter.first].get())->set_desc(desc);
        }
    }


    void infer_tmp_memory_size() override
    {
        this->tmpElements.clear();
        this->maxTmpElements = 0;

        for (auto op: this->ops) {
            auto len = op->infer_tmp_memory_size();
            this->tmpElements.push_back(len);
            if (len > (this->maxTmpElements)) {
                this->maxTmpElements = len;
            }
        }
    }

    void assign_tmp_tensor() override
    {
        // design for serial , if parallel running should redesign
#ifdef _USE_MALI
        if(this->schedule == MALI) {
            this->temp = std::shared_ptr<Memory_>(new OclMemory(this->handle));
        } else {
#endif        
            this->temp = std::shared_ptr<Memory_>(new CpuMemory());
#ifdef _USE_MALI
        }
#endif       
        temp->alloc(this->maxTmpElements);
        for (auto op: this->ops) {
            op->set_tmp_memory(this->maxTmpElements, temp);
        }
    }


private:
    HashMap<std::string, std::shared_ptr<Tensor>> tensorMap;
    HashMap<std::string, std::shared_ptr<Operator>> operatorMap;
    HashMap<std::string, std::tuple<Vec<std::string>, Vec<std::string>>> operatorTensorMap;

    std::set<std::string> weightOpOutputNames;

    //input & output tensors    
    HashMap<std::string, std::shared_ptr<Tensor>> inputTensors;
    HashMap<std::string, std::shared_ptr<Tensor>> outputTensors;
#ifdef _USE_MALI
    HashMap<std::string, std::shared_ptr<Tensor>> inputTensorsHost; 
#endif
    Vec<U32> storageSizes;

    Vec<std::string> sortedOps;

    U32 maxTmpElements;
    Vec<U32> tmpElements;
    std::shared_ptr<Memory_> temp;

    Vec<TensorDesc> modelInDims;

    Vec<std::string> modelInputTensorNames;
    Vec<std::string> modelOutputTensorNames;
};
#endif
