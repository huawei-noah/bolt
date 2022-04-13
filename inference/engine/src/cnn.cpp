// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <unordered_map>
#include "cnn.h"
#ifdef _USE_CPU
#include "cpu/factory_cpu.hpp"
#endif
#ifdef _USE_GPU
#include "ocl/factory_ocl.hpp"
#endif
#include "profiling.h"

bool is_same_tensor(Tensor a, Tensor b)
{
    auto ptr_a = ((CpuMemory *)a.get_memory())->get_ptr();
    auto ptr_b = ((CpuMemory *)b.get_memory())->get_ptr();
    bool ret;
    if (ptr_a != nullptr && ptr_a == ptr_b) {
        ret = true;
    } else {
        ret = false;
    }
    return ret;
}

void CNN::check_dynamic_output_size(OperatorType type)
{
    std::set<OperatorType> types = {OT_Shape, OT_NonMaxSuppression};
    if (types.find(type) != types.end()) {
        this->dynamicOutputSize = true;
    }
    if (type == OT_Shape) {
        UNI_WARNING_LOG("model contains Shape operator, this will use dynamic output size "
                        "inference(may encounter error). If you don't want to use it, you can use "
                        "onnx-simplifier to simplify original onnx model.\n");
        if (IS_GPU(this->deviceInfo.schedule)) {
            UNI_ERROR_LOG("gpu currently not support dynamic output size inference.\n");
        }
    }
}

CNN CNN::clone()
{
    CNN cnn = *this;
    for (U32 i = 0; i < cnn.ops.size(); i++) {
        cnn.ops[i] = cnn.ops[i]->clone();
        cnn.operatorMap[cnn.ops[i]->get_name()] = cnn.ops[i];
    }
    for (auto &tensor : cnn.tensorMap) {
        std::shared_ptr<Tensor> cloneTensor = std::shared_ptr<Tensor>(new Tensor());
        *cloneTensor = tensor.second->clone(false);
        tensor.second = cloneTensor;
    }
    std::map<std::string, TensorDesc> inputDescMap;
    for (auto &iter : cnn.inputTensors) {
        inputDescMap[iter.first] = iter.second->get_desc();
    }
    cnn.infer_output_tensors_size(inputDescMap);
    cnn.assign_output_tensor();
    cnn.tmpTensor = cnn.tmpTensor.clone();
    cnn.infer_tmp_memory_size();
    cnn.tmpTensor.alloc();
    for (auto &operatorTensor : cnn.operatorTensorMap) {
        std::string operatorName = operatorTensor.first;
        std::vector<std::vector<Tensor>> tensors(operatorTensor.second.size());
        for (U32 i = 0; i < operatorTensor.second.size(); i++) {
            for (U32 j = 0; j < operatorTensor.second[i].size(); j++) {
                std::string tensorName = operatorTensor.second[i][j];
                if (cnn.weightOpOutputNames.find(tensorName) != cnn.weightOpOutputNames.end()) {
                    cnn.tensorMap[tensorName] = this->tensorMap[tensorName];
                }
                tensors[i].push_back(*(cnn.tensorMap[tensorName].get()));
            }
        }
        cnn.operatorMap[operatorName]->set_input_output_tensors(tensors[0], tensors[1]);
        cnn.operatorMap[operatorName]->set_tmp_memory(cnn.tmpTensor);
    }
    for (auto &tensor : cnn.inputTensors) {
        tensor.second = cnn.tensorMap[tensor.first];
    }
    for (auto &tensor : cnn.outputTensors) {
        tensor.second = cnn.tensorMap[tensor.first];
    }

    // check
    CHECK_REQUIREMENT(!is_same_tensor(this->tmpTensor, cnn.tmpTensor));
    for (U32 i = 0; i < this->storageMemory.size(); i++) {
        CHECK_REQUIREMENT(
            !is_same_tensor(*(this->storageMemory[i].get()), *(cnn.storageMemory[i].get())));
    }
    for (auto &iter : this->tensorMap) {
        if (cnn.weightOpOutputNames.find(iter.first) == cnn.weightOpOutputNames.end()) {
            CHECK_REQUIREMENT(
                !is_same_tensor(*(iter.second.get()), *(cnn.tensorMap[iter.first].get())));
        }
    }
    for (auto &iter : this->inputTensors) {
        CHECK_REQUIREMENT(
            !is_same_tensor(*(iter.second.get()), *(cnn.inputTensors[iter.first].get())));
    }
    for (auto &iter : this->outputTensors) {
        CHECK_REQUIREMENT(
            !is_same_tensor(*(iter.second.get()), *(cnn.outputTensors[iter.first].get())));
    }
    for (auto &iter : this->operatorMap) {
        std::shared_ptr<Operator> op1 = iter.second;
        std::shared_ptr<Operator> op2 = cnn.operatorMap[iter.first];
        for (int i = 0; i < 2; i++) {
            std::vector<std::string> names = this->operatorTensorMap[iter.first][i];
            std::vector<Tensor> tensor1, tensor2;
            if (i == 0) {
                tensor1 = op1->get_input_tensors();
                tensor2 = op2->get_input_tensors();
            } else {
                tensor1 = op1->get_output_tensors();
                tensor2 = op2->get_output_tensors();
            }
            CHECK_REQUIREMENT(tensor1.size() == tensor2.size());
            for (U32 j = 0; j < tensor1.size(); j++) {
                if (tensor1[j].bytes() != 0) {
                    CHECK_REQUIREMENT(
                        is_same_tensor(tensor1[j], *(this->tensorMap[names[j]].get())));
                    CHECK_REQUIREMENT(is_same_tensor(tensor2[j], *(cnn.tensorMap[names[j]].get())));
                    if (this->weightOpOutputNames.find(names[j]) == this->weightOpOutputNames.end()) {
                        CHECK_REQUIREMENT(!is_same_tensor(tensor1[j], tensor2[j]));
                    }
                }
            }
        }
    }
    return cnn;
}

void CNN::sort_operators_sequential(const ModelSpec *ms)
{
    int opNum = ms->num_operator_specs;
    this->sortedOps.clear();
    for (int i = 0; i < opNum; i++) {
        std::string opName = ms->ops[i].name;
        if (opName.compare("data") == 0) {
            continue;
        }
        this->sortedOps.push_back(opName);
    }
}

void CNN::initialize_ops(const ModelSpec *ms)
{
    UNI_DEBUG_LOG("Initialize inference...\n");
    int opNum = ms->num_operator_specs;

    for (int i = 0; i < ms->num_inputs; i++) {
        this->inputTensors[ms->input_names[i]] = this->allocate_tensor();
        this->inputTensors[ms->input_names[i]]->resize(ms->input_dims[i]);
        this->tensorMap[ms->input_names[i]] = this->allocate_tensor();
    }
    for (int i = 0; i < ms->num_outputs; i++) {
        this->outputTensors[ms->output_names[i]] = this->allocate_tensor();
    }

    U32 operatorIndex = 0;
    std::map<std::string, U32> operatorIndexMap;
    for (int i = 0; i < opNum; i++) {
        OperatorSpec curOps = ms->ops[i];
        std::string opName = curOps.name;
        if (opName.compare("data") == 0) {
            continue;
        }
        operatorIndexMap[opName] = operatorIndex++;
    }

    std::shared_ptr<Factory> factory;
    if (IS_GPU(this->deviceInfo.schedule)) {
#ifdef _USE_GPU
        factory = std::shared_ptr<Factory>(new FactoryOCL());
        this->tmpTensor = Tensor(OCLMem);
#else
        UNI_ERROR_LOG("This library not support ARM GPU, please rebuild library with --gpu "
                      "option.\n");
        exit(1);
#endif
    } else {
        factory = std::shared_ptr<Factory>(new FactoryCPU());
        this->tmpTensor = Tensor();
    }

    for (int i = 0; i < opNum; i++) {
        OperatorSpec curOps = ms->ops[i];
        std::string opName = curOps.name;
        UNI_DEBUG_LOG("create operator:%s type:%s.\n", curOps.name, OperatorTypeName()[curOps.type]);
        if (opName.compare("data") == 0) {
            continue;
        }
        this->check_dynamic_output_size(curOps.type);
        std::vector<std::string> inputTensorsName;
        std::vector<std::string> outputTensorsName;
        for (U32 j = 0; j < curOps.num_inputs; j++) {
            inputTensorsName.push_back(curOps.input_tensors_name[j]);
        }
        for (U32 j = 0; j < curOps.num_outputs; j++) {
            outputTensorsName.push_back(curOps.output_tensors_name[j]);
        }
        // create op object
        std::shared_ptr<Operator> op = factory->createOperators(curOps, this->dt, operatorIndexMap,
            &this->tensorMap, inputTensorsName, outputTensorsName, &weightOpOutputNames);
        // setup operatorMap, tensorMap, operatorTensorMap
        op->set_name(opName);
        this->add(op, inputTensorsName, outputTensorsName);
        this->set_op_tensors_positions(
            op, curOps.tensor_positions, inputTensorsName, outputTensorsName);
        op->set_schedule(this->deviceInfo.schedule);
        op->init_feature_scale(curOps.num_quant_feature, curOps.feature_scale);
        op->set_algorithm_map(this->algorithmMap);
        this->ops.push_back(op);
    }

    // setup WeightSpec ptr in WeightOperator
    for (int i = 0; i < ms->num_weight_specs; i++) {
        WeightSpec curOpWs = ms->ws[i];
        std::string opName = curOpWs.op_name;
        UNI_DEBUG_LOG("set operator:%s's weight parameter.\n", curOpWs.op_name);
        if (this->operatorMap.find(opName) == this->operatorMap.end()) {
            UNI_WARNING_LOG("unsed weight %s in model.\n", opName.c_str());
            continue;
        }
        auto op = this->operatorMap[opName];
        auto weightOp = dynamic_cast<WeightOperator *>(op.get());
        weightOp->set_weightspec_ptr(curOpWs);
        if (curOpWs.bytes_of_vec != 0) {
            CHECK_REQUIREMENT(curOpWs.vec != nullptr);
            weightOp->set_hasBias(true);
        }
    }
    UNI_DEBUG_LOG("Initialize inference end.\n");
}

void CNN::ready(std::map<std::string, TensorDesc> inputDescMap)
{
    UNI_DEBUG_LOG("Inference ready...\n");
    UNI_PROFILE(
        {
            this->infer_output_tensors_size(inputDescMap);
            // handle the weight ops
            for (auto &op : this->ops) {
                if (op->is_weight()) {
                    UNI_DEBUG_LOG("op: %s init weight\n", op->get_name().c_str());
                    auto weightOpPtr = dynamic_cast<WeightOperator *>(op.get());
                    CHECK_STATUS(weightOpPtr->init_weight_bias_from_model());
                }
                UNI_DEBUG_LOG("op: %s infer forward algorithm\n", op->get_name().c_str());
                //need process for qualcomm
                CHECK_STATUS(op->infer_forward_algorithm(this->algorithmMap));
            }

            this->infer_tmp_memory_size();
            this->assign_tmp_tensor();
            // transform filter
            for (auto &op : this->ops) {
                if (op->is_weight()) {
                    UNI_DEBUG_LOG("op: %s transform filter\n", op->get_name().c_str());
                    auto weightOpPtr = dynamic_cast<WeightOperator *>(op.get());
                    CHECK_STATUS(weightOpPtr->transform_filter());
                }
            }
            this->infer_tmp_memory_size();
            this->tmpTensor.alloc();
            this->assign_output_tensor();
        },
        std::string("ready"), std::string("prepare"));
    UNI_DEBUG_LOG("Inference ready end.\n");
}

void CNN::reready(std::map<std::string, TensorDesc> inputDescMap)
{
    UNI_DEBUG_LOG("Inference reready for dynamic input...\n");
    this->infer_output_tensors_size(inputDescMap);
    if (this->memoryTracker.getMemoryNeedAssign()) {
        this->assign_output_tensor();
    }
    this->infer_tmp_memory_size();
    this->tmpTensor.alloc();
    UNI_DEBUG_LOG("Inference reready end.\n");
}

EE CNN::mark_input_output()
{
    for (auto &iter : this->inputTensors) {
        std::string str = iter.first;
        if (tensorMap.find(str) != tensorMap.end()) {
            this->inputTensors[str] = tensorMap[str];
        } else {
            UNI_ERROR_LOG(
                "can not find tensor(name: %s) to be marked as model input.\n", str.c_str());
            return NOT_MATCH;
        }
    }
    for (auto &iter : this->outputTensors) {
        std::string str = iter.first;
        if (tensorMap.find(str) != tensorMap.end()) {
            this->outputTensors[str] = tensorMap[str];
        } else {
            UNI_ERROR_LOG("can not find tensor(name: %s) to be marked as model output. Maybe this "
                          "tensor is removed by graph optimizer.\n",
                str.c_str());
            return NOT_MATCH;
        }
    }
    return SUCCESS;
}

void CNN::set_input_by_copy(std::map<std::string, U8 *> modelTensorsInput)
{
    UNI_DEBUG_LOG("Copy input...\n");
    for (auto &modelTensorInput : modelTensorsInput) {
        std::string inputName = modelTensorInput.first;
        UNI_DEBUG_LOG("    Copy input %s...\n", inputName.c_str());
        U8 *data = modelTensorInput.second;
        if (this->inputTensors.find(inputName) == this->inputTensors.end()) {
            UNI_ERROR_LOG("Can not find input:%s to set.\n", inputName.c_str());
            return;
        }
        auto tensorPtr = this->inputTensors[inputName];
        Tensor input;
        input.resize(tensorPtr->get_desc());
        std::shared_ptr<U8> shared_data(data, [](U8 *ptr) {});
        ((CpuMemory *)(input.get_memory()))->set_shared_ptr(shared_data);
        UNI_PROFILE(
            { tensorPtr->copy_from(&input); }, "copy " + inputName, std::string("input::copy"));
        UNI_DEBUG_LOG("    Copy input: %s %s\n", inputName.c_str(), tensorPtr->string(8).c_str());
    }
    UNI_DEBUG_LOG("Copy input end.\n");
}

void CNN::set_input_by_assign(std::map<std::string, std::shared_ptr<U8>> modelTensorsInput)
{
    UNI_DEBUG_LOG("Set input...\n");
    for (auto &modelTensorInput : modelTensorsInput) {
        std::string inputName = modelTensorInput.first;
        std::shared_ptr<U8> data = modelTensorInput.second;
        if (this->inputTensors.find(inputName) == this->inputTensors.end()) {
            UNI_ERROR_LOG("Can not find input:%s to set.\n", inputName.c_str());
            return;
        }
        auto tensorPtr = this->inputTensors[inputName];
        UNI_PROFILE(
            {
                if (data != ((CpuMemory *)(tensorPtr->get_memory()))->get_shared_ptr()) {
                    Tensor input;
                    input.resize(tensorPtr->get_desc());
                    ((CpuMemory *)(input.get_memory()))->set_shared_ptr(data);
                    tensorPtr->reuse(&input);
                }
            },
            "copy " + inputName, std::string("input::copy"));
        UNI_DEBUG_LOG("    Set input: %s %s\n", inputName.c_str(), tensorPtr->string(8).c_str());
    }
    UNI_DEBUG_LOG("Set input end.\n");
}

std::map<std::string, std::shared_ptr<Tensor>> CNN::get_input()
{
    std::map<std::string, std::shared_ptr<Tensor>> ret;
    if (IS_GPU(this->deviceInfo.schedule)) {
#ifdef _USE_GPU
        for (auto &iter : this->inputTensors) {
            std::shared_ptr<Tensor> tmpTensorCPU(new Tensor());
            tmpTensorCPU->resize(iter.second->get_desc());
            tmpTensorCPU->alloc();
            ret[iter.first] = tmpTensorCPU;
        }
#endif
    } else {
        ret = this->inputTensors;
    }
    return ret;
}

std::map<std::string, std::shared_ptr<Tensor>> CNN::get_output()
{
    return this->outputTensors;
}

Tensor CNN::get_tensor_by_name(std::string tensorName)
{
    Tensor ret;
    if (this->tensorMap.find(tensorName) == this->tensorMap.end()) {
        UNI_ERROR_LOG("Can not find output:%s to get.\n", tensorName.c_str());
    } else {
        ret = *(this->tensorMap[tensorName].get());
    }
    return ret;
}

TensorDesc CNN::get_tensor_desc_by_name(std::string tensorName)
{
    TensorDesc desc = tensor4d(DT_U8, 0, 0, 0, 0);
    if (this->tensorMap.find(tensorName) != this->tensorMap.end()) {
        desc = this->tensorMap[tensorName]->get_desc();
    }
    return desc;
}

std::map<std::string, TensorDesc> CNN::get_input_desc()
{
    std::map<std::string, TensorDesc> descs;
    for (auto &iter : this->inputTensors) {
        descs[iter.first] = iter.second->get_desc();
    }
    return descs;
}

std::map<std::string, TensorDesc> CNN::get_output_desc()
{
    std::map<std::string, TensorDesc> descs;
    for (auto iter : this->outputTensors) {
        descs[iter.first] = iter.second->get_desc();
    }
    return descs;
}

void CNN::update_tensor_positions()
{
    std::unordered_map<std::string, I32> m;
    for (auto &opName : this->sortedOps) {
        auto op = this->operatorMap[opName];
        if (op->get_type() == OT_Reshape) {
            std::vector<std::string> curOpInputTensorName = this->operatorTensorMap[opName][0];
            std::vector<std::string> curOpOutputTensorName = this->operatorTensorMap[opName][1];
            auto tensor = this->tensorMap[curOpInputTensorName[0]];
            if ((tensor->get_desc().df != DF_NCHWC8) &&
                (tensor->get_desc().df != DF_NCHWC16))
            {
                std::vector<I32> tensorPositions = op->get_tensor_positions();
                m[curOpInputTensorName[0]] = m[curOpOutputTensorName[0]] = tensorPositions[0] = -1;
                tensorPositions[1] = -3;
                // when slot is -3, reuse the input tensor mem.
                op->set_tensor_positions(tensorPositions);
            }
        }
    }
    if (!m.empty()) {
        for (auto &opName : this->sortedOps) {
            auto op = this->operatorMap[opName];
            std::vector<I32> tensorPositions = op->get_tensor_positions();
            if (tensorPositions.size() > 1 && tensorPositions[1] == -3) {
                continue;
            }
            bool update = false;
            for (U32 i = 0, tensorIter = 0; i < 2; ++i) {
                U32 iterSize = this->operatorTensorMap[opName][i].size();
                for (U32 j = 0; j < iterSize; ++j) {
                    std::string tensorName = this->operatorTensorMap[opName][i][j];
                    if (m.count(tensorName)) {
                        tensorPositions[tensorIter] = m[tensorName];
                        update = true;
                    }
                    ++tensorIter;
                }
            }
            if (update) {
                op->set_tensor_positions(tensorPositions);
            }
        }
    }
}

EE CNN::infer_output_tensors_size(std::map<std::string, TensorDesc> inputDescMap)
{
    UNI_DEBUG_LOG("Infer tensor dimension...\n");
    this->set_input_desc(inputDescMap);
    for (auto &iter : inputDescMap) {
        UNI_DEBUG_LOG(
            "model input: %s desc %s\n", iter.first.c_str(), tensorDesc2Str(iter.second).c_str());
    }
    this->infer_layout_desc();
#ifndef _USE_GPU
    this->update_tensor_positions();
#endif
    this->update_op_tensors();
    UNI_DEBUG_LOG("Infer tensor dimension end.\n");
    return SUCCESS;
}

void CNN::assign_output_tensor()
{
    this->storageMemory.clear();
    auto storageSize = this->memoryTracker.getStorageSize();
    for (U32 size : storageSize) {
        auto tensor = this->allocate_tensor(size);
        this->storageMemory.push_back(tensor);
    }
#ifdef _USE_GPU
    this->storageImage.clear();
    auto oclImageSize = this->memoryTracker.getOclImageSize();
    for (auto &p : oclImageSize) {
        auto slot = p.first;
        auto imageSizes = p.second;
        std::vector<std::shared_ptr<Tensor>> tensors;
        for (auto &size : imageSizes) {
            auto tensor = this->allocate_tensor(size.data());
            tensors.push_back(tensor);
        }
        this->storageImage[slot] = tensors;
    }
#endif

    for (std::string opName : this->sortedOps) {
        std::shared_ptr<Operator> op = this->operatorMap[opName];
        std::vector<I32> tensorPositions = op->get_tensor_positions();
        std::vector<std::vector<Tensor>> tensors(this->operatorTensorMap[opName].size());
        for (U32 i = 0, tensorIter = 0; i < this->operatorTensorMap[opName].size(); i++) {
            std::vector<std::string> &tensorNames = this->operatorTensorMap[opName][i];
            for (std::string &tensorName : tensorNames) {
                //UNI_DEBUG_LOG("Reuse tensor %s slot %d\n", tensorName.c_str(),
                //    tensorPositions[tensorIter]);
                auto tensor = this->tensorMap[tensorName];
                bool needAssign = true;
                if (i == 0 && (this->inputTensors.find(tensorName) == this->inputTensors.end())) {
                    needAssign = false;
                }
                if (this->weightOpOutputNames.find(tensorName) != this->weightOpOutputNames.end()) {
                    needAssign = false;
                }
                if (needAssign) {
                    I32 slot = tensorPositions[tensorIter];
                    if (slot >= 0) {
                        tensor->reuse(get_reuse_memory(slot, tensor.get()));

                    } else if (slot == -1) {
                        tensor->alloc();
#ifdef _USE_GPU
                    } else if (slot == -2) {
                        auto mem = (OclMemory *)tensor->get_memory();
                        mem->mapped_alloc();
#endif
                    } else if (slot == -3) {
                        // when slot is -3, reuse the input tensor mem.
                        tensor->reuse(&(tensors[0][0]));
                    }
                }
                tensorIter++;
                tensors[i].push_back(*(tensor.get()));
            }
        }
        op->set_input_output_tensors(tensors[0], tensors[1]);
    }
    this->memoryTracker.setMemoryAssigned();
}

void CNN::run()
{
    for (U32 opIndex = 0; opIndex < ops.size();) {
        std::shared_ptr<Operator> op = this->ops[opIndex];
        UNI_DEBUG_LOG(
            "Run op: %s type: %s\n", op->get_name().c_str(), OperatorTypeName()[op->get_type()]);
        if (op->get_type() == OT_Repeat || op->get_type() == OT_Jump) {
            opIndex = op->get_next_operator_index();
        } else {
            if (this->dynamicOutputSize) {
                std::vector<Tensor> inputs = op->get_input_tensors();
                std::vector<Tensor> outputs = op->get_output_tensors();
                std::vector<Tensor *> in, out;
                for (U32 i = 0; i < inputs.size(); i++) {
                    in.push_back(&inputs[i]);
                }
                for (U32 i = 0; i < outputs.size(); i++) {
                    out.push_back(&outputs[i]);
                }
                op->infer_output_tensors_size(in, out);
            }
#ifdef _DEBUG
            std::vector<Tensor> inputTensors = op->get_input_tensors();
            std::vector<std::string> inputNames = operatorTensorMap[op->get_name()][0];
            for (U32 i = 0; i < inputTensors.size(); i++) {
                Tensor inputTensor = inputTensors[i];
                std::string line = inputTensor.string(8);
                UNI_DEBUG_LOG("    input:%s %s\n", inputNames[i].c_str(), line.c_str());
            }
#endif
            UNI_PROFILE(
                {
                    op->run();
#if defined(_USE_GPU) && defined(_PROFILE)
                    if (IS_GPU(this->deviceInfo.schedule)) {
                        gcl_finish(OCLContext::getInstance().handle.get());
                    }
#endif
                },
                op->get_name(),
                std::string(OperatorTypeName()[op->get_type()]) + std::string("::run"));
            opIndex++;
        }
#ifdef _DEBUG
        std::vector<Tensor> outputTensors = op->get_output_tensors();
        std::vector<std::string> outputNames = operatorTensorMap[op->get_name()][1];
        for (U32 i = 0; i < outputTensors.size(); i++) {
            Tensor outputTensor = outputTensors[i];
            std::string line = outputTensor.string(8);
            UNI_DEBUG_LOG("    output:%s %s\n", outputNames[i].c_str(), line.c_str());
        }
#endif
    }
}

std::shared_ptr<Tensor> CNN::allocate_tensor(U32 size)
{
    MemoryType type = CPUMem;
    if (IS_GPU(this->deviceInfo.schedule)) {
        type = OCLMem;
    }
    std::shared_ptr<Tensor> tensor = std::shared_ptr<Tensor>(new Tensor(type));
    tensor->resize(tensor1d(DT_U8, size));
    tensor->alloc();
    return tensor;
}

#ifdef _USE_GPU
std::shared_ptr<Tensor> CNN::allocate_tensor(U32 *size)
{
    std::shared_ptr<Tensor> tensor = std::shared_ptr<Tensor>(new Tensor(OCLMemImg));
    auto mem = (OclMemoryImg *)tensor->get_memory();
    mem->alloc(size[0], size[1], size[2]);
    return tensor;
}
#endif

void CNN::add(std::shared_ptr<Operator> op,
    std::vector<std::string> &inputTensorsName,
    std::vector<std::string> &outputTensorsName)
{
    std::string operatorName = op->get_name();
    this->operatorMap[operatorName] = op;

    if (this->operatorTensorMap.find(operatorName) == this->operatorTensorMap.end()) {
        this->operatorTensorMap[operatorName] = {inputTensorsName, outputTensorsName};
    } else {
        UNI_ERROR_LOG("duplicate tensor: %s\n", operatorName.c_str());
    }

    for (std::string &inputName : inputTensorsName) {
        if (this->tensorMap.find(inputName) == this->tensorMap.end()) {
            this->tensorMap[inputName] = this->allocate_tensor();
        }
    }

    for (std::string &outputName : outputTensorsName) {
        if (this->tensorMap.find(outputName) == this->tensorMap.end()) {
            this->tensorMap[outputName] = this->allocate_tensor();
        }
    }
}

void CNN::set_op_tensors_positions(std::shared_ptr<Operator> op,
    I32 *tensor_positions,
    std::vector<std::string> &inputTensorsName,
    std::vector<std::string> &outputTensorsName)
{
    U32 inputTensorsNum = inputTensorsName.size();
    U32 outputTensorsNum = outputTensorsName.size();
    U32 numTensors = inputTensorsNum + outputTensorsNum;
    std::vector<I32> tensorPositions(numTensors);
    UNI_MEMCPY(tensorPositions.data(), tensor_positions, numTensors * bytesOf(DT_I32));
    if (IS_GPU(this->deviceInfo.schedule)) {
        for (U32 j = 0; j < numTensors; j++) {
            std::string curTensorName;
            if (j < inputTensorsNum) {
                curTensorName = inputTensorsName[j];
            } else {
                curTensorName = outputTensorsName[j - inputTensorsNum];
            }
            if (this->inputTensors.find(curTensorName) != this->inputTensors.end() ||
                this->outputTensors.find(curTensorName) != this->outputTensors.end()) {
                tensorPositions[j] = -2;
            }
        }
    }
    op->set_tensor_positions(tensorPositions);
}

void CNN::infer_layout_desc()
{
    for (std::string &opName : this->sortedOps) {
        auto op = this->operatorMap[opName];
        UNI_DEBUG_LOG("op: %s type: %s\n", opName.c_str(), OperatorTypeName()[op->get_type()]);
        std::vector<std::string> curOpInputTensorName = this->operatorTensorMap[opName][0];
        std::vector<std::string> curOpOutputTensorName = this->operatorTensorMap[opName][1];
        std::vector<Tensor *> inputTensors;
        std::vector<Tensor *> outputTensors;
        for (std::string &inputTensorName : curOpInputTensorName) {
            auto tensor = this->tensorMap[inputTensorName].get();
            inputTensors.push_back(tensor);
            UNI_DEBUG_LOG("    input: %s desc %s\n", inputTensorName.c_str(),
                tensorDesc2Str(tensor->get_desc()).c_str());
        }
        for (std::string &outputTensorName : curOpOutputTensorName) {
            auto tensor = this->tensorMap[outputTensorName].get();
            outputTensors.push_back(tensor);
        }
        CHECK_STATUS(op->infer_output_tensors_size(inputTensors, outputTensors));
        for (std::string &outputTensorName : curOpOutputTensorName) {
            UNI_DEBUG_LOG("    output: %s desc %s\n", outputTensorName.c_str(),
                tensorDesc2Str(this->tensorMap[outputTensorName]->get_desc()).c_str());
        }
    }
}

void CNN::update_op_tensors()
{
    for (auto &opName : this->sortedOps) {
        auto op = this->operatorMap[opName];
        std::vector<std::string> curOpInputTensorName = this->operatorTensorMap[opName][0];
        std::vector<std::string> curOpOutputTensorName = this->operatorTensorMap[opName][1];
        std::vector<Tensor> inTensors, outTensors;
        for (std::string &inputTensorName : curOpInputTensorName) {
            auto tensorTmp = this->tensorMap[inputTensorName];
            inTensors.push_back(*tensorTmp.get());
        }

        for (std::string &outputTensorName : curOpOutputTensorName) {
            auto tensorTmp = this->tensorMap[outputTensorName];
            outTensors.push_back(*tensorTmp.get());
        }
        op->set_input_output_tensors(inTensors, outTensors);

        curOpInputTensorName.insert(
            curOpInputTensorName.end(), curOpOutputTensorName.begin(), curOpOutputTensorName.end());
        memoryTracker.trackOpTensorSizes(op, curOpInputTensorName);
    }
    check_memory_reuse_ratio();
}

void CNN::set_input_desc(std::map<std::string, TensorDesc> inputDescMap)
{
    for (auto &iter : inputDescMap) {
        if (tensorMap.find(iter.first) == tensorMap.end()) {
            UNI_WARNING_LOG("unused model input node: %s\n", iter.first.c_str());
            continue;
        }
        TensorDesc desc = iter.second;
        (this->tensorMap[iter.first].get())->resize(desc);
    }
}

void CNN::infer_tmp_memory_size()
{
    U32 tmpSize = this->tmpTensor.bytes();
    // input data format transform tmp buffer
#ifdef _USE_GPU
    if (IS_GPU(this->deviceInfo.schedule)) {
        for (auto &iter : this->inputTensors) {
            tmpSize = UNI_MAX(tmpSize, tensorNumBytes(iter.second->get_desc()));
        }
        for (auto &op : this->ops) {
            op->set_tmp_images(&this->tmpImages);
        }
    }
#endif

    // operator tmp buffer
    for (auto &op : this->ops) {
        auto len = op->infer_tmp_memory_size();
        tmpSize = UNI_MAX(tmpSize, len);
    }
    this->tmpTensor.resize(tensor1d(DT_U8, tmpSize));
}

void CNN::assign_tmp_tensor()
{
    this->tmpTensor.alloc();
#ifdef _USE_GPU
    if (IS_GPU(this->deviceInfo.schedule)) {
        this->tmpImages.alloc();
    }
#endif
    for (auto &op : this->ops) {
        op->set_tmp_memory(this->tmpTensor);
    }
}

Tensor *CNN::get_reuse_memory(U32 slot, Tensor *tensor)
{
    auto mem = tensor->get_memory();
    auto type = mem->get_mem_type();
    Tensor *reuseTensor = nullptr;
    if (type == CPUMem || type == OCLMem) {
        reuseTensor = this->storageMemory[slot].get();
#ifdef _USE_GPU
    } else {
        OclMemoryImg *img = (OclMemoryImg *)mem;
        U32 str[3];
        img->stride(str);
        I32 subSlot = this->memoryTracker.getOclImageSubSlotId(slot, str);
        reuseTensor = this->storageImage[slot][subSlot].get();
#endif
    }
    return reuseTensor;
}

void CNN::check_memory_reuse_ratio()
{
    U32 originalSize = 0;
    U32 standaloneSize = 0;
    for (auto &tensor : this->tensorMap) {
        U32 tensorSize = tensor.second->bytes();
        originalSize += tensorSize;
        if (weightOpOutputNames.find(tensor.first) != weightOpOutputNames.end()) {
            standaloneSize += tensorSize;
        }
    }
    UNI_DEBUG_LOG("tensor memory: originally %d tensors take %u bytes.\n",
        (int)this->tensorMap.size(), originalSize);
    UNI_DEBUG_LOG("tensor memory: now %u tensors take %u bytes, and %u bytes are reserved "
                  "for standalone tensors (e.g. loop topology). reuse rate: %f\n",
        this->memoryTracker.getNumSlots(), this->memoryTracker.getSizeSum(), standaloneSize,
        (F32)originalSize / (this->memoryTracker.getSizeSum() + standaloneSize));
}
