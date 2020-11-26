// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cnn.h"
#if defined(_USE_CPU)
#include "cpu/factory_cpu.hpp"
#endif
#ifdef _USE_MALI
#include "ocl/factory_ocl.hpp"
#endif

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
    cnn.assign_output_tensor();
    cnn.tmpTensor = this->tmpTensor.clone();
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
    for (auto iter : this->tensorMap) {
        if (cnn.weightOpOutputNames.find(iter.first) == cnn.weightOpOutputNames.end()) {
            CHECK_REQUIREMENT(
                !is_same_tensor(*(iter.second.get()), *(cnn.tensorMap[iter.first].get())));
        }
    }
    for (auto iter : this->inputTensors) {
        CHECK_REQUIREMENT(
            !is_same_tensor(*(iter.second.get()), *(cnn.inputTensors[iter.first].get())));
    }
    for (auto iter : this->outputTensors) {
        CHECK_REQUIREMENT(
            !is_same_tensor(*(iter.second.get()), *(cnn.outputTensors[iter.first].get())));
    }
    for (auto iter : this->operatorMap) {
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
    int opNum = ms->num_operator_specs;

    for (int i = 0; i < ms->num_inputs; i++) {
        this->modelInputTensorNames.push_back(ms->input_names[i]);
        this->modelInputTensorDescs.push_back(ms->input_dims[i]);
    }
    for (int i = 0; i < ms->num_outputs; i++) {
        this->modelOutputTensorNames.push_back(ms->output_names[i]);
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

    for (int i = 0; i < opNum; i++) {
        OperatorSpec curOps = ms->ops[i];
        std::string opName = curOps.name;
        if (opName.compare("data") == 0) {
            continue;
        }
        std::vector<std::string> inputTensorsName;
        std::vector<std::string> outputTensorsName;
        int inputTensorsNum = curOps.num_inputs;
        for (int j = 0; j < inputTensorsNum; j++) {
            inputTensorsName.push_back(curOps.input_tensors_name[j]);
        }

        int outputTensorsNum = curOps.num_outputs;
        for (int j = 0; j < outputTensorsNum; j++) {
            outputTensorsName.push_back(curOps.output_tensors_name[j]);
        }

        int numTensors = inputTensorsNum + outputTensorsNum;
        std::vector<I32> tensorPositions(numTensors);
        memcpy(tensorPositions.data(), curOps.tensor_positions, numTensors * bytesOf(DT_I32));
        // create op object
        std::shared_ptr<Factory> factory;
        if (this->deviceInfo.schedule == MALI) {
#ifdef _USE_MALI
            auto factory_ocl = (Factory *)(new FactoryOCL());
            factory = std::shared_ptr<Factory>(factory_ocl);
            for (int j = 0; j < outputTensorsNum; j++) {
                auto curOutputTensorName = outputTensorsName[j];
                for (auto modelOutputTensorName : modelOutputTensorNames) {
                    if (modelOutputTensorName == curOutputTensorName) {
                        tensorPositions[j + inputTensorsNum] = -1;
                    }
                }
            }
#endif
        } else {
            auto factory_cpu = (Factory *)(new FactoryCPU());
            factory = std::shared_ptr<Factory>(factory_cpu);
        }
        std::shared_ptr<Operator> op = factory->createOperators(curOps, this->dt, operatorIndexMap,
            &this->tensorMap, inputTensorsName, outputTensorsName, &weightOpOutputNames);
        op->set_name(opName);
        op->set_schedule(this->deviceInfo.schedule);
        op->set_tensor_positions(tensorPositions);
        op->init_feature_scale(curOps.num_quant_feature, curOps.feature_scale);
        op->set_algorithm_map(this->algorithmMap);
        this->ops.push_back(op);

        // setup operatorMap, tensorMap, operatorTensorMap
        this->add(op, inputTensorsName, outputTensorsName);
    }

    // setup WeightSpec ptr in WeightOperator
    for (int i = 0; i < ms->num_weight_specs; i++) {
        WeightSpec curOpWs = ms->ws[i];
        std::string opName = curOpWs.op_name;
        auto op = this->operatorMap[opName];
        auto weightOp = dynamic_cast<WeightOperator *>(op.get());
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

void CNN::ready(std::map<std::string, TensorDesc> inputDescMap)
{
    UNI_DEBUG_LOG("ready() schedule: %d\n", (int)(this->deviceInfo.schedule));
    UNI_PROFILE(
        {
            this->infer_output_tensors_size(inputDescMap);
            // handle the weight ops
            for (auto op : this->ops) {
                UNI_DEBUG_LOG("ready() op: %s init weight and infer forward algorithm\n",
                    op->get_name().c_str());
                if (op->is_weight()) {
                    auto weightOpPtr = dynamic_cast<WeightOperator *>(op.get());
                    CHECK_STATUS(weightOpPtr->init_weight_bias_from_model(nullptr));
                }
                CHECK_STATUS(op->infer_forward_algorithm(this->algorithmMap));
            }

            this->tmpTensor = *(this->allocate_tensor().get());
            this->infer_tmp_memory_size();
            this->assign_tmp_tensor();
            // transform filter
            for (auto op : this->ops) {
                UNI_DEBUG_LOG("ready() op: %s transform filter\n", op->get_name().c_str());
                if (op->is_weight()) {
                    auto weightOpPtr = dynamic_cast<WeightOperator *>(op.get());
                    CHECK_STATUS(weightOpPtr->transform_filter());
                }
            }
            this->infer_tmp_memory_size();
            this->tmpTensor.alloc();
            this->assign_output_tensor();
        },
        std::string("ready"), std::string("prepare"));
}

void CNN::reready(std::map<std::string, TensorDesc> inputDescMap)
{
    this->infer_output_tensors_size(inputDescMap);
    if (this->memoryTracker.getMemoryNeedAssign()) {
        this->assign_output_tensor();
    }
    this->infer_tmp_memory_size();
    this->tmpTensor.alloc();
}

EE CNN::mark_input_output()
{
    this->inputTensors.clear();
    for (U32 i = 0; i < this->modelInputTensorNames.size(); i++) {
        std::string str = this->modelInputTensorNames[i];
        if (tensorMap.find(str) != tensorMap.end()) {
            inputTensors[str] = tensorMap[str];
        } else {
            return NOT_MATCH;
        }
    }
    this->outputTensors.clear();
    for (U32 i = 0; i < this->modelOutputTensorNames.size(); i++) {
        std::string str = this->modelOutputTensorNames[i];
        if (tensorMap.find(str) != tensorMap.end()) {
            outputTensors[str] = tensorMap[str];
        } else {
            return NOT_MATCH;
        }
    }
    return SUCCESS;
}

void CNN::copy_to_named_input(std::string inputName, const U8 *data)
{
    if (inputTensors.find(inputName) == inputTensors.end()) {
        CHECK_STATUS(NOT_MATCH);
    }
    auto tensorPtr = this->inputTensors[inputName];
    Tensor input;
    input.resize(tensorPtr->get_desc());
    std::shared_ptr<U8> shared_data((U8 *)data, [](U8 *ptr) {});
    ((CpuMemory *)(input.get_memory()))->set_shared_ptr(shared_data);
    tensorPtr->copy_from(&input);
}

void CNN::set_input_tensors_value(std::map<std::string, std::shared_ptr<U8>> modelTensorsInput)
{
    for (auto &modelTensorInput : modelTensorsInput) {
        std::string inputName = modelTensorInput.first;
        std::shared_ptr<U8> data = modelTensorInput.second;
        if (inputTensors.find(inputName) == inputTensors.end()) {
            CHECK_STATUS(NOT_MATCH);
        }
        auto tensorPtr = this->inputTensors[inputName];
        Tensor input;
        input.resize(tensorPtr->get_desc());
        ((CpuMemory *)(input.get_memory()))->set_shared_ptr(data);
        tensorPtr->reuse(&input);
    }
}

std::map<std::string, std::shared_ptr<Tensor>> CNN::get_inputs()
{
    std::map<std::string, std::shared_ptr<Tensor>> ret;
    if (this->deviceInfo.schedule == MALI) {
#ifdef _USE_MALI
        for (U32 i = 0; i < modelInputTensorNames.size(); i++) {
            std::shared_ptr<Tensor> tmpTensorCPU(new Tensor());
            tmpTensorCPU->resize(modelInputTensorDescs[i]);
            tmpTensorCPU->alloc();
            auto p = std::pair<std::string, std::shared_ptr<Tensor>>(
                modelInputTensorNames[i], tmpTensorCPU);
            ret.insert(p);
        }
#endif
    } else {
        ret = this->inputTensors;
    }
    return ret;
}

std::map<std::string, std::shared_ptr<Tensor>> CNN::get_outputs()
{
    return this->outputTensors;
}

Tensor CNN::get_tensor_by_name(std::string tensorName)
{
    if (this->tensorMap.find(tensorName) == this->tensorMap.end()) {
        CHECK_STATUS(NOT_MATCH);
    }
    return *(this->tensorMap[tensorName].get());
}

TensorDesc CNN::get_tensor_desc_by_name(std::string tensorName)
{
    TensorDesc desc = tensor4d(DT_U8, 0, 0, 0, 0);
    if (this->tensorMap.find(tensorName) != this->tensorMap.end()) {
        desc = this->tensorMap[tensorName]->get_desc();
    }
    return desc;
}

std::vector<std::string> CNN::get_model_input_tensor_names()
{
    return this->modelInputTensorNames;
}

std::vector<TensorDesc> CNN::get_model_input_tensor_descs()
{
    return this->modelInputTensorDescs;
}

std::vector<std::string> CNN::get_model_output_tensor_names()
{
    return this->modelOutputTensorNames;
}

EE CNN::infer_output_tensors_size(std::map<std::string, TensorDesc> inputDescMap)
{
    this->set_input_tensors_desc(inputDescMap);
    for (auto iter : inputDescMap) {
        UNI_DEBUG_LOG("infer_output_tensors_size() model input: %s desc %s\n", iter.first.c_str(),
            tensorDesc2Str(iter.second).c_str());
    }
    this->infer_layout_desc();
    this->update_op_tensors();
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

    std::set<std::string> input_set(modelInputTensorNames.begin(), modelInputTensorNames.end());
    std::set<std::string> output_set(modelOutputTensorNames.begin(), modelOutputTensorNames.end());
    for (std::string opName : this->sortedOps) {
        std::shared_ptr<Operator> op = this->operatorMap[opName];
        std::vector<I32> tensorPositions = op->get_tensor_positions();
        std::vector<std::vector<Tensor>> tensors(this->operatorTensorMap[opName].size());
        for (U32 i = 0, tensorIter = 0; i < this->operatorTensorMap[opName].size(); i++) {
            std::vector<std::string> &tensorNames = this->operatorTensorMap[opName][i];
            for (std::string &tensorName : tensorNames) {
                UNI_DEBUG_LOG("assign_output_tensor() tensor %s slot %d\n", tensorName.c_str(),
                    tensorPositions[tensorIter]);
                auto tensor = this->tensorMap[tensorName];
                if (i == 1 || input_set.find(tensorName) != input_set.end()) {
                    if (tensorPositions[tensorIter] != -1) {
                        auto mem = this->storageMemory[tensorPositions[tensorIter]].get();
                        tensor->reuse(mem);
                    } else if (this->weightOpOutputNames.find(tensorName) ==
                        this->weightOpOutputNames.end()) {
                        if (this->deviceInfo.schedule == MALI &&
                            output_set.find(tensorName) != output_set.end()) {
#ifdef _USE_MALI
                            auto mem = (OclMemory *)tensor->get_memory();
                            mem->mapped_alloc();
#endif
                        } else {
                            tensor->alloc();
                        }
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
            "run() op: %s type: %s\n", op->get_name().c_str(), OperatorTypeName()[op->get_type()]);
        if (op->get_type() == OT_Repeat || op->get_type() == OT_Jump) {
            opIndex = op->get_next_operator_index();
        } else {
            UNI_PROFILE(op->run(), op->get_name(),
                std::string(OperatorTypeName()[op->get_type()]) + std::string("::run"));
            opIndex++;
        }
#ifdef _DEBUG
        std::vector<Tensor> outputTensors = op->get_output_tensors();
        for (U32 i = 0; i < outputTensors.size(); i++) {
            Tensor outputTensor = outputTensors[i];
            std::string line = outputTensor.string(32);
            UNI_DEBUG_LOG("    output: %s\n", line.c_str());
        }
#endif
    }
}

std::shared_ptr<Tensor> CNN::allocate_tensor(U32 size)
{
    MemoryType type = CPUMem;
    if (this->deviceInfo.schedule == MALI) {
        type = OCLMem;
    }
    std::shared_ptr<Tensor> tensor = std::shared_ptr<Tensor>(new Tensor(type));
    tensor->resize(tensor1d(DT_U8, size));
    tensor->alloc();
    return tensor;
}

void CNN::add(std::shared_ptr<Operator> op,
    std::vector<std::string> inputTensorsName,
    std::vector<std::string> outputTensorsName)
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

void CNN::infer_layout_desc()
{
    for (std::string opName : this->sortedOps) {
        auto op = this->operatorMap[opName];
        UNI_DEBUG_LOG("op: %s type: %s\n", opName.c_str(), OperatorTypeName()[op->get_type()]);
        std::vector<std::string> curOpInputTensorName = this->operatorTensorMap[opName][0];
        std::vector<std::string> curOpOutputTensorName = this->operatorTensorMap[opName][1];
        std::vector<Tensor *> inputTensors;
        std::vector<Tensor *> outputTensors;
        for (std::string inputTensorName : curOpInputTensorName) {
            auto tensor = this->tensorMap[inputTensorName].get();
            inputTensors.push_back(tensor);
            UNI_DEBUG_LOG("    input: %s desc %s\n", inputTensorName.c_str(),
                tensorDesc2Str(tensor->get_desc()).c_str());
        }
        for (std::string outputTensorName : curOpOutputTensorName) {
            auto tensor = this->tensorMap[outputTensorName].get();
            outputTensors.push_back(tensor);
        }
        CHECK_STATUS(op->infer_output_tensors_size(inputTensors, outputTensors));
        for (std::string outputTensorName : curOpOutputTensorName) {
            UNI_DEBUG_LOG("    output: %s desc %s\n", outputTensorName.c_str(),
                tensorDesc2Str(this->tensorMap[outputTensorName]->get_desc()).c_str());
        }
    }
}

void CNN::update_op_tensors()
{
    for (auto opName : this->sortedOps) {
        auto op = this->operatorMap[opName];
        UNI_DEBUG_LOG("update_op_tensors() op: %s type: %s\n", opName.c_str(),
            OperatorTypeName()[op->get_type()]);
        std::vector<std::string> curOpInputTensorName = this->operatorTensorMap[opName][0];
        std::vector<std::string> curOpOutputTensorName = this->operatorTensorMap[opName][1];
        std::vector<Tensor> inTensors, outTensors;
        for (std::string inputTensorName : curOpInputTensorName) {
            auto tensorTmp = this->tensorMap[inputTensorName];
            inTensors.push_back(*tensorTmp.get());
        }

        for (std::string outputTensorName : curOpOutputTensorName) {
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

void CNN::set_input_tensors_desc(std::map<std::string, TensorDesc> inputDescMap)
{
    for (auto iter : inputDescMap) {
        if (tensorMap.find(iter.first) == tensorMap.end()) {
            UNI_WARNING_LOG("Unused model input node: %s\n", iter.first.c_str());
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
    if (this->deviceInfo.schedule == MALI) {
        for (auto desc : modelInputTensorDescs) {
            tmpSize = UNI_MAX(tmpSize, tensorNumBytes(desc));
        }
    }

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
    for (auto &op : this->ops) {
        op->set_tmp_memory(this->tmpTensor);
    }
}

void CNN::check_memory_reuse_ratio()
{
    U32 originalSize = 0;
    U32 standaloneSize = 0;
    for (auto tensor : this->tensorMap) {
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
