// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _SEQUENTIAL_HPP
#define _SEQUENTIAL_HPP

#include "sys.h"
#include "error.h"
#include "types.h"
#include "tensor.hpp"
#include "operator.hpp"
#include "convolution.hpp"
#include "fully_connected.hpp"
#include "cnn.h"
#include "op_type.h"
#include "tensor_desc.h"
#include "sequential.hpp"
#include "cpu/rnn_cpu.hpp"

class Sequential : public CNN {
public:
    Sequential(AffinityPolicy affinityPolicy, DataType dt, std::string name)
        : CNN(affinityPolicy, dt, name)
    {}

    void initialize_weight(std::shared_ptr<U8> _modelPtr)
    {
        this->modelPtr = _modelPtr;
    }

    EE infer_output_tensors_size(std::map<std::string, TensorDesc> inputDescMap) override
    {
        if (inputDescMap.size() != 1) {
            return NOT_SUPPORTED;
        }
        std::vector<Tensor> inputTensors;
        std::vector<Tensor *> inputTensorsPtr;
        std::vector<TensorDesc> inDims;
        std::vector<Tensor> outputTensors;
        std::vector<Tensor *> outputTensorsPtr(1);
        U32 count = 0;
        for (auto iter : inputDescMap) {
            Tensor tensor;
            tensor.resize(iter.second);
            inputTensors.push_back(tensor);
            inDims.push_back(iter.second);
            inputTensorsPtr.push_back(&inputTensors[count]);
            count++;
        }
        this->dimsOp = {inDims};
        auto num = [](std::vector<TensorDesc> inDims) -> U32 {
            U32 ret = 0;
            for (auto d : inDims) {
                ret += tensorNumElements(d);
            }
            return ret;
        };
        maxOutputElements = num(inDims);

        count = 0;
        for (auto op : this->ops) {
            Tensor tensor;
            outputTensors.push_back(tensor);
            outputTensorsPtr[0] = &outputTensors[count];
            CHECK_STATUS(op->infer_output_tensors_size(inputTensorsPtr, outputTensorsPtr));
            auto outDesc = outputTensorsPtr[0]->get_desc();
            std::vector<TensorDesc> outDescVec;
            outDescVec.push_back(outDesc);
            dimsOp.push_back(outDescVec);
            U32 numElements = tensorNumElements(outDesc);
            if (maxOutputElements < numElements) {
                maxOutputElements = numElements;
            }
            inputTensorsPtr[0] = &outputTensors[count];
            count++;
        }
        return SUCCESS;
    }

    void assign_output_tensor() override
    {
        auto firstPtr = (U8 *)operator new(bytesOf(this->dt) * maxOutputElements);
        std::shared_ptr<U8> firstSharedPtr(firstPtr);
        auto secondPtr = (U8 *)operator new(bytesOf(this->dt) * maxOutputElements);
        std::shared_ptr<U8> secondSharedPtr(secondPtr);
        for (U32 i = 0; i < this->ops.size(); i++) {
            auto op = this->ops[i];
            auto inDims = dimsOp[i];
            auto outDims = dimsOp[i + 1];

            std::vector<Tensor> inTensors;
            U32 index = 0;
            for (auto d : inDims) {
                auto val =
                    std::shared_ptr<U8>(firstSharedPtr, (U8 *)firstPtr + index * bytesOf(this->dt));
                Tensor tensor;
                tensor.resize(d);
                ((CpuMemory *)tensor.get_memory())->set_shared_ptr(val);
                inTensors.push_back(tensor);
                index += tensorNumElements(d);
            }

            std::vector<Tensor> outTensors;
            index = 0;
            for (auto d : outDims) {
                auto val = std::shared_ptr<U8>(
                    secondSharedPtr, (U8 *)secondPtr + index * bytesOf(this->dt));
                Tensor tensor;
                tensor.resize(d);
                ((CpuMemory *)tensor.get_memory())->set_shared_ptr(val);
                outTensors.push_back(tensor);
                index += tensorNumElements(d);
            }

            op->set_input_output_tensors(inTensors, outTensors);

            std::swap(firstPtr, secondPtr);
            std::swap(firstSharedPtr, secondSharedPtr);
        }
    }

    EE ConvBiasAssignmentAndWeightTransform()
    {
        return SUCCESS;
    }

    EE FCBiasAssignmentAndWeight()
    {
        return SUCCESS;
    }

    void ready(std::map<std::string, TensorDesc> inputDescMap) override
    {
        for (auto op : this->ops) {
            op->set_schedule(this->deviceInfo.schedule);
        }
        this->infer_output_tensors_size(inputDescMap);
        this->assign_output_tensor();

        for (auto op : this->ops) {
            if (op->is_weight()) {
                if (op->get_type() == OT_Conv) {
                    auto convOpPtr = dynamic_cast<Convolution *>(op.get());
                    CHECK_STATUS(convOpPtr->init_weight_bias_from_model(&modelPtr));
                    CHECK_STATUS(convOpPtr->infer_forward_algorithm(this->algorithmMap));
                    CHECK_STATUS(convOpPtr->transform_filter());
                } else if (op->get_type() == OT_FC) {
                    auto fcOpPtr = dynamic_cast<FullyConnected *>(op.get());
                    CHECK_STATUS(fcOpPtr->init_weight_bias_from_model(&modelPtr));
                    CHECK_STATUS(fcOpPtr->transform_filter());
                } else if (op->get_type() == OT_RNN) {
                    auto rnnOpPtr = dynamic_cast<RNNCell *>(op.get());
                    CHECK_STATUS(rnnOpPtr->init_weight_bias_from_model(&modelPtr));
                    CHECK_STATUS(rnnOpPtr->transform_filter());
                }
            }
        }

        this->infer_tmp_memory_size();
        this->assign_tmp_tensor();
    }

    void infer_tmp_memory_size() override
    {
        tmpElements.clear();
        maxTmpElements = 0;

        for (auto op : this->ops) {
            auto len = op->infer_tmp_memory_size();
            tmpElements.push_back(len);
            if (len > maxTmpElements) {
                maxTmpElements = len;
            }
        }
    }

    void assign_tmp_tensor() override
    {
        temp.resize(tensor1d(DT_U8, maxTmpElements));
        temp.alloc();
        for (auto op : this->ops) {
            op->set_tmp_memory(temp);
        }
    }

    void add(std::shared_ptr<Operator> op)
    {
        this->ops.push_back(op);
    }

    std::vector<Tensor> get_inputTensors()
    {
        auto op = this->ops[0].get();
        return op->get_input_tensors();
    }

    std::vector<Tensor> get_output_tensors()
    {
        auto len = this->ops.size();
        auto op = this->ops[len - 1].get();
        return op->get_output_tensors();
    }

    void set_input_tensors(std::vector<Tensor> inputTensors)
    {
        auto op = this->ops[0].get();
        op->set_input_tensors(inputTensors);
    }

private:
    std::shared_ptr<U8> modelPtr;
    U32 maxOutputElements;
    std::vector<std::vector<TensorDesc>> dimsOp;
    U32 maxTmpElements;
    std::vector<U32> tmpElements;
    Tensor temp;
};
#endif
