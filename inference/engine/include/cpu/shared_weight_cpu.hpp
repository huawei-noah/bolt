// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _SHARED_WEIGHT_CPU_H
#define _SHARED_WEIGHT_CPU_H

#include "shared_weight.hpp"

class SharedWeightCPU : public SharedWeight {
public:
    SharedWeightCPU(DataType dt,
        TensorDesc desc,
        std::string outputTensorName,
        std::map<std::string, std::shared_ptr<Tensor>> *tensorMapPtr)
        : SharedWeight(dt, desc, outputTensorName, tensorMapPtr)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<SharedWeightCPU> mem = std::shared_ptr<SharedWeightCPU>(
            new SharedWeightCPU(this->dt, this->desc, this->outputTensorName, this->tensorMapPtr));
        *mem = *this;
        return mem;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        UNUSED(inTensors);
#ifdef _USE_INT8
        if (featureScale.size() > 0 && 0 < (featureScale.back())[0]) {
            this->desc.dt = get_activation_quant_data_type();
        }
#endif
        outTensors[0]->resize(this->desc);
        return SUCCESS;
    }

    void run() override
    {}

    EE transform_filter() override
    {
        return SUCCESS;
    }

    EE init_weight_bias_from_model(std::shared_ptr<U8> *modelPtrShared) override
    {
        U8 *modelPtr = nullptr;
        if (modelPtrShared != nullptr) {
            modelPtr = (*modelPtrShared).get();
        }
        TensorDesc weightDesc = this->desc;
        Tensor modelWeightTensor;
        modelWeightTensor.resize(weightDesc);
        U32 weightBytes = modelWeightTensor.bytes();
        modelWeightTensor.alloc();
        if (modelPtr != nullptr) {
            UNI_MEMCPY(
                ((CpuMemory *)(modelWeightTensor.get_memory()))->get_ptr(), modelPtr, weightBytes);
            *modelPtrShared = std::shared_ptr<U8>(*modelPtrShared, modelPtr + weightBytes);
        } else {
            UNI_MEMCPY(((CpuMemory *)(modelWeightTensor.get_memory()))->get_ptr(), this->ws.weight,
                weightBytes);
        }
        this->weightTensors.push_back(modelWeightTensor);
        (*this->tensorMapPtr)[this->outputTensorName]->reuse(&(this->weightTensors[0]));

        if ((this->ws.num_quant_scale > 0) && (this->ws.weight_scale[0].num_scale > 0) && (this->ws.weight_scale[0].scale[0] > 0)) {
            (*this->tensorMapPtr)[this->outputTensorName]->set_scale(
                this->ws.weight_scale[0].scale[0]);
        }

        return SUCCESS;
    }
};

#endif  // _SHARED_WEIGHT_CPU_H
