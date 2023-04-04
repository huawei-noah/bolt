// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _INSTANCE_NORM_OCL_H
#define _INSTANCE_NORM_OCL_H

#include "instance_norm.hpp"

class InstanceNormOCL : public InstanceNorm {
public:
    InstanceNormOCL(DataType dt, InstanceNormParamSpec p) : InstanceNorm(dt, p)
    {
        INIT_GPU_INFO(nullptr)
    }

    ~InstanceNormOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<InstanceNormOCL> mem =
            std::shared_ptr<InstanceNormOCL>(new InstanceNormOCL(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    EE infer_weight_desc() override
    {
        int num = this->get_channels_num();
        TensorDesc weightDesc = tensor1d(this->dt, num);
        TensorDesc biasDesc = tensor1d(this->dt, num);
        Tensor modelWeightTensor = Tensor(OCLMem);
        Tensor modelBiasTensor = Tensor(OCLMem);
        modelWeightTensor.resize(weightDesc);
        modelBiasTensor.resize(biasDesc);
        this->weightTensors.push_back(modelWeightTensor);
        this->biasTensors.push_back(modelBiasTensor);
        return SUCCESS;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        CHECK_STATUS(instance_norm(this->inputTensors[0], this->weightTensors[0],
            this->biasTensors[0], this->p, this->temp, this->outputTensors[0], &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        outTensors[0]->resize(inTensors[0]->get_desc());
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes = 0;
        CHECK_STATUS(
            instance_norm_infer_forward_tmp_bytes(this->inputTensors[0], this->p, &bytes, &this->archInfo));
        return bytes;
    }

    REGISTER_OCL_OPERATOR_RUN
};

#endif  // _INSTANCE_NORM_OCL_H
