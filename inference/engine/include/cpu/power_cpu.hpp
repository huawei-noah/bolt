// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _POWER_CPU_H
#define _POWER_CPU_H

#include "power.hpp"

class PowerCPU : public Power {
public:
    PowerCPU(DataType dt, PowerParamSpec p) : Power(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<PowerCPU> mem = std::shared_ptr<PowerCPU>(new PowerCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        Tensor outputTensor = this->outputTensors[0];

        if (DT_I8 == inputDesc.dt) {
#ifdef _USE_INT8
            CHECK_REQUIREMENT(0 == this->p.shift);
            F32 scaleO = inputTensor.get_scale() / this->p.scale;
            outputTensor.set_scale(scaleO);
            auto inPtr = ((CpuMemory *)(inputTensor.get_memory()))->get_ptr();
            auto outPtr = ((CpuMemory *)(outputTensor.get_memory()))->get_ptr();
            if (inPtr != outPtr) {
                memcpy(outPtr, inPtr, tensorNumBytes(inputDesc));
            }
#endif
        } else {
            CHECK_STATUS(power(inputTensor, this->p, outputTensor, &this->archInfo));
        }
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        return power_infer_output_size(inTensors[0], outTensors[0], &this->archInfo);
    }
};

#endif  // _POWER_CPU_H
