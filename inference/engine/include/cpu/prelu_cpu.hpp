// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _PRELU_CPU_H
#define _PRELU_CPU_H

#include "prelu.hpp"

class PReLUCPU : public PReLU {
public:
    PReLUCPU(DataType dt) : PReLU(dt)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<PReLUCPU> mem = std::shared_ptr<PReLUCPU>(new PReLUCPU(this->dt));
        *mem = *this;
        return mem;
    }

    EE infer_weight_desc() override
    {
        U32 weightNum = (this->ws.weight == nullptr)
            ? 0
            : this->ws.bytes_of_weight / UNI_MAX(1, bytesOf(this->ws.mdt));
        if (weightNum > 0) {
            Tensor weightTensor;
            weightTensor.resize(tensor1d(this->dt, weightNum));
            this->weightTensors.push_back(weightTensor);
        }
        return SUCCESS;
    }

    void run() override
    {
        Tensor weight;
        if (this->weightTensors.size() > 0) {
            weight = this->weightTensors[0];
        } else if (this->inputTensors.size() > 1) {
            weight = this->inputTensors[1];
        } else {
            UNI_ERROR_LOG("operator:%s type:%s doesn't have weight.\n", this->name.c_str(),
                OperatorTypeName()[this->get_type()]);
        }
        if (weight.length() == 1) {
            this->p.propagate_down = true;
        } else {
            this->p.propagate_down = false;
        }
        CHECK_STATUS(
            prelu(this->inputTensors[0], weight, this->p, this->outputTensors[0], &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        return prelu_infer_output_size(inTensors[0], outTensors[0], &this->archInfo);
    }
};
#endif  // _PRELU_CPU_H
