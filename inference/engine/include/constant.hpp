// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _CONSTANT_H
#define _CONSTANT_H
#include "operator.hpp"

class Constant : public Operator {
public:
    Constant(TensorDesc constDesc, void *data)
    {
        this->constDesc = constDesc;
        this->data = data;
    }

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<Constant> mem =
            std::shared_ptr<Constant>(new Constant(this->constDesc, this->data));
        *mem = *this;
        return mem;
    }

    OperatorType get_type() override
    {
        return OT_Constant;
    }

    void run() override
    {
        Tensor outputTensor = this->outputTensors[0];
        auto outputPtr = ((CpuMemory *)outputTensor.get_memory())->get_ptr();
        memcpy(outputPtr, data, tensorNumBytes(constDesc));
    }

    EE infer_output_tensors_size(std::vector<TensorDesc> *outDims) override
    {
        (*outDims)[0] = constDesc;
        return SUCCESS;
    }

private:
    TensorDesc constDesc;
    void *data;
};

#endif  // _CONSTANT__H
