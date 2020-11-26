// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _JUMP_H
#define _JUMP_H

#include "operator.hpp"

class Jump : public Operator {
public:
    Jump(DataType dt, I32 jumpOperatorIndex, I32 currentOperatorIndex)
    {
        this->dt = dt;
        this->jumpOperatorIndex = jumpOperatorIndex;
        this->nextOperatorIndex = currentOperatorIndex + 1;
    }

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<Jump> mem = std::shared_ptr<Jump>(
            new Jump(this->dt, this->jumpOperatorIndex, this->nextOperatorIndex));
        *mem = *this;
        return mem;
    }

    OperatorType get_type() override
    {
        return OT_Jump;
    }

    void run() override
    {}

    int get_next_operator_index() override
    {
        // check status
        if (this->inputTensors.size() > 1) {
            Tensor inputTensor = this->inputTensors[1];
            I32 *ptr = (I32 *)((CpuMemory *)(inputTensor.get_memory()))->get_ptr();
            U32 length = inputTensor.length();
            for (U32 i = 0; i < length; i++) {
                if (ptr[i]) {
                    return this->jumpOperatorIndex;
                }
            }
        }
        return this->nextOperatorIndex;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        UNUSED(inTensors);
        auto outDim = outTensors[0]->get_desc();
        outDim.dt = this->dt;
        outDim.nDims = 0;
        outTensors[0]->resize(outDim);
        return SUCCESS;
    }

private:
    int jumpOperatorIndex;
    int nextOperatorIndex;
};

#endif  // _JUMP_H
