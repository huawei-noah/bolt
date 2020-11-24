// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _REPEAT_CPU_H
#define _REPEAT_CPU_H

#include "repeat.hpp"

class RepeatCPU : public Repeat {
public:
    RepeatCPU(DataType dt, RepeatParamSpec p, I32 jumpOperatorIndex, I32 currentOperatorIndex)
        : Repeat(dt, p, jumpOperatorIndex, currentOperatorIndex)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<RepeatCPU> mem = std::shared_ptr<RepeatCPU>(
            new RepeatCPU(this->dt, this->p, this->jumpOperatorIndex, this->nextOperatorIndex - 1));
        *mem = *this;
        return mem;
    }

    void run() override
    {}

    int get_next_operator_index() override
    {
        // check status
        if (this->inputTensors.size() > 1) {
            Tensor inputTensor = this->inputTensors[1];
            TensorDesc inputDesc = inputTensor.get_desc();
            I32 *ptr = (I32 *)(((CpuMemory *)(inputTensor.get_memory()))->get_ptr());
            U32 length = tensorNumElements(inputDesc);
            for (U32 i = 0; i < length; i++) {
                // end loop
                if (ptr[i]) {
                    this->iter = 0;
                    return this->nextOperatorIndex;
                }
            }
        }

        // check loop
        if (this->iter < this->p.loops) {
            this->iter++;
            return this->jumpOperatorIndex;
        } else {
            this->iter = 0;
            return this->nextOperatorIndex;
        }
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->iter = 0;
        if (this->p.axis >= 0) {
            int axisIndex = 0;
            if (inTensors.size() > 2) {
                axisIndex = 2;
            } else {
                UNI_ERROR_LOG("[ERROR] set to use axis feature of Repeat must meet input tensors "
                              ">= 3 requirement\n");
            }
            TensorDesc desc = inTensors[axisIndex]->get_desc();
            this->p.loops = desc.dims[desc.nDims - 1 - this->p.axis];
        }
        TensorDesc outDesc = outTensors[0]->get_desc();
        outDesc.dt = this->dt;
        outDesc.nDims = 0;
        outTensors[0]->resize(outDesc);
        return SUCCESS;
    }
};

#endif  // _REPEAT_CPU_H
