// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _PREALLOCATED_MEMORY_CPU_H
#define _PREALLOCATED_MEMORY_CPU_H

#include "preallocated_memory.hpp"

class PreAllocatedMemoryCPU : public PreAllocatedMemory {
public:
    PreAllocatedMemoryCPU(DataType dt, TensorDesc desc) : PreAllocatedMemory(dt, desc)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<PreAllocatedMemoryCPU> mem =
            std::shared_ptr<PreAllocatedMemoryCPU>(new PreAllocatedMemoryCPU(this->dt, this->desc));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        CHECK_STATUS(preallocated_memory(this->outputTensors[0], &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        if (inTensors.size() > 0) {
            CHECK_STATUS(NOT_MATCH);
        }
        outTensors[0]->resize(this->desc);
        return SUCCESS;
    }
};

#endif  // _PREALLOCATED_MEMORY_CPU_H
