// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _SLICE_CPU_H
#define _SLICE_CPU_H

#include "slice.hpp"

class SliceCPU : public Slice {
public:
    SliceCPU(DataType dt, SliceParamSpec p) : Slice(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<SliceCPU> mem = std::shared_ptr<SliceCPU>(new SliceCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        CHECK_STATUS(slice(inputTensors[0], this->p, this->temp, outputTensors, &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        CHECK_STATUS(slice_infer_output_size(inTensors[0], this->p, outTensors, &this->archInfo));
        auto outDesc = outTensors[0]->get_desc();
        if (outDesc.nDims == 3 && outDesc.dims[1] == 1 && outDesc.dims[2] == 1) {
            outDesc.nDims = 2;
            outDesc.df = DF_NORMAL;
            outTensors[0]->resize(outDesc);
        }
        return SUCCESS;
    }
};

#endif  // _SLICE_CPU_H
