// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _CAST_CPU_H
#define _CAST_CPU_H

#include "cast.hpp"

class CastCPU : public Cast {
public:
    CastCPU(DataType dt, CastParamSpec p) : Cast(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<CastCPU> mem = std::shared_ptr<CastCPU>(new CastCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        CHECK_STATUS(cast(this->inputTensors[0], this->p, this->outputTensors[0], &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        CHECK_STATUS(cast_infer_output_size(inTensors[0], this->p, outTensors[0], &this->archInfo));
        return SUCCESS;
    }
};

#endif  // _CASTCPU_H
