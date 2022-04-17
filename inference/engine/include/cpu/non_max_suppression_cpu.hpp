// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _NON_MAX_SUPPRESSION_CPU_H
#define _NON_MAX_SUPPRESSION_CPU_H

#include "non_max_suppression.hpp"

class NonMaxSuppressionCPU : public NonMaxSuppression {
public:
    NonMaxSuppressionCPU(DataType dt, NonMaxSuppressionParamSpec p) : NonMaxSuppression(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<NonMaxSuppressionCPU> mem =
            std::shared_ptr<NonMaxSuppressionCPU>(new NonMaxSuppressionCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        CHECK_STATUS(non_max_suppression(inputTensors, this->p, outputTensors[0], &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        return non_max_suppression_infer_output_size(
            inTensors, this->p, outTensors[0], &this->archInfo);
    }
};

#endif  // NON_MAX_SUPPRESSION_CPU_H
