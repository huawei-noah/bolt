// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _REDUCTION_OCL_H
#define _REDUCTION_OCL_H

#include "reduction.hpp"

class ReductionOCL : public Reduction {
public:
    ReductionOCL(DataType dt, ReductionParamSpec p) : Reduction(dt, p)
    {
        setMALIArchInfo(
            &(this->archInfo), nullptr, &this->needSetKernelVec, &this->needSelectKernelLS);
    }

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<ReductionOCL> mem =
            std::shared_ptr<ReductionOCL>(new ReductionOCL(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        Tensor inputTensor = this->inputTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        Tensor maskTensor(OCLMem);
        if (this->inputTensors.size() > 1) {
            maskTensor = this->inputTensors[1];
        } else {
            TensorDesc maskDesc;
            maskDesc.nDims = 0;
            maskTensor.resize(maskDesc);
        }
        CHECK_STATUS(
            reduction(inputTensor, maskTensor, this->p, this->temp, outputTensor, &this->archInfo));
    }
    REGISTER_OCL_OPERATOR_RUN
};

#endif
