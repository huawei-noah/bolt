// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _WHERE_CPU_H
#define _WHERE_CPU_H

#include <math.h>
#include "where.hpp"

class WhereCPU : public Where {
public:
    WhereCPU(DataType dt) : Where(dt)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<WhereCPU> mem = std::shared_ptr<WhereCPU>(new WhereCPU(this->dt));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        CHECK_STATUS(where(this->inputTensors[1], this->inputTensors[0], this->biasTensors[0],
            this->outputTensors[0], &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        //inTensors[0] is condition now, 2021/2/3
        CHECK_STATUS(where_infer_output_size(
            inTensors[inTensors.size() - 1], outTensors[0], &this->archInfo));
        return SUCCESS;
    }

    EE infer_weight_desc() override
    {
        auto curOpWs = this->get_weightspec();
        int weightBytes = curOpWs.bytes_of_weight;
        int Lw = sqrt(weightBytes / bytesOf(curOpWs.mdt));
        int biasBytes = curOpWs.bytes_of_vec;
        int Lb = biasBytes / bytesOf(curOpWs.mdt);
        this->weightTensors = std::vector<Tensor>(1);
        this->weightTensors[0].resize(tensor4d(this->dt, 1, 1, Lw, Lw));
        this->biasTensors = std::vector<Tensor>(1);
        this->biasTensors[0].resize(tensor2d(this->dt, 1, Lb));
        return SUCCESS;
    }
};

#endif  // _WHERECPU_H
