// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _EXPAND_CPU_H
#define _EXPAND_CPU_H

#include "expand.hpp"

class ExpandCPU : public Expand {
public:
    ExpandCPU(DataType dt, ExpandParamSpec p) : Expand(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<ExpandCPU> mem =
            std::shared_ptr<ExpandCPU>(new ExpandCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    ExpandParamSpec get_param(TensorDesc desc)
    {
        ExpandParamSpec ps = this->p;
        if (ps.num_shape == 0) {
            ps.num_shape = desc.dims[0];
            for (int i = 0; i < ps.num_shape; i++) {
                ps.shape[i] = desc.dims[desc.nDims + i];
            }
        }
        return ps;
    }

    void run() override
    {
        ExpandParamSpec ps = p;
        if (ps.num_shape == 0 && inputTensors.size() > 1) {
            ps = get_param(inputTensors[1].get_desc());
        }
        CHECK_STATUS(
            expand(this->inputTensors[0], ps, this->temp, this->outputTensors[0], &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        ExpandParamSpec ps = p;
        if (ps.num_shape == 0 && inTensors.size() > 1) {
            ps = get_param(inTensors[1]->get_desc());
        }
        return expand_infer_output_size(inTensors[0], ps, outTensors[0], &this->archInfo);
    }
};

#endif  // _EXPAND_CPU_H
