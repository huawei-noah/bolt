// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _RANGE_CPU_H
#define _RANGE_CPU_H

#include "range.hpp"

class RangeCPU : public Range {
public:
    RangeCPU(DataType dt, RangeParamSpec p) : Range(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<RangeCPU> mem = std::shared_ptr<RangeCPU>(new RangeCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        //CHECK_STATUS(non_zero(inputTensors[0], outputTensors[0], &this->archInfo));
        int idx = outputTensors.size() - 1;
        TensorDesc desc = outputTensors[idx].get_desc();
        I32 length = (p.limit - p.start) / p.delta;
        switch (desc.dt) {
            case DT_I32: {
                I32 *ptr = (I32 *)((CpuMemory *)(outputTensors[idx].get_memory()))->get_ptr();
                for (int i = 0; i < length; i++) {
                    ptr[i] = p.start + p.delta * i;
                }
                break;
            }
            default:
                UNI_ERROR_LOG("not support Range(%s).\n", OperatorTypeName()[desc.dt]);
                break;
        }
        if (outputTensors.size() > 1) {
            U32 *ptr = (U32 *)((CpuMemory *)(outputTensors[0].get_memory()))->get_ptr();
            *ptr = length;
        }
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        U32 length = (p.limit - p.start) / p.delta;
        TensorDesc desc0 = tensor1d(DT_U32, length);
        desc0.df = DF_SCALAR;
        desc0.dims[1] = length;
        TensorDesc desc1 = tensor1d(p.dt, length);
        if (outTensors.size() >= 1) {
            outTensors[outTensors.size() - 1]->resize(desc1);
        }
        if (outTensors.size() == 2) {
            outTensors[outTensors.size() - 2]->resize(desc0);
        }
        return SUCCESS;
    }
};

#endif  // RANGE_CPU_H
