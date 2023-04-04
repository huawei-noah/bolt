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

    bool is_unknown(const float &value)
    {
        return value == (float)UNI_RESERVE;
    }

    template <typename T>
    float get_param(float value, int *count)
    {
        if (is_unknown(value)) {
            T *ptr = (T *)((CpuMemory *)(inputTensors[*count].get_memory()))->get_ptr();
            *count = *count + 1;
            value = ptr[0];
        }
        return value;
    }

    float get_param(float value, std::vector<Tensor *> inTensors, int *count)
    {
        if (is_unknown(value)) {
            TensorDesc desc = inTensors[*count]->get_desc();
            if (tensorIsShape(desc)) {
                transformToFloat(desc.dt, desc.dims + desc.nDims, &value, 1);
            }
            *count = *count + 1;
        }
        return value;
    }

    void run() override
    {
        int count = 0;
        int idx = outputTensors.size() - 1;
        TensorDesc desc = outputTensors[idx].get_desc();
        I32 length;
        switch (desc.dt) {
            case DT_I32: {
                float start = get_param<I32>(p.start, &count);
                float limit = get_param<I32>(p.limit, &count);
                float delta = get_param<I32>(p.delta, &count);
                length = (limit - start) / delta;
                I32 *ptr = (I32 *)((CpuMemory *)(outputTensors[idx].get_memory()))->get_ptr();
                for (int i = 0; i < length; i++) {
                    ptr[i] = start + delta * i;
                }
                break;
            }
            case DT_F32: {
                float start = get_param<F32>(p.start, &count);
                float limit = get_param<F32>(p.limit, &count);
                float delta = get_param<F32>(p.delta, &count);
                length = (limit - start) / delta;
                F32 *ptr = (F32 *)((CpuMemory *)(outputTensors[idx].get_memory()))->get_ptr();
                for (int i = 0; i < length; i++) {
                    ptr[i] = start + delta * i;
                }
                break;
            }
            default:
                UNI_ERROR_LOG("not support Range(%s).\n", DataTypeName()[desc.dt]);
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
        int count = 0;
        float start = get_param(p.start, inTensors, &count);
        float limit = get_param(p.limit, inTensors, &count);
        float delta = get_param(p.delta, inTensors, &count);
        U32 length = UNI_DYNAMIC_SHAPE;
        if (!(is_unknown(start) || is_unknown(limit) || is_unknown(delta))) {
            U32 value = (limit - start) / delta;
            if (value != 0) {
                length = value;
            }
        }
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
