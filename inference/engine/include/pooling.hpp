// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _POOLING_H
#define _POOLING_H

#include "operator.hpp"
#include "tensor_computing.h"

class Pooling : public Operator {
public:
    Pooling(PoolingParamSpec p)
    {
        this->p = p;
    }

    OperatorType get_type() override
    {
        return OT_Pooling;
    }

    void set_kernelSize(U32 globalKernelSizeH, U32 globalKernelSizeW)
    {
        this->p.kernel_h = globalKernelSizeH;
        this->p.kernel_w = globalKernelSizeW;
    }

    void set_stride(U32 globalStrideH, U32 globalStrideW)
    {
        this->p.stride_h = globalStrideH;
        this->p.stride_w = globalStrideW;
    }

protected:
    PoolingParamSpec p;
};

#endif  // _POOLING_H
