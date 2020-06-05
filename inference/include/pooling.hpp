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

class Pooling: public Operator {
public:
    Pooling(PoolingMode mode, U32 ksH, U32 ksW, U32 strideH, U32 strideW,
            U32 paddingT, U32 paddingB, U32 paddingL, U32 paddingR, RoundMode rm)
    {
        this->mode = mode;
        this->kernelSizeH = ksH;
        this->kernelSizeW = ksW;
        this->strideH = strideH;
        this->strideW = strideW;
        this->paddingT = paddingT;
        this->paddingB = paddingB;
        this->paddingL = paddingL;
        this->paddingR = paddingR;
        this->rm = rm;
    }

    OperatorType get_op_type() override
    {
        return OT_Pooling;
    }

    PoolingDesc create_PoolingDesc(PoolingMode pm, U32 ksH, U32 ksW, U32 strideH, U32 strideW,
                                U32 paddingT, U32 paddingB, U32 paddingL, U32 paddingR, RoundMode rm)
    {
        PoolingDesc poolingDesc;
        poolingDesc.pm = pm;
        poolingDesc.kernelSize_h = ksH;
        poolingDesc.kernelSize_w = ksW;
        poolingDesc.stride_h = strideH;
        poolingDesc.stride_w = strideW;
        poolingDesc.padding_top = paddingT;
        poolingDesc.padding_bottom = paddingB;
        poolingDesc.padding_left = paddingL;
        poolingDesc.padding_right = paddingR;
        poolingDesc.rm = rm;
        return poolingDesc;
    }

    void set_kernelSize(U32 globalKernelSizeH, U32 globalKernelSizeW) {
        this->kernelSizeH = globalKernelSizeH;
        this->kernelSizeW = globalKernelSizeW;
    }

    void set_stride(U32 globalStrideH, U32 globalStrideW) {
        this->strideH = globalStrideH;
        this->strideW = globalStrideW;
    }

protected:
    PoolingMode mode;
    RoundMode rm;
    U32 kernelSizeH;
    U32 kernelSizeW;
    U32 strideH;
    U32 strideW;
    U32 paddingT;
    U32 paddingB;
    U32 paddingL;
    U32 paddingR;
};

#endif //_POOLING_H
