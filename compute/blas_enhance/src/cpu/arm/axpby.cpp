// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "error.h"

#include "cpu/arm/blas_arm.h"
#ifdef _USE_FP16
#include "cpu/arm/fp16/blas_fp16.h"
#endif
#ifdef _USE_FP32
#include "cpu/arm/fp32/blas_fp32.h"
#endif

EE axpby_arm(U32 len, DataType dt, F32 a, const void *x, F32 b, void *y, Arch arch)
{
    EE ret = SUCCESS;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            if (ARM_A55 != arch && ARM_A76 != arch) {
                return NOT_SUPPORTED;
            }
            ret = axpby_fp16(len, a, (F16 *)x, b, (F16 *)y);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            ret = axpby_fp32(len, a, (F32 *)x, b, (F32 *)y);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
