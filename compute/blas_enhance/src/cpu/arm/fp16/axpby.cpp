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
#include "cpu/arm/fp16/blas_fp16.h"

EE axpby_fp16(U32 len, F32 a, const F16 *x, F32 b, F16 *y)
{
    EE ret = SUCCESS;
    float16x8_t alpha = vdupq_n_f16(a);
    float16x8_t beta = vdupq_n_f16(b);
    I32 i = 0;
    for (; i < ((I32)len) - 7; i += 8) {
        float16x8_t out = vld1q_f16(y + i);
        float16x8_t in = vld1q_f16(x + i);
        out = vmulq_f16(out, beta);
        out = vfmaq_f16(out, alpha, in);
        vst1q_f16(y + i, out);
    }
    for (; i < (I32)len; i++) {
        y[i] = a * x[i] + b * y[i];
    }
    return ret;
}
