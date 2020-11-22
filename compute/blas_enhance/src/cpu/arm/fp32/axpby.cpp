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
#include "cpu/arm/fp32/blas_fp32.h"

EE axpby_fp32(U32 len, F32 a, const F32 *x, F32 b, F32 *y)
{
    EE ret = SUCCESS;
    float32x4_t alpha = vdupq_n_f32(a);
    float32x4_t beta = vdupq_n_f32(b);
    I32 i = 0;
    for (; i < ((I32)len) - 3; i += 4) {
        float32x4_t out = vld1q_f32(y + i);
        float32x4_t in = vld1q_f32(x + i);
        out = vmulq_f32(out, beta);
        out = vmlaq_f32(out, alpha, in);
        vst1q_f32(y + i, out);
    }
    for (; i < (I32)len; i++) {
        y[i] = a * x[i] + b * y[i];
    }
    return ret;
}
