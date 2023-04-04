// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/tensor_computing_x86.h"
#ifdef _USE_FP32
#include "cpu/x86/fp32/tensor_computing_fp32.h"
#endif

template <typename T>
EE decode_priorbox_x86(const T *location,
    const T *priorbox,
    const T *variance,
    I32 num_total_priorbox,
    T *xmin,
    T *ymin,
    T *xmax,
    T *ymax)
{
    EE ret = NOT_SUPPORTED;
    if (0) {
#ifdef _USE_FP32
    } else if (sizeof(T) == 4) {
        ret = decode_priorbox_fp32((const F32 *)location, (const F32 *)priorbox,
            (const F32 *)variance, num_total_priorbox, (F32 *)xmin, (F32 *)ymin, (F32 *)xmax,
            (F32 *)ymax);
#endif
    }
    return ret;
}

#ifdef _USE_FP32
template EE decode_priorbox_x86(const F32 *location,
    const F32 *priorbox,
    const F32 *variance,
    I32 num_total_priorbox,
    F32 *xmin,
    F32 *ymin,
    F32 *xmax,
    F32 *ymax);
#endif
#ifdef _USE_FP16
template EE decode_priorbox_x86(const F16 *location,
    const F16 *priorbox,
    const F16 *variance,
    I32 num_total_priorbox,
    F16 *xmin,
    F16 *ymin,
    F16 *xmax,
    F16 *ymax);
#endif
