// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <arm_neon.h>
#include "cpu/arm/int32/tensor_computing_int32.h"

EE clip_int32(I32 *input, I32 *output, I32 len, I32 minValue, I32 maxValue)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }

    int32x4_t min_v = vdupq_n_s32(minValue);
    int32x4_t max_v = vdupq_n_s32(maxValue);
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (int i = 0; i < len - 3; i += 4) {
        int32x4_t in = vld1q_s32(input + i);
        int32x4_t tmp_v = vminq_s32(max_v, vmaxq_s32(min_v, in));
        vst1q_s32(output + i, tmp_v);
    }
    for (int i = len / 4 * 4; i < len; i++) {
        F32 value = input[i];
        value = (value > minValue) ? value : minValue;
        value = (value < maxValue) ? value : maxValue;
        output[i] = value;
    }
    return SUCCESS;
}
