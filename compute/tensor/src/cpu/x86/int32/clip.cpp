// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/int32/tensor_computing_int32.h"

EE clip_int32(I32 *input, I32 *output, I32 len, I32 minValue, I32 maxValue)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    __m256i min_v = _mm256_set1_epi32(minValue);
    __m256i max_v = _mm256_set1_epi32(maxValue);
    I32 i = 0;
    for (i = 0; i < len - 7; i += 8) {
        __m256i in = _mm256_loadu_si256((const __m256i *)(input + i));
        __m256i tmp_v = _mm256_min_epi32(max_v, _mm256_max_epi32(min_v, in));
        _mm256_storeu_si256((__m256i *)(output + i), tmp_v);
    }
    for (; i < len; i++) {
        I32 value = input[i];
        value = (value > minValue) ? value : minValue;
        value = (value < maxValue) ? value : maxValue;
        output[i] = value;
    }
    return SUCCESS;
}
