// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/fp32/tensor_computing_fp32.h"

EE attention_mask_fp32(TensorDesc inputDesc,
    const F32 *input,
    AttentionMaskParamSpec p,
    TensorDesc outputDesc,
    F32 *output)
{
    UNUSED(outputDesc);
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    I32 attentionLength = p.attention_length;
    bool sameLength = p.same_length;
    float maskValue = p.mask;
    int qlen = inputDesc.dims[1];
    int klen = inputDesc.dims[0];
    int mlen = klen - qlen;
    I32 length = qlen * klen;
    std::vector<F32> mask;
    if (attentionLength < 0) {
        mask = std::vector<F32>(length, 0);
    } else {
        mask = std::vector<F32>(length, 1);
        for (int i = 0; i < qlen; i++) {
            int start, loops;
            if (attentionLength > 0) {
                int end = mlen + i;
                start = UNI_MAX(end - attentionLength, 0);
                loops = end - start + 1;
            } else {
                if (sameLength) {
                    start = i;
                    loops = qlen + 1;
                } else {
                    start = 0;
                    loops = i + qlen + 1;
                }
            }
            loops = UNI_MAX(loops, 0);
            start = UNI_MIN(start, klen);
            if (start + loops > klen) {
                loops = UNI_MAX(klen - start, 0);
            }
            UNI_MEMSET(&mask[i * klen + start], 0, sizeof(F32) * loops);
        }
    }
    I32 loops = tensorNumElements(inputDesc) / length;
    __m256 one_v = _mm256_set1_ps(1.0f);
    __m256 mask_value_v = _mm256_set1_ps(maskValue);
    for (int i = 0, index = 0; i < loops; i++) {
        int j = 0;
        for (; j < length - 7; j += 8) {
            __m256 in = _mm256_loadu_ps(input + index);
            __m256 mask_v = _mm256_loadu_ps(&mask[j]);
            __m256 tmp_v = _mm256_sub_ps(one_v, mask_v);
            mask_v = _mm256_mul_ps(mask_value_v, mask_v);
            tmp_v = _mm256_fmsub_ps(in, tmp_v, mask_v);
            _mm256_storeu_ps(output + index, tmp_v);
            index += 8;
        }
        for (; j < length; j++) {
            output[index] = input[index] * (1 - mask[j]) - maskValue * mask[j];
            index++;
        }
    }
    return SUCCESS;
}
