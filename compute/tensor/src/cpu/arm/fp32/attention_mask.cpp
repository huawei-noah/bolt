// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <string.h>
#include "cpu/arm/fp32/tensor_computing_fp32.h"

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
            memset(&mask[i * klen + start], 0, sizeof(F32) * loops);
        }
    }
    I32 loops = tensorNumElements(inputDesc) / length;
    float32x4_t one_v = vdupq_n_f32(1);
    float32x4_t mask_value_v = vdupq_n_f32(maskValue);
    for (int i = 0, index = 0; i < loops; i++) {
        int j = 0;
        for (; j < length - 3; j += 4) {
            float32x4_t in = vld1q_f32(input + index);
            float32x4_t mask_v = vld1q_f32(&mask[j]);
            float32x4_t tmp_v = vsubq_f32(one_v, mask_v);
            tmp_v = vmulq_f32(in, tmp_v);
            tmp_v = vfmsq_f32(tmp_v, mask_value_v, mask_v);
            vst1q_f32(output + index, tmp_v);
            index += 4;
        }
        for (; j < length; j++) {
            output[index] = input[index] * (1 - mask[j]) - maskValue * mask[j];
            index++;
        }
    }
    return SUCCESS;
}
