// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/tensor_computing_arm.h"
#include "arm_neon_expand.h"

EE requantize_arm(TensorDesc inputDesc,
    INT8 *input,
    F32 inputScale,
    TensorDesc outputDesc,
    INT8 *output,
    F32 outputScale)
{
    int num = tensorNumElements(outputDesc);
    if (num <= 0) {
        return SUCCESS;
    }
    F32 rescale = outputScale / inputScale;
    INT8 factor = rescale * 128;
    if (outputScale == inputScale || rescale >= 0.9921 || factor < 2) {
        UNI_MEMCPY(output, input, num * sizeof(INT8));
        return SUCCESS;
    }

    int8x8_t fact = vdup_n_s8(factor);

    int8x8_t in[4];
    int16x8_t in16[4];
    U32 i32 = num / 32;
    for (U32 i = 0; i < i32; i++) {
        for (U32 j = 0; j < 4; j++) {
            in[j] = vld1_s8(input + j * 8);
        }
        for (U32 j = 0; j < 4; j++) {
            in16[j] = vmull_s8(in[j], fact);
        }
        in[0] = vqshrn_n_s16(in16[0], 7);
        for (U32 j = 1; j < 4; j++) {
            in[j] = vqshrn_n_s16(in16[j], 7);
            vst1_s8(output + j * 8 - 8, in[j - 1]);
        }
        vst1_s8(output + 24, in[3]);
        input += 32;
        output += 32;
    }

    U32 remainder = num - i32 * 32;
    for (U32 j = 0; j < remainder; j += 8) {
        int8x8_t in = vld1_s8(input + j);
        int16x8_t in16 = vmull_s8(in, fact);
        in = vqshrn_n_s16(in16, 7);
        vst1_s8(output + j, in);
    }
    return SUCCESS;
}
