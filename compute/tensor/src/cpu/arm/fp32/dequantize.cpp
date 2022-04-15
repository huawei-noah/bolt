// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/fp32/tensor_computing_fp32.h"

inline static void dequantize_i8_f32(I32 len, INT8 *q, F32 scale, F32 *d)
{
    F32 factor = 1 / scale;
    int i = 0;
    for (; i < len - 7; i += 8) {
        int8x8_t in0 = vld1_s8(q + i);
        int16x8_t s0 = vmovl_s8(in0);
        float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0)));
        float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0)));
        f0 = vmulq_n_f32(f0, factor);
        f1 = vmulq_n_f32(f1, factor);
        vst1q_f32(d + i, f0);
        vst1q_f32(d + i + 4, f1);
    }
    for (; i < len; i++) {
        d[i] = q[i] * factor;
    }
}

inline static void dequantize_i32_f32(I32 len, I32 *q, F32 scale, I32 biasLen, F32 *biasPtr, F32 *d)
{
    if (0 != biasLen) {
        CHECK_REQUIREMENT(nullptr != biasPtr);
        CHECK_REQUIREMENT(len % biasLen == 0);
    }
    F32 factor = 1 / scale;
    if (biasLen > 0) {
        for (int i = 0; i < len; i += biasLen) {
            int j = 0;
            for (; j < biasLen - 3; j += 4) {
                int32x4_t in0 = vld1q_s32(q + i + j);
                float32x4_t bias = vld1q_f32(biasPtr + j);
                float32x4_t f0 = vcvtq_f32_s32(in0);
                f0 = vmulq_n_f32(f0, factor);
                f0 = vaddq_f32(f0, bias);
                //f0 = vmlaq_n_f32(bias, f0, factor);
                vst1q_f32(d + i + j, f0);
            }
            for (; j < biasLen; j++) {
                d[i + j] = q[i + j] * factor + biasPtr[j];
            }
        }
    } else {
        int i = 0;
        for (; i < len - 3; i += 4) {
            int32x4_t in0 = vld1q_s32(q + i);
            float32x4_t f0 = vcvtq_f32_s32(in0);
            f0 = vmulq_n_f32(f0, factor);
            vst1q_f32(d + i, f0);
        }
        for (; i < len; i++) {
            d[i] = q[i] * factor;
        }
    }
}

EE dequantize_to_f32(TensorDesc qDesc,
    void *qData,
    const F32 *scale,
    TensorDesc bDesc,
    void *bData,
    TensorDesc dDesc,
    void *data)
{
    if (nullptr == data || nullptr == qData || nullptr == scale) {
        CHECK_STATUS(NULL_POINTER);
    }
    int length = tensorNumElements(qDesc);
    int biasLength = tensorNumElements(bDesc);
    EE ret = SUCCESS;
    switch (qDesc.dt) {
        case DT_I8:
            CHECK_REQUIREMENT(biasLength == 0);
            dequantize_i8_f32(length, (INT8 *)qData, scale[0], (F32 *)data);
            break;
        case DT_I32:
            dequantize_i32_f32(
                length, (I32 *)qData, scale[0], biasLength, (F32 *)bData, (F32 *)data);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
