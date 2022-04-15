// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/fp16/tensor_computing_fp16.h"

inline static void dequantize_i8_f16(I32 len, INT8 *q, F32 scale, F16 *d)
{
    F16 factor = 1 / scale;
    int i = 0;
    for (; i < len - 15; i += 16) {
        int8x8_t in0 = vld1_s8(q + i);
        int8x8_t in1 = vld1_s8(q + i + 8);
        int16x8_t s0 = vmovl_s8(in0);
        int16x8_t s1 = vmovl_s8(in1);
        float16x8_t f0 = vcvtq_f16_s16(s0);
        float16x8_t f1 = vcvtq_f16_s16(s1);
        f0 = vmulq_n_f16(f0, factor);
        f1 = vmulq_n_f16(f1, factor);
        vst1q_f16(d + i, f0);
        vst1q_f16(d + i + 8, f1);
    }
    for (; i < len; i++) {
        d[i] = q[i] * factor;
    }
}

inline static void dequantize_i32_f16(I32 len, I32 *q, F32 scale, I32 biasLen, F16 *biasPtr, F16 *d)
{
    if (0 != biasLen) {
        CHECK_REQUIREMENT(nullptr != biasPtr);
        CHECK_REQUIREMENT(len % biasLen == 0);
    }
    float16x4_t bias[4];
    F32 factor = 1 / scale;
    if (biasLen % 4 == 0) {
        int i = 0;
        for (; i < len - 15; i += 16) {
            int32x4_t in0 = vld1q_s32(q + i);
            int32x4_t in1 = vld1q_s32(q + i + 4);
            int32x4_t in2 = vld1q_s32(q + i + 8);
            int32x4_t in3 = vld1q_s32(q + i + 12);
            if (0 != biasLen) {
                I32 offset = i % biasLen;
                for (U32 j = 0; j < 4; j++) {
                    bias[j] = vld1_f16(biasPtr + offset);
                    offset += 4;
                    if (offset >= biasLen) {
                        offset = 0;
                    }
                }
            }
            float32x4_t f0 = vcvtq_f32_s32(in0);
            float32x4_t f1 = vcvtq_f32_s32(in1);
            float32x4_t f2 = vcvtq_f32_s32(in2);
            float32x4_t f3 = vcvtq_f32_s32(in3);
            f0 = vmulq_n_f32(f0, factor);
            f1 = vmulq_n_f32(f1, factor);
            f2 = vmulq_n_f32(f2, factor);
            f3 = vmulq_n_f32(f3, factor);
            float16x4_t h0 = vcvt_f16_f32(f0);
            float16x4_t h1 = vcvt_f16_f32(f1);
            float16x4_t h2 = vcvt_f16_f32(f2);
            float16x4_t h3 = vcvt_f16_f32(f3);
            if (0 != biasLen) {
                h0 = vadd_f16(h0, bias[0]);
                h1 = vadd_f16(h1, bias[1]);
                h2 = vadd_f16(h2, bias[2]);
                h3 = vadd_f16(h3, bias[3]);
            }
            vst1_f16(d + i, h0);
            vst1_f16(d + i + 4, h1);
            vst1_f16(d + i + 8, h2);
            vst1_f16(d + i + 12, h3);
        }

        for (; i < len; i++) {
            d[i] = q[i] * factor;
            if (0 != biasLen) {
                d[i] += biasPtr[i % biasLen];
            }
        }
    } else {
        for (int i = 0; i < len; i += biasLen) {
            int j = 0;
            for (; j < biasLen - 3; j += 4) {
                int32x4_t in0 = vld1q_s32(q + i + j);
                bias[0] = vld1_f16(biasPtr + j);
                float32x4_t f0 = vcvtq_f32_s32(in0);
                f0 = vmulq_n_f32(f0, factor);
                float16x4_t h0 = vcvt_f16_f32(f0);
                h0 = vadd_f16(h0, bias[0]);
                vst1_f16(d + i + j, h0);
            }
            for (; j < biasLen; j++) {
                d[i + j] = q[i + j] * factor + biasPtr[j];
            }
        }
    }
}

EE dequantize_to_f16(TensorDesc qDesc,
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
    EE ret = SUCCESS;
    int length = tensorNumElements(qDesc);
    int biasLength = tensorNumElements(bDesc);
    switch (qDesc.dt) {
        case DT_I8:
            CHECK_REQUIREMENT(biasLength == 0);
            dequantize_i8_f16(length, (INT8 *)qData, scale[0], (F16 *)data);
            break;
        case DT_I32:
            dequantize_i32_f16(
                length, (I32 *)qData, scale[0], biasLength, (F16 *)bData, (F16 *)data);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
