// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/image_arm.h"
#include "arm_neon_expand.h"

#ifdef _USE_FP16
EE resize_bilinear_fp16(TensorDesc inputDesc, F16 *inArray, TensorDesc outputDesc, F16 *outArray)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (idf != DF_NCHWC8 || odf != DF_NCHWC8) {
        CHECK_STATUS(NOT_MATCH);
    }

    F32 strideH = (F32)(ih - 1) / (F32)(oh - 1);
    F32 strideW = (F32)(iw - 1) / (F32)(ow - 1);

    oc /= 8;

    for (U32 n = 0; n < on; n++) {
        for (U32 c = 0; c < oc; c++) {
            I32 outBase = n * oc * oh * ow + c * oh * ow * 8;
            I32 inBase = n * oc * ih * iw + c * ih * iw * 8;
            for (U32 h = 0; h < oh; h++) {
                for (U32 w = 0; w < ow; w++) {
                    if (h == 0 && w == 0) {
                        memcpy(outArray + outBase, inArray + inBase, 8 * bytesOf(DT_F16));
                        continue;
                    }
                    if (h == 0 && w == ow - 1) {
                        memcpy(outArray + outBase + w * 8, inArray + inBase + (iw - 1) * 8,
                            8 * bytesOf(DT_F16));
                        continue;
                    }
                    if (h == oh - 1 && w == 0) {
                        memcpy(outArray + outBase + h * ow * 8,
                            inArray + inBase + (ih - 1) * iw * 8, 8 * bytesOf(DT_F16));
                        continue;
                    }
                    if (h == oh - 1 && w == ow - 1) {
                        memcpy(outArray + outBase + h * ow * 8 + w * 8,
                            inArray + inBase + (ih - 1) * iw * 8 + (iw - 1) * 8,
                            8 * bytesOf(DT_F16));
                        continue;
                    }

                    F32 hC = strideH * h;
                    F32 wC = strideW * w;

                    I32 hT = floor(hC);
                    I32 hB = ceil(hC);
                    I32 wL = floor(wC);
                    I32 wR = ceil(wC);

                    if (hT == hB && wL == wR) {
                        memcpy(outArray + outBase + h * ow * 8 + w * 8,
                            inArray + inBase + hT * iw * 8 + wL * 8, 8 * bytesOf(DT_F16));
                    } else if (hT == hB) {
                        float16x8_t res = {0};
                        float16x8_t vecL = vld1q_f16(inArray + inBase + hT * iw * 8 + wL * 8);
                        float16x8_t vecR = vld1q_f16(inArray + inBase + hT * iw * 8 + wR * 8);
                        res = vfmaq_n_f16(res, vecL, wR - wC);
                        res = vfmaq_n_f16(res, vecR, wC - wL);
                        vst1q_f16(outArray + outBase + h * ow * 8 + w * 8, res);
                    } else if (wL == wR) {
                        float16x8_t res = {0};
                        float16x8_t vecT = vld1q_f16(inArray + inBase + hT * iw * 8 + wL * 8);
                        float16x8_t vecB = vld1q_f16(inArray + inBase + hB * iw * 8 + wL * 8);
                        res = vfmaq_n_f16(res, vecT, hB - hC);
                        res = vfmaq_n_f16(res, vecB, hC - hT);
                        vst1q_f16(outArray + outBase + h * ow * 8 + w * 8, res);
                    } else {
                        float16x8_t res = {0};
                        float16x8_t vecTL = vld1q_f16(inArray + inBase + hT * iw * 8 + wL * 8);
                        float16x8_t vecTR = vld1q_f16(inArray + inBase + hT * iw * 8 + wR * 8);
                        float16x8_t vecBL = vld1q_f16(inArray + inBase + hB * iw * 8 + wL * 8);
                        float16x8_t vecBR = vld1q_f16(inArray + inBase + hB * iw * 8 + wR * 8);
                        res = vfmaq_n_f16(res, vecTL, (hB - hC) * (wR - wC));
                        res = vfmaq_n_f16(res, vecTR, (hB - hC) * (wC - wL));
                        res = vfmaq_n_f16(res, vecBL, (hC - hT) * (wR - wC));
                        res = vfmaq_n_f16(res, vecBR, (hC - hT) * (wC - wL));
                        vst1q_f16(outArray + outBase + h * ow * 8 + w * 8, res);
                    }
                }
            }
        }
    }
    return SUCCESS;
}
#endif

#ifdef _USE_FP32
EE resize_bilinear_fp32(TensorDesc inputDesc, F32 *inArray, TensorDesc outputDesc, F32 *outArray)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (idf != DF_NCHWC8 || odf != DF_NCHWC8) {
        CHECK_STATUS(NOT_MATCH);
    }

    F32 strideH = (F32)(ih - 1) / (F32)(oh - 1);
    F32 strideW = (F32)(iw - 1) / (F32)(ow - 1);

    oc /= 8;

    for (U32 n = 0; n < on; n++) {
        for (U32 c = 0; c < oc; c++) {
            I32 outBase = n * oc * oh * ow + c * oh * ow * 8;
            I32 inBase = n * oc * ih * iw + c * ih * iw * 8;
            for (U32 h = 0; h < oh; h++) {
                for (U32 w = 0; w < ow; w++) {
                    if (h == 0 && w == 0) {
                        memcpy(outArray + outBase, inArray + inBase, 8 * bytesOf(DT_F32));
                        continue;
                    }
                    if (h == 0 && w == ow - 1) {
                        memcpy(outArray + outBase + w * 8, inArray + inBase + (iw - 1) * 8,
                            8 * bytesOf(DT_F32));
                        continue;
                    }
                    if (h == oh - 1 && w == 0) {
                        memcpy(outArray + outBase + h * ow * 8,
                            inArray + inBase + (ih - 1) * iw * 8, 8 * bytesOf(DT_F32));
                        continue;
                    }
                    if (h == oh - 1 && w == ow - 1) {
                        memcpy(outArray + outBase + h * ow * 8 + w * 8,
                            inArray + inBase + (ih - 1) * iw * 8 + (iw - 1) * 8,
                            8 * bytesOf(DT_F32));
                        continue;
                    }

                    F32 hC = strideH * h;
                    F32 wC = strideW * w;

                    I32 hT = floor(hC);
                    I32 hB = ceil(hC);
                    I32 wL = floor(wC);
                    I32 wR = ceil(wC);

                    if (hT == hB && wL == wR) {
                        memcpy(outArray + outBase + h * ow * 8 + w * 8,
                            inArray + inBase + hT * iw * 8 + wL * 8, 8 * bytesOf(DT_F32));
                    } else if (hT == hB) {
                        float32x4_t res[2] = {0};
                        float32x4_t vecL = vld1q_f32(inArray + inBase + hT * iw * 8 + wL * 8);
                        float32x4_t vecL1 = vld1q_f32(inArray + inBase + hT * iw * 8 + wL * 8 + 4);
                        float32x4_t vecR = vld1q_f32(inArray + inBase + hT * iw * 8 + wR * 8);
                        float32x4_t vecR1 = vld1q_f32(inArray + inBase + hT * iw * 8 + wR * 8 + 4);
                        res[0] = vfmaq_n_f32(res[0], vecL, wR - wC);
                        res[1] = vfmaq_n_f32(res[1], vecL1, wR - wC);
                        res[0] = vfmaq_n_f32(res[0], vecR, wC - wL);
                        res[1] = vfmaq_n_f32(res[1], vecR1, wC - wL);
                        vst1q_f32(outArray + outBase + h * ow * 8 + w * 8, res[0]);
                        vst1q_f32(outArray + outBase + h * ow * 8 + w * 8 + 4, res[1]);
                    } else if (wL == wR) {
                        float32x4_t res[2] = {0};
                        float32x4_t vecT = vld1q_f32(inArray + inBase + hT * iw * 8 + wL * 8);
                        float32x4_t vecT1 = vld1q_f32(inArray + inBase + hT * iw * 8 + wL * 8 + 4);
                        float32x4_t vecB = vld1q_f32(inArray + inBase + hB * iw * 8 + wL * 8);
                        float32x4_t vecB1 = vld1q_f32(inArray + inBase + hB * iw * 8 + wL * 8 + 4);
                        res[0] = vfmaq_n_f32(res[0], vecT, hB - hC);
                        res[1] = vfmaq_n_f32(res[1], vecT1, hB - hC);
                        res[0] = vfmaq_n_f32(res[0], vecB, hC - hT);
                        res[1] = vfmaq_n_f32(res[1], vecB1, hC - hT);
                        vst1q_f32(outArray + outBase + h * ow * 8 + w * 8, res[0]);
                        vst1q_f32(outArray + outBase + h * ow * 8 + w * 8 + 4, res[1]);
                    } else {
                        float32x4_t res[2] = {0};
                        float32x4_t vecTL = vld1q_f32(inArray + inBase + hT * iw * 8 + wL * 8);
                        float32x4_t vecTL1 = vld1q_f32(inArray + inBase + hT * iw * 8 + wL * 8 + 4);
                        float32x4_t vecTR = vld1q_f32(inArray + inBase + hT * iw * 8 + wR * 8);
                        float32x4_t vecTR1 = vld1q_f32(inArray + inBase + hT * iw * 8 + wR * 8 + 4);
                        float32x4_t vecBL = vld1q_f32(inArray + inBase + hB * iw * 8 + wL * 8);
                        float32x4_t vecBL1 = vld1q_f32(inArray + inBase + hB * iw * 8 + wL * 8 + 4);
                        float32x4_t vecBR = vld1q_f32(inArray + inBase + hB * iw * 8 + wR * 8);
                        float32x4_t vecBR1 = vld1q_f32(inArray + inBase + hB * iw * 8 + wR * 8 + 4);
                        res[0] = vfmaq_n_f32(res[0], vecTL, (hB - hC) * (wR - wC));
                        res[1] = vfmaq_n_f32(res[1], vecTL1, (hB - hC) * (wR - wC));
                        res[0] = vfmaq_n_f32(res[0], vecTR, (hB - hC) * (wC - wL));
                        res[1] = vfmaq_n_f32(res[1], vecTR1, (hB - hC) * (wC - wL));
                        res[0] = vfmaq_n_f32(res[0], vecBL, (hC - hT) * (wR - wC));
                        res[1] = vfmaq_n_f32(res[1], vecBL1, (hC - hT) * (wR - wC));
                        res[0] = vfmaq_n_f32(res[0], vecBR, (hC - hT) * (wC - wL));
                        res[1] = vfmaq_n_f32(res[1], vecBR1, (hC - hT) * (wC - wL));
                        vst1q_f32(outArray + outBase + h * ow * 8 + w * 8, res[0]);
                        vst1q_f32(outArray + outBase + h * ow * 8 + w * 8 + 4, res[1]);
                    }
                }
            }
        }
    }
    return SUCCESS;
}
#endif

EE resize_bilinear_arm(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP16
        case DT_F16:
            ret = resize_bilinear_fp16(inputDesc, (F16 *)input, outputDesc, (F16 *)output);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            ret = resize_bilinear_fp32(inputDesc, (F32 *)input, outputDesc, (F32 *)output);
            break;
#endif
        default:
            return NOT_SUPPORTED;
    }
    return ret;
}
