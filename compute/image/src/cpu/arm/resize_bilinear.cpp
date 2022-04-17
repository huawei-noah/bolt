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
#include "uni.h"

#ifdef _USE_FP16
EE resize_bilinear_fp16(
    TensorDesc inputDesc, F16 *inArray, ResizeParamSpec p, TensorDesc outputDesc, F16 *outArray)
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
    F32 strideH, strideW;
    if (p.trans_mode == COORDINATE_TRANS_ALIGN_CORNERS) {
        strideH = ((F32)ih - 1) / (oh - 1);
        strideW = ((F32)iw - 1) / (ow - 1);
    } else {
        strideH = ((F32)ih) / oh;
        strideW = ((F32)iw) / ow;
    }
    U32 ic_align = 8, oc_align = 8;
    ic /= ic_align;
    oc /= oc_align;
    for (U32 n = 0, dst = 0; n < on; n++) {
        for (U32 c = 0; c < oc; c++) {
            for (U32 h = 0; h < oh; h++) {
                F32 hC = strideH * h;
                U32 hT = floor(hC);
                U32 hB = ceil(hC);
                if (hB == hT) {
                    hB = hT + 1;
                }
                U32 hBB = UNI_MIN(hB, ih - 1);
                for (U32 w = 0; w < ow; w++, dst += 8) {
                    F32 wC = strideW * w;
                    U32 wL = floor(wC);
                    U32 wR = ceil(wC);
                    if (wL == wR) {
                        wR = wL + 1;
                    }
                    U32 wRR = UNI_MIN(wR, iw - 1);
                    F32 factorTL = (hB - hC) * (wR - wC);
                    F32 factorTR = (hB - hC) * (wC - wL);
                    F32 factorBL = (hC - hT) * (wR - wC);
                    F32 factorBR = (hC - hT) * (wC - wL);

                    U32 srcTL = (((n * ic + c) * ih + hT) * iw + wL) * ic_align;
                    U32 srcTR = (((n * ic + c) * ih + hT) * iw + wRR) * ic_align;
                    U32 srcBL = (((n * ic + c) * ih + hBB) * iw + wL) * ic_align;
                    U32 srcBR = (((n * ic + c) * ih + hBB) * iw + wRR) * ic_align;
                    float16x8_t a0 = vld1q_f16(inArray + srcTL);
                    float16x8_t a1 = vld1q_f16(inArray + srcTR);
                    float16x8_t a2 = vld1q_f16(inArray + srcBL);
                    float16x8_t a3 = vld1q_f16(inArray + srcBR);
                    float16x8_t res = vmulq_n_f16(a0, factorTL);
                    res = vfmaq_n_f16(res, a1, factorTR);
                    res = vfmaq_n_f16(res, a2, factorBL);
                    res = vfmaq_n_f16(res, a3, factorBR);
                    vst1q_f16(outArray + dst, res);
                }
            }
        }
    }
    return SUCCESS;
}
#endif

#ifdef _USE_FP32
EE resize_bilinear_fp32(
    TensorDesc inputDesc, F32 *inArray, ResizeParamSpec p, TensorDesc outputDesc, F32 *outArray)
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
    F32 strideH, strideW;
    if (p.trans_mode == COORDINATE_TRANS_ALIGN_CORNERS) {
        strideH = ((F32)ih - 1) / (oh - 1);
        strideW = ((F32)iw - 1) / (ow - 1);
    } else {
        strideH = ((F32)ih) / oh;
        strideW = ((F32)iw) / ow;
    }
    U32 ic_align = 8, oc_align = 8;
    ic /= ic_align;
    oc /= oc_align;
    for (U32 n = 0, dst = 0; n < on; n++) {
        for (U32 c = 0; c < oc; c++) {
            for (U32 h = 0; h < oh; h++) {
                F32 hC = strideH * h;
                U32 hT = floor(hC);
                U32 hB = ceil(hC);
                if (hB == hT) {
                    hB = hT + 1;
                }
                U32 hBB = UNI_MIN(hB, ih - 1);
                for (U32 w = 0; w < ow; w++, dst += 8) {
                    F32 wC = strideW * w;
                    U32 wL = floor(wC);
                    U32 wR = ceil(wC);
                    if (wL == wR) {
                        wR = wL + 1;
                    }
                    U32 wRR = UNI_MIN(wR, iw - 1);
                    F32 factorTL = (hB - hC) * (wR - wC);
                    F32 factorTR = (hB - hC) * (wC - wL);
                    F32 factorBL = (hC - hT) * (wR - wC);
                    F32 factorBR = (hC - hT) * (wC - wL);

                    U32 srcTL = (((n * ic + c) * ih + hT) * iw + wL) * ic_align;
                    U32 srcTR = (((n * ic + c) * ih + hT) * iw + wRR) * ic_align;
                    U32 srcBL = (((n * ic + c) * ih + hBB) * iw + wL) * ic_align;
                    U32 srcBR = (((n * ic + c) * ih + hBB) * iw + wRR) * ic_align;
                    float32x4_t a00 = vld1q_f32(inArray + srcTL);
                    float32x4_t a01 = vld1q_f32(inArray + srcTL + 4);
                    float32x4_t a10 = vld1q_f32(inArray + srcTR);
                    float32x4_t a11 = vld1q_f32(inArray + srcTR + 4);
                    float32x4_t a20 = vld1q_f32(inArray + srcBL);
                    float32x4_t a21 = vld1q_f32(inArray + srcBL + 4);
                    float32x4_t a30 = vld1q_f32(inArray + srcBR);
                    float32x4_t a31 = vld1q_f32(inArray + srcBR + 4);
                    float32x4_t res0 = vmulq_n_f32(a00, factorTL);
                    float32x4_t res1 = vmulq_n_f32(a01, factorTL);
                    res0 = vfmaq_n_f32(res0, a10, factorTR);
                    res1 = vfmaq_n_f32(res1, a11, factorTR);
                    res0 = vfmaq_n_f32(res0, a20, factorBL);
                    res1 = vfmaq_n_f32(res1, a21, factorBL);
                    res0 = vfmaq_n_f32(res0, a30, factorBR);
                    res1 = vfmaq_n_f32(res1, a31, factorBR);
                    vst1q_f32(outArray + dst, res0);
                    vst1q_f32(outArray + dst + 4, res1);
                }
            }
        }
    }
    return SUCCESS;
}
#endif

EE resize_bilinear_arm(
    TensorDesc inputDesc, void *input, ResizeParamSpec p, TensorDesc outputDesc, void *output)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP16
        case DT_F16:
            ret = resize_bilinear_fp16(inputDesc, (F16 *)input, p, outputDesc, (F16 *)output);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            ret = resize_bilinear_fp32(inputDesc, (F32 *)input, p, outputDesc, (F32 *)output);
            break;
#endif
        default:
            return NOT_SUPPORTED;
    }
    return ret;
}
