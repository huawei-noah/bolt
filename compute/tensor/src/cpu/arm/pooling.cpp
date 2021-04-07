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
#ifdef _USE_FP32
#include "cpu/arm/fp32/tensor_computing_fp32.h"
#endif
#ifdef _USE_FP16
#include "cpu/arm/fp16/tensor_computing_fp16.h"
#endif
#ifdef _USE_INT8
#include "cpu/arm/int8/tensor_computing_int8.h"
#endif

EE pooling_arm(TensorDesc inputDesc,
    const void *input,
    PoolingParamSpec p,
    void *scale,
    TensorDesc outputDesc,
    void *output)
{
    if (nullptr == input || nullptr == output) {
        return NULL_POINTER;
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, it, ih, iw;
    U32 on, oc, ot, oh, ow;
    if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
        it = ot = 1;
        p.padding_before = p.padding_after = 0;
        p.kernel_t = p.stride_t = 1;
    } else if (tensorIs5d(inputDesc)) {
        CHECK_STATUS(tensor5dGet(inputDesc, &idt, &idf, &in, &ic, &it, &ih, &iw));
        CHECK_STATUS(tensor5dGet(outputDesc, &odt, &odf, &on, &oc, &ot, &oh, &ow));
    } else {
        return NOT_SUPPORTED;
    }
    EE ret = SUCCESS;
    if (idt != odt) {
        ret = NOT_MATCH;
    }
    if (in != on || ic != oc) {
        ret = NOT_MATCH;
    }
    if (idf != DF_NCHWC8 || odf != idf) {
        ret = NOT_MATCH;
    }
    if (p.padding_before >= p.kernel_t || p.padding_top >= p.kernel_h ||
        p.padding_left >= p.kernel_w) {
        ret = NOT_SUPPORTED;
    }

    ic /= 8;
    const U8 *inputPtr = (const U8 *)input;
    U8 *outputPtr = (U8 *)output;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 t = 0; t < ot; t++) {
                for (U32 h = 0; h < oh; h++) {
                    for (U32 w = 0; w < ow; w++, outputPtr += 8 * bytesOf(odt)) {
                        int tstart = t * (int)p.stride_t - (int)p.padding_before;
                        int hstart = h * (int)p.stride_h - (int)p.padding_top;
                        int wstart = w * (int)p.stride_w - (int)p.padding_left;
                        int tend = UNI_MIN(tstart + (int)p.kernel_t, (int)it);
                        int hend = UNI_MIN(hstart + (int)p.kernel_h, (int)ih);
                        int wend = UNI_MIN(wstart + (int)p.kernel_w, (int)iw);
                        tstart = UNI_MAX(tstart, 0);
                        hstart = UNI_MAX(hstart, 0);
                        wstart = UNI_MAX(wstart, 0);
                        int poolSize = (tend - tstart) * (hend - hstart) * (wend - wstart);
                        switch (idt) {
#ifdef _USE_FP32
                            case DT_F32:
                                ret = pooling_c8_fp32(tstart, tend, hstart, hend, wstart, wend,
                                    poolSize, (const F32 *)inputPtr, it, ih, iw, p,
                                    (F32 *)outputPtr);
                                break;
#endif
#ifdef _USE_FP16
                            case DT_F16:
                                ret = pooling_c8_fp16(tstart, tend, hstart, hend, wstart, wend,
                                    poolSize, (const F16 *)inputPtr, it, ih, iw, p,
                                    (F16 *)outputPtr);
                                break;
#endif
#ifdef _USE_INT8
                            case DT_I8:
                                ret = pooling_c8_int8(tstart, tend, hstart, hend, wstart, wend,
                                    poolSize, (const INT8 *)inputPtr, it, ih, iw, p,
                                    (INT8 *)outputPtr, scale);
                                break;
#endif
                            default:
                                ret = NOT_SUPPORTED;
                                break;
                        }
                    }
                }
            }
            inputPtr += it * ih * iw * 8 * bytesOf(idt);
        }
    }
    return ret;
}

EE pooling_bp_arm(
    TensorDesc inputDesc, const void *input, PoolingParamSpec p, TensorDesc outputDesc, void *output)
{
    EE ret = SUCCESS;
    if (nullptr == input || nullptr == output) {
        ret = NULL_POINTER;
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in = 0, ic = 0, ih = 0, iw = 0, on = 0, oc = 0, oh = 0, ow = 0;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (idt != odt) {
        ret = NOT_MATCH;
    }
    if (in != on || ic != oc) {
        ret = NOT_MATCH;
    }
    if (idf != DF_NCHWC8 || odf != idf) {
        ret = NOT_MATCH;
    }
    if (p.padding_top >= p.kernel_h || p.padding_left >= p.kernel_w) {
        ret = NOT_SUPPORTED;
    }

    ic /= 8;
    const U8 *inputPtr = (const U8 *)input;
    U8 *outputPtr = (U8 *)output;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 h = 0; h < ih; h++) {
                for (U32 w = 0; w < iw; w++, inputPtr += 8 * bytesOf(idt)) {
                    int hstart = (int)h * (int)p.stride_h - (int)p.padding_top;
                    int wstart = (int)w * (int)p.stride_w - (int)p.padding_left;
                    int hend = UNI_MIN(hstart + p.kernel_h, oh);
                    int wend = UNI_MIN(wstart + p.kernel_w, ow);
                    hstart = UNI_MAX(hstart, 0);
                    wstart = UNI_MAX(wstart, 0);
                    switch (idt) {
#ifdef _USE_FP32
                        case DT_F32:
                            ret = pooling_bp_c8_fp32((const F32 *)inputPtr, hstart, hend, wstart,
                                wend, (F32 *)outputPtr, ow, p);
                            break;
#endif
                        default:
                            ret = NOT_SUPPORTED;
                            break;
                    }
                }
            }
            outputPtr += oh * ow * 8 * bytesOf(odt);
        }
    }
    return ret;
}
