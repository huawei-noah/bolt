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
    PoolingParamSpec poolingParamSpec,
    void *scale,
    TensorDesc outputDesc,
    void *output)
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

    U32 strideH = poolingParamSpec.stride_h;
    U32 strideW = poolingParamSpec.stride_w;
    U32 paddingT = poolingParamSpec.padding_top;
    U32 paddingL = poolingParamSpec.padding_left;
    U32 kernelSizeH = poolingParamSpec.kernel_h;
    U32 kernelSizeW = poolingParamSpec.kernel_w;
    if (paddingT >= kernelSizeH || paddingL >= kernelSizeW) {
        ret = NOT_SUPPORTED;
    }

    ic /= 8;
    const U8 *inputPtr = (const U8 *)input;
    U8 *outputPtr = (U8 *)output;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 h = 0; h < oh; h++) {
                for (U32 w = 0; w < ow; w++, outputPtr += 8 * bytesOf(odt)) {
                    int hstart = UNI_MAX((int)h * (int)strideH - (int)paddingT, 0);
                    int wstart = UNI_MAX((int)w * (int)strideW - (int)paddingL, 0);
                    int hend = UNI_MIN(hstart + kernelSizeH, ih);
                    int wend = UNI_MIN(wstart + kernelSizeW, iw);
                    int poolSize = (hend - hstart) * (wend - wstart);
                    switch (idt) {
#ifdef _USE_FP32
                        case DT_F32:
                            ret = pooling_c8_fp32((const F32 *)inputPtr, iw, hstart, hend, wstart,
                                wend, (F32 *)outputPtr, poolingParamSpec);
                            break;
#endif
#ifdef _USE_FP16
                        case DT_F16:
                            // Global average pooling kernel can be very big. Accumulate to FP32 to protect accurracy
                            if (poolSize > 256 && poolingParamSpec.mode == POOLING_MEAN) {
                                ret = pooling_c8_big_fp16((const F16 *)inputPtr, iw, hstart, hend,
                                    wstart, wend, (F16 *)outputPtr, poolSize);
                            } else {
                                ret = pooling_c8_fp16((const F16 *)inputPtr, iw, hstart, hend,
                                    wstart, wend, (F16 *)outputPtr, poolingParamSpec);
                            }
                            break;
#endif
#ifdef _USE_INT8
                        case DT_I8:
                            ret = pooling_c8_int8((const INT8 *)inputPtr, iw, hstart, hend, wstart,
                                wend, (INT8 *)outputPtr, poolingParamSpec, scale);
                            break;
#endif
                        default:
                            ret = NOT_SUPPORTED;
                            break;
                    }
                }
            }
            inputPtr += ih * iw * 8 * bytesOf(idt);
        }
    }
    return ret;
}

EE pooling_bp_arm(TensorDesc inputDesc,
    const void *input,
    PoolingParamSpec poolingParamSpec,
    TensorDesc outputDesc,
    void *output)
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

    U32 strideH = poolingParamSpec.stride_h;
    U32 strideW = poolingParamSpec.stride_w;
    U32 paddingT = poolingParamSpec.padding_top;
    U32 paddingL = poolingParamSpec.padding_left;
    U32 kernelSizeH = poolingParamSpec.kernel_h;
    U32 kernelSizeW = poolingParamSpec.kernel_w;
    if (paddingT >= kernelSizeH || paddingL >= kernelSizeW) {
        ret = NOT_SUPPORTED;
    }

    ic /= 8;
    const U8 *inputPtr = (const U8 *)input;
    U8 *outputPtr = (U8 *)output;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 h = 0; h < ih; h++) {
                for (U32 w = 0; w < iw; w++, inputPtr += 8 * bytesOf(idt)) {
                    int hstart = (int)h * (int)strideH - (int)paddingT;
                    int wstart = (int)w * (int)strideW - (int)paddingL;
                    int hend = UNI_MIN(hstart + kernelSizeH, oh);
                    int wend = UNI_MIN(wstart + kernelSizeW, ow);
                    hstart = UNI_MAX(hstart, 0);
                    wstart = UNI_MAX(wstart, 0);
                    switch (idt) {
#ifdef _USE_FP32
                        case DT_F32:
                            ret = pooling_bp_c8_fp32((const F32 *)inputPtr, hstart, hend, wstart,
                                wend, (F32 *)outputPtr, ow, poolingParamSpec);
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
