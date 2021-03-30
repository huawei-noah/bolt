// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/general/tensor_computing_general.h"
#include "cpu/general/general_functions.h"

template <typename T>
inline EE deconvolution(TensorDesc inputDesc,
    T *inArray,
    TensorDesc filterDesc,
    const T *filterArray,
    ConvolutionParamSpec convParamSpec,
    const T *biasArray,
    TensorDesc outputDesc,
    T *outArray,
    ActivationParamSpec activationDesc)
{
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 group = convParamSpec.group;
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingL = convParamSpec.padding_left;
    U32 ocGroupSize = oc / group;

    // initialize outputs to 0
    memset(outArray, 0, tensorNumBytes(outputDesc));
    U32 ic8 = ic / 8;
    U32 oc8 = oc / 8;
    for (U32 n = 0; n < in; n++) {
        for (U32 o = 0; o < oc; o++) {
            U32 groupId = o / ocGroupSize;
            U32 icStart = groupId * fn;
            U32 icEnd = (groupId + 1) * fn;
            for (U32 c = icStart; c < icEnd; c++) {
                for (U32 h = 0; h < ih; h++) {
                    for (U32 w = 0; w < iw; w++) {
                        U32 i_off;
                        if (idf == DF_NCHW) {
                            i_off = ((n * ic + c) * ih + h) * iw + w;
                        } else {
                            i_off = (((n * ic8 + (c / 8)) * ih + h) * iw + w) * 8 + c % 8;
                        }
                        for (I32 fh_idx = 0; fh_idx < (I32)fh; fh_idx++) {
                            for (I32 fw_idx = 0; fw_idx < (I32)fw; fw_idx++) {
                                I32 oh_idx = fh_idx + strideH * h - paddingT;
                                I32 ow_idx = fw_idx + strideW * w - paddingL;
                                if (oh_idx >= 0 && oh_idx < (I32)oh && ow_idx >= 0 &&
                                    ow_idx < (I32)ow) {
                                    U32 o_off;
                                    if (odf == DF_NCHW) {
                                        o_off = ((n * oc + o) * oh + oh_idx) * ow + ow_idx;
                                    } else {
                                        o_off =
                                            (((n * oc8 + (o / 8)) * oh + oh_idx) * ow + ow_idx) * 8 +
                                            o % 8;
                                    }
                                    U32 f_off =
                                        (((c - icStart) * fc + o) * fh + fh_idx) * fw + fw_idx;
                                    outArray[o_off] += inArray[i_off] * filterArray[f_off];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // bias
    U32 ohow = oh * ow;
    for (U32 i = 0; i < tensorNumElements(outputDesc); i++) {
        U32 o;
        if (odf == DF_NCHW) {
            o = (i / ohow) % oc;
        } else {
            o = (i / (ohow * 8)) % oc8 * 8 + i % 8;
        }
        outArray[i] += biasArray[o];
        switch (activationDesc.mode) {
            case ACTIVATION_NULL: {
                break;
            }
            case ACTIVATION_RELU: {
                F32 tmp = activationDesc.value[0] * outArray[i];
                if (outArray[i] < tmp) {
                    outArray[i] = tmp;
                }
                break;
            }
            default:
                return NOT_SUPPORTED;
        }
    }
    return SUCCESS;
}

EE deconvolution_general(TensorDesc inputDesc,
    void *input,
    TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    TensorDesc scaleDesc,
    const void *scale,
    TensorDesc biasDesc,
    const void *bias,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec activationDesc)
{
    UNUSED(scaleDesc);
    UNUSED(scale);
    UNUSED(biasDesc);

    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP16
        case DT_F16:
            ret = deconvolution<F16>(inputDesc, (F16 *)input, filterDesc, (F16 *)filter,
                convParamSpec, (F16 *)bias, outputDesc, (F16 *)output, activationDesc);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            ret = deconvolution<F32>(inputDesc, (F32 *)input, filterDesc, (F32 *)filter,
                convParamSpec, (F32 *)bias, outputDesc, (F32 *)output, activationDesc);
            break;
#endif
        default:
            return NOT_SUPPORTED;
    }
    return ret;
}
