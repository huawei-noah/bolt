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

EE depthwise_pointwise_convolution_infer_forward_tmp_bytes_general(TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    U32 *bytes)
{
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(dwFilterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    U32 elementSize = bytesOf(fdt);
    if (fdt == DT_I8) {
        elementSize = bytesOf(DT_I32);
    }
    *bytes = ic * oh * ow * elementSize;
    return SUCCESS;
}

template <typename T1, typename T2, typename T3>
inline EE depthwise_pointwise_convolution(TensorDesc inputDesc,
    T1 *inArray,
    TensorDesc dwFilterDesc,
    const T2 *dwFilterArray,
    TensorDesc pwFilterDesc,
    const T2 *pwFilterArray,
    ConvolutionParamSpec convParamSpec,
    const T3 *dwBiasArray,
    const T3 *pwBiasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    T3 *outArray,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec)
{
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(dwFilterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingL = convParamSpec.padding_left;
    U32 dilatedRateH = convParamSpec.dilatedRate_h;
    U32 dilatedRateW = convParamSpec.dilatedRate_w;

    bool fuseDepthwisePointwise = (pwFilterArray == nullptr) ? false : true;

    T3 *pwArray;
    if (fuseDepthwisePointwise) {
        CHECK_REQUIREMENT(tmpBytes >= ic * oh * ow * sizeof(T3));
        pwArray = (T3 *)tmp;
    } else {
        pwArray = outArray;
    }
    U32 ic8 = ic / 8;
    U32 oc8 = oc / 8;
    for (U32 n = 0; n < in; n++) {
        // dw conv
        for (U32 c = 0, pw_off = 0; c < ic; c++) {
            for (U32 h = 0; h < oh; h++) {
                for (U32 w = 0; w < ow; w++, pw_off++) {
                    T3 value = dwBiasArray[c];
                    for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                        for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                            I32 ih_idx = h * strideH - paddingT + fh_idx * dilatedRateH;
                            I32 iw_idx = w * strideW - paddingL + fw_idx * dilatedRateW;
                            if (ih_idx >= 0 && ih_idx < (I32)ih && iw_idx >= 0 && iw_idx < (I32)iw) {
                                U32 i_off;
                                if (idf == DF_NCHW) {
                                    i_off = ((n * ic + c) * ih + ih_idx) * iw + iw_idx;
                                } else {
                                    i_off = (((n * ic8 + (c / 8)) * ih + ih_idx) * iw + iw_idx) * 8 +
                                        c % 8;
                                }
                                value += inArray[i_off] *
                                    dwFilterArray[c * fh * fw + fh_idx * fw + fw_idx];
                            }
                        }
                    }
                    CHECK_STATUS(
                        activation_template<T3>(depthwiseActivationParamSpec, value, &value));

                    if (fuseDepthwisePointwise || odf == DF_NCHW) {
                        pwArray[pw_off] = value;
                    } else {
                        pwArray[(((n * ic8 + (c / 8)) * oh + h) * ow + w) * 8 + c % 8] = value;
                    }
                }
            }
        }
        if (fuseDepthwisePointwise) {
            // pw conv
            for (U32 o = 0; o < oc; o++) {
                for (U32 hw = 0; hw < oh * ow; hw++) {
                    T3 value = pwBiasArray[o];
                    for (U32 c = 0; c < ic; c++) {
                        U32 pw_off = c * oh * ow + hw;
                        value += pwArray[pw_off] * pwFilterArray[o * ic + c];
                    }
                    CHECK_STATUS(
                        activation_template<T3>(pointwiseActivationParamSpec, value, &value));
                    U32 o_off;
                    if (odf == DF_NCHW) {
                        o_off = (n * oc + o) * oh * ow + hw;
                    } else {
                        o_off = ((n * oc8 + (o / 8)) * oh * ow + hw) * 8 + o % 8;
                    }
                    outArray[o_off] = value;
                }
            }
        }
    }
    return SUCCESS;
}

EE depthwise_pointwise_convolution_general(TensorDesc inputDesc,
    void *input,
    TensorDesc dwFilterDesc,
    const void *dwFilter,
    TensorDesc pwFilterDesc,
    const void *pwFilter,
    ConvolutionParamSpec convParamSpec,
    TensorDesc dwBiasDesc,
    const void *dwBias,
    TensorDesc pwBiasDesc,
    const void *pwBias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP16
        case DT_F16:
            ret = depthwise_pointwise_convolution<F16, F16, F16>(inputDesc, (F16 *)input,
                dwFilterDesc, (F16 *)dwFilter, pwFilterDesc, (F16 *)pwFilter, convParamSpec,
                (F16 *)dwBias, (F16 *)pwBias, tmpBytes, tmp, outputDesc, (F16 *)output,
                depthwiseActivationParamSpec, pointwiseActivationParamSpec);
            break;
#endif
#ifdef _USE_INT8
        case DT_I8:
            ret = depthwise_pointwise_convolution<INT8, INT8, I32>(inputDesc, (INT8 *)input,
                dwFilterDesc, (INT8 *)dwFilter, pwFilterDesc, (INT8 *)pwFilter, convParamSpec,
                (I32 *)dwBias, (I32 *)pwBias, tmpBytes, tmp, outputDesc, (I32 *)output,
                depthwiseActivationParamSpec, pointwiseActivationParamSpec);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            ret = depthwise_pointwise_convolution<F32, F32, F32>(inputDesc, (F32 *)input,
                dwFilterDesc, (F32 *)dwFilter, pwFilterDesc, (F32 *)pwFilter, convParamSpec,
                (F32 *)dwBias, (F32 *)pwBias, tmpBytes, tmp, outputDesc, (F32 *)output,
                depthwiseActivationParamSpec, pointwiseActivationParamSpec);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
