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

EE deconvolution_transform_filter_arm(TensorDesc filterDesc,
    const void *filter,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    void *filterTransformed)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = deconvolution_transform_filter_fp32(
                filterDesc, (F32 *)filter, algorithm, ftmDesc, (F32 *)filterTransformed);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = deconvolution_transform_filter_fp16(
                filterDesc, (F16 *)filter, algorithm, ftmDesc, (F16 *)filterTransformed);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

template <typename T>
EE deconvolution_overlap_crop_arm_kernel(T *input,
    T *output,
    TensorDesc inputDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 fh = convParamSpec.kernel_h;
    U32 fw = convParamSpec.kernel_w;
    U32 fhfw = fh * fw;
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingL = convParamSpec.padding_left;
    U32 tileSize = bytesOf(odt);
    for (U32 kn = 0; kn < in; ++kn) {
        for (U32 kh = 0; kh < ih; ++kh) {
            for (U32 kw = 0; kw < iw; ++kw) {
                for (U32 kc = 0; kc < oc; kc += 8) {
                    for (U32 jh = 0; jh < fh; ++jh) {
                        for (U32 jw = 0; jw < fw; ++jw) {
                            for (U32 kc8 = 0; kc8 < 8; ++kc8) {
                                U32 ohIdx = kh * strideH + jh;
                                U32 owIdx = kw * strideW + jw;
                                if ((ohIdx < paddingT) || (ohIdx >= oh + paddingT) ||
                                    (owIdx < paddingL) || (owIdx >= ow + paddingL)) {
                                    continue;
                                }
                                ohIdx -= paddingT;
                                owIdx -= paddingL;
                                U32 oidx = (kc * oh + ohIdx * 8) * ow + owIdx * 8 + kc8;
                                U32 iidx = ((kh * iw + kw) * oc + kc + kc8) * fhfw + jh * fw + jw;
                                output[oidx] += input[iidx];
                            }
                        }
                    }
                }
            }
        }
        output += oc * oh * ow * tileSize;
        input += ic * ih * iw * tileSize;
    }

    return SUCCESS;
}

EE deconvolution_overlap_crop_arm(void *input,
    void *output,
    TensorDesc inputDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = deconvolution_overlap_crop_arm_kernel<F32>(
                (F32 *)input, (F32 *)output, inputDesc, outputDesc, convParamSpec);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = deconvolution_overlap_crop_arm_kernel<F16>(
                (F16 *)input, (F16 *)output, inputDesc, outputDesc, convParamSpec);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}