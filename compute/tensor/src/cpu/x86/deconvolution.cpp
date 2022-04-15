// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/tensor_computing_x86.h"
#ifdef _USE_FP32
#include "cpu/x86/fp32/tensor_computing_fp32.h"
#include "cpu/x86/fp32/transform_functions_fp32.h"
#endif
#ifdef _USE_FP16
#include "cpu/x86/fp16/tensor_computing_fp16.h"
#endif
#include "cpu/x86/x86_functions.h"
#include "blas_enhance.h"

EE deconvolution_transform_filter_x86(TensorDesc filterDesc,
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
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE deconvolution_overlap_crop_x86(void *input,
    void *output,
    TensorDesc inputDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec)
{
    switch (outputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            deconvOverlapAndCrop((F32 *)input, (F32 *)output, inputDesc, outputDesc, convParamSpec);
            break;
        }
#endif
        default:
            return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE deconvolution_overlap_crop_c8_x86(void *input,
    void *output,
    TensorDesc inputDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec)
{
    switch (outputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            deconvOverlapAndCropNCHWC8(
                (F32 *)input, (F32 *)output, inputDesc, outputDesc, convParamSpec);
            break;
        }
#endif
        default:
            return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE deconvolution_overlap_crop_equal_c8_x86(void *input,
    void *output,
    const void *bias,
    TensorDesc inputDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec)
{
    switch (outputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            deconvOverlapAndCropEqualNCHWC8((F32 *)input, (F32 *)output, (const F32 *)bias,
                inputDesc, outputDesc, convParamSpec);
            break;
        }
#endif
        default:
            return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE deconvolution_pointwise_x86(TensorDesc inputDesc,
    void *input,
    TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec activationDesc,
    Arch arch)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 fh = convParamSpec.kernel_h;
    U32 fw = convParamSpec.kernel_w;
    CHECK_REQUIREMENT(idf == DF_NCHWC8);

    ConvolutionParamSpec p = createConvolutionParamSpec(
        1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, oc, Convolution_Pointwise);
    TensorDesc nullDesc;
    U8 *convBias = (U8 *)tmp;
    if (fh == convParamSpec.stride_h && fw == convParamSpec.stride_w) {
        for (U32 ii = 0; ii < fh * fw; ++ii) {
            memcpy(convBias + ii * oc * bytesOf(odt), bias, oc * bytesOf(odt));
        }
    } else {
        memset(convBias, 0, oc * fh * fw * bytesOf(odt));
    }
    TensorDesc convOutDesc = tensor4df(odt, DF_NCHWC8, in, oc * fh * fw, ih, iw);
    U8 *convOut = (U8 *)tmp + oc * fh * fw * bytesOf(odt);
    ActivationParamSpec convActivationDesc;
    convActivationDesc.mode = ACTIVATION_NULL;

    convolution_x86(inputDesc, input, nullptr, filterDesc, filter, p,
        CONVOLUTION_ALGORITHM_POINTWISE, nullDesc, nullptr, nullDesc, convBias, 0, tmp, convOutDesc,
        convOut, convActivationDesc, arch);

    if (fh == convParamSpec.stride_h && fw == convParamSpec.stride_w) {
        deconvolution_overlap_crop_equal_c8_x86(
            convOut, output, bias, inputDesc, outputDesc, convParamSpec);
    } else {
        U8 *tmpOutputPtr = (U8 *)output;
        U32 biasTileSize = bytesOf(biasDesc.dt) * 8;
        U8 *biasPtr = (U8 *)bias;
        for (U32 c = 0; c < oc / 8; c++, biasPtr += biasTileSize) {
            for (U32 n = 0; n < oh * ow; n++) {
                memcpy(tmpOutputPtr, biasPtr, biasTileSize);
                tmpOutputPtr += biasTileSize;
            }
        }
        deconvolution_overlap_crop_c8_x86(convOut, output, inputDesc, outputDesc, convParamSpec);
    }

    if (activationDesc.mode != ACTIVATION_NULL) {
        CHECK_STATUS(array_activation_x86(
            odt, output, tensorNumElements(outputDesc), activationDesc, output));
    }
    return SUCCESS;
}