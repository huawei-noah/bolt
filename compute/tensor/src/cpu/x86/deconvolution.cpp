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
#ifdef _USE_INT8
#include "cpu/x86/int8/tensor_computing_int8.h"
#include "cpu/tensor_computing_cpu.h"
#endif
#include "cpu/x86/x86_functions.h"
#include "blas_enhance.h"
#include "tensor_transpose.h"

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
#ifdef _USE_INT8
        case DT_I8: {
            ret = deconvolution_transform_filter_int8(
                filterDesc, (const INT8 *)filter, algorithm, ftmDesc, (INT8 *)filterTransformed);
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
            deconvOverlapAndCropF32((F32 *)input, (F32 *)output, inputDesc, outputDesc, convParamSpec);
            break;
        }
#endif
        case DT_I32: {
            deconvOverlapAndCropI32((I32 *)input, (I32 *)output, inputDesc, outputDesc, convParamSpec);
            break;
        }
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
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            deconvOverlapAndCropNCHWC8F32(
                (F32 *)input, (F32 *)output, inputDesc, outputDesc, convParamSpec);
            break;
        }
#endif
        case DT_I32: {
            deconvOverlapAndCropNCHWC8I32(
                (I32 *)input, (I32 *)output, inputDesc, outputDesc, convParamSpec);
            break;
        }
        default:
            return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE deconvolution_overlap_crop_equal_c8_x86(void *input,
    void *output,
    TensorDesc inputDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec)
{
    switch (outputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            deconvOverlapAndCropEqualNCHWC8(
                (F32 *)input, (F32 *)output, inputDesc, outputDesc, convParamSpec);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8:
        case DT_U8_Q: {
            deconvOverlapAndCropEqualNCHWC8(
                (INT8 *)input, (INT8 *)output, inputDesc, outputDesc, convParamSpec);
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
    TensorDesc scaleDesc,
    void *scale,
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
    CHECK_REQUIREMENT(idf == DF_NCHWC8 || idf == DF_NCHWC16);

    U32 cx = (idf == DF_NCHWC16) ? 16 : 8;

    ConvolutionParamSpec p = createConvolutionParamSpec(
        1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, oc, CONVOLUTION_POINTWISE);
    TensorDesc convBiasDesc = tensor1d(DT_F32, oc * fh * fw);
    U8 *convBias = (U8 *)tmp;
    DataType convOdt = odt;
    if (fh == convParamSpec.stride_h && fw == convParamSpec.stride_w) {
        for (U32 ii = 0; ii < fh * fw; ++ii) {
            UNI_MEMCPY(convBias + ii * oc * bytesOf(DT_F32), bias, oc * bytesOf(DT_F32));
        }
        if (idt == DT_U8_Q) {
            convOdt = DT_U8_Q;
        }
    } else {
        UNI_MEMSET(convBias, 0, oc * fh * fw * bytesOf(DT_F32));
        if (convOdt == DT_U8_Q) {
            convOdt = DT_I32;
        }
    }
    TensorDesc convOutDesc = tensor4df(convOdt, inputDesc.df, in, oc * fh * fw, ih, iw);
    U8 *convOut = (U8 *)tmp + oc * fh * fw * bytesOf(DT_F32);
    ActivationParamSpec convActivationDesc;
    convActivationDesc.mode = ACTIVATION_NULL;

    tmp = (void *)(convOut + tensorNumBytes(convOutDesc));
    convolution_x86(inputDesc, input, nullptr, filterDesc, filter, p,
        CONVOLUTION_ALGORITHM_POINTWISE, scaleDesc, scale, convBiasDesc, convBias, tmpBytes, tmp,
        convOutDesc, convOut, convActivationDesc, arch);

    void *OriTmpOutputPtr = output;
    if (fh == convParamSpec.stride_h && fw == convParamSpec.stride_w) {
        if (convOdt != odt) {
            OriTmpOutputPtr = tmp;
        }
        TensorDesc iDesc = inputDesc;
        TensorDesc oDesc = outputDesc;
        iDesc.dt = convOdt;
        oDesc.dt = convOdt;
        CHECK_STATUS(deconvolution_overlap_crop_equal_c8_x86(
            convOut, OriTmpOutputPtr, iDesc, oDesc, convParamSpec));
    } else {

#ifdef _USE_INT8
        if (convOdt == DT_I32) {
            // quantize bias to I32
            I32 *biasI = (I32 *)tmp;
            tmp = (U8 *)tmp + tensorNumBytes(biasDesc);
            const F32 *biasF = (const F32 *)bias;
            F32 *factor = (F32 *)scale;
            for (U32 i = 0; i < tensorNumElements(biasDesc); i++) {
                biasI[i] = round(factor[0] * factor[2] * biasF[i]);
            }
            bias = biasI;
        }
#endif

        U8 *tmpOutputPtr = (U8 *)output;
        if (convOdt != odt) {
            tmpOutputPtr = (U8 *)tmp;
        }
        OriTmpOutputPtr = tmpOutputPtr;
        U32 biasTileSize = bytesOf(biasDesc.dt) * cx;
        for (U32 n = 0; n < on; ++n) {
            const U8 *biasPtr = (const U8 *)bias;
            for (U32 c = 0; c < oc / cx; c++, biasPtr += biasTileSize) {
                for (U32 hw = 0; hw < oh * ow; hw++) {
                    UNI_MEMCPY(tmpOutputPtr, biasPtr, biasTileSize);
                    tmpOutputPtr += biasTileSize;
                }
            }
        }
        inputDesc.dt = convOutDesc.dt;
        CHECK_STATUS(deconvolution_overlap_crop_c8_x86(
            convOut, OriTmpOutputPtr, inputDesc, outputDesc, convParamSpec));
    }

#ifdef _USE_INT8
        if (convOdt != odt) {
            TensorDesc desc = outputDesc;
            desc.dt = convOdt;
            if (odt == DT_F32) {
                TensorDesc nullDesc;
                dequantize_x86(
                    desc, OriTmpOutputPtr, (F32 *)scale + 1, nullDesc, nullptr, outputDesc, output);
            } else {
                F32 *scaleRaw = (F32 *)scale;
                scaleRaw[2] = scaleRaw[1];
                scaleRaw[1] = -1;
                quantize_cpu(desc, OriTmpOutputPtr, &outputDesc, output, scaleRaw + 1, arch);
            }
        }
#endif

    if (activationDesc.mode != ACTIVATION_NULL) {
        CHECK_STATUS(array_activation_x86(
            odt, output, tensorNumElements(outputDesc), activationDesc, output, nullptr));
    }
    return SUCCESS;
}
