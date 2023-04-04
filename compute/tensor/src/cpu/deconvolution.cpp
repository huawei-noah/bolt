// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/tensor_computing_cpu.h"
#include "cpu/cpu_functions.h"
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif
#include "blas_enhance.h"
#include "tensor_transpose.h"

EE deconvolution_infer_forward_algorithm_cpu(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    ConvolutionForwardAlgorithm *algorithm,
    DataType targetDataType,
    Arch arch)
{
    if (nullptr == algorithm) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, fdt;
    DataFormat idf, fdf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    if (1 == fn && ic != fn) {
        *algorithm = CONVOLUTION_ALGORITHM_GROUP_DECONV;
        return SUCCESS;
    }
#ifdef _USE_X86
    if (IS_X86(arch) && (idf == DF_NCHWC8 || idf == DF_NCHWC16)) {
        *algorithm = CONVOLUTION_ALGORITHM_POINTWISE;
        return SUCCESS;
    }
#endif
    *algorithm = CONVOLUTION_ALGORITHM_IM2COL_GEMM;
    return SUCCESS;
}

static TensorDesc get_gemm_a(TensorDesc inputDesc)
{
    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    return tensor2df(idt, DF_NORMAL, in * ih * iw, ic);
}

static TensorDesc get_gemm_b(TensorDesc filterDesc)
{
    TensorDesc desc = filterDesc;
    if (desc.nDims != 2) {
        DataType fdt;
        DataFormat fdf;
        U32 fn, fc, fh, fw;
        CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
        desc = tensor2df(fdt, DF_NORMAL, fn, fc * fh * fw);
    }
    return desc;
}

static TensorDesc get_gemm_c(
    TensorDesc inputDesc, TensorDesc outputDesc, ConvolutionParamSpec convParamSpec)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 fh = convParamSpec.kernel_h;
    U32 fw = convParamSpec.kernel_w;
#ifdef _USE_INT8
    if (idt == DT_I8 || idt == DT_U8_Q) {
        odt = DT_I32;
    }
#endif
    return tensor2df(odt, DF_NORMAL, in * ih * iw, fw * fh * oc);
}

static TensorDesc get_conv_pad(TensorDesc inputDesc, ConvolutionParamSpec convParamSpec)
{
    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    U32 fh = convParamSpec.kernel_h;
    U32 fw = convParamSpec.kernel_w;

    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingB = convParamSpec.pad_bottom;
    U32 paddingL = convParamSpec.pad_left;
    U32 paddingR = convParamSpec.pad_right;
    U32 tPadding = fh - 1 - paddingT;
    U32 bPadding = fh - 1 - paddingB;
    U32 lPadding = fw - 1 - paddingL;
    U32 rPadding = fw - 1 - paddingR;

    ih = ih + (ih - 1) * (strideH - 1) + tPadding + bPadding;
    iw = iw + (iw - 1) * (strideW - 1) + lPadding + rPadding;
    return tensor4df(idt, idf, in, ic, ih, iw);
}

EE deconvolution_transform_filter_bytes_cpu(TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    U32 *bytes,
    Arch arch)
{
    EE ret = NOT_SUPPORTED;
    if (algorithm == CONVOLUTION_ALGORITHM_IM2COL_GEMM ||
        algorithm == CONVOLUTION_ALGORITHM_POINTWISE) {
        *bytes = tensorNumBytes(filterDesc) + 32;
#ifdef _USE_INT8
        if (IS_X86(arch)) {
            DataType fdt;
            DataFormat fdf;
            U32 fn, fc, fh, fw;
            CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
            *bytes = UNI_ALIGN(fn, 16) * fc * fh * fw * bytesOf(fdt) + 32;
            *bytes += fc * fh * fw * bytesOf(DT_I32);  //offsetC
        }
#endif
        ret = SUCCESS;
    } else if (algorithm == CONVOLUTION_ALGORITHM_GROUP_DECONV) {
        ret = depthwise_convolution_transform_filter_bytes_cpu(
            filterDesc, DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT, bytes);
    }
    return ret;
}

EE deconvolution_transform_filter_cpu(TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    void *filterTransformed,
    Arch arch)
{
    if (algorithm == CONVOLUTION_ALGORITHM_IM2COL_GEMM) {
        TensorDesc desc = get_gemm_b(filterDesc);
        return matrix_matrix_multiply_transform_rhs(desc, filter, ftmDesc, filterTransformed, arch);
    }
    EE ret = NOT_SUPPORTED;
    if (IS_ARM(arch)) {
#ifdef _USE_NEON
        ret = deconvolution_transform_filter_arm(
            filterDesc, filter, algorithm, ftmDesc, filterTransformed);
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = deconvolution_transform_filter_x86(
            filterDesc, filter, algorithm, ftmDesc, filterTransformed);
#endif
    }
    return ret;
}

EE deconvolution_infer_forward_tmp_bytes_cpu(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    U32 *bytes,
    Arch arch)
{
    if (nullptr == bytes) {
        CHECK_STATUS(NULL_POINTER);
    }

    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw, fn, fc, fh, fw, on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    fh = convParamSpec.kernel_h;
    fw = convParamSpec.kernel_w;

    if (algorithm == CONVOLUTION_ALGORITHM_POINTWISE) {
        *bytes = in * ih * iw * oc * fh * fw * bytesOf(odt);
#ifdef _USE_INT8
        if (idt == DT_U8_Q) {
            DataType convOdt = DT_I32;
            if (fh == convParamSpec.stride_h && fw == convParamSpec.stride_w) {
                convOdt = DT_U8_Q;
                *bytes = *bytes / bytesOf(odt) * bytesOf(DT_U8_Q); // conv U8 out
            } else {
                *bytes = *bytes / bytesOf(odt) * bytesOf(DT_I32); // conv I32 out
            }

            if (odt != convOdt) {
                *bytes += on * oc * oh * ow * bytesOf(convOdt);  //crop out, results before quantization
            }

            *bytes += oc * fh * fw * bytesOf(DT_I32);  // quant bias

            if (ic % 16 != 0) {
                *bytes += in * ih * iw * UNI_ALIGN(ic, 16) * bytesOf(idt); // transform input
            }
        }
#endif
        *bytes += oc * fh * fw * bytesOf(DT_F32) + 32;
        if (idf == DF_NCHWC16 && idt == DT_F32) {
            *bytes += in * ih * iw * ic * bytesOf(idt);
        }

        return SUCCESS;
    }
    if (algorithm == CONVOLUTION_ALGORITHM_IM2COL_GEMM) {
        TensorDesc matrixADesc = get_gemm_a(inputDesc);
        TensorDesc matrixBDesc = get_gemm_b(filterDesc);
        TensorDesc matrixCDesc = get_gemm_c(inputDesc, outputDesc, convParamSpec);
        CHECK_STATUS(matrix_matrix_multiply_tmp_bytes(matrixADesc, matrixBDesc, bytes, arch));
#ifdef _USE_INT8
        if (inputDesc.dt == DT_U8_Q || inputDesc.dt == DT_I8) {
            *bytes =
                UNI_MAX(*bytes, (fw * fh * oc + tensorNumElements(outputDesc)) * bytesOf(DT_I32));
        }
#endif
        *bytes += tensorNumBytes(inputDesc) + tensorNumBytes(matrixCDesc);
        return SUCCESS;
    }

    TensorDesc inPaddedDesc = get_conv_pad(inputDesc, convParamSpec);
    *bytes = tensorNumBytes(inPaddedDesc) * 2 + 32;
    return SUCCESS;
}

static EE deconvolution_gemm(TensorDesc inputDesc,
    void *input,
    TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    TensorDesc scaleDesc,
    F32 *scale,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec activationDesc,
    Arch arch)
{
    TensorDesc desc = inputDesc;
    desc.df = DF_NHWC;
    transformFormat(inputDesc, input, desc, tmp);
    input = tmp;
    tmp = (U8 *)tmp + tensorNumBytes(desc);

    TensorDesc matrixADesc = get_gemm_a(inputDesc);
    TensorDesc matrixCDesc = get_gemm_c(inputDesc, outputDesc, convParamSpec);
    void *out = tmp;
    tmp = (U8 *)tmp + tensorNumBytes(matrixCDesc);

#ifdef _USE_X86
    if (IS_X86(arch) && filterDesc.dt == DT_I8) {
        desc = get_gemm_b(filterDesc);
        U32 offsetCBytes = UNI_ALIGN(desc.dims[0], 16) * UNI_ALIGN(desc.dims[1], 8);
        UNI_MEMCPY(tmp, (U8 *)filter + offsetCBytes, desc.dims[0] * 4);
    }
#endif

    UNI_MEMSET(out, 0, tensorNumBytes(matrixCDesc));
    CHECK_STATUS(matrix_matrix_multiply(
        matrixADesc, input, filterDesc, filter, tmpBytes, tmp, matrixCDesc, out, scale, arch));

    U8 *outputPtr = (U8 *)output;
    TensorDesc realDesc = outputDesc;
#ifdef _USE_INT8
    if (matrixCDesc.dt == DT_I32) {
        biasDesc.dt = DT_I32;
        I32 *biasI = (I32 *)tmp;
#ifdef _USE_FP16
        const F16 *biasF = (const F16 *)bias;
#else
        const F32 *biasF = (const F32 *)bias;
#endif
        for (U32 i = 0; i < tensorNumElements(biasDesc); i++) {
            biasI[i] = round(scale[0] * scale[2] * biasF[i]);
        }
        bias = tmp;
        tmp = (U8 *)tmp + tensorNumBytes(biasDesc);

        realDesc.dt = matrixCDesc.dt;
        outputPtr = (U8 *)tmp;
        tmp = (U8 *)tmp + tensorNumBytes(realDesc);
    }
#endif

    DataType odt;
    DataFormat odf;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(realDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 cx = 8;
    if (IS_X86(arch) && filterDesc.dt == DT_I8) {
        cx = 16;
    }
    U8 *dst = (U8 *)outputPtr;
    U32 size = bytesOf(biasDesc.dt) * cx;
    for (U32 n = 0; n < on; ++n) {
        U8 *src = (U8 *)bias;
        for (U32 c = 0; c < oc / cx; c++, src += size) {
            for (U32 hw = 0; hw < oh * ow; hw++) {
                UNI_MEMCPY(dst, src, size);
                dst += size;
            }
        }
    }

    EE ret = NOT_SUPPORTED;
    if (IS_ARM(arch)) {
#ifdef _USE_NEON
        ret = deconvolution_overlap_crop_arm(out, outputPtr, inputDesc, realDesc, convParamSpec);
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = deconvolution_overlap_crop_x86(out, outputPtr, inputDesc, realDesc, convParamSpec);
#endif
    }

    if (activationDesc.mode != ACTIVATION_NULL) {
        ArrayActivationFunction activation_func = get_array_activation_function(arch);
        CHECK_STATUS(
            activation_func(odt, outputPtr, tensorNumElements(realDesc), activationDesc, outputPtr, nullptr));
    }

    if (realDesc.dt != outputDesc.dt) {
        F32 *scaleRaw = (F32 *)scale;
        if (outputDesc.dt != DT_I8 && outputDesc.dt != DT_U8_Q) {
            F32 scaleO = scaleRaw[0] * scaleRaw[2];
            TensorDesc nullDesc;
            if (IS_ARM(arch)) {
#ifdef _USE_NEON
                ret = dequantize_arm(
                    realDesc, outputPtr, &scaleO, nullDesc, nullptr, outputDesc, output);
#endif
#ifdef _USE_X86
            } else if (IS_X86(arch)) {
                ret = dequantize_x86(
                    realDesc, outputPtr, &scaleO, nullDesc, nullptr, outputDesc, output);
#endif
            }
        } else {
            scaleRaw[2] *= scaleRaw[0];
            quantize_cpu(realDesc, outputPtr, &outputDesc, output, scaleRaw + 1, arch);
        }
    }

    return ret;
}

static EE deconvolution_convolution(TensorDesc inputDesc,
    void *input,
    TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec activationDesc,
    Arch arch)
{
    TensorDesc inPaddedDesc = get_conv_pad(inputDesc, convParamSpec);
    TensorDesc outDesc = outputDesc;
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    U32 ihPadded, iwPadded;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(inPaddedDesc, &idt, &idf, &in, &ic, &ihPadded, &iwPadded));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    inPaddedDesc.dims[inPaddedDesc.nDims - 1] = 1;
    outDesc.dims[outDesc.nDims - 1] = 1;

    if (!(idf == DF_NCHWC8 && odf == DF_NCHWC8)) {
        CHECK_STATUS(NOT_MATCH);
    }

    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingB = convParamSpec.pad_bottom;
    U32 paddingL = convParamSpec.pad_left;
    U32 paddingR = convParamSpec.pad_right;

    ConvolutionParamSpec transposedCD = convParamSpec;
    transposedCD.stride_h = 1;
    transposedCD.stride_w = 1;
    transposedCD.pad_top = 0;
    transposedCD.pad_bottom = 0;
    transposedCD.pad_left = 0;
    transposedCD.pad_right = 0;
    transposedCD.dilatedRate_h = 1;
    transposedCD.dilatedRate_w = 1;

    U32 tPadding = fh - 1 - paddingT;
    U32 bPadding = fh - 1 - paddingB;
    U32 lPadding = fw - 1 - paddingL;
    U32 rPadding = fw - 1 - paddingR;
    U32 stuffH = strideH - 1;
    U32 stuffW = strideW - 1;

    U32 memUnit = 8 * bytesOf(idt);
    U32 ic8 = ic / 8;
    EE ret = NOT_SUPPORTED;
    TensorDesc blankTensorDesc;
    ActivationParamSpec blankActivationParamSpec;
    for (U32 n = 0; n < in; ++n) {
        U8 *inputMov = (U8 *)input + n * ih * iw * ic * bytesOf(idt);
        U8 *outputMov = (U8 *)output + n * oh * ow * oc * bytesOf(odt);
        U8 *inPad = (U8 *)tmp;
        U8 *inPadMov = inPad;

        for (U32 c = 0; c < ic8; c++) {
            for (U32 h = 0; h < tPadding; h++) {
                UNI_MEMSET(inPadMov, 0, iwPadded * memUnit);
                inPadMov += iwPadded * memUnit;
            }
            for (U32 h = 0; h < ih - 1; h++) {
                UNI_MEMSET(inPadMov, 0, lPadding * memUnit);
                inPadMov += lPadding * memUnit;
                for (U32 w = 0; w < iw - 1; w++) {
                    UNI_MEMCPY(inPadMov, inputMov, memUnit);
                    inPadMov += memUnit;
                    inputMov += memUnit;
                    UNI_MEMSET(inPadMov, 0, stuffW * memUnit);
                    inPadMov += stuffW * memUnit;
                }
                UNI_MEMCPY(inPadMov, inputMov, memUnit);
                inPadMov += memUnit;
                inputMov += memUnit;
                UNI_MEMSET(inPadMov, 0, rPadding * memUnit);
                inPadMov += rPadding * memUnit;

                // stuffH
                UNI_MEMSET(inPadMov, 0, iwPadded * stuffH * memUnit);
                inPadMov += iwPadded * stuffH * memUnit;
            }
            UNI_MEMSET(inPadMov, 0, lPadding * memUnit);
            inPadMov += lPadding * memUnit;
            for (U32 w = 0; w < iw - 1; w++) {
                UNI_MEMCPY(inPadMov, inputMov, memUnit);
                inPadMov += memUnit;
                inputMov += memUnit;
                UNI_MEMSET(inPadMov, 0, stuffW * memUnit);
                inPadMov += stuffW * memUnit;
            }
            UNI_MEMCPY(inPadMov, inputMov, memUnit);
            inPadMov += memUnit;
            inputMov += memUnit;
            UNI_MEMSET(inPadMov, 0, rPadding * memUnit);
            inPadMov += rPadding * memUnit;

            for (U32 h = ihPadded - bPadding; h < ihPadded; h++) {
                UNI_MEMSET(inPadMov, 0, iwPadded * memUnit);
                inPadMov += iwPadded * memUnit;
            }
        }

        ret = depthwise_pointwise_convolution_cpu(inPaddedDesc, inPad, filterDesc, filter,
            blankTensorDesc, nullptr, transposedCD,
            DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT, biasDesc, bias, blankTensorDesc,
            nullptr, tmpBytes - tensorNumBytes(inPaddedDesc), inPad + tensorNumBytes(inPaddedDesc),
            outDesc, outputMov, activationDesc, blankActivationParamSpec, arch);
    }
    return ret;
}

EE deconvolution_cpu(TensorDesc inputDesc,
    void *input,
    TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
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
    if (nullptr == input || nullptr == filter || nullptr == output || nullptr == bias ||
        nullptr == tmp) {
        CHECK_STATUS(NULL_POINTER);
    }

    if (algorithm == CONVOLUTION_ALGORITHM_IM2COL_GEMM) {
        return deconvolution_gemm(inputDesc, input, filterDesc, filter, convParamSpec, scaleDesc,
            (F32 *)scale, biasDesc, bias, tmpBytes, tmp, outputDesc, output, activationDesc, arch);
    }
#ifdef _USE_X86
    if ((inputDesc.df == DF_NCHWC16) && (inputDesc.dt == DT_F32)) {
        TensorDesc desc = inputDesc;
        desc.df = DF_NCHWC8;
        transformFormat(inputDesc, input, desc, tmp);
        input = tmp;
        inputDesc.df = DF_NCHWC8;
        tmp = (U8 *)tmp + tensorNumBytes(desc);
    }
    if (IS_X86(arch) && algorithm == CONVOLUTION_ALGORITHM_POINTWISE) {
        return deconvolution_pointwise_x86(inputDesc, input, filterDesc, filter, convParamSpec,
            scaleDesc, scale, biasDesc, bias, tmpBytes, tmp, outputDesc, output, activationDesc,
            arch);
    }
#endif
    return deconvolution_convolution(inputDesc, input, filterDesc, filter, convParamSpec, algorithm,
        biasDesc, bias, tmpBytes, tmp, outputDesc, output, activationDesc, arch);
}
