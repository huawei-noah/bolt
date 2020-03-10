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

EE deconvolution_infer_forward_algorithm_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, ConvolutionForwardAlgorithm *algorithm, DataType targetDataType)
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

    U32 strideH = convDesc.stride_h;
    U32 strideW = convDesc.stride_w;
    U32 paddingT = convDesc.padding_top;
    U32 paddingB = convDesc.padding_bottom;
    U32 paddingL = convDesc.padding_left;
    U32 paddingR = convDesc.padding_right;

    if (fh < paddingT + 2 || fh < paddingB + 2 || fw < paddingL + 2 || fw < paddingR + 2) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    U32 tPadding = (fh - 1 - paddingT) - 1;  // Leave out padding of length 1 to activate Winograd
    U32 bPadding = (fh - 1 - paddingB) - 1;
    U32 lPadding = (fw - 1 - paddingL) - 1;
    U32 rPadding = (fw - 1 - paddingR) - 1;

    ih = ih + (ih - 1) * (strideH - 1) + tPadding + bPadding;
    iw = iw + (iw - 1) * (strideW - 1) + lPadding + rPadding;
    
    TensorDesc inPaddedDesc = tensor4df(idt, idf, in, ic, ih, iw);

    ConvolutionDesc transposedCD;
    transposedCD.stride_h = 1;
    transposedCD.stride_w = 1;
    transposedCD.padding_top = 1;
    transposedCD.padding_bottom = 1;
    transposedCD.padding_left = 1;
    transposedCD.padding_right = 1;
    transposedCD.dilatedRate_h = 1;
    transposedCD.dilatedRate_w = 1;

    return convolution_infer_forward_algorithm_arm(inPaddedDesc, filterDesc, outputDesc, transposedCD, policy, algorithm, targetDataType);
}

EE deconvolution_transform_filter_bytes_arm(TensorDesc filterDesc, ConvolutionForwardAlgorithm algorithm, U32* bytes) {
    return convolution_transform_filter_bytes_arm(filterDesc, algorithm, bytes);
}

EE deconvolution_transform_filter_arm(TensorDesc filterDesc, const void* filter,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, void* filterTransformed)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        //TODO: confirm
/*#ifdef _USE_FP32
        case DT_F32: {
            ret = deconvolution_transform_filter_fp32(filterDesc, (F32*)filter, algorithm, ftmDesc, (F32*)filterTransformed);
            break;
        }
#endif*/
#ifdef _USE_FP16
        case DT_F16: {
            ret = deconvolution_transform_filter_fp16(filterDesc, (F16*)filter, algorithm, ftmDesc, (F16*)filterTransformed);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE deconvolution_infer_forward_tmp_bytes_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = deconvolution_infer_forward_tmp_bytes_fp32(inputDesc, filterDesc, outputDesc, convDesc, algorithm, bytes);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = deconvolution_infer_forward_tmp_bytes_fp16(inputDesc, filterDesc, outputDesc, convDesc, algorithm, bytes);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;

}

EE deconvolution_arm(TensorDesc inputDesc, void* input,
    TensorDesc filterDesc, const void* filter,
    ConvolutionDesc convDesc,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc scaleDesc, const void* scale,
    TensorDesc biasDesc, const void* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, void* output,
    ActivationMode activationMode,
    Arch arch)
{
    UNUSED(scaleDesc);
    UNUSED(scale);
    
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = deconvolution_fp32(inputDesc, (F32*)input,
                                   filterDesc, (F32*)filter,
                                   convDesc,
                                   algorithm,
                                   biasDesc, (F32*)bias,
                                   tmpBytes, tmp,
                                   outputDesc, (F32*)output,
                                   activationMode,
                                   arch);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = deconvolution_fp16(inputDesc, (F16*)input,
                                   filterDesc, (F16*)filter,
                                   convDesc,
                                   algorithm,
                                   biasDesc, (F16*)bias,
                                   tmpBytes, tmp,
                                   outputDesc, (F16*)output,
                                   activationMode,
                                   arch);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
