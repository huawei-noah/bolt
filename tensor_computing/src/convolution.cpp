// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <cmath>
#include "sys.h"
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "tensor_computing.h"
#include "tensor_computing_type.h"
#include "cpu/general/tensor_computing_general.h"
#include "cpu/arm/tensor_computing_arm.h"

EE convolution_infer_output_size(TensorDesc inputDesc, TensorDesc filterDesc, ConvolutionDesc convDesc,
    TensorDesc* outputDesc, DataType targetDataType, U32* outputBytes)
{
    if (nullptr == outputDesc || nullptr == outputBytes)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    DataType idt, fdt;
    DataFormat idf, fdf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 oh, ow;
    CHECK_STATUS_WITH_RETURN(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    if (fh < 1 || fw < 1)
        CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
    
    U32 stride = convDesc.stride;
    U32 padding = convDesc.padding;
    U32 dilatedRate = convDesc.dilatedRate;
    U32 fh_dilated = (fh - 1) * dilatedRate + 1;
    U32 fw_dilated = (fw - 1) * dilatedRate + 1;
    oh = (ih + 2 * padding - fh_dilated) / stride + 1;
    ow = (iw + 2 * padding - fw_dilated) / stride + 1;

    if (fn % 8 != 0)
        CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);

    *outputDesc = tensor4df(targetDataType, DF_NCHWC8, in, fn, oh, ow);
    *outputBytes = tensorNumBytes(*outputDesc);
    return SUCCESS;
}

EE convolution_infer_forward_algorithm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, ConvolutionForwardAlgorithm *algorithm, DataType targetDataType, Arch arch)
{
    EE ret = SUCCESS;
    if (arch == ARM_A55 || arch == ARM_A76)
        ret = convolution_infer_forward_algorithm_arm(inputDesc, filterDesc, outputDesc, convDesc, policy, algorithm, targetDataType);
    else
        ret = NOT_SUPPORTED;
    return ret;
}

EE convolution_transform_filter_bytes(TensorDesc filterDesc, ConvolutionForwardAlgorithm algorithm, U32* bytes, Arch arch){
    EE ret = SUCCESS;
    if (arch == ARM_A55 || arch == ARM_A76)
        ret = convolution_transform_filter_bytes_arm(filterDesc, algorithm, bytes);
    else
        ret = NOT_SUPPORTED;
    return ret;
}

EE convolution_transform_filter(TensorDesc filterDesc, const void* filter, ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, void* filterTransformed, Arch arch)
{
    EE ret = SUCCESS;
    if (arch == ARM_A55 || arch == ARM_A76)
        ret = convolution_transform_filter_arm(filterDesc, filter, algorithm, ftmDesc, filterTransformed);
    else
        ret = NOT_SUPPORTED;
    return ret;
}

EE convolution_infer_forward_tmp_bytes(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes, Arch arch) {
    EE ret = SUCCESS;
    if (arch == ARM_A55 || arch == ARM_A76)
        ret = convolution_infer_forward_tmp_bytes_arm(inputDesc, filterDesc, outputDesc, convDesc, algorithm, bytes);
    else
        ret = NOT_SUPPORTED;
    return ret;
}

EE convolution(TensorDesc inputDesc, void* input,
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
    EE ret = SUCCESS;
    switch (arch) {
        case CPU_GENERAL:
            ret = convolution_general(inputDesc, input,
                                      filterDesc, filter,
                                      convDesc,
                                      scaleDesc, scale,
                                      biasDesc, bias,
                                      outputDesc, output,
                                      activationMode);
            break;
        case ARM_A55:
            ret = convolution_arm(inputDesc, input,
                                  filterDesc, filter,
                                  convDesc,
                                  algorithm,
                                  scaleDesc, scale,
                                  biasDesc, bias,
                                  tmpBytes, tmp,
                                  outputDesc, output,
                                  activationMode,
                                  arch);
            break;
        case ARM_A76:
            ret = convolution_arm(inputDesc, input,
                                  filterDesc, filter,
                                  convDesc,
                                  algorithm,
                                  scaleDesc, scale,
                                  biasDesc, bias,
                                  tmpBytes, tmp,
                                  outputDesc, output,
                                  activationMode,
                                  arch);
            break;
        default:
            ret = NOT_SUPPORTED;
    }
    return ret;
}
