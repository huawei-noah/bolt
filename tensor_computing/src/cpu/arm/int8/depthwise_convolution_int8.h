// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_DEPTHWISE_CONVOLUTION_INT8
#define _H_DEPTHWISE_CONVOLUTION_INT8
#include "sys.h"
#include "type.h"
#include "tensor_desc.h"
#include "error.h"

#include "cpu/arm/int8/depthwise_convolution_transform_int8.h"

EE depthwise_convolution_infer_forward_algorithm_int8(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, DepthwiseConvolutionForwardAlgorithm *algorithm);

EE depthwise_convolution_infer_forward_tmp_bytes_int8(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32 *bytes);

EE depthwise_convolution_int8(TensorDesc inputDesc, INT8* input,
    TensorDesc filterDesc, const INT8* filter,
    ConvolutionDesc convDesc, DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc, const I32* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, I32* output,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode,
    Arch arch);

EE depthwise_pointwise_convolution_direct(TensorDesc inputDesc, INT8* inArray,
    TensorDesc filterDesc, const INT8* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const I32* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, I32* outArray,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode,
    Arch arch);
#endif
