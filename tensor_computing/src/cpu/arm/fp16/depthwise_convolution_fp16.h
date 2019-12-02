// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_DEPTHWISE_CONVOLUTION_FP16
#define _H_DEPTHWISE_CONVOLUTION_FP16
#include "sys.h"
#include "type.h"
#include "tensor_desc.h"
#include "error.h"

#include "cpu/arm/fp16/depthwise_convolution_transform_fp16.h"

EE depthwise_convolution_infer_forward_algorithm_fp16(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, DepthwiseConvolutionForwardAlgorithm *algorithm);

EE depthwise_convolution_infer_forward_tmp_bytes_fp16(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32 *bytes);

EE depthwise_convolution_fp16(TensorDesc inputDesc, F16* input,
    TensorDesc filterDesc, const F16* filter,
    ConvolutionDesc convDesc, DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc, const F16* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F16* output,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode,
    Arch arch);
#endif
