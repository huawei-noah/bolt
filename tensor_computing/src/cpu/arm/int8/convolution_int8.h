// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_CONVOLUTION_INT8
#define _H_CONVOLUTION_INT8
#include "tensor_desc.h"
#include "type.h"
#include "error.h"
#include "cpu/arm/int8/convolution_transform_int8.h"


EE convolution_infer_forward_algorithm_int8(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, ConvolutionForwardAlgorithm *algorithm);

EE convolution_infer_forward_tmp_bytes_int8(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes);

EE convolution_int8(TensorDesc inputDesc, const INT8* input,
    TensorDesc filterDesc, const INT8* filter, F16* scales,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc, const F16* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, void* output,
    ActivationMode activationMode,
    Arch arch);

template<typename OT>
EE convolution_gemm_A55(TensorDesc inputDesc, const void* input, F16* inputScale, TensorDesc filterDesc, const void* filter, F16* filterScale,
    ConvolutionDesc convDesc, TensorDesc biasDesc, const void* bias, U32 tmpBytes, void* tmp, TensorDesc outputDesc,
    void* output, F16* outputScale, ActivationMode am);

template<typename OT>
EE convolution_winograd_A55(TensorDesc inputDesc, const void* input, F16* input_scale, TensorDesc filterDesc, const void* filter, F16* filterScale,
    ConvolutionDesc convDesc, TensorDesc biasDesc, const void* bias, U32 tmpBytes, void* tmp, TensorDesc outputDesc,
    void* output, F16* outputScale, ActivationMode am);
#endif
