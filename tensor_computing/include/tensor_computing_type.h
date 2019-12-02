// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_TENSOR_COMPUTING_TYPE
#define _H_TENSOR_COMPUTING_TYPE

#include "type.h"

typedef struct {
    U32 stride;
    U32 padding;
    U32 dilatedRate;
} ConvolutionDesc;

typedef enum {
    CONVOLUTION_NO_TMP_MEM,
    CONVOLUTION_FASTEST,
} ConvolutionPolicy;

typedef enum {
    CONVOLUTION_ALGORITHM_DIRECT,
    CONVOLUTION_ALGORITHM_GEMM,
    CONVOLUTION_ALGORITHM_GEMM_IC1OR3,
    CONVOLUTION_ALGORITHM_GEMM_DILATED,
    CONVOLUTION_ALGORITHM_WINOGRAD,
    CONVOLUTION_ALGORITHM_BNN,
} ConvolutionForwardAlgorithm;

typedef struct {
    PoolingMode pm;
    U32 stride;
    U32 padding;
    U32 kernelSize;
    RoundMode rm;
} PoolingDesc;

typedef struct {
    U32 num_output;
} LSTMDesc;

typedef enum {
    DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT,
    DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT,
    DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT_NO_PADDING,
    DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_3X3S1P1,
} DepthwiseConvolutionForwardAlgorithm;
#endif
