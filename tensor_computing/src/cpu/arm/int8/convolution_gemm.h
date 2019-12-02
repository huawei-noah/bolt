// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_CONVOLUTION_GEMM_INT8
#define _H_CONVOLUTION_GEMM_INT8

#include <cmath>
#include "sys.h"
#include "type.h"
#include "error.h"
#include "tensor_desc.h"
#include "tensor_computing_type.h"

inline EE convolution_gemm(TensorDesc inputDesc, const void* input, F16* inputScale, TensorDesc filterDesc, const void* filter, F16* filterScale,
    ConvolutionDesc convDesc, TensorDesc biasDesc, const void* bias, U32 tmpBytes, void* tmp, TensorDesc outputDesc,
    void* output, F16* outputScale, ActivationMode am, Arch arch)
{
    EE ret = SUCCESS;
    switch (arch) {
        case ARM_A55:
            ret = convolution_gemm_A55<INT8>(inputDesc, input, inputScale,
                                       filterDesc, filter, filterScale,
                                       convDesc,
                                       biasDesc, bias,
                                       tmpBytes, tmp,
                                       outputDesc, output, outputScale,
                                       am);
            break;
        case ARM_A76:
            ret = convolution_gemm_A55<INT8>(inputDesc, input, inputScale,
                                       filterDesc, filter, filterScale,
                                       convDesc,
                                       biasDesc, bias,
                                       tmpBytes, tmp,
                                       outputDesc, output, outputScale,
                                       am);
            break;
        default:
            return NOT_SUPPORTED;
    }
    return ret;
}
#endif
