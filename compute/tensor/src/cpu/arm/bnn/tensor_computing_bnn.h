// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TENSOR_COMPUTING_BNN
#define _H_TENSOR_COMPUTING_BNN

#ifdef _USE_FP16
#include "cpu/arm/bnn/convolution_transform_bnn.h"
#include "cpu/arm/bnn/convolution_dorefa.h"
#include "cpu/arm/bnn/convolution_xnor.h"

EE convolution_bnn(TensorDesc inputDesc,
    const F16 *input,
    TensorDesc filterDesc,
    const BIN8 *filter,
    ConvolutionParamSpec convParamSpec,
    TensorDesc scaleDesc,
    const F16 *scale,
    TensorDesc biasDesc,
    const F16 *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F16 *output,
    ActivationParamSpec activationDesc,
    Arch arch);
#endif
#endif
