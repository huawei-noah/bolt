// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CONVOLUTION_DOREFA
#define _H_CONVOLUTION_DOREFA

#ifdef _USE_FP16
#include "sys.h"
#include "uni.h"
#include "tensor_desc.h"
#include "parameter_spec.h"

EE convolution_dorefa_A55(TensorDesc inputDesc,
    const F16 *input,
    TensorDesc filterDesc,
    const BIN8 *filterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc scaleDesc,
    const F16 *scaleArray,
    TensorDesc biasDesc,
    const F16 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F16 *outArray,
    ActivationParamSpec activationDesc);

EE convolution_dorefa_A76(TensorDesc inputDesc,
    const F16 *input,
    TensorDesc filterDesc,
    const BIN8 *filterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc scaleDesc,
    const F16 *scaleArray,
    TensorDesc biasDesc,
    const F16 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F16 *outArray,
    ActivationParamSpec activationDesc);

inline EE convolution_dorefa(TensorDesc inputDesc,
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
    Arch arch)
{
    EE ret = SUCCESS;
    switch (arch) {
        case ARM_A55:
            ret = convolution_dorefa_A55(inputDesc, input, filterDesc, filter, convParamSpec,
                scaleDesc, scale, biasDesc, bias, tmpBytes, tmp, outputDesc, output, activationDesc);
            break;
        case ARM_A76:
            ret = convolution_dorefa_A76(inputDesc, input, filterDesc, filter, convParamSpec,
                scaleDesc, scale, biasDesc, bias, tmpBytes, tmp, outputDesc, output, activationDesc);
            break;
        default:
            return NOT_SUPPORTED;
    }
    return ret;
}
#endif
#endif
