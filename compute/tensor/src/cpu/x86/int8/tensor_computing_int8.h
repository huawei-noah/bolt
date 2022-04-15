// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef CHEETAH_TENSOR_COMPUTING_INT8_H
#define CHEETAH_TENSOR_COMPUTING_INT8_H

#include "sys.h"
#include "error.h"
#include "data_type.h"
#include "parameter_spec.h"

EE dequantizeI32ToF32(TensorDesc qDesc, I32 *qData, const F32 *scale, TensorDesc dDesc, F32 *data);

EE quantizeBiasOffsetCI32(F32 *bias,
    TensorDesc biasDesc,
    INT8 *filter,
    TensorDesc filterDesc,
    const F32 *scale,
    I32 *offsetCBias);

EE quantizeF32ToI8(TensorDesc dDesc, const F32 *data, TensorDesc *qDesc, INT8 *qData, F32 *scale);

EE quantizeF32ToU8(TensorDesc dDesc, const F32 *data, TensorDesc *qDesc, UINT8 *qData, F32 *scale);

EE transformU8ToI8(TensorDesc dDesc, const UINT8 *data, TensorDesc *qDesc, INT8 *qData);

EE quantizeI32ToU8(TensorDesc dDesc, const I32 *data, TensorDesc *qDesc, UINT8 *qData, F32 *scale);

EE quantizeI32ToI8(TensorDesc dDesc, const I32 *data, TensorDesc *qDesc, INT8 *qData, F32 *scale);

EE convolution_int8(TensorDesc inputDesc,
    UINT8 *input,
    TensorDesc filterDesc,
    const INT8 *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc,
    const I32 *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    F32 *scale,
    ActivationParamSpec activationDesc,
    Arch arch);

EE convolution_direct(TensorDesc inputDesc,
    UINT8 *inArray,
    TensorDesc filterDesc,
    const INT8 *filterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const I32 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *outArray,
    F32 *scale,
    ActivationParamSpec activationDesc);

EE convolution_transform_filter_int8(TensorDesc filterDesc,
    const INT8 *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    INT8 *filterTransformed);

EE convolution_infer_forward_tmp_bytes_int8(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    U32 *bytes);

EE convolution_1x1_direct(TensorDesc inputDesc,
    UINT8 *inArray,
    TensorDesc filterDesc,
    const INT8 *filterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const I32 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *outArray,
    F32 *scale,
    ActivationParamSpec activationDesc);

#endif  //CHEETAH_TENSOR_COMPUTING_INT8_H