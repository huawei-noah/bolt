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
#ifdef _USE_INT32
#include "cpu/arm/int32/tensor_computing_int32.h"
#endif
#ifdef _USE_FP32
#include "cpu/arm/fp32/tensor_computing_fp32.h"
#endif
#ifdef _USE_FP16
#include "cpu/arm/fp16/tensor_computing_fp16.h"
#endif

EE scale_arm(TensorDesc inputDesc,
    void *input,
    void *alpha,
    void *beta,
    ScaleParamSpec p,
    TensorDesc outputDesc,
    void *output)
{
    U32 length = tensorNumElements(outputDesc);
    int axis = (p.axis + outputDesc.nDims) % outputDesc.nDims;
    I32 on = outputDesc.dims[outputDesc.nDims - 1];
    I32 oc = outputDesc.dims[outputDesc.nDims - 1 - axis];
    I32 elements_per_channel = length / (on * oc);
    I32 ic = inputDesc.dims[inputDesc.nDims - 1 - axis];
    if (outputDesc.df == DF_NCHWC8) {
        axis = outputDesc.nDims;
    }
    EE ret = NOT_SUPPORTED;
    switch (outputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = scale_fp32((F32 *)input, axis, outputDesc.nDims, (F32 *)alpha, (F32 *)beta, on,
                oc, elements_per_channel, ic, (F32 *)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = scale_fp16((F16 *)input, axis, outputDesc.nDims, (F16 *)alpha, (F16 *)beta, on,
                oc, elements_per_channel, ic, (F16 *)output);
            break;
        }
#endif
#ifdef _USE_INT32
        case DT_I32: {
            ret = scale_int32((I32 *)input, axis, outputDesc.nDims, (I32 *)alpha, (I32 *)beta, on,
                oc, elements_per_channel, ic, (I32 *)output);
            break;
        }
#endif
        default:
            break;
    }

    return ret;
}
