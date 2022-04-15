// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/tensor_computing_x86.h"
#ifdef _USE_FP32
#include "cpu/x86/fp32/tensor_computing_fp32.h"
#endif

EE instance_norm_x86(TensorDesc inputDesc,
    void *input,
    void *tmp,
    void *scale,
    void *bias,
    InstanceNormParamSpec p,
    void *output)
{
    DataType idt = inputDesc.dt;
    EE ret = SUCCESS;
    switch (idt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = instance_norm_fp32(
                inputDesc, (F32 *)input, (F32 *)tmp, (F32 *)scale, (F32 *)bias, p, (F32 *)output);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE instance_norm_infer_forward_tmp_bytes_x86(
    TensorDesc inputDesc, InstanceNormParamSpec p, U32 *bytes)
{
    EE ret = SUCCESS;

    I32 axis = (p.axis + inputDesc.nDims) % inputDesc.nDims;
    axis = inputDesc.nDims - 1 - axis;

    // support axisDim != inputDesc.dims[axis]
    I32 axisDim = (p.axis_dim > 0) ? p.axis_dim : inputDesc.dims[axis];

    I32 loopInner = 1;
    for (I32 i = 0; i < axis; ++i) {
        loopInner *= inputDesc.dims[i];
    }

    I32 loopOuter = 1;
    for (U32 i = axis; i < inputDesc.nDims; ++i) {
        loopOuter *= inputDesc.dims[i];
    }

    if (axisDim == (I32)inputDesc.dims[axis]) {
        *bytes = loopOuter * 2 * bytesOf(inputDesc.dt);
        return ret;
    }

    I32 loopInnerIn = loopInner * inputDesc.dims[axis] * 1.0f / axisDim;
    I32 loopOuterIn = loopOuter * axisDim * 1.0f / inputDesc.dims[axis];
    *bytes = loopOuterIn * loopInnerIn * bytesOf(inputDesc.dt);
    *bytes += UNI_MAX(loopOuterIn, loopOuter) * 4 * bytesOf(inputDesc.dt);

    return ret;
}