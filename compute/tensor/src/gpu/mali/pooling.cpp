// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "sys.h"

#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/pooling_mali_fp16.h"

EE pooling_padding_input_mali(TensorDesc inputDesc,
    PoolingParamSpec poolingParamSpec,
    TensorDesc *outputDesc,
    OclMemory *inputMem,
    OclMemory *outputMem)
{
    if (inputMem == nullptr || outputMem == nullptr || outputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    U32 pl = poolingParamSpec.pad_left;
    U32 pr = poolingParamSpec.pad_right;
    U32 pt = poolingParamSpec.pad_top;
    U32 pb = poolingParamSpec.pad_bottom;
    U32 pf = poolingParamSpec.pad_before;
    U32 pa = poolingParamSpec.pad_after;
    inputMem->padding(pl, pr, pt, pb, pf, pa);
    return SUCCESS;
}

EE pooling_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    PoolingParamSpec poolingParamSpec,
    const void *scale,
    GCLMem_t temp,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    UNUSED(scale);
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = pooling_mali_fp16(
                handle, inputDesc, input, poolingParamSpec, outputDesc, output, temp);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE pooling_infer_forward_tmp_bytes_mali(
    TensorDesc inputDesc, U32 *bytes, ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = pooling_infer_forward_tmp_bytes_mali_fp16(inputDesc, bytes, forwardRunInfo);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
