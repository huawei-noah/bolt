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
#include "gpu/mali/fp16/bilateral_slice_apply_mali_fp16.h"

inline EE bilateral_slice_apply_checkpara_mali_common(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc guideDesc,
    const GCLMem_t guide,
    TensorDesc gridDesc,
    const GCLMem_t grid,
    BilateralSliceApplyParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (nullptr == handle || nullptr == input || nullptr == grid || nullptr == output) {
        return NULL_POINTER;
    }
    if (p.mode == BILATERAL_SLICE_APPLY_NULL && nullptr == guide) {
        return NULL_POINTER;
    }
    if (inputDesc.df != guideDesc.df || inputDesc.df != gridDesc.df) {
        return NOT_SUPPORTED;
    }
    if (inputDesc.df != outputDesc.df || inputDesc.df != DF_NHWC) {
        return NOT_SUPPORTED;
    }
    if (inputDesc.dims[0] != guideDesc.dims[0] || inputDesc.dims[1] != guideDesc.dims[1]) {
        return NOT_MATCH;
    }
    if (inputDesc.dims[0] != outputDesc.dims[0] || inputDesc.dims[1] != outputDesc.dims[1]) {
        return NOT_MATCH;
    }
    if (inputDesc.dims[2] != outputDesc.dims[2]) {
        return NOT_MATCH;
    }
    U32 coefficient = inputDesc.dims[2] * (inputDesc.dims[2] + 1);
    if (gridDesc.dims[2] % coefficient != 0) {
        return NOT_MATCH;
    }
    if (coefficient != 12) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE bilateral_slice_padding_input_mali(TensorDesc inputDesc,
    TensorDesc guideDesc,
    TensorDesc gridDesc,
    BilateralSliceApplyParamSpec p,
    TensorDesc *outputDesc,
    OclMemory *inputMem,
    OclMemory *guideMem,
    OclMemory *gridMem,
    OclMemory *outputMem)
{
    if (outputDesc == nullptr || inputMem == nullptr || guideMem == nullptr || gridMem == nullptr ||
        outputMem == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, gdt;
    DataFormat idf, gdf;
    U32 gw, gh, gc, gn;
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    if (inputDesc.df != DF_NHWC || guideDesc.df != DF_NHWC) {
        return NOT_MATCH;
    }
    tensorSelectGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw);
    tensorSelectGet(guideDesc, &gdt, &gdf, &gn, &gc, &gh, &gw);
    ow = gw;
    oh = gh;
    oc = ic;
    on = gn;
    *outputDesc = tensor4df(idt, idf, on, oc, oh, ow);
    return SUCCESS;
}

EE bilateral_slice_apply_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc guideDesc,
    TensorDesc gridDesc,
    BilateralSliceApplyParamSpec p,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    UNUSED(inputDesc);
    UNUSED(guideDesc);
    UNUSED(gridDesc);
    UNUSED(p);
    UNUSED(forwardRunInfo);

    DataType dt;
    U32 gc, gw;
    U32 ih;
    tensorSelectGet(gridDesc, &dt, NULL, NULL, &gc, NULL, &gw);
    tensorSelectGet(inputDesc, NULL, NULL, NULL, NULL, &ih, NULL);
    *bytes = gc * gw * ih * bytesOf(dt);
    return SUCCESS;
}

EE bilateral_slice_apply_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc guideDesc,
    const GCLMem_t guide,
    TensorDesc gridDesc,
    const GCLMem_t grid,
    BilateralSliceApplyParamSpec p,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(bilateral_slice_apply_checkpara_mali_common(
        handle, inputDesc, input, guideDesc, guide, gridDesc, grid, p, outputDesc, output));
    return bilateral_slice_apply_mali_fp16(handle, inputDesc, input, guideDesc, guide,
                gridDesc, grid, p, forwardRunInfo, tmpBytes, tmpBuf, outputDesc, output);
}
