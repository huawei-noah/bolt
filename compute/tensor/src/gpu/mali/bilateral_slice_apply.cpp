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
#include "gpu/mali/uchar/bilateral_slice_apply_mali_uchar.h"

inline EE bilateral_slice_apply_checkpara_mali_common(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc guideDesc,
    const GCLMem_t guide,
    TensorDesc gridDesc,
    const GCLMem_t grid,
    BilateralSliceApplyParamSpec bilateralSliceApplyParamSpec,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (nullptr == handle || nullptr == input || nullptr == grid || nullptr == output) {
        return NULL_POINTER;
    }
    if (bilateralSliceApplyParamSpec.mode == BSLICE_APPLY_NULL && nullptr == guide) {
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
    if ((gridDesc.dims[2] % bilateralSliceApplyParamSpec.coefficient) != 0) {
        return NOT_MATCH;
    }
    if (bilateralSliceApplyParamSpec.has_offset == true) {
        if (bilateralSliceApplyParamSpec.coefficient != inputDesc.dims[2] * (inputDesc.dims[2] + 1)) {
            return NOT_MATCH;
        }
        if (bilateralSliceApplyParamSpec.coefficient != 12) {
            return NOT_SUPPORTED;
        }
    } else {
        return NOT_SUPPORTED;
        // if(bilateralSliceApplyParamSpec.coefficient_len != inputDesc.dims[2] *  inputDesc.dims[2])      return NOT_MATCH;
        // if(bilateralSliceApplyParamSpec.coefficient_len != 9) return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE bilateral_slice_padding_input_mali(TensorDesc inputDesc,
    TensorDesc guideDesc,
    TensorDesc gridDesc,
    BilateralSliceApplyParamSpec bilateralSliceApplyParamSpec,
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
    DataType idt, guide_dt;
    DataFormat idf, guide_df;
    U32 guide_w, guide_h, guide_c, guide_n;
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    if (inputDesc.df != DF_NHWC || guideDesc.df != DF_NHWC) {
        return NOT_MATCH;
    }
    tensorSelectGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw);
    tensorSelectGet(guideDesc, &guide_dt, &guide_df, &guide_n, &guide_c, &guide_h, &guide_w);
    ow = guide_w;
    oh = guide_h;
    oc = ic;
    on = guide_n;
    *outputDesc = tensor4df(idt, idf, on, oc, oh, ow);
    return SUCCESS;
}

EE bilateral_slice_apply_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc guideDesc,
    TensorDesc gridDesc,
    BilateralSliceApplyParamSpec bilateralSliceApplyParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    UNUSED(inputDesc);
    UNUSED(guideDesc);
    UNUSED(gridDesc);
    UNUSED(bilateralSliceApplyParamSpec);
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
    BilateralSliceApplyParamSpec bilateralSliceApplyParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(bilateral_slice_apply_checkpara_mali_common(handle, inputDesc, input, guideDesc,
        guide, gridDesc, grid, bilateralSliceApplyParamSpec, outputDesc, output));
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = bilateral_slice_apply_mali_fp16(handle, inputDesc, input, guideDesc, guide,
                gridDesc, grid, bilateralSliceApplyParamSpec, forwardRunInfo, tmpBytes, tmpBuf,
                outputDesc, output);
            break;
        }
        case DT_U8: {
            ret = bilateral_slice_apply_mali_uchar(handle, inputDesc, input, guideDesc, guide,
                gridDesc, grid, bilateralSliceApplyParamSpec, forwardRunInfo, tmpBytes, tmpBuf,
                outputDesc, output);
            break;
        }
        case DT_I8: {
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
