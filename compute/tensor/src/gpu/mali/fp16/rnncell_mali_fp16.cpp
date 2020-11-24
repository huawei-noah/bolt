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
#include "error.h"
#include "types.h"
#include "gpu/mali/fp16/rnncell_mali_fp16.h"
#define get_xDim(xDesc, xDim)                       \
    {                                               \
        if (xDesc.nDims == 2 || xDesc.df == DF_MTK) \
            xDim = xDesc.dims[0];                   \
        if (xDesc.df == DF_MKT)                     \
            xDim = xDesc.dims[1];                   \
    }

inline EE rnncell_checkpara_mali_fp16(
    TensorDesc xDesc, TensorDesc filterDesc, TensorDesc biasDesc, TensorDesc hDesc)
{
    if (xDesc.dt != filterDesc.dt || xDesc.dt != biasDesc.dt || xDesc.dt != hDesc.dt ||
        xDesc.dt != DT_F16) {
        return NOT_MATCH;
    }
    return SUCCESS;
}

inline EE rnncell_core_mali_fp16(GCLHandle_t handle,
    TensorDesc xDesc,
    const GCLMem_t currentX,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc biasDesc,
    GCLMem_t bias,
    GCLMem_t state,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    RNNParamSpec rnncellDesc,
    U32 batchStrideX,
    U32 batchStrideH,
    TensorDesc hDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    UNUSED(batchStrideX);
    UNUSED(batchStrideH);
    UNUSED(hDesc);
    U32 item_c = forwardRunInfo->best_c[0];
    U32 hDim = rnncellDesc.numOutput;
    U32 col = (rnncellDesc.numProjection > 0) ? rnncellDesc.numProjection : hDim;
    bool project = (rnncellDesc.numProjection > 0) ? true : false;
    float fbias = rnncellDesc.forgetBias;
    float zonecell = rnncellDesc.zoneoutCell;
    float zoneout = rnncellDesc.zoneoutOutput;

    DataType dt = xDesc.dt;
    U32 xDim;
    get_xDim(xDesc, xDim);
    Mem xMem = currentX->mem;
    Mem sMem = state->mem;
    Mem xhMem;
    U32 offset = 0;
    U32 xhNum, xhSize;
    xhNum = (xDim + hDim + item_c - 1) / item_c * item_c;
    xhSize = xhNum * bytesOf(dt);
    CHECK_STATUS(gcl_create_sub_buffer(xhSize, &offset, tmpBuf, &xhMem));

    Mem interMem;
    U32 interNum, interSize;
    U32 filterRow, filterCol;
    tensorSelectGet(filterDesc, NULL, NULL, NULL, NULL, &filterRow, &filterCol);
    interNum = filterRow + 4;
    interSize = interNum * bytesOf(dt);
    CHECK_STATUS(gcl_create_sub_buffer(interSize, &offset, tmpBuf, &interMem));

    Mem tmpOut;
    Mem outbuf = output->mem;
    if (project) {
        U32 item_cp = forwardRunInfo->best_c[1];
        U32 tmpOutNum = (col + item_cp - 1) / item_cp * item_cp;
        U32 tmpOutSize = tmpOutNum * bytesOf(dt);
        CHECK_STATUS(gcl_create_sub_buffer(tmpOutSize, &offset, tmpBuf, &tmpOut));
        outbuf = tmpOut;
    }

    U32 xh_str, xw_str, xh_off, xw_off;
    get_gclmem_dim(currentX->desc, &xw_str, &xh_str, NULL, &xw_off, &xh_off);
    if (xw_str != 1 || xh_str != 1 || xw_off != 0 || xh_off != 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    U32 gs1 = xhNum;
    U32 ls1 = 0;
    U32 dim = 1;
    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel(handle, "rnncell_build_xh", &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, xDim, xDim + hDim, col, gs1, xMem, sMem, xhMem));
    gcl_set_kernelVec(handle, kernel, dim, &gs1, &ls1, "rnncell_build_xh");
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs1, &ls1, "rnncell_build_xh"));
    handle->t_total += handle->t_execute;
#endif

    Mem fltbuf = filter[0].mem;
    Mem biasMem = bias->mem;
    char kernelname[128];
    U32 ic_str = filter[0].desc.stride[1];
    sprintf(kernelname, "conv_direct_spe_fwhs1_%d", item_c);
    gs1 = filterRow;
    CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, 1, 1, ic_str, 0, 0, 1, 1, 0, 0, filterRow, 0, 0, gs1, 1,
        xhMem, fltbuf, biasMem, interMem));
    gcl_set_kernelVec(handle, kernel, dim, &gs1, &ls1, kernelname);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs1, &ls1, kernelname));
    handle->t_total += handle->t_execute;
#endif

    U8 noproject = (project) ? 0 : 1;
    gs1 = (col + 3) / 4;
    CHECK_STATUS(gcl_create_kernel(handle, "rnncell_update_res", &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(
        kernel, col, noproject, gs1, fbias, zonecell, zoneout, sMem, interMem, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, &gs1, &ls1, "rnncell_update_res");
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs1, &ls1, "rnncell_update_res"));
    handle->t_total += handle->t_execute;
#endif

    if (project) {
        item_c = forwardRunInfo->best_c[1];
        filterRow = rnncellDesc.numOutput;
        ic_str = filter[1].desc.stride[1];
        Mem fltbuf = filter[1].mem;
        sprintf(kernelname, "conv_direct_spe_fwhs1_nobias_%d", item_c);
        gs1 = filterRow;
        CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, 1, 1, ic_str, 0, 0, 1, 1, 0, 0, filterRow, 0, 0, gs1, 1,
            outbuf, fltbuf, biasMem, output->mem));
        gcl_set_kernelVec(handle, kernel, dim, &gs1, &ls1, kernelname);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs1, &ls1, kernelname));
        handle->t_total += handle->t_execute;
#endif

        gs1 = (hDim + 3) / 4;
        CHECK_STATUS(gcl_create_kernel(handle, "rnncell_update_project_state", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, hDim, col, gs1, zoneout, output->mem, sMem));
        gcl_set_kernelVec(handle, kernel, dim, &gs1, &ls1, "rnncell_update_project_state");
#ifdef _DEBUG
        CHECK_STATUS(
            gcl_run_kernel(handle, kernel, dim, &gs1, &ls1, "rnncell_update_project_state"));
        handle->t_total += handle->t_execute;
#endif
    }
    return SUCCESS;
}

EE rnncell_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    RNNParamSpec rnncellDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    UNUSED(outputDesc);
    U32 item_c = forwardRunInfo->best_c[0];
    DataType dt = inputDesc.dt;
    U32 xDim;
    get_xDim(inputDesc, xDim);
    U32 hDim = rnncellDesc.numOutput;
    U32 xhNum = (xDim + hDim + item_c - 1) / item_c * item_c;
    U32 xhSize = (xhNum * bytesOf(dt) + 1023) / 1024 * 1024;

    U32 filterRow;
    tensorSelectGet(filterDesc, NULL, NULL, NULL, NULL, &filterRow, NULL);
    U32 interNum = filterRow + 4;
    U32 interSize = (interNum * bytesOf(dt) + 1023) / 1024 * 1024;

    U32 tmpOutSize = 0;
    if (rnncellDesc.numProjection > 0) {
        U32 tmpOutNum = rnncellDesc.numProjection;
        tmpOutSize = (tmpOutNum * bytesOf(dt) + 1023) / 1024 * 1024;
    }
    *bytes = xhSize + interSize + tmpOutSize;
    return SUCCESS;
}

EE rnncell_mali_fp16(GCLHandle_t handle,
    TensorDesc xDesc,
    const GCLMem_t currentX,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc biasDesc,
    GCLMem_t bias,
    GCLMem_t state,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    RNNParamSpec rnncellDesc,
    U32 batchStrideX,
    U32 batchStrideH,
    TensorDesc hDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    CHECK_STATUS(rnncell_checkpara_mali_fp16(xDesc, filterDesc, biasDesc, hDesc));
    CHECK_STATUS(fill_output_zero(handle, output, hDesc));
    CHECK_STATUS(rnncell_core_mali_fp16(handle, xDesc, currentX, filterDesc, filter, biasDesc, bias,
        state, tmpBytes, tmpBuf, rnncellDesc, batchStrideX, batchStrideH, hDesc, output,
        forwardRunInfo));
    return SUCCESS;
}
