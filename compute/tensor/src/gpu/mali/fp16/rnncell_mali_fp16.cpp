// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/rnncell_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/conv_direct_opt.h"

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
    U32 item_c = forwardRunInfo->best_c[0];
    U32 hDim = rnncellDesc.numOutput;
    U32 col = (rnncellDesc.numProjection > 0) ? rnncellDesc.numProjection : hDim;
    bool project = (rnncellDesc.numProjection > 0) ? true : false;
    float fbias = rnncellDesc.forgetBias;
    float zonecell = rnncellDesc.zoneoutCell;
    float zoneout = rnncellDesc.zoneoutOutput;
    DataType dt = xDesc.dt;
    U32 xDim = xDesc.dims[0];
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

    U32 xw_str, xh_str, xh_off, xw_off;
    U32 hw_str, hh_str, hh_off, hw_off;
    CHECK_STATUS(gclmem_get_desc_padding(currentX->desc, &xw_str, &xh_str, NULL, &xw_off, &xh_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &hw_str, &hh_str, NULL, &hw_off, &hh_off));
    U32 x_off = xh_off * xw_str + xw_off;
    U32 h_off = hh_off * hw_str + hw_off;

    Mem tmpOut;
    Mem outbuf = output->mem;
    if (project) {
        U32 item_cp = forwardRunInfo->best_c[1];
        U32 tmpOutNum = (col + item_cp - 1) / item_cp * item_cp;
        U32 tmpOutSize = tmpOutNum * bytesOf(dt);
        CHECK_STATUS(gcl_create_sub_buffer(tmpOutSize, &offset, tmpBuf, &tmpOut));
        outbuf = tmpOut;
        h_off = 0;
    }

    U32 gs[3] = {xhNum, 1, 1};
    U32 ls[3] = {0, 0, 0};
    U32 ls_update[3] = {16, 1, 1};
    U32 dim = 1;
    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel(handle, "rnncell_build_xh", &kernel));
    CHECK_STATUS(
        gcl_set_kernelArgs(kernel, xDim, xDim + hDim, x_off, col, gs[0], xMem, sMem, xhMem));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, "rnncell_build_xh");

    Mem fltbuf = filter[0].mem;
    Mem biasMem = bias[0].mem;
    char kernelName[128];
    KernelOpt kernelOpt;
    CHECK_STATUS(set_conv_direct_spe_fwhs1_opt_mali(
        1, 1, 1, 1, item_c, false, true, ACTIVATION_NULL, DT_F16, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));

    gs[0] = filterRow;
    U32 ic_str = filter[0].desc.stride[1];
    CHECK_STATUS(gcl_set_kernelArgs(kernel, 1, 1, ic_str, 0, 0, 1, filterRow, 0, 0, filterRow, 0, 0,
        gs[0], 1, xhMem, fltbuf, biasMem, interMem));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);

    U8 noproject = (project) ? 0 : 1;
    gs[0] = (col + 3) / 4;
    CHECK_STATUS(gcl_create_kernel(handle, "rnncell_update_res", &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(
        kernel, col, noproject, h_off, gs[0], fbias, zonecell, zoneout, sMem, interMem, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls_update, "rnncell_update_res");

    if (project) {
        item_c = forwardRunInfo->best_c[1];
        filterRow = rnncellDesc.numOutput;
        ic_str = filter[1].desc.stride[1];
        fltbuf = filter[1].mem;
        biasMem = bias[1].mem;
        CHECK_STATUS(set_conv_direct_spe_fwhs1_opt_mali(
            1, 1, 1, 1, item_c, true, true, ACTIVATION_NULL, DT_F16, kernelName, &kernelOpt));
        gs[0] = filterRow;
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, 1, 1, ic_str, 0, 0, hh_str, hw_str, hh_off, hw_off,
            filterRow, 0, 0, gs[0], 1, outbuf, fltbuf, biasMem, output->mem));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
        gs[0] = (hDim + 3) / 4;
        CHECK_STATUS(gcl_create_kernel(handle, "rnncell_update_project_state", &kernel));
        CHECK_STATUS(
            gcl_set_kernelArgs(kernel, hDim, col, h_off, gs[0], zoneout, output->mem, sMem));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls_update, "rnncell_update_project_state");
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
    U32 xDim = inputDesc.dims[0];
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
