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
#include "gpu/mali/fp16/gemv_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/conv_direct_opt.h"
#include "gpu/mali/cl/kernel_option/rnncell_update_res_opt.h"

inline EE rnncell_checkpara_mali_fp16(
    TensorDesc xDesc, TensorDesc filterDesc, TensorDesc biasDesc, TensorDesc hDesc)
{
    if (xDesc.dt != filterDesc.dt || xDesc.dt != biasDesc.dt || xDesc.dt != hDesc.dt) {
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
    U32 hDim = rnncellDesc.num_outputs;
    U32 col = (rnncellDesc.num_projection > 0) ? rnncellDesc.num_projection : hDim;
    bool project = (rnncellDesc.num_projection > 0) ? true : false;
    float fbias = rnncellDesc.forget_bias;
    float zonecell = rnncellDesc.zoneout_cell;
    float zoneout = rnncellDesc.zoneout_output;
    U32 xw_str, xh_str, xh_off, xw_off;
    U32 hw_str, hh_str, hh_off, hw_off;
    CHECK_STATUS(gclmem_get_desc_padding(currentX->desc, &xw_str, &xh_str, NULL, &xw_off, &xh_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &hw_str, &hh_str, NULL, &hw_off, &hh_off));
    U32 x_off = xh_off * xw_str + xw_off;
    U32 h_off = hh_off * hw_str + hw_off;

    DataType dt = xDesc.dt;
    U32 xDim = xDesc.dims[0];
    Mem xMem = currentX->mem;
    Mem sMem = state->mem;
    Mem xhMem;
    U32 offset = 0;
    U32 xhNum, xhSize;
    U32 c_align = (item_c > 16) ? (item_c >> 4) : item_c;
    xhNum = UNI_ALIGN(xDim + hDim, c_align);
    xhSize = xhNum * bytesOf(dt);
    CHECK_STATUS(gcl_create_sub_buffer(xhSize, &offset, tmpBuf, &xhMem));

    Mem interMem;
    U32 interNum, interSize;
    U32 filterRow, filterCol;
    filterCol = hDim + xDim;
    filterRow = 4 * col;
    interNum = filterRow + 4;
    interSize = interNum * bytesOf(dt);
    CHECK_STATUS(gcl_create_sub_buffer(interSize, &offset, tmpBuf, &interMem));

    Mem tmpOut;
    Mem outbuf = output->mem;
    if (project) {
        U32 item_cp = forwardRunInfo->best_c[1];
        U32 cp_align = (item_cp > 16) ? (item_cp >> 4) : item_cp;
        U32 tmpOutNum = UNI_ALIGN(col, cp_align);
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

    U32 tmpOff = offset;
    Mem reduceMem = tmpBuf->mem;
    char kernelName[128];
    KernelOpt kernelOpt;
    CHECK_STATUS(gemv_build_run_info(handle, item_c, filterRow, 1, {}, true, false, dt,
        &tmpOff, tmpBuf, &reduceMem, kernelName, &kernelOpt));

    Mem fltbuf = filter[0].mem;
    Mem biasMem = bias[0].mem;
    CHECK_STATUS(gemv_run(handle, item_c, filterRow, xhNum, 1, 0, 0, 0, 0, xhMem, fltbuf, biasMem,
        reduceMem, interMem, kernelName, &kernelOpt));

    CHECK_STATUS(set_rnncell_update_res_opt_mali(
        project, false, dt, GCL_MEM_BUF, GCL_MEM_BUF, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));

    gs[0] = (col + 3) / 4;
    CHECK_STATUS(gcl_set_kernelArgs(
        kernel, col, h_off, gs[0], fbias, zonecell, zoneout, sMem, interMem, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls_update, kernelName);

    if (project) {
        item_c = forwardRunInfo->best_c[1];
        filterRow = rnncellDesc.num_outputs;
        fltbuf = filter[1].mem;
        tmpOff = offset;
        //biasMem = bias[1].mem;
        CHECK_STATUS(gemv_build_run_info(handle, item_c, filterRow, 1, {}, false,
            false, dt, &tmpOff, tmpBuf, &reduceMem, kernelName, &kernelOpt));
        CHECK_STATUS(gemv_run(handle, item_c, filterRow, col, 1, 0, 0, 0, 0, outbuf, fltbuf,
            biasMem, reduceMem, output->mem, kernelName, &kernelOpt));

        gs[0] = (hDim + 3) / 4;
        CHECK_STATUS(gcl_create_kernel(handle, "rnncell_update_project_state", &kernel));
        CHECK_STATUS(
            gcl_set_kernelArgs(kernel, hDim, col, h_off, gs[0], zoneout, output->mem, sMem));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls_update, "rnncell_update_project_state");
    }
    return SUCCESS;
}

inline void transform_filter_desc(TensorDesc filterDesc,
    RNNParamSpec rnnParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *ftmDesc)
{
    U32 item_h = forwardRunInfo->best_h[0];
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    ftmDesc[0] = gemv_transform_filter_desc(filterDesc, item_h, item_c, item_k);
    bool useProject = (rnnParamSpec.num_projection > 0) ? true : false;
    if (useProject) {
        item_h = forwardRunInfo->best_h[1];
        item_c = forwardRunInfo->best_c[1];
        item_k = forwardRunInfo->best_k[1];
        TensorDesc filterDescPro = tensor2df(
            filterDesc.dt, DF_NORMAL, rnnParamSpec.num_outputs, rnnParamSpec.num_projection);
        ftmDesc[1] = gemv_transform_filter_desc(filterDescPro, item_h, item_c, item_k);
    }
}

EE rnncell_transform_filter_bytes_mali_fp16(TensorDesc filterDesc,
    RNNParamSpec rnnParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *ftmDesc)
{
    transform_filter_desc(filterDesc, rnnParamSpec, forwardRunInfo, ftmDesc);
    return SUCCESS;
}

EE rnncell_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    RNNParamSpec rnnParamSpec,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem,
    ForwardRunInfoMali_t forwardRunInfo)
{
    U32 filterNum = (rnnParamSpec.num_projection > 0) ? 2 : 1;
    for (U32 i = 0; i < filterNum; i++) {
        ForwardRunInfoMali runInfo = *forwardRunInfo;
        if (i == 1) {
            runInfo.best_h[i - 1] = runInfo.best_h[i];
            runInfo.best_c[i - 1] = runInfo.best_c[i];
            runInfo.best_k[i - 1] = runInfo.best_k[i];
            filterDesc.dims[0] = rnnParamSpec.num_projection;
            filterDesc.dims[1] = rnnParamSpec.num_outputs;
        }
        CHECK_STATUS(gemv_transform_filter_mali_fp16(
            handle, filterDesc, &filter[i], &fltmemDesc[i], &fltmem[i], &runInfo));
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
    U32 item_c = forwardRunInfo->best_c[0];
    DataType dt = inputDesc.dt;
    U32 xDim = inputDesc.dims[0];
    U32 hDim = rnncellDesc.num_outputs;
    U32 c_align = (item_c > 16) ? (item_c >> 4) : item_c;
    U32 xhNum = UNI_ALIGN(xDim + hDim, c_align);
    U32 xhSize = UNI_ALIGN(xhNum * bytesOf(dt), BUFFER_ALIGN_BASE);

    U32 col = (rnncellDesc.num_projection > 0) ? rnncellDesc.num_projection : hDim;
    U32 filterRow = col * 4;
    U32 interNum = filterRow + 4;
    U32 interSize = UNI_ALIGN(interNum * bytesOf(dt), BUFFER_ALIGN_BASE);

    U32 tmpOutSize = 0;
    U32 filterRowPro = 0;
    U32 item_cp = item_c;
    if (rnncellDesc.num_projection > 0) {
        item_cp = forwardRunInfo->best_c[1];
        U32 cp_align = (item_cp > 16) ? (item_cp >> 4) : item_cp;
        U32 tmpOutNum = UNI_ALIGN(col, cp_align);
        tmpOutSize = UNI_ALIGN(tmpOutNum * bytesOf(dt), BUFFER_ALIGN_BASE);
        filterRowPro = rnncellDesc.num_outputs;
    }

    U32 reduceSize = 0;
    if (item_c > 16 || item_cp > 16) {
        U32 row = (filterRow > filterRowPro) ? filterRow : filterRowPro;
        reduceSize = UNI_ALIGN(row * 32 * bytesOf(dt), BUFFER_ALIGN_BASE);
    }
    *bytes = xhSize + interSize + tmpOutSize + reduceSize;
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
