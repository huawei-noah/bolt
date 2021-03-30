// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/rnn_mali_fp16.h"

inline EE rnn_checkpara_mali_fp16(
    TensorDesc xDesc, TensorDesc filterDesc, TensorDesc biasDesc, TensorDesc hDesc)
{
    if (xDesc.dt != filterDesc.dt || xDesc.dt != biasDesc.dt || xDesc.dt != hDesc.dt ||
        xDesc.dt != DT_F16) {
        return NOT_MATCH;
    }
    return SUCCESS;
}

inline EE rnn_core_mali_fp16(GCLHandle_t handle,
    TensorDesc xDesc,
    const GCLMem_t currentX,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc biasDesc,
    GCLMem_t bias,
    GCLMem_t state,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    RNNParamSpec rnnParamSpec,
    U32 batchStrideX,
    U32 batchStrideH,
    TensorDesc hDesc,
    GCLMem_t currentH,
    ForwardRunInfoMali_t forwardRunInfo)
{
    UNUSED(handle);
    UNUSED(xDesc);
    UNUSED(currentX);
    UNUSED(filterDesc);
    UNUSED(filter);
    UNUSED(biasDesc);
    UNUSED(bias);
    UNUSED(state);
    UNUSED(tmpBytes);
    UNUSED(tmpBuf);
    UNUSED(rnnParamSpec);
    UNUSED(batchStrideX);
    UNUSED(batchStrideH);
    UNUSED(hDesc);
    UNUSED(currentH);
    UNUSED(forwardRunInfo);
    return NOT_SUPPORTED;
}

EE rnn_transform_filter_bytes_mali_fp16(TensorDesc filterDesc,
    RNNParamSpec rnnParamSpec,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    U32 filterRow, filterCol;
    tensorSelectGet(filterDesc, NULL, NULL, NULL, NULL, &filterRow, &filterCol);
    U32 s0, s1, s2, num, byteSize, item_c;
    U32 filterNum = (rnnParamSpec.numProjection > 0) ? 2 : 1;
    for (U32 i = 0; i < filterNum; ++i) {
        item_c = forwardRunInfo->best_c[i];
        if (i == 0) {
            s0 = filterRow;
            s1 = (filterCol + item_c - 1) / item_c;
        } else {
            s0 = rnnParamSpec.numOutput;
            s1 = (rnnParamSpec.numProjection + item_c - 1) / item_c;
        }
        s2 = 1;
        num = s0 * s1 * s2 * item_c;
        byteSize = num * bytesOf(DT_F16);
        gclmemFilterDesc[i].stride[0] = s0;
        gclmemFilterDesc[i].stride[1] = s1;
        gclmemFilterDesc[i].stride[2] = s2;
        gclmemFilterDesc[i].offset[0] = 0;
        gclmemFilterDesc[i].offset[1] = 0;
        gclmemFilterDesc[i].offset[2] = 0;
        gclmemFilterDesc[i].num = num;
        gclmemFilterDesc[i].byteSize = byteSize;
        gclmemFilterDesc[i].memType = GCL_MEM_BUF;
        gclmemFilterDesc[i].flags = CL_MEM_READ_WRITE;
        gclmemFilterDesc[i].memFormat = DF_CHWNC4;
        if (item_c == 8) {
            gclmemFilterDesc[i].memFormat = DF_CHWNC8;
        }
        if (item_c == 16) {
            gclmemFilterDesc[i].memFormat = DF_CHWNC16;
        }
        gclmemFilterDesc[i].host_ptr = NULL;
    }
    *bytes = 0;
    return SUCCESS;
}

EE rnn_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    RNNParamSpec rnnParamSpec,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem,
    ForwardRunInfoMali_t forwardRunInfo)
{
    DataType fdt;
    U32 filterRow, filterCol;
    tensorSelectGet(filterDesc, &fdt, NULL, NULL, NULL, &filterRow, &filterCol);
    U32 filterNum = (rnnParamSpec.numProjection > 0) ? 2 : 1;
    U32 item_c, item_k;

    char kernelname[128];
    Kernel kernel;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    U32 fwh = 1;
    for (U32 i = 0; i < filterNum; i++) {
        item_c = forwardRunInfo->best_c[i];
        item_k = forwardRunInfo->best_k[i];
        sprintf(kernelname, "conv_direct_trans_fltbuf_%d%d", item_c, item_k);
        CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
        if (i == 1) {
            filterCol = rnnParamSpec.numProjection;
            filterRow = rnnParamSpec.numOutput;
        }
        CHECK_STATUS(
            gcl_set_kernelArgs(kernel, fwh, filterCol, filterRow, filter[i].mem, fltmem[i].mem));
        gs[0] = fwh;
        gs[1] = (filterCol + item_c - 1) / item_c;
        gs[2] = filterRow;
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
        fltmemDesc[i] = tensor2df(fdt, DF_NORMAL, filterRow, filterCol);
    }
    return SUCCESS;
}

EE rnn_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    RNNParamSpec rnnParamSpec,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    UNUSED(inputDesc);
    UNUSED(filterDesc);
    UNUSED(outputDesc);
    UNUSED(rnnParamSpec);
    UNUSED(bytes);
    UNUSED(forwardRunInfo);
    return SUCCESS;
}

EE rnn_mali_fp16(GCLHandle_t handle,
    TensorDesc xDesc,
    const GCLMem_t currentX,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc biasDesc,
    GCLMem_t bias,
    GCLMem_t state,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    RNNParamSpec rnnParamSpec,
    U32 batchStrideX,
    U32 batchStrideH,
    TensorDesc hDesc,
    GCLMem_t currentH,
    ForwardRunInfoMali_t forwardRunInfo)
{
    CHECK_STATUS(rnn_checkpara_mali_fp16(xDesc, filterDesc, biasDesc, hDesc));
    CHECK_STATUS(rnn_core_mali_fp16(handle, xDesc, currentX, filterDesc, filter, biasDesc, bias,
        state, tmpBytes, tmpBuf, rnnParamSpec, batchStrideX, batchStrideH, hDesc, currentH,
        forwardRunInfo));
    return SUCCESS;
}
