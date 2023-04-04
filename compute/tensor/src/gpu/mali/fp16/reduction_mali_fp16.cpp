// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "tensor_desc.h"

#include "gpu/mali/fp16/reduction_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/reduction_opt.h"

inline EE reduction_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return SUCCESS;
}

inline EE reduction_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc maskDesc,
    GCLMem_t mask,
    ReductionParamSpec p,
    GCLMem_t tmp,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    int axisTran[6];
    int axis = 0;
    for (int i = 0; i < p.num_axes; i++) {
        axis = p.axes[i];
        if (axis < 0) {
            axis = inputDesc.nDims + axis;
        }
        axis = inputDesc.nDims - 1 - axis;
        axisTran[i] = axis;
    }

    DataType idt;
    DataFormat imf, omf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    U32 iw_str, ih_str, ic_str, iw_off, ih_off, i_off;
    U32 ow_str, oh_str, oc_str, ow_off, oh_off, o_off;
    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off);
    i_off = ih_off * iw_str + iw_off;
    o_off = oh_off * ow_str + ow_off;
    if (in > 1 && axis > 2) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    axis = axisTran[0];
    U32 od = outputDesc.nDims;
    imf = input->desc.memFormat;
    omf = output->desc.memFormat;
    Mem inbuf = input->mem;
    Mem outbuf = output->mem;
    Mem tmpbuf = tmp->mem;
    int keep_dim = (p.keep_dim) ? 1 : 0;
    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    U32 gs[3] = {0, 0, 0};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    bool useOc4 = false;
    bool useNchw = false;
    U32 edge;
    if (imf == DF_NCHWC4 && omf == DF_NCHWC4 && keep_dim) {
        useOc4 = true;
        gs[0] = ow;
        gs[1] = oh;
        gs[2] = (oc + 3) / 4 * on;
        edge = oc;
    } else if (imf == DF_NCHWC4) {
        gs[0] = iw;
        gs[1] = ih;
        gs[2] = (ic + 3) / 4 * in;
        gs[axis] = 1;
        edge = ic;
    } else {
        gs[0] = (iw + 3) >> 2;
        gs[1] = ih;
        gs[2] = ic * in;
        gs[axis] = 1;
        useNchw = true;
        edge = ow;
    }
    CHECK_STATUS(set_reduction_opt_mali(
        useNchw, useOc4, axis, p.mode, idt, GCL_MEM_BUF, GCL_MEM_BUF, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, ow_str, oh_str, i_off, o_off, iw, ih,
        ic, edge, keep_dim, od, gs[0], gs[1], inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

EE reduction_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    ReductionParamSpec p,
    TensorDesc outputDesc,
    GCLMemDesc gclmemInputDesc,
    GCLMemDesc gclmemOutputDesc,
    U32 *bytes)
{
    U32 size = 0;
    *bytes = size;
    return SUCCESS;
}

EE reduction_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc maskDesc,
    GCLMem_t mask,
    ReductionParamSpec p,
    GCLMem_t tmp,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(reduction_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(reduction_core_mali_fp16(
        handle, inputDesc, input, maskDesc, mask, p, tmp, outputDesc, output));
    return ret;
}
