// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/bilateral_slice_apply_mali_fp16.h"

inline EE bilateral_slice_apply_checkpara_mali_fp16(
    TensorDesc inputDesc, TensorDesc guideDesc, TensorDesc gridDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != guideDesc.dt || inputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    if (inputDesc.dt != gridDesc.dt || inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE bilateral_slice_apply_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc guideDesc,
    const GCLMem_t guide,
    TensorDesc gridDesc,
    const GCLMem_t grid,
    BilateralSliceApplyParamSpec bilateralSliceApplyParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    UNUSED(guideDesc);
    UNUSED(forwardRunInfo);
    U32 iw, ih, ic, in;
    U32 gw, gh, gc, gn;
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(gridDesc, NULL, NULL, &gn, &gc, &gh, &gw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);

    U32 coe = bilateralSliceApplyParamSpec.coefficient_len;
    BilateralSliceApplyMode mode = bilateralSliceApplyParamSpec.mode;
    //    bool has_offset = bilateralSliceApplyParamSpec.has_offset;
    U32 dep = gc / coe;
    U32 gcw = gc * gw;
    U32 wh = iw * ih;
    F32 scale_x = (F32)gw / iw;
    F32 scale_y = (F32)gh / ih;
    Mem inbuf, gridbuf, guidebuf, outbuf, gridTran;
    inbuf = input->mem;
    gridbuf = grid->mem;
    outbuf = output->mem;
    gridTran = tmpBuf->mem;
    if (mode == BSliceApply_NULL) {
        guidebuf = guide->mem;
    } else {
        guidebuf = inbuf;
    }

    U32 gs0[3] = {gc / 4, gw, ih};
    U32 ls0[3] = {0, 0, 0};
    U32 dim0 = 3;
    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel(handle, "bilateral_slice_apply_pre", &kernel));
    CHECK_STATUS(
        gcl_set_kernelArgs(kernel, gh, gc, gcw, gs0[0], gs0[1], scale_y, gridbuf, gridTran));
    gcl_set_kernelVec(handle, kernel, dim0, gs0, ls0, "bilateral_slice_apply_pre");

#ifdef _DEBUG
    CHECK_STATUS(
        gcl_run_kernel_profiling(handle, kernel, dim0, gs0, ls0, "bilateral_slice_apply_pre"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, grid, "bilateral_slice_apply_grid"));
#endif
    char kernelname[128];
    if (mode == BSliceApply_CONV) {
        sprintf(kernelname, "bilateral_slice_apply_c12_conv");
    } else {
        sprintf(kernelname, "bilateral_slice_apply_c12");
    }
    U32 gs[2] = {ow, oh};
    U32 ls[2] = {0, 0};
    U32 dim = 2;
    CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw, wh, gc, gw, gh, gcw, dep, coe, gs[0], gs[1],
        scale_x, scale_y, guidebuf, gridTran, inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);

#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel_profiling(handle, kernel, dim, gs, ls, kernelname));
    CHECK_STATUS(gcl_print_memory<F16>(handle, input, "bilateral_slice_apply_input"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, output, "bilateral_slice_apply_output"));
    if (mode == BSliceApply_NULL) {
        CHECK_STATUS(gcl_print_memory<F16>(handle, guide, "bilateral_slice_apply_guide"));
    }
#endif
    return SUCCESS;
}

EE bilateral_slice_apply_mali_fp16(GCLHandle_t handle,
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
    UNUSED(tmpBytes);
    CHECK_STATUS(
        bilateral_slice_apply_checkpara_mali_fp16(inputDesc, guideDesc, gridDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(bilateral_slice_apply_core_mali_fp16(handle, inputDesc, input, guideDesc, guide,
        gridDesc, grid, bilateralSliceApplyParamSpec, forwardRunInfo, tmpBuf, outputDesc, output));
    return SUCCESS;
}
