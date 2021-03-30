// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/resize_bilinear_mali_fp16.h"

inline EE resize_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    if (outputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE resize_bilinear_core_mali_fp16(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);

    F32 v_ratio[2] = {(F32)(ih - 1) / (F32)(oh - 1), (F32)(iw - 1) / (F32)(ow - 1)};

    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);

    cl_mem inbuf = input->mem;
    cl_mem outbuf = output->mem;

    U32 gs[3] = {oh, ow, (oc + 3) / 4};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel(handle, "resize_bilinear", &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, ih_str, ih_off, iw, iw_str, iw_off, oh, oh_str,
        oh_off, ow, ow_str, ow_off, v_ratio[0], v_ratio[1], inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, "resize_bilinear");
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "resize_bilinear"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, input, "resize_input"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, output, "resize_output"));
#endif
    return SUCCESS;
}

inline EE resize_bilinear_core_nchw_mali_fp16(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);

    F32 v_ratio[2] = {(F32)(ih - 1) / (F32)(oh - 1), (F32)(iw - 1) / (F32)(ow - 1)};

    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);

    cl_mem inbuf = input->mem;
    cl_mem outbuf = output->mem;
    char kernelname[128];
    sprintf(kernelname, "resize_bilinear_nchw");
    U32 gs[3] = {ow, oh, oc};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, ih_str, ih_off, iw, iw_str, iw_off, oh, oh_str,
        oh_off, ow, ow_str, ow_off, v_ratio[0], v_ratio[1], inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

EE resize_bilinear_mali_fp16(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    CHECK_STATUS(resize_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    if (input->desc.memFormat == DF_NCHW) {
        CHECK_STATUS(
            resize_bilinear_core_nchw_mali_fp16(handle, inputDesc, input, outputDesc, output));
    } else {
        CHECK_STATUS(resize_bilinear_core_mali_fp16(handle, inputDesc, input, outputDesc, output));
    }
    return SUCCESS;
}
