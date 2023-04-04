// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/depth2space_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/depth2space_opt.h"

inline EE depth2space_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE depth2space_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    Depth2SpaceParamSpec p,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    DataType dt;
    U32 iw, ih, ic, in;
    U32 oc;
    tensorSelectGet(inputDesc, &dt, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, NULL, &oc, NULL, NULL);
    U32 iw_str, ih_str, iw_off, ih_off, ihw_str, ic_str, i_off;
    U32 ow_str, oh_str, ow_off, oh_off, ohw_str, o_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    ihw_str = iw_str * ih_str;
    ohw_str = ow_str * oh_str;
    cl_mem inbuf, outbuf, tmp;
    inbuf = input->mem;
    outbuf = output->mem;
    tmp = tmpBuf->mem;
    DataFormat imf = input->desc.memFormat;
    DataFormat omf = output->desc.memFormat;
    i_off = ih_off * iw_str + iw_off;
    o_off = oh_off * ow_str + ow_off;
    Kernel kernel;
    char kernelName[128];
    KernelOpt kernelOpt;

    if (imf == DF_NCHWC4 && p.block_size == 2) {
        U32 gs[3] = {iw, ih, (ic_str + 3) / 4};
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        bool useOutputNchw = (omf == DF_NCHW) ? true : false;
        CHECK_STATUS(set_depth2space_nchwc4_2x2_opt(
            useOutputNchw, dt, input->desc.memType, GCL_MEM_BUF, kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, p.block_size, iw_str, ihw_str, ic_str, i_off,
            ow_str, oh_str, ohw_str, o_off, iw, ih, oc, inbuf, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    } else {
        if (imf == DF_NCHWC4) {
            GCLMem tMem;
            GCLMemDesc desc = input->desc;
            U32 str[3] = {iw, ih, ic * in};
            U32 off[3] = {0, 0, 0};
            MemFlags flag = CL_MEM_READ_WRITE;
            CHECK_STATUS(
                gclmem_set_desc_padding(&desc, str, off, dt, DF_NCHW, GCL_MEM_BUF, flag));
            tMem.desc = desc;
            tMem.mem = tmp;
            CHECK_STATUS(ocl_data_trans_form(handle, input, &tMem, 0, 0, NCHWC4_TO_NCHW));
            inbuf = tmp;
        }
        U32 gs[3] = {
            iw, ih, (ic / (p.block_size * p.block_size) + 3) / 4 * (p.block_size * p.block_size)};
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        CHECK_STATUS(set_common_opt(
            dt, GCL_MEM_BUF, GCL_MEM_BUF, "depth2space_nchw", kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, p.block_size, iw_str, ihw_str, ow_str, ohw_str,
            i_off, o_off, iw, ih, ic, inbuf, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    }
    return SUCCESS;
}

EE depth2space_infer_tmpBuf_size_mali_fp16(
    TensorDesc inputDesc, Depth2SpaceParamSpec p, TensorDesc outputDesc, U32 *bytes)
{
    UNUSED(outputDesc);
    DataFormat idf;
    DataType idt;
    U32 iw, ih, ic, in;
    tensorSelectGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw);
    *bytes = 0;
    if (p.block_size != 2) {
        *bytes = in * ic * ih * iw * bytesOf(idt);
    }
    return SUCCESS;
}

EE depth2space_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    Depth2SpaceParamSpec p,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(depth2space_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(
        depth2space_core_mali_fp16(handle, inputDesc, input, p, tmpBuf, outputDesc, output));
    return ret;
}
