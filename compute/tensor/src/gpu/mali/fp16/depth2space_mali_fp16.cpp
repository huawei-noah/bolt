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

inline EE depth2space_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    if (outputDesc.dt != DT_F16) {
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
    UNUSED(outputDesc);
    U32 iw, ih, ic, in;
    U32 oc;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, NULL, &oc, NULL, NULL);
    U32 iw_str, ih_str, iw_off, ih_off, iwh_str, ic_str;
    U32 ow_str, oh_str, ow_off, oh_off, owh_str;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    iwh_str = iw_str * ih_str;
    owh_str = ow_str * oh_str;
    cl_mem inbuf, outbuf, tmp;
    inbuf = input->mem;
    outbuf = output->mem;
    tmp = tmpBuf->mem;
    DataFormat imf = input->desc.memFormat;
    DataFormat omf = output->desc.memFormat;

    if (imf == DF_NCWHC4 && p.blockSize == 2) {
        U32 gs[3] = {ih, iw, (ic_str + 3) / 4};
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        Kernel kernel;
        char kernelname[128];
        if (omf == DF_NCHW) {
            sprintf(kernelname, "depth2space_ncwhc4_2x2_nchw");
        } else {
            sprintf(kernelname, "depth2space_ncwhc4_2x2");
        }

        CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, p.blockSize, ih_str, iwh_str, ic_str, ih_off,
            iw_off, oh_str, ow_str, owh_str, oh_off, ow_off, ih, iw, oc, inbuf, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
#endif
    } else {
        if (imf == DF_NCWHC4) {
            U32 gs0[3] = {ih, (iw + 3) / 4, (ic + 3) / 4};
            U32 ls0[3] = {0, 0, 0};
            U32 dim0 = 3;
            Kernel kernel0;
            CHECK_STATUS(gcl_create_kernel(handle, "mem_trans_ncwhc4_to_nchw", &kernel0));
            CHECK_STATUS(gcl_set_kernelArgs(kernel0, iw_str, ih_str, iw_off, ih_off, iw, ih, 0, 0,
                iw, ih, ic, iw, ih, ic, 0, 0, inbuf, tmp));
            gcl_set_kernelVec(handle, kernel0, dim0, gs0, ls0, "mem_trans_ncwhc4_to_nchw");
#ifdef _DEBUG
            CHECK_STATUS(
                gcl_run_kernel(handle, kernel0, dim0, gs0, ls0, "mem_trans_ncwhc4_to_nchw"));
#endif
            inbuf = tmp;
        }
        U32 gs[3] = {
            iw, ih, (ic / (p.blockSize * p.blockSize) + 3) / 4 * (p.blockSize * p.blockSize)};
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        Kernel kernel;
        CHECK_STATUS(gcl_create_kernel(handle, "depth2space_nchw", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, p.blockSize, iw_str, iwh_str, iw_off, ih_off,
            oh_str, owh_str, oh_off, ow_off, iw, ih, ic, inbuf, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, "depth2space_nchw");
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "depth2space_nchw"));
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
    if (idf == DF_NCHW && p.blockSize != 2) {
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
