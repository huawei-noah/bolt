// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/expand_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/expand_opt.h"

inline EE expand_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE expand_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ExpandParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    DataType idt;
    U32 iw, ih, ic, in, it;
    U32 ow, oh, oc, on, ot;
    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw, &it);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow, &ot);
    U32 iDims = inputDesc.nDims;
    U32 oDims = outputDesc.nDims;
    U32 iDim[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    U32 oDim[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    for (U32 i = 0; i < iDims; i++) {
        iDim[i] = inputDesc.dims[i];
    }
    for (U32 i = 0; i < oDims; i++) {
        oDim[i] = outputDesc.dims[i];
    }
    U32 iw_str, ih_str, ic_str, iw_off, ih_off;
    U32 ow_str, oh_str, oc_str, ow_off, oh_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off);
    DataFormat imf = input->desc.memFormat;
    GCLMemType inputMemType = input->desc.memType;
    bool use3dMode = (iDims == 5 && imf == DF_NCHWC4) ? true : false;
    cl_mem inbuf = input->mem;
    cl_mem outbuf = output->mem;
    U32 irDim = 1;
    U32 orDim = 1;
    for (U32 i = 2; i < iDims; i++) {
        irDim = irDim * iDim[i];
    }
    for (U32 i = 2; i < oDims; i++) {
        orDim = orDim * oDim[i];
    }
    if (imf == DF_NCHWC4) {
        GCLMem tMem;
        GCLMemDesc desc;
        desc = input->desc;
        U32 str[3] = {iDim[0], iDim[1], irDim};
        U32 off[3] = {0, 0, 0};
        MemFlags flag = CL_MEM_READ_WRITE;
        CHECK_STATUS(gclmem_set_desc_padding(&desc, str, off, idt, DF_NCHW, GCL_MEM_BUF, flag));
        tMem.desc = desc;
        tMem.mem = tmpbuf->mem;
        if (use3dMode) {
            CHECK_STATUS(ocl_data_trans_form_3d(handle, input, &tMem, 0, 0, NCHWC4_TO_NCHW));
        } else {
            CHECK_STATUS(ocl_data_trans_form(handle, input, &tMem, 0, 0, NCHWC4_TO_NCHW));
        }
        iw_str = iDim[0];
        ih_str = iDim[1];
        iw_off = 0;
        ih_off = 0;
        inbuf = tMem.mem;
        inputMemType = GCL_MEM_BUF;
    }
    char kernelName[128];
    Kernel kernel;
    KernelOpt kernelOpt;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    CHECK_STATUS(set_expand_opt_mali(
        oDims, idt, inputMemType, output->desc.memType, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    gs[0] = (oDim[0] + 3) / 4;
    gs[1] = oDim[1];
    gs[2] = orDim;
    if (oDims < 4) {
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str, oh_str,
            ow_off, oh_off, iDim[0], iDim[1], iDim[2], oDim[0], gs[0], gs[1], inbuf, outbuf));
    } else {
        switch (oDims) {
            case 4:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str,
                    oh_str, ow_off, oh_off, iDim[0], iDim[1], iDim[2], iDim[3], oDim[2], oDim[0],
                    gs[0], gs[1], inbuf, outbuf));
                break;
            case 5:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str,
                    oh_str, ow_off, oh_off, iDim[0], iDim[1], iDim[2], iDim[3], oDim[2], iDim[4],
                    oDim[3], oDim[0], gs[0], gs[1], inbuf, outbuf));
                break;
            case 6:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str,
                    oh_str, ow_off, oh_off, iDim[0], iDim[1], iDim[2], iDim[3], oDim[2], iDim[4],
                    oDim[3], iDim[5], oDim[4], oDim[0], gs[0], gs[1], inbuf, outbuf));
                break;
            case 7:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str,
                    oh_str, ow_off, oh_off, iDim[0], iDim[1], iDim[2], iDim[3], oDim[2], iDim[4],
                    oDim[3], iDim[5], oDim[4], iDim[6], oDim[5], oDim[0], gs[0], gs[1], inbuf,
                    outbuf));
                break;
            case 8:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str,
                    oh_str, ow_off, oh_off, iDim[0], iDim[1], iDim[2], iDim[3], oDim[2], iDim[4],
                    oDim[3], iDim[5], oDim[4], iDim[6], oDim[5], iDim[7], oDim[6], oDim[0], gs[0],
                    gs[1], inbuf, outbuf));
                break;
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

EE expand_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc outputDesc,
    GCLMemDesc gclmemInputDesc,
    GCLMemDesc gclmemOutputDesc,
    U32 *bytes)
{
    U32 size = 0;
    if (gclmemInputDesc.memFormat == DF_NCHWC4) {
        size = tensorNumBytes(inputDesc);
    }
    *bytes = size;
    return SUCCESS;
}

EE expand_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ExpandParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(expand_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(expand_core_mali_fp16(handle, inputDesc, input, p, tmpbuf, outputDesc, output));
    return SUCCESS;
}
