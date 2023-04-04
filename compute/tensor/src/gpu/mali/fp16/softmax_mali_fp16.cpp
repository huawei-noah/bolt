// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/softmax_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/softmax_opt.h"

inline EE softmax_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

bool matchVecCase(GCLMemDesc desc)
{
    bool matchCase = false;
    U32 iw_str, ih_str, ic_str;
    CHECK_STATUS(gclmem_get_desc_padding(desc, &iw_str, &ih_str, &ic_str, NULL, NULL));
    if (iw_str == 1 && ih_str == 1) {
        matchCase = true;
    }
    if (ih_str == 1 && ic_str == 1 && desc.memFormat == DF_NCHW) {
        matchCase = true;
    }
    if (iw_str == 1 && ic_str == 1 && desc.memFormat == DF_NCHW && desc.memType == GCL_MEM_BUF) {
        matchCase = true;
    }
    return matchCase;
}
inline EE softmax_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t tmp,
    int axis,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    DataType idt;
    U32 iw, ih, ic, in;
    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw);
    U32 iw_str, ih_str, ic_str, iw_off, ih_off, ihw_str, i_off;
    U32 ow_str, oh_str, ow_off, oh_off, ohw_str, o_off;
    gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    ihw_str = ih_str * iw_str;
    ohw_str = oh_str * ow_str;
    i_off = ih_off * iw_str + iw_off;
    o_off = oh_off * ow_str + ow_off;

    U32 nDims = inputDesc.nDims;
    I32 axisTran = (axis + nDims) % nDims;
    axisTran = nDims - 1 - axisTran;
    cl_mem inbuf, outbuf;
    inbuf = input->mem;
    outbuf = output->mem;
    Kernel kernel;
    char kernelName[128];
    KernelOpt kernelOpt;
    U32 gs[2];
    U32 ls[2] = {0, 0};
    U32 dim = 2;
    U32 len = 0;
    bool useNchw = (input->desc.memFormat == DF_NCHWC4) ? false : true;
    bool matchCase = matchVecCase(input->desc);
    if (matchCase) {
        for (U32 i = 0; i < input->desc.nDims; i++) {
            if (input->desc.dims[i] != 1) {
                len = input->desc.dims[i];
                if (axisTran != (I32)i) {
                    //matchCase = false; //may cause err when inputDesc dims not reduce
                }
                if (len != tensorNumElements(inputDesc)) {
                    matchCase = false;
                }
                break;
            }
        }
    }
    if (matchCase) {
        CHECK_STATUS(set_softmax_vec_reduce_opt_mali(
            useNchw, idt, input->desc, output->desc, kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        U32 d4 = (len + 3) / 4;
        U32 e4 = ((len & 3) == 0) ? 4 : (len & 3);
        gs[0] = 16;
        if (d4 > 128) {
            gs[0] = 32;
        }
        if (d4 > 256) {
            gs[0] = 64;
        }
        if (d4 > 1024) {
            gs[0] = 128;
        }
        ls[0] = gs[0];
        dim = 1;
        CHECK_STATUS(gcl_set_kernelArgs(kernel, d4, e4, gs[0], inbuf, tmp->mem, outbuf));
        CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName));
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
        return SUCCESS;
    }

    if (axisTran != 0 && axisTran != 1 && axisTran != 2) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    CHECK_STATUS(set_softmax_opt_mali(
        axisTran, useNchw, idt, input->desc.memType, output->desc.memType, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    if (axisTran == 0) {
        if (useNchw) {
            gs[0] = ih;
            gs[1] = ic * in;
        } else {
            gs[0] = ih;
            gs[1] = ((ic + 3) >> 2) * in;
        }
    } else if (axisTran == 1) {
        if (useNchw) {
            gs[0] = (iw + 3) >> 2;
            gs[1] = ic * in;
        } else {
            gs[0] = iw;
            gs[1] = ((ic + 3) >> 2) * in;
        }
    } else if (axisTran == 2) {
        if (in > 1) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        if (useNchw) {
            gs[0] = (iw + 3) >> 2;
            gs[1] = ih;
        } else {
            gs[0] = iw;
            gs[1] = ih;
        }
    }
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, ow_str, oh_str, i_off, o_off, iw, ih,
        ic, gs[0], gs[1], inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

EE softmax_infer_forward_tmp_bytes_mali_fp16(
    TensorDesc inputDesc, GCLMemDesc gclmemInputDesc, int axis, U32 *bytes)
{
    U32 size = 0;
    bool matchCase = matchVecCase(gclmemInputDesc);
    U32 nDims = inputDesc.nDims;
    I32 axisTran = (axis + nDims) % nDims;
    axisTran = nDims - 1 - axisTran;
    if (matchCase) {
        for (U32 i = 0; i < nDims; i++) {
            if (inputDesc.dims[i] != 1) {
                U32 len = inputDesc.dims[i];
                if (axisTran != (I32)i) {
                    matchCase = false;
                }
                if (len != tensorNumElements(inputDesc)) {
                    matchCase = false;
                }
                break;
            }
        }
    }
    if (matchCase) {
        size = (128 + 2) * bytesOf(inputDesc.dt);
    }
    *bytes = size;
    return SUCCESS;
}

EE softmax_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t tmp,
    int axis,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(softmax_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(softmax_core_mali_fp16(handle, inputDesc, input, tmp, axis, outputDesc, output));
    return SUCCESS;
}
