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

namespace {
constexpr int SOFTMAX_KERNEL_ITEM_NUM = 16;
constexpr int SOFTMAX_KERNEL_TMPBUF_EXPAND = 2;
}  // namespace

inline EE softmax_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    if (outputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE softmax_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t tmp,
    int axis,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    UNUSED(outputDesc);
    U32 iw, ih, ic, in;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
    U32 iw_str, ih_str, ic_str, iw_off, ih_off, ihw_str;
    U32 ow_str, oh_str, ow_off, oh_off, ohw_str;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    ihw_str = ih_str * iw_str;
    ohw_str = oh_str * ow_str;
    U32 nDims = inputDesc.nDims;
    I32 axisTran = (axis + nDims) % nDims;
    axisTran = nDims - 1 - axisTran;
    cl_mem inbuf, outbuf;
    inbuf = input->mem;
    outbuf = output->mem;
    Kernel kernel;
    char kernelname[128];
    U32 gs[2];
    U32 ls[2] = {0, 0};
    U32 dim = 2;
    if (iw_off == 0 && ih_off == 0) {
        bool matchCase = false;
        I32 icd4;
        I32 ice4;
        if (iw_str == 1 && ih_str == 1) {
            icd4 = (ic + 3) >> 2;
            ice4 = ((ic & 3) == 0) ? 4 : (ic & 3);
            matchCase = true;
        }
        if (iw_str == 1 && ic_str == 1) {
            icd4 = (ih + 3) >> 2;
            ice4 = ((ih & 3) == 0) ? 4 : (ih & 3);
            matchCase = true;
        }
        if (ih_str == 1 && ic_str == 1) {
            icd4 = (iw + 3) >> 2;
            ice4 = ((iw & 3) == 0) ? 4 : (iw & 3);
            matchCase = true;
        }

        if (matchCase) {
            gs[0] = SOFTMAX_KERNEL_ITEM_NUM;
            dim = 1;
            Mem clTmpBuf = tmp->mem;
            sprintf(kernelname, "softmax_h1w1_max_part");
            CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
            CHECK_STATUS(
                gcl_set_kernelArgs(kernel, icd4, ice4, SOFTMAX_KERNEL_ITEM_NUM, inbuf, clTmpBuf));
            CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname));
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
#endif
            dim = 1;
            gs[0] = 1;
            sprintf(kernelname, "softmax_h1w1_max_all");
            CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, SOFTMAX_KERNEL_ITEM_NUM, clTmpBuf));
            CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname));
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
#endif
            dim = 1;
            gs[0] = SOFTMAX_KERNEL_ITEM_NUM;
            sprintf(kernelname, "softmax_h1w1_sum_part");
            CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
            CHECK_STATUS(
                gcl_set_kernelArgs(kernel, icd4, ice4, SOFTMAX_KERNEL_ITEM_NUM, inbuf, clTmpBuf));
            CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname));
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
#endif
            dim = 1;
            gs[0] = 1;
            sprintf(kernelname, "softmax_h1w1_sum_all");
            CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, SOFTMAX_KERNEL_ITEM_NUM, clTmpBuf));
            CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname));
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
#endif
            dim = 1;
            gs[0] = icd4;
            sprintf(kernelname, "softmax_h1w1_output");
            CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(
                kernel, icd4, ice4, SOFTMAX_KERNEL_ITEM_NUM, inbuf, clTmpBuf, outbuf));
            CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname));
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
#endif
            return SUCCESS;
        }
    }

    if (input->desc.memFormat == DF_NCWHC4) {
        if ((nDims == 4 && axisTran == 1) || (inputDesc.df == DF_MTK && axisTran == 0) ||
            (inputDesc.df == DF_MKT && axisTran == 1)) {
            gs[0] = ih;
            gs[1] = iw;
            I32 icd4 = (ic + 3) >> 2;
            I32 ice4 = ((ic & 3) == 0) ? 4 : (ic & 3);
            sprintf(kernelname, "softmax");
            CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, icd4, ice4, ih_str, ihw_str, ih_off, iw_off,
                oh_str, ohw_str, oh_off, ow_off, gs[0], gs[1], inbuf, outbuf));
        } else {
            return NOT_SUPPORTED;
        }
    } else if (input->desc.memFormat == DF_NCHW) {
        if (axisTran == 2) {  // on c axis
            gs[0] = (iw + 3) / 4;
            gs[1] = ih;
            sprintf(kernelname, "softmax_nchw_c");
            CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, ic, iw_str, ihw_str, iw_off, ih_off, ow_str,
                ohw_str, ow_off, oh_off, iw, gs[0], gs[1], inbuf, outbuf));
        } else if (axisTran == 0) {  // on w axis
            gs[0] = ih;
            gs[1] = ic;
            I32 iwd4 = (iw + 3) >> 2;
            I32 iwe4 = ((iw & 3) == 0) ? 4 : (iw & 3);
            sprintf(kernelname, "softmax_nchw_w");
            CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, iwd4, iwe4, iw_str, ih_str, iw_off, ih_off,
                ow_str, oh_str, ow_off, oh_off, gs[0], gs[1], inbuf, outbuf));
        } else {
            return NOT_SUPPORTED;
        }
    } else {
        return NOT_SUPPORTED;
    }
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
#endif
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

EE softmax_infer_forward_tmp_bytes_mali_fp16(
    TensorDesc inputDesc, U32 *bytes, ForwardRunInfoMali_t forwardRunInfo)
{
    UNUSED(forwardRunInfo);
    U32 in, ic, ih, iw;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
    if (ih != 1 || iw != 1 || in != 1) {
        *bytes = 0;
    } else {
        *bytes = SOFTMAX_KERNEL_ITEM_NUM + SOFTMAX_KERNEL_TMPBUF_EXPAND;
    }

    return SUCCESS;
}
