// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "sys.h"
#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/cl/kernel_option/space2depth_opt.h"

inline EE space2depth_checkpara_mali(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    if (handle == nullptr || nullptr == input || nullptr == output) {
        return NULL_POINTER;
    }
    if (inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE space2depth_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    Space2DepthParamSpec space2DepthPara,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    UNUSED(outputDesc);
    DataType idt;
    U32 iw, ih, ic, in;
    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw);
    U32 iw_str, ih_str, iw_off, ih_off, i_off;
    U32 ow_str, oh_str, ow_off, oh_off, o_off;
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    i_off = ih_off * iw_str + iw_off;
    o_off = oh_off * ow_str + ow_off;
    cl_mem inbuf, outbuf;
    inbuf = input->mem;
    outbuf = output->mem;
    bool useNchw = (inputDesc.df == DF_NCHWC4) ? false : true;
    U32 blockSize = space2DepthPara.block_size;

    U32 gs[3] = {iw, ih, (ic + 3) / 4};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    if (useNchw) {
        gs[2] = ic;
    }
    if (in > 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    Kernel kernel;
    KernelOpt kernelOpt;
    char kernelName[128];
    CHECK_STATUS(
        set_space2depth_opt(useNchw, idt, GCL_MEM_BUF, GCL_MEM_BUF, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(
        kernel, iw_str, ih_str, ow_str, oh_str, i_off, o_off, gs[0], gs[1], inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

EE space2depth_padding_input_mali(TensorDesc inputDesc,
    Space2DepthParamSpec space2DepthPara,
    TensorDesc *outputDesc,
    OclMemory *inputMem,
    OclMemory *outputMem)
{
    if (inputMem == nullptr || outputMem == nullptr || outputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    U32 blockSize = space2DepthPara.block_size;
    DataType idt;
    DataFormat idf;
    U32 iw, ih, ic, in;
    tensorSelectGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw);
    if (iw % blockSize != 0) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (ih % blockSize != 0) {
        CHECK_STATUS(NOT_MATCH);
    }
    U32 on = in;
    U32 oc = ic * blockSize * blockSize;
    U32 oh = ih / blockSize;
    U32 ow = iw / blockSize;
    *outputDesc = tensor4df(idt, DF_NCHW, on, oc, oh, ow);
    return SUCCESS;
}

EE space2depth_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    Space2DepthParamSpec space2DepthPara,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(space2depth_checkpara_mali(handle, inputDesc, input, outputDesc, output));
    return space2depth_core_mali_fp16(handle, inputDesc, input, space2DepthPara, outputDesc, output);
}
