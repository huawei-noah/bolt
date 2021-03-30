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

EE check_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputADesc,
    GCLMemDesc_t gclmemInputBDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    /*tensorDesc record cpu org data format info*/
    /*gclmemDesc record gpu trans data format info*/
    if (outputDesc) {
        (*outputDesc).dt = DT_I32;
        (*outputDesc).nDims = 1;
        (*outputDesc).dims[0] = inputDesc.dims[inputDesc.nDims - 1];
    }
    DataType idt = inputDesc.dt;
    U32 ndims = inputDesc.nDims;
    U32 iw = inputDesc.dims[0];
    U32 ih = (ndims > 1) ? inputDesc.dims[1] : 1;
    U32 ic = (ndims > 2) ? inputDesc.dims[2] : 1;
    U32 in = (ndims > 3) ? inputDesc.dims[3] : 1;
    if (in > 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    CHECK_STATUS(infer_gclmem_desc_nchw(
        iw, ih, ic, 0, 0, 1, 1, 1, idt, DT_I32, gclmemInputADesc, gclmemOutputDesc));
    CHECK_STATUS(
        infer_gclmem_desc_nchw(iw, ih, ic, 0, 0, 0, 0, 0, idt, idt, gclmemInputBDesc, NULL));
    return SUCCESS;
}

inline EE check_checkpara_mali(GCLHandle_t handle,
    TensorDesc inputDescA,
    GCLMem_t inputA,
    TensorDesc inputDescB,
    GCLMem_t inputB,
    CheckParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (handle == nullptr || inputA == nullptr || inputB == nullptr || output == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputA->desc.memFormat != output->desc.memFormat ||
        inputB->desc.memFormat != output->desc.memFormat || inputA->desc.memFormat != DF_NCHW) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (inputDescA.dt == DT_I32 || inputDescA.dt == DT_U32) {
        if (inputDescB.dt != DT_I32 && inputDescB.dt != DT_U32) {
            CHECK_STATUS(NOT_MATCH);
        }
    }
    if (outputDesc.dt != DT_I32) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (p.check_mode != CHECK_EQUAL) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return SUCCESS;
}

inline EE check_core_mali(GCLHandle_t handle,
    TensorDesc inputDescA,
    GCLMem_t inputA,
    TensorDesc inputDescB,
    GCLMem_t inputB,
    CheckParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    U32 ndims = inputDescA.nDims;
    U32 iw = inputDescA.dims[0];
    U32 ih = (ndims > 1) ? inputDescA.dims[1] : 1;
    U32 ic = (ndims > 2) ? inputDescA.dims[2] : 1;
    if (iw == 1 && ih == 1 && ic == 1) {
        U32 aw_str, ah_str, aw_off, ah_off;
        U32 bw_str, bh_str, bw_off, bh_off;
        U32 ow_str, oh_str, ow_off, oh_off;
        get_gclmem_dim(inputA->desc, &aw_str, &ah_str, NULL, &aw_off, &ah_off);
        get_gclmem_dim(inputB->desc, &bw_str, &bh_str, NULL, &bw_off, &bh_off);
        get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
        U32 gs = 1;
        U32 ls = 0;
        U32 dim = 1;
        Kernel kernel;
        CHECK_STATUS(gcl_create_kernel(handle, "check_int_spe", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(
            kernel, aw_off, bw_off, ow_off, gs, inputA->mem, inputB->mem, output->mem));
        gcl_set_kernelVec(handle, kernel, dim, &gs, &ls, "check_int_spe");
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs, &ls, "check_int_spe"));
        CHECK_STATUS(gcl_print_memory<I32>(handle, inputA, "clip_inputA"));
        CHECK_STATUS(gcl_print_memory<I32>(handle, inputB, "clip_inputB"));
        CHECK_STATUS(gcl_print_memory<I32>(handle, output, "clip_output"));
#endif
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE check_mali(GCLHandle_t handle,
    TensorDesc inputDescA,
    GCLMem_t inputA,
    TensorDesc inputDescB,
    GCLMem_t inputB,
    CheckParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(
        check_checkpara_mali(handle, inputDescA, inputA, inputDescB, inputB, p, outputDesc, output));
    DataType dt = inputDescA.dt;
    if (dt == DT_U32) {
        dt = DT_I32;
    }
    switch (dt) {
        case DT_F16: {
            ret = NOT_SUPPORTED;
            break;
        }
        case DT_I8: {
            ret = NOT_SUPPORTED;
            break;
        }
        case DT_I32: {
            ret = check_core_mali(
                handle, inputDescA, inputA, inputDescB, inputB, p, outputDesc, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
