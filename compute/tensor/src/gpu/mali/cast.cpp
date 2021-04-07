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

inline EE cast_checkpara_mali(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    if (handle == nullptr || input == nullptr || output == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.nDims != outputDesc.nDims) {
        CHECK_STATUS(NOT_MATCH);
    }
    for (U32 i = 0; i < inputDesc.nDims; i++) {
        if (inputDesc.dims[i] != outputDesc.dims[i]) {
            CHECK_STATUS(NOT_MATCH);
        }
    }
    if (input->desc.memFormat != DF_NCHW && input->desc.memFormat != DF_NCWHC4) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (input->desc.memFormat != output->desc.memFormat) {
        CHECK_STATUS(NOT_MATCH);
    }
    return SUCCESS;
}

inline void set_dt_name(TensorDesc desc, char *name)
{
    DataType dt = desc.dt;
    if (dt == DT_F16) {
        strcpy(name, "f16");
    } else if (dt == DT_I32) {
        strcpy(name, "i32");
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
}
inline EE cast_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    CastParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    UNUSED(p);
    U32 n, c, h, w;
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    CHECK_STATUS(gclmem_get_desc_dim(output->desc, NULL, NULL, &n, &c, &h, &w));
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    char kernelName[128];
    char idtName[16];
    char odtName[16];
    char formatName[16];
    set_dt_name(inputDesc, idtName);
    set_dt_name(outputDesc, odtName);
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Kernel kernel;

    if (input->desc.memFormat == DF_NCHW) {
        gs[0] = (w + 3) / 4;
        gs[1] = h;
        gs[2] = n * c;
        strcpy(formatName, "_nchw");
    } else {
        gs[0] = h;
        gs[1] = w;
        gs[2] = (c + 3) / 4 * n;
        strcpy(formatName, "");
    }
    sprintf(kernelName, "cast_%s_to_%s%s", idtName, odtName, formatName);
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, w, iw_str, ih_str, iw_off, ih_off, ow_str, oh_str,
        ow_off, oh_off, gs[0], gs[1], input->mem, output->mem))
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

EE cast_infer_output_size_mali(TensorDesc inputDesc,
    CastParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    if (gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr || outputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    (*outputDesc) = inputDesc;
    if (p.targetDt == DT_I32) {
        (*outputDesc).dt = DT_I32;
    } else if (p.targetDt == DT_F16) {
        (*outputDesc).dt = DT_F16;
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    DataType dt;
    U32 w, h, c, n;
    tensorSelectGet(inputDesc, &dt, NULL, &n, &c, &h, &w);
    if (gclmemInputDesc->memFormat == DF_NCHW || gclmemInputDesc->byteSize == 0) {
        CHECK_STATUS(infer_gclmem_desc_nchw(
            w, h, c, 0, 0, w, h, c, dt, dt, gclmemInputDesc, gclmemOutputDesc));
    } else {
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(
            w, h, c, 0, 0, w, h, c, dt, dt, gclmemInputDesc, gclmemOutputDesc));
    }
    return SUCCESS;
}

EE cast_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    CastParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(cast_checkpara_mali(handle, inputDesc, input, outputDesc, output));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(cast_core_mali_fp16(handle, inputDesc, input, p, outputDesc, output));
    return ret;
}
