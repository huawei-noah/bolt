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
#include "types.h"
#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"

inline void check_tensordesc_dims(
    U32 sn, U32 sc, U32 sh, U32 sw, U32 dn, U32 dc, U32 dh, U32 dw, U32 srcOffset, U32 dstOffset, U32 length)
{
    U32 srcElementNum = sw * sh * sc * sn;
    U32 dstElementNum = dw * dh * dc * dn;
    if (sn > 1 || dn > 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (length + srcOffset > srcElementNum) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (length + dstOffset > dstElementNum) {
        CHECK_STATUS(NOT_MATCH);
    }
}

inline EE copy_checkpara_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    U32 srcOffset,
    U32 dstOffset,
    U32 length)
{
    if (handle == nullptr) {
        return NULL_POINTER;
    }
    if (inputDesc.size() != 2) {
        return NOT_SUPPORTED;
    }
    if (input.size() != 2 && input.size() != 4) {
        return NOT_SUPPORTED;
    }
    if (input[0] == nullptr || input[1] == nullptr) {
        return NOT_SUPPORTED;
    }
    GCLMem_t srcMem = (GCLMem_t)input[0];
    GCLMem_t dstMem = (GCLMem_t)input[1];
    U32 sn, sc, sh, sw, sw_off, sh_off;
    U32 dn, dc, dh, dw, dw_off, dh_off;
    sn = 1;
    dn = 1;
    get_gclmem_dim(srcMem->desc, &sw, &sh, &sc, &sw_off, &sh_off);
    get_gclmem_dim(dstMem->desc, &dw, &dh, &dc, &dw_off, &dh_off);
    if (sw_off != 0 || sh_off != 0 || dw_off != 0 || dh_off != 0) {
        return NOT_SUPPORTED;
    }
    check_tensordesc_dims(sn, sc, sh, sw, dn, dc, dh, dw, srcOffset, dstOffset, length);
    return SUCCESS;
}

inline EE copy_core_mali_fp16(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    U32 srcOffset,
    U32 dstOffset,
    U32 srcStride,
    U32 dstStride,
    U32 length)
{
    DataType sdt = inputDesc[0].dt;
    DataType ddt = inputDesc[1].dt;
    if (sdt == DT_U32 && ddt == DT_I32) {
        sdt = DT_I32;
    }
    cl_mem srcbuf = ((GCLMem_t)(input[0]))->mem;
    cl_mem dstbuf = ((GCLMem_t)(input[1]))->mem;
    cl_mem srcBlockIndex = NULL;
    cl_mem dstBlockIndex = NULL;
    bool useBlockIndex = false;
    if (input.size() == 4) {
        srcBlockIndex = ((GCLMem_t)(input[2]))->mem;
        dstBlockIndex = ((GCLMem_t)(input[3]))->mem;
        useBlockIndex = true;
    }
    U32 gs = (length + 3) / 4;
    U32 ls = 0;
    U32 dim = 1;
    Kernel kernel;
    char dataType[16];
    if (sdt == DT_I32) {
        strcpy(dataType, "i32");
    }
    if (sdt == DT_U32) {
        strcpy(dataType, "u32");
    }
    if (sdt == DT_F16) {
        strcpy(dataType, "f16");
    }
    char kernelName[128];
    if (!useBlockIndex) {
        sprintf(kernelName, "copy_%s", dataType);
    } else {
        sprintf(kernelName, "copy_with_block_index_%s", dataType);
    }

    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    if (!useBlockIndex) {
        CHECK_STATUS(
            gcl_set_kernelArgs(kernel, length, length, srcOffset, dstOffset, gs, srcbuf, dstbuf));
    } else {
        CHECK_STATUS(gcl_set_kernelArgs(kernel, length, length, srcOffset, dstOffset, srcStride,
            dstStride, gs, srcBlockIndex, dstBlockIndex, srcbuf, dstbuf));
    }
    gcl_set_kernelVec(handle, kernel, dim, &gs, &ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs, &ls, kernelName));
    CHECK_STATUS(gcl_print_memory<U8>(handle, (GCLMem_t)input[0], "copy_srcbuf"));
    CHECK_STATUS(gcl_print_memory<U8>(handle, (GCLMem_t)input[1], "copy_dstbuf"));
#endif
    return SUCCESS;
}

EE copy_infer_output_size_mali(std::vector<TensorDesc> inputDesc, GCLMemDesc_t gclmemInputDesc)
{
    if (inputDesc.size() != 2) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    DataType sdt, ddt;
    U32 sw, sh, sc, sn;
    U32 dw, dh, dc, dn;
    TensorDesc srcDesc = inputDesc[0];
    TensorDesc dstDesc = inputDesc[1];
    tensorSelectGet(srcDesc, &sdt, NULL, &sn, &sc, &sh, &sw);
    tensorSelectGet(dstDesc, &ddt, NULL, &dn, &dc, &dh, &dw);
    if (sdt == DT_U32 && ddt == DT_I32) {
        sdt = DT_I32;
    }
    if (sdt != DT_F16 && sdt != DT_I32 && sdt != DT_U32) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (sdt != ddt) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (gclmemInputDesc) {
        CHECK_STATUS(
            infer_gclmem_desc_nchw(sw, sh, sc, 0, 0, 0, 0, 0, sdt, sdt, &gclmemInputDesc[0], NULL));
        CHECK_STATUS(
            infer_gclmem_desc_nchw(dw, dh, dc, 0, 0, 0, 0, 0, ddt, ddt, &gclmemInputDesc[1], NULL));
    }
    return SUCCESS;
}

EE copy_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    U32 srcOffset,
    U32 dstOffset,
    U32 srcStride,
    U32 dstStride,
    U32 length)
{
    EE ret = SUCCESS;
    CHECK_STATUS(copy_checkpara_mali(handle, inputDesc, input, srcOffset, dstOffset, length));
    CHECK_STATUS(fill_output_zero(handle, (GCLMem_t)input[1], inputDesc[1]));
    CHECK_STATUS(copy_core_mali_fp16(
        handle, inputDesc, input, srcOffset, dstOffset, srcStride, dstStride, length));
    return ret;
}
