// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/gather_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/common_opt.h"

inline EE gather_checkpara_mali_fp16(
    TensorDesc inputDesc, TensorDesc indexDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (indexDesc.dt != DT_I32 && indexDesc.dt != DT_U32) {
        CHECK_STATUS(NOT_MATCH);
    }
    return SUCCESS;
}

inline bool needReshapeInput(TensorDesc inputDesc, GCLMemDesc gclmemInputDesc, int gatherMode)
{
    if (inputDesc.df == DF_NCHWC4 || gclmemInputDesc.memType != GCL_MEM_BUF) {
        return true;
    }
    if (gatherMode != 1) {
        if (tensorNumElements(inputDesc) != gclmemInputDesc.num) {
            return true;
        }
    }
    return false;
}

inline bool needReshapeOutput(TensorDesc outputDesc, GCLMemDesc gclmemOutputDesc, int gatherMode)
{
    if (gatherMode != 1) {
        if (tensorNumElements(outputDesc) != gclmemOutputDesc.num) {
            return true;
        }
    }
    return false;
}

inline EE reshapeInput(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t inputTran,
    GCLMem_t tmpBuf,
    U32 *tmpOff)
{
    DataType dt;
    DataFormat df;
    U32 in, ic, ih, iw;
    tensorSelectGet(inputDesc, &dt, &df, &in, &ic, &ih, &iw);
    for (U32 i = 4; i < inputDesc.nDims; i++) {
        in *= inputDesc.dims[i];
    }
    inputTran->desc = input->desc;
    MemTransFormType type = (df == DF_NCHWC4) ? NCHWC4_TO_NCHW : NCHW_TO_NCHW;
    U32 str[3] = {iw, ih, ic * in};
    U32 off[3] = {0, 0, 0};
    MemFlags flag = CL_MEM_READ_WRITE;
    CHECK_STATUS(
        gclmem_set_desc_padding(&(inputTran->desc), str, off, dt, DF_NCHW, GCL_MEM_BUF, flag));
    U32 size = tensorNumBytes(inputDesc);
    CHECK_STATUS(gcl_create_sub_buffer(size, tmpOff, tmpBuf, &(inputTran->mem)));
    CHECK_STATUS(ocl_data_trans_form(handle, input, inputTran, 0, 0, type));
    return SUCCESS;
}

inline EE buildOutput(
    TensorDesc outputDesc, GCLMem_t output, GCLMem_t outputTran, GCLMem_t tmpBuf, U32 *tmpOff)
{
    DataType dt;
    DataFormat df;
    U32 on, oc, oh, ow;
    tensorSelectGet(outputDesc, &dt, &df, &on, &oc, &oh, &ow);
    for (U32 i = 4; i < outputDesc.nDims; i++) {
        on *= outputDesc.dims[i];
    }
    outputTran->desc = output->desc;
    U32 str[3] = {ow, oh, oc * on};
    U32 off[3] = {0, 0, 0};
    MemFlags flag = CL_MEM_READ_WRITE;
    CHECK_STATUS(
        gclmem_set_desc_padding(&(outputTran->desc), str, off, dt, DF_NCHW, GCL_MEM_BUF, flag));
    U32 size = tensorNumBytes(outputDesc);
    CHECK_STATUS(gcl_create_sub_buffer(size, tmpOff, tmpBuf, &(outputTran->mem)));
    return SUCCESS;
}

inline EE gather_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc indexDesc,
    GCLMem_t index,
    GatherParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    DataType dt = inputDesc.dt;
    U32 nDims = inputDesc.nDims;
    U32 axis = (p.axis + nDims) % nDims;
    axis = nDims - 1 - axis;
    U32 axisBeforeLen = 1;
    for (U32 i = 0; i < axis; i++) {
        axisBeforeLen *= inputDesc.dims[i];
    }
    U32 axisAfterLen = 1;
    for (U32 i = axis + 1; i < nDims; i++) {
        axisAfterLen *= inputDesc.dims[i];
    }

    U32 inAxisLen = inputDesc.dims[axis];
    U32 outAxisLen = tensorNumElements(indexDesc);
    U32 index_w_str, index_h_str, index_w_off, index_h_off;
    CHECK_STATUS(gclmem_get_desc_padding(
        index->desc, &index_w_str, &index_h_str, NULL, &index_w_off, &index_h_off));
    U32 index_off = index_h_off * index_w_str + index_w_off;
    U32 index_w = indexDesc.dims[0];
    U32 index_h = (indexDesc.nDims > 1) ? indexDesc.dims[1] : 1;

    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    CHECK_STATUS(set_common_opt(dt, GCL_MEM_BUF, GCL_MEM_BUF, "gather", kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));

    U32 gs[3] = {axisBeforeLen, outAxisLen, axisAfterLen};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    CHECK_STATUS(
        gcl_set_kernelArgs(kernel, axisBeforeLen, inAxisLen, outAxisLen, index_w_str, index_h_str,
            index_off, index_w, index_h, gs[0], gs[1], input->mem, index->mem, output->mem));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

EE gather_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    GCLMemDesc gclmemInputDesc,
    TensorDesc indexDesc,
    GatherParamSpec p,
    TensorDesc outputDesc,
    GCLMemDesc gclmemOutputDesc,
    U32 *bytes)
{
    int gatherMode = getGatherMode(p);
    U32 size = 0;
    if (needReshapeInput(inputDesc, gclmemInputDesc, gatherMode)) {
        size += UNI_ALIGN(tensorNumBytes(inputDesc), BUFFER_ALIGN_BASE);
    }
    if (indexDesc.df == DF_NCHWC4) {
        size += UNI_ALIGN(tensorNumBytes(indexDesc), BUFFER_ALIGN_BASE);
    }
    if (needReshapeOutput(outputDesc, gclmemOutputDesc, gatherMode)) {
        size += UNI_ALIGN(tensorNumBytes(outputDesc), BUFFER_ALIGN_BASE);
    }
    *bytes = size;
    return SUCCESS;
}

EE gather_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc indexDesc,
    GCLMem_t index,
    GatherParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(gather_checkpara_mali_fp16(inputDesc, indexDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    GCLMem_t inputPtr = input;
    GCLMem_t indexPtr = index;
    GCLMem_t outputPtr = output;
    GCLMem tmpMem[3];
    U32 tmpOff = 0;
    int gatherMode = getGatherMode(p);
    if (needReshapeInput(inputDesc, input->desc, gatherMode)) {
        CHECK_STATUS(reshapeInput(handle, inputDesc, input, &(tmpMem[0]), tmpbuf, &tmpOff));
        inputPtr = &(tmpMem[0]);
    }
    if (indexDesc.df == DF_NCHWC4) {
        CHECK_STATUS(reshapeInput(handle, indexDesc, index, &(tmpMem[1]), tmpbuf, &tmpOff));
        indexPtr = &(tmpMem[1]);
    }
    if (needReshapeOutput(outputDesc, output->desc, gatherMode)) {
        CHECK_STATUS(buildOutput(outputDesc, output, &(tmpMem[2]), tmpbuf, &tmpOff));
        outputPtr = &(tmpMem[2]);
    }
    if (gatherMode == 0) {
    } else if (gatherMode == 1) {
    } else {
        CHECK_STATUS(gather_core_mali_fp16(
            handle, inputDesc, inputPtr, indexDesc, indexPtr, p, outputDesc, outputPtr));
    }
    if (needReshapeOutput(outputDesc, output->desc, gatherMode)) {
        CHECK_STATUS(ocl_data_trans_form(handle, outputPtr, output, 0, 0, NCHW_TO_NCHW));
    }
    return SUCCESS;
}
