// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/fully_connected_mali_fp16.h"
#include "gpu/mali/fp16/gemv_mali_fp16.h"
#include "gpu/mali/fp16/matmul_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/conv_direct_opt.h"
#include "gpu/mali/cl/kernel_option/gemm_tn_opt.h"
#include "gpu/mali/cl/kernel_option/transpose_opt.h"

inline EE fully_connected_checkpara_mali_fp16(
    TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != filterDesc.dt) {
        return NOT_MATCH;
    }
    return SUCCESS;
}

inline EE fully_connected_gemv_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc biasDesc,
    GCLMem_t bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    U32 tmpOff = 0;
    CHECK_STATUS(gemv(handle, inputDesc, outputDesc, {}, false, &tmpOff, tmpBuf, input,
        bias, filter, output, forwardRunInfo));
    return SUCCESS;
}

inline bool gemm_need_reshape_input(GCLMemDesc desc)
{
    bool needReshape = false;
    if (desc.memFormat == DF_NCHWC4) {
        needReshape = true;
    } else {
        U32 iw_str, ih_str;
        gclmem_get_desc_padding(desc, &iw_str, &ih_str, NULL, NULL, NULL);
        if (iw_str != desc.dims[0] || ih_str != desc.dims[1] || desc.memType != GCL_MEM_BUF) {
            for (U32 i = 2; i < desc.nDims; i++) {
                if (desc.dims[i] > 1) {
                    needReshape = true;
                }
            }
        }
    }
    return needReshape;
}

inline EE gemm_reshape_input(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t inputTran,
    U32 *tmpOff,
    GCLMem_t tmp)
{
    MemTransFormType type = (input->desc.memFormat == DF_NCHW) ? NCHW_TO_NCHW : NCHWC4_TO_NCHW;
    DataType dt = inputDesc.dt;
    U32 dim[3] = {1, 1, 1};
    for (U32 i = 0; i < inputDesc.nDims; i++) {
        if (i < 2) {
            dim[i] = inputDesc.dims[i];
        } else {
            dim[2] = dim[2] * inputDesc.dims[i];
        }
    }
    inputTran->desc = input->desc;
    inputTran->desc.df = DF_NCHW;
    U32 str[3] = {dim[0], dim[1], dim[2]};
    U32 off[3] = {0, 0, 0};
    MemFlags flag = CL_MEM_READ_WRITE;
    CHECK_STATUS(gclmem_set_desc_padding(
        &(inputTran->desc), str, off, input->desc.dt, DF_NCHW, GCL_MEM_BUF, flag));
    CHECK_STATUS(gcl_create_sub_buffer(inputTran->desc.byteSize, tmpOff, tmp, &inputTran->mem));
    CHECK_STATUS(ocl_data_trans_form(handle, input, inputTran, 0, 0, type));
    return SUCCESS;
}

inline EE gemm_trans_fc_input(GCLHandle_t handle,
    TensorDesc inputDesc,
    U32 item,
    GCLMem_t input,
    GCLMem_t inputTran,
    U32 *tmpOff,
    GCLMem_t tmpBuf,
    GCLMem_t tmpImg)
{
    GCLMem_t inputPtr = input;
    GCLMem curMem;
    OclMemory tmpOclMem;
    OclMemoryImg tmpOclImg;
    bool needReshapeDesc = false;
    for (U32 i = 2; i < inputDesc.nDims; i++) {
        if (inputDesc.dims[i] > 1) {
            needReshapeDesc = true;
        }
    }
    if (needReshapeDesc) {
        for (U32 i = 2; i < inputDesc.nDims; i++) {
            inputDesc.dims[1] *= inputDesc.dims[i];
            inputDesc.dims[i] = 1;
        }
        tmpOclMem.resize(inputDesc);
        curMem.desc = tmpOclMem.get_desc();
        curMem.mem = input->mem;
        inputPtr = &curMem;
    }

    DataType dt = inputDesc.dt;
    U32 w = inputDesc.dims[0];
    U32 h = inputDesc.dims[1];
    U32 c = 1;

    inputDesc.dims[0] = h;
    inputDesc.dims[1] = w;
    OclMemory *memPtr = (tmpImg) ? (OclMemory *)(&tmpOclImg) : &tmpOclMem;
    memPtr->resize(inputDesc);
    U32 pr = UNI_ALIGN(h, item) - h;
    memPtr->padding(0, pr, 0, 0);
    inputTran->desc = memPtr->get_desc();
    if (tmpImg) {
        inputTran->mem = tmpImg->mem;
    } else {
        CHECK_STATUS(
            gcl_create_sub_buffer(inputTran->desc.byteSize, tmpOff, tmpBuf, &inputTran->mem));
    }
    CHECK_STATUS(trans_matmul_input(handle, inputPtr, inputTran, dt, w, h, c));
    return SUCCESS;
}

inline EE fully_connected_gemm_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc biasDesc,
    GCLMem_t bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    GCLMem_t tmpImg,
    TensorDesc outputDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    U32 M, N, K;
    cl_mem matrixA, matrixB, biasbuf, matrixC, tmp;
    matrixB = filter->mem;
    matrixC = output->mem;
    tmp = tmpBuf->mem;
    OclGemmBiasMode biasMode = (bias) ? USE_BIAS_MATCH_B : NO_BIAS;
    biasbuf = (biasMode == USE_BIAS_MATCH_B) ? bias->mem : tmp;
    U32 item_n = forwardRunInfo->best_h[0];
    U32 item_m = forwardRunInfo->best_k[0];
    GCLMem_t inputPtr = input;
    GCLMem inputTran[2];
    U32 tmpOff = 0;
    if (gemm_need_reshape_input(input->desc)) {
        CHECK_STATUS(gemm_reshape_input(handle, inputDesc, inputPtr, inputTran, &tmpOff, tmpBuf));
        inputPtr = inputTran;
    }
    CHECK_STATUS(gemm_trans_fc_input(
        handle, inputDesc, item_m, inputPtr, inputTran + 1, &tmpOff, tmpBuf, tmpImg));
    matrixA = inputTran[1].mem;
    M = 1;
    for (U32 i = 1; i < inputDesc.nDims; i++) {
        M *= inputDesc.dims[i];
    }
    if (M != outputDesc.dims[1]) {
        CHECK_STATUS(NOT_MATCH);
    }
    M = UNI_ALIGN(M, item_m);
    K = inputDesc.dims[0];
    N = UNI_ALIGN(outputDesc.dims[0], item_n);

    DataType odt;
    U32 ow, oh, oc;
    U32 oh_str, ow_str, oh_off, ow_off;
    CHECK_STATUS(gclmem_get_desc_dim(output->desc, &odt, NULL, NULL, &oc, &oh, &ow));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));

    U32 A_str = M * K;
    U32 B_str = N * K;
    U32 C_str = ow_str * oh_str;
    U32 A_off = 0;
    U32 B_off = 0;
    U32 C_off = oh_off * ow_str + ow_off;

    U32 gs[3] = {(ow + item_n - 1) / item_n, (oh + item_m - 1) / item_m, 1};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    char kernelName[128];
    Kernel kernel;
    KernelOpt kernelOpt;
    GCLMemType amt = inputTran[1].desc.memType;
    GCLMemType bmt = filter->desc.memType;
    CHECK_STATUS(set_gemm_tn_opt_mali(item_m, item_n, biasMode, false, {}, odt, amt,
        bmt, output->desc.memType, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, M, N, K, A_str, B_str, C_str, A_off, B_off, C_off,
        ow_str, ow, oh, oc, gs[0], gs[1], matrixA, matrixB, biasbuf, matrixC));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
    return SUCCESS;
}

inline TensorDesc gemm_transform_filter_desc(
    TensorDesc filterDesc, U32 item_h, U32 item_c, U32 item_k)
{
    U32 fc = filterDesc.dims[filterDesc.nDims - 2];
    U32 fn = filterDesc.dims[filterDesc.nDims - 1];
    TensorDesc desc;
    desc.df = DF_NCHW;
    desc.dt = filterDesc.dt;
    desc.nDims = 4;
    desc.dims[3] = 1;
    desc.dims[2] = 1;
    desc.dims[1] = fc;
    desc.dims[0] = UNI_ALIGN(fn, item_h);
    return desc;
}

inline TensorDesc transform_filter_desc(TensorDesc filterDesc, U32 item_h, U32 item_c, U32 item_k)
{
    if (item_k == 0) {  //spe direct for gemv
        return gemv_transform_filter_desc(filterDesc, item_h, item_c, item_k);
    } else {
        return gemm_transform_filter_desc(filterDesc, item_h, item_c, item_k);
    }
}

EE fully_connected_transform_filter_bytes_mali_fp16(
    TensorDesc filterDesc, ForwardRunInfoMali_t forwardRunInfo, TensorDesc *ftmDesc)
{
    U32 item_h = forwardRunInfo->best_h[0];
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    *ftmDesc = transform_filter_desc(filterDesc, item_h, item_c, item_k);
    return SUCCESS;
}

EE fully_connected_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem,
    ForwardRunInfoMali_t forwardRunInfo)
{
    U32 item_h = forwardRunInfo->best_h[0];
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    if (item_k == 0) {
        CHECK_STATUS(gemv_transform_filter_mali_fp16(
            handle, filterDesc, filter, fltmemDesc, fltmem, forwardRunInfo));
    } else {
        U32 w = filterDesc.dims[0];
        U32 h = filterDesc.dims[1];
        U32 c = 1;
        CHECK_STATUS(trans_matmul_input(handle, filter, fltmem, filterDesc.dt, w, h, c, false));
        *fltmemDesc = transform_filter_desc(filterDesc, item_h, item_c, item_k);
    }
    return SUCCESS;
}

EE fully_connected_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    GCLMemDesc gclmemInputDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    U32 item_k = forwardRunInfo->best_k[0];
    if (item_k == 0) {
        CHECK_STATUS(
            gemv_infer_forward_tmp_bytes_mali_fp16(inputDesc, outputDesc, bytes, forwardRunInfo));
    } else {
        U32 size = 0;
        bool useImg = check_qualcomm_device();
        if (gemm_need_reshape_input(gclmemInputDesc)) {
            size += UNI_ALIGN(tensorNumBytes(inputDesc), BUFFER_ALIGN_BASE);
        }
        DataType dt = inputDesc.dt;
        U32 iw = inputDesc.dims[0];
        U32 ih = 1;
        for (U32 i = 1; i < inputDesc.nDims; i++) {
            ih = ih * inputDesc.dims[i];
        }
        if (useImg) {
            U32 width = (ih + 3) / 4;
            U32 height = iw;
            U32 depth = 1;
            if (CHECK_MEET_IMAGE_LIMITS(width, height, depth)) {
                bytes[1] = width;
                bytes[2] = height;
                bytes[3] = depth;
            } else {
                useImg = false;
            }
        }
        if (!useImg) {
            size += UNI_ALIGN(ih, item_k) * iw * bytesOf(dt);
        }
        bytes[0] = size;
    }
    return SUCCESS;
}

EE fully_connected_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc biasDesc,
    GCLMem_t bias,
    U32 tmpBytes,
    std::vector<GCLMem_t> tmp,
    TensorDesc outputDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    CHECK_STATUS(fully_connected_checkpara_mali_fp16(inputDesc, filterDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    U32 row = outputDesc.dims[1];
    if (row == 1) {
        CHECK_STATUS(fully_connected_gemv_mali_fp16(handle, inputDesc, input, filterDesc, filter,
            biasDesc, bias, tmpBytes, tmp[0], outputDesc, output, forwardRunInfo));
    } else {
        CHECK_STATUS(fully_connected_gemm_mali_fp16(handle, inputDesc, input, filterDesc, filter,
            biasDesc, bias, tmpBytes, tmp[0], tmp[1], outputDesc, output, forwardRunInfo));
    }
    return SUCCESS;
}
