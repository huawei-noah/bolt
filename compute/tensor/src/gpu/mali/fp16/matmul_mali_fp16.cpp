// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/matmul_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/gemm_tn_opt.h"

inline EE matmul_checkpara_mali_fp16(
    TensorDesc matrixADesc, TensorDesc matrixBDesc, TensorDesc matrixCDesc)
{
    if (matrixADesc.dt != matrixBDesc.dt || matrixADesc.dt != matrixCDesc.dt ||
        matrixADesc.dt != DT_F16) {
        return NOT_MATCH;
    }
    return SUCCESS;
}

inline EE transpose_matrix(GCLHandle_t handle,
    U32 iw_str,
    U32 ih_str,
    U32 iw_off,
    U32 ih_off,
    U32 ow_str,
    U32 oh_str,
    U32 ow_off,
    U32 oh_off,
    U32 w,
    U32 h,
    U32 c,
    Mem inbuf,
    Mem outbuf)
{
    U32 gs[3] = {(w + 3) / 4, h, c};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    U32 dimTran[3] = {1, 0, 2};
    Kernel kernel;
    char kernelName[128];
    sprintf(kernelName, "transpose_nchw");
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str, oh_str, ow_off,
        oh_off, dimTran[0], dimTran[1], dimTran[2], w, gs[0], gs[1], inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

inline EE matmul_core_mali_fp16(GCLHandle_t handle,
    TensorDesc matrixADesc,
    bool transposeA,
    const GCLMem_t matrixA,
    TensorDesc matrixBDesc,
    bool transposeB,
    const GCLMem_t matrixB,
    GCLMem_t tmp,
    TensorDesc matrixCDesc,
    GCLMem_t matrixC,
    ForwardRunInfoMali_t forwardRunInfo)
{
    DataType dt;
    U32 ac, ah, aw;
    U32 bc, bh, bw;
    U32 cc, ch, cw;
    U32 aw_str, ah_str, aw_off, ah_off;
    U32 bw_str, bh_str, bw_off, bh_off;
    U32 cw_str, ch_str, cw_off, ch_off;
    CHECK_STATUS(gclmem_get_desc_dim(matrixA->desc, &dt, NULL, NULL, &ac, &ah, &aw));
    CHECK_STATUS(gclmem_get_desc_dim(matrixB->desc, NULL, NULL, NULL, &bc, &bh, &bw));
    CHECK_STATUS(gclmem_get_desc_dim(matrixC->desc, NULL, NULL, NULL, &cc, &ch, &cw));
    CHECK_STATUS(gclmem_get_desc_padding(matrixA->desc, &aw_str, &ah_str, NULL, &aw_off, &ah_off));
    CHECK_STATUS(gclmem_get_desc_padding(matrixB->desc, &bw_str, &bh_str, NULL, &bw_off, &bh_off));
    CHECK_STATUS(gclmem_get_desc_padding(matrixC->desc, &cw_str, &ch_str, NULL, &cw_off, &ch_off));

    U32 M, N, K;
    U32 A_str, B_str, C_str;
    U32 A_off, B_off, C_off;
    U32 item_a = forwardRunInfo->best_k[0];
    U32 item_b = forwardRunInfo->best_w[0];
    Mem A = matrixA->mem;
    Mem B = matrixB->mem;
    Mem C = matrixC->mem;
    Mem tmpbuf = tmp->mem;
    U32 offset = 0;
    A_off = 0;
    B_off = 0;
    C_off = ch_off * cw_str + cw_off;
    C_str = cw_str * ch_str;

    if (!transposeA) {
        Mem subA;
        M = ALIGN(ah, item_a);
        K = aw;
        U32 size_A = M * K * ac * bytesOf(dt);
        CHECK_STATUS(gcl_create_sub_buffer(size_A, &offset, tmp, &subA));
        CHECK_STATUS(transpose_matrix(
            handle, aw_str, ah_str, aw_off, ah_off, M, K, 0, 0, aw, ah, ac, A, subA));
        A_str = M * K;
        A = subA;
    } else {
        A_off = ah_off * aw_str + aw_off;
        A_str = aw_str * ah_str;
        M = aw_str;
        K = ah;
    }

    if (transposeB) {
        Mem subB;
        N = ALIGN(bh, item_b);
        U32 size_B = N * K * bc * bytesOf(dt);
        CHECK_STATUS(gcl_create_sub_buffer(size_B, &offset, tmp, &subB));
        CHECK_STATUS(transpose_matrix(
            handle, bw_str, bh_str, bw_off, bh_off, N, K, 0, 0, bw, bh, bc, B, subB));
        B_str = N * K;
        B = subB;
    } else {
        B_off = bh_off * bw_str + bw_off;
        B_str = bw_str * bh_str;
        N = bw_str;
    }

    char kernelName[128];
    Kernel kernel;
    KernelOpt kernelOpt;
    U32 gs[3] = {(cw + item_b - 1) / item_b, (ch + item_a - 1) / item_a, cc};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;

    CHECK_STATUS(set_gemm_tn_opt_mali(
        item_a, item_b, false, false, false, ACTIVATION_NULL, DT_F16, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, M, N, K, A_str, B_str, C_str, A_off, B_off, C_off,
        cw_str, cw, ch, cc, gs[0], gs[1], A, B, C));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

EE matmul_infer_forward_tmp_bytes_mali_fp16(TensorDesc matrixADesc,
    bool transposeA,
    TensorDesc matrixBDesc,
    bool transposeB,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    U32 item_a = forwardRunInfo->best_k[0];
    U32 item_b = forwardRunInfo->best_w[0];
    DataType dt;
    U32 ac, ah, aw;
    U32 bc, bh, bw;
    CHECK_STATUS(tensorSelectGet(matrixADesc, &dt, NULL, NULL, &ac, &ah, &aw));
    CHECK_STATUS(tensorSelectGet(matrixBDesc, NULL, NULL, NULL, &bc, &bh, &bw));
    U32 size = 0;
    if (!transposeA) {
        size += ALIGN(ah, item_a) * aw * ac * bytesOf(dt);
        size = ALIGN(size, 1024);
    }

    if (transposeB) {
        size += ALIGN(bh, item_b) * bw * bc * bytesOf(dt);
    }
    *bytes = size;
    return SUCCESS;
}

EE matmul_mali_fp16(GCLHandle_t handle,
    TensorDesc matrixADesc,
    bool transposeA,
    const GCLMem_t matrixA,
    TensorDesc matrixBDesc,
    bool transposeB,
    const GCLMem_t matrixB,
    GCLMem_t tmp,
    TensorDesc matrixCDesc,
    GCLMem_t matrixC,
    ForwardRunInfoMali_t forwardRunInfo)
{
    CHECK_STATUS(matmul_checkpara_mali_fp16(matrixADesc, matrixBDesc, matrixCDesc));
    CHECK_STATUS(fill_output_zero(handle, matrixC, matrixCDesc));
    CHECK_STATUS(matmul_core_mali_fp16(handle, matrixADesc, transposeA, matrixA, matrixBDesc,
        transposeB, matrixB, tmp, matrixCDesc, matrixC, forwardRunInfo));
    return SUCCESS;
}
