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
#include "error.h"
#include "types.h"
#include "gpu/mali/fp16/matmul_mali_fp16.h"

inline EE matmul_checkpara_mali_fp16(
    TensorDesc matrixADesc, TensorDesc matrixBDesc, TensorDesc matrixCDesc)
{
    if (matrixADesc.dt != matrixBDesc.dt || matrixADesc.dt != matrixCDesc.dt ||
        matrixADesc.dt != DT_F16) {
        return NOT_MATCH;
    }
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
    UNUSED(tmp);
    UNUSED(matrixCDesc);
    U32 adims = matrixADesc.nDims;
    U32 ac = (adims > 2) ? matrixADesc.dims[2] : 1;
    U32 ah = matrixADesc.dims[1];
    U32 aw = matrixADesc.dims[0];
    U32 bh = matrixBDesc.dims[1];
    U32 bw = matrixBDesc.dims[0];

    U32 item_w = forwardRunInfo->best_w[0];
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    cl_mem A, B, C;
    A = matrixA->mem;
    B = matrixB->mem;
    C = matrixC->mem;
    char kernelname[128];
    Kernel kernel;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    if (matrixA->desc.offset[0] != 0 || matrixA->desc.offset[1] != 0 ||
        matrixB->desc.offset[0] != 0 || matrixB->desc.offset[1] != 0 ||
        matrixC->desc.offset[0] != 0 || matrixC->desc.offset[1] != 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (transposeA && !transposeB) {
        U32 M = matrixA->desc.stride[0];
        U32 N = matrixB->desc.stride[0];
        U32 K = ah;
        U32 ow_str = matrixC->desc.stride[0];
        U32 A_str = M * matrixA->desc.stride[1];
        U32 B_str = N * matrixB->desc.stride[1];
        U32 C_str = ow_str * matrixC->desc.stride[1];
        U32 batch = ac;
        gs[0] = (bw + item_w - 1) / item_w;
        gs[1] = (aw + item_k - 1) / item_k;
        gs[2] = batch;
        sprintf(kernelname, "gemm_tn_nobias_%d%d", item_k, item_w);
        CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, M, N, K, ow_str, A_str, B_str, C_str, 0, 0, bw, aw,
            gs[0], gs[1], 0, 0, A, B, C));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
        CHECK_STATUS(gcl_print_memory<F16>(handle, matrixA, "gemm_tn_a"));
        CHECK_STATUS(gcl_print_memory<F16>(handle, matrixB, "gemm_tn_b"));
        CHECK_STATUS(gcl_print_memory<F16>(handle, matrixC, "gemm_tn_c"));
        handle->t_total += handle->t_execute;
#endif
        return SUCCESS;
    }

    if (!transposeA && transposeB) {
        U32 KA = matrixA->desc.stride[0];
        U32 KB = matrixB->desc.stride[0];
        U32 K = (aw + item_c - 1) / item_c * item_c;
        U32 ow_str = matrixC->desc.stride[0];
        U32 A_str = KA * matrixA->desc.stride[1];
        U32 B_str = KB * matrixB->desc.stride[1];
        U32 C_str = ow_str * matrixC->desc.stride[1];
        U32 batch = ac;
        gs[0] = (bh + item_w - 1) / item_w;
        gs[1] = (ah + item_k - 1) / item_k;
        gs[2] = batch;
        sprintf(kernelname, "gemm_nt_nobias_%d%d%d", item_k, item_w, (item_c >> 1));
        CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(
            kernel, KA, KB, K, ow_str, A_str, B_str, C_str, 0, 0, bh, ah, gs[0], gs[1], A, B, C));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
        CHECK_STATUS(gcl_print_memory<F16>(handle, matrixA, "gemm_nt_a"));
        CHECK_STATUS(gcl_print_memory<F16>(handle, matrixB, "gemm_nt_b"));
        CHECK_STATUS(gcl_print_memory<F16>(handle, matrixC, "gemm_nt_c"));
        handle->t_total += handle->t_execute;
#endif
        return SUCCESS;
    }

    if (transposeA && transposeB) {
        if (matrixADesc.df != DF_MKT) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        U32 m, k, t;
        get_nlp_mkt_val(matrixADesc, NULL, &m, &k, &t);
        if (t != 1) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        if (m != 1) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        if (aw != 1) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        if (ah != k) {
            CHECK_STATUS(NOT_MATCH);
        }
        U32 KA = matrixA->desc.stride[2] * 4;
        U32 KB = matrixB->desc.stride[0];
        U32 K = (ah + item_c - 1) / item_c * item_c;
        U32 ow_str = matrixC->desc.stride[0];
        U32 batch = 1;
        gs[0] = (bh + item_w - 1) / item_w;
        gs[1] = 1;
        gs[2] = batch;
        sprintf(kernelname, "gemm_nt_nobias_%d%d%d", item_k, item_w, (item_c >> 1));
        CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(
            kernel, KA, KB, K, ow_str, 0, 0, 0, 0, 0, bh, 1, gs[0], gs[1], A, B, C));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
        CHECK_STATUS(gcl_print_memory<F16>(handle, matrixA, "gemm_nt_a"));
        CHECK_STATUS(gcl_print_memory<F16>(handle, matrixB, "gemm_nt_b"));
        CHECK_STATUS(gcl_print_memory<F16>(handle, matrixC, "gemm_nt_c"));
        handle->t_total += handle->t_execute;
#endif
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE matmul_infer_forward_tmp_bytes_mali_fp16(TensorDesc matrixADesc,
    bool transposeA,
    TensorDesc matrixBDesc,
    bool transposeB,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    UNUSED(matrixADesc);
    UNUSED(transposeA);
    UNUSED(matrixBDesc);
    UNUSED(transposeB);
    UNUSED(forwardRunInfo);
    *bytes = 0;
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
