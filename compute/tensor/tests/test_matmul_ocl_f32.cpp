// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <string.h>
#include "tensor_computing.h"
#include "ut_util.h"
#include "gcl.h"
#include "libkernelsource.h"
#include "../src/gpu/mali/cl/kernel_option/gemm_tn_opt.h"

std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
GCLHandle_t handle = handleSharedPtr.get();
Tensor matrixCTensorCpu;
Tensor matrixCTensor = Tensor(OCLMem);

#ifdef _USE_FP32
inline GCLMem_t alloc(Tensor tensor)
{
    auto mem = (OclMemory *)tensor.get_memory();
    mem->alloc();
    return (GCLMem_t)mem->get_ptr();
}

inline GCLMem_t alloc_map(Tensor tensor)
{
    auto mem = (OclMemory *)tensor.get_memory();
    mem->mapped_alloc();
    return (GCLMem_t)mem->get_ptr();
}

inline GCLMem_t alloc_bytes(Tensor tensor, U32 size)
{
    auto mem = (OclMemory *)tensor.get_memory();
    GCLMem_t ptr = NULL;
    if (size > 0) {
        mem->resize(tensor1d(DT_U8, size));
        mem->alloc();
        ptr = (GCLMem_t)mem->get_ptr();
    }
    return ptr;
}

inline GCLMem_t alloc_desc(Tensor tensor, GCLMemDesc desc)
{
    auto mem = (OclMemory *)tensor.get_memory();
    mem->padding(desc);
    mem->alloc();
    return (GCLMem_t)mem->get_ptr();
}

inline GCLMem_t alloc_map_desc(Tensor tensor, GCLMemDesc desc)
{
    auto mem = (OclMemory *)tensor.get_memory();
    mem->padding(desc);
    mem->mapped_alloc();
    return (GCLMem_t)mem->get_ptr();
}

inline U8 *matmulF32Cpu(TensorDesc matrixADesc,
    TensorDesc matrixBDesc,
    TensorDesc matrixCDesc,
    bool transposeA,
    bool transposeB,
    U8 *matrixA_cpu,
    U8 *matrixB_cpu,
    DataType dt)
{
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;
    Tensor matrixATensorCpu;
    matrixATensorCpu.resize(matrixADesc);
    matrixATensorCpu.alloc();
    memcpy(get_ptr_from_tensor(matrixATensorCpu, UT_ARCH), matrixA_cpu, tensorNumBytes(matrixADesc));

    Tensor matrixBTensorCpu;
    matrixBTensorCpu.resize(matrixBDesc);
    matrixBTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(matrixBTensorCpu, UT_ARCH), matrixB_cpu, tensorNumBytes(matrixBDesc));

    CHECK_STATUS(matmul_infer_output_size(&matrixATensorCpu, transposeA, &matrixBTensorCpu,
        transposeB, &matrixCTensorCpu, &archInfo_org));
    matrixCTensorCpu.alloc();

    Tensor tmpTensorCpu;
    U32 tmpBytes = 0;
    CHECK_STATUS(matmul_infer_forward_tmp_bytes(
        matrixATensorCpu, transposeA, matrixBTensorCpu, transposeB, &tmpBytes, &archInfo_org));
    tmpTensorCpu.resize(tensor1d(dt, tmpBytes / bytesOf(dt)));
    tmpTensorCpu.alloc();

    CHECK_STATUS(matmul(matrixATensorCpu, transposeA, matrixBTensorCpu, transposeB, tmpTensorCpu,
        matrixCTensorCpu, &archInfo_org));
    return (U8 *)get_ptr_from_tensor(matrixCTensorCpu, UT_ARCH);
}

inline U8 *matmulF32Gpu(TensorDesc matrixADesc,
    TensorDesc matrixBDesc,
    TensorDesc matrixCDesc,
    bool transposeA,
    bool transposeB,
    U8 *matrixACpu,
    U8 *matrixBCpu,
    DataType dt)
{
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    U8 *out_val = nullptr;
    if (!transposeA && !transposeB) {
        U32 batch = matrixADesc.dims[2];
        U32 k = matrixADesc.dims[0];
        U32 m = matrixADesc.dims[1];
        U32 n = matrixBDesc.dims[0];
        if (k != matrixBDesc.dims[1]) {
            CHECK_STATUS(NOT_MATCH);
        }
        if (batch != matrixBDesc.dims[2]) {
            CHECK_STATUS(NOT_MATCH);
        }
        /*(m k) * (k n) = (m n) */
        TensorDesc matrixCDesc = tensor4df(dt, DF_NCHW, 1, batch, m, n);

        Tensor matrixATensorOrg = Tensor(OCLMem);
        Tensor matrixBTensorOrg = Tensor(OCLMem);
        Tensor matrixATensor = Tensor(OCLMem);
        Tensor matrixBTensor = Tensor(OCLMem);
        Tensor tmpTensor = Tensor(OCLMem);
        matrixATensorOrg.resize(matrixADesc);
        matrixBTensorOrg.resize(matrixBDesc);
        matrixATensor.resize(matrixADesc);
        matrixBTensor.resize(matrixBDesc);
        matrixCTensor.resize(matrixCDesc);
        GCLMemDesc matrixAGclDescOrg = gclmem_build_desc();
        GCLMemDesc matrixBGclDescOrg = gclmem_build_desc();
        GCLMemDesc matrixAGclDesc = gclmem_build_desc();
        GCLMemDesc matrixBGclDesc = gclmem_build_desc();
        GCLMemDesc matrixCGclDesc = gclmem_build_desc();
        U32 m_align = m;
        U32 n_align = n;
        for (U32 i = 1; i <= 8; i++) {
            U32 j = (m + i - 1) / i * i;
            if (m_align < j) {
                m_align = j;
            }
            j = (n + i - 1) / i * i;
            if (n_align < j) {
                n_align = j;
            }
        }

        U32 stride[3] = {1, 1, 1};
        U32 offset[3] = {0, 0, 0};
        /*set gpu memory properties, and alloc gpu memory*/
        stride[0] = k;
        stride[1] = m;
        stride[2] = batch;
        gclmem_set_desc_padding(
            &matrixAGclDescOrg, stride, offset, DT_F32, DF_NCHW, GCL_MEM_BUF, CL_MEM_READ_WRITE);
        GCLMem_t matrixAOrg = alloc_desc(matrixATensorOrg, matrixAGclDescOrg);

        stride[0] = n;
        stride[1] = k;
        stride[2] = batch;
        gclmem_set_desc_padding(
            &matrixBGclDescOrg, stride, offset, DT_F32, DF_NCHW, GCL_MEM_BUF, CL_MEM_READ_WRITE);
        GCLMem_t matrixBOrg = alloc_desc(matrixBTensorOrg, matrixBGclDescOrg);

        stride[0] = m_align;
        stride[1] = k;
        stride[2] = batch;
        gclmem_set_desc_padding(
            &matrixAGclDesc, stride, offset, DT_F32, DF_NCHW, GCL_MEM_BUF, CL_MEM_READ_WRITE);
        GCLMem_t matrixA = alloc_desc(matrixATensor, matrixAGclDesc);

        stride[0] = n_align;
        stride[1] = k;
        stride[2] = batch;
        gclmem_set_desc_padding(
            &matrixBGclDesc, stride, offset, DT_F32, DF_NCHW, GCL_MEM_BUF, CL_MEM_READ_WRITE);
        GCLMem_t matrixB = alloc_desc(matrixBTensor, matrixBGclDesc);

        stride[0] = n;
        stride[1] = m;
        stride[2] = batch;
        gclmem_set_desc_padding(&matrixCGclDesc, stride, offset, DT_F32, DF_NCHW, GCL_MEM_BUF,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR);
        GCLMem_t matrixC = alloc_map_desc(matrixCTensor, matrixCGclDesc);

        GCLMem_t tmp = alloc_bytes(tmpTensor, matrixAGclDescOrg.byteSize);

        /*trans cpu org val to gpu*/
        U32 size = matrixAGclDescOrg.byteSize;
        CHECK_STATUS(
            gcl_trans_memory(handle, matrixACpu, matrixAOrg, &size, HOST_TO_DEVICE_BUF, CL_TRUE));
        size = matrixBGclDescOrg.byteSize;
        CHECK_STATUS(
            gcl_trans_memory(handle, matrixBCpu, matrixBOrg, &size, HOST_TO_DEVICE_BUF, CL_TRUE));
        gcl_finish(handle);

        std::vector<U32> kernelIndex;
        Kernel kernel;
        char kernelName[512];
        U32 gs[3] = {0, 0, 0};
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        /*transpose matrix A*/
        sprintf(kernelName, "transpose_nchw");
        gs[0] = (k + 3) / 4;
        gs[1] = m;
        gs[2] = batch;
        U32 dimTran[3] = {1, 0, 2};
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, k, m, 0, 0, m, k, 0, 0, dimTran[0], dimTran[1],
            dimTran[2], k, gs[0], gs[1], matrixAOrg->mem, tmp->mem));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);

        /*kernel local size auto tunning, fine best kernel ls val*/
        kernelIndex.push_back(kernelVec.size() - 1);
        CHECK_STATUS(gcl_run_kernelVec_select_ls(handle, kernelIndex));
#if defined _DEBUG
        CHECK_STATUS(gcl_run_kernelVec_timing(handle, kernelVec.size() - 1, kernelVec.size()));
        double transATime = handle->t_execute * 0.001;
        double paddingATime = 0;
        double paddingBTime = 0;
#endif

        /*padding matrix A and B*/
        Mem matrixAMem = matrixA->mem;
        Mem matrixBMem = matrixB->mem;
        sprintf(kernelName, "padding_nchw_constant");
        if (m_align != m) {
            gs[0] = (m_align + 3) / 4;
            gs[1] = k;
            gs[2] = batch;
            CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, m, k, 0, 0, m_align, k, 0, 0, m, k, m_align, k,
                0, 0, 0, 0, 0, 0, gs[0], gs[1], tmp->mem, matrixAMem));
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
            /*kernel local size auto tunning, fine best kernel ls val*/
            kernelIndex[0] = kernelVec.size() - 1;
            CHECK_STATUS(gcl_run_kernelVec_select_ls(handle, kernelIndex));
#if defined _DEBUG
            CHECK_STATUS(gcl_run_kernelVec_timing(handle, kernelVec.size() - 1, kernelVec.size()));
            paddingATime = handle->t_execute * 0.001;
#endif
        } else {
            matrixAMem = tmp->mem;
        }

        if (n_align != n) {
            gs[0] = (n_align + 3) / 4;
            gs[1] = k;
            gs[2] = batch;
            CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, n, k, 0, 0, n_align, k, 0, 0, n, k, n_align, k,
                0, 0, 0, 0, 0, 0, gs[0], gs[1], matrixBOrg->mem, matrixBMem));
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
            /*kernel local size auto tunning, fine best kernel ls val*/
            kernelIndex[0] = kernelVec.size() - 1;
            CHECK_STATUS(gcl_run_kernelVec_select_ls(handle, kernelIndex));
#if defined _DEBUG
            CHECK_STATUS(gcl_run_kernelVec_timing(handle, kernelVec.size() - 1, kernelVec.size()));
            paddingBTime = handle->t_execute * 0.001;
#endif
        } else {
            matrixBMem = matrixBOrg->mem;
        }

        /*TN GEMM*/
        /*item_m: calculate points number on matrix A for each thread*/
        /*item_n: calculate points number on matrix B for each thread*/
        /*each thread calculate item_m * item_n points on matrix C*/

        /*auto tunning to find best item_m & item_n*/
        std::vector<GCLKernelInfo> kernelVecTunning;
        handle->kernelVec = &kernelVecTunning;
        CHECK_STATUS(gcl_enable_queue_profiling(handle));
        U32 a_str = m_align * k;
        U32 b_str = n_align * k;
        U32 c_str = m * n;
        U32 a_off = 0;
        U32 b_off = 0;
        U32 c_off = 0;
        double minTime = DBL_MAX;
        U32 best_m = 0;
        U32 best_n = 0;
        KernelOpt kernelOpt;
        for (U32 item_m = 1; item_m <= 8; item_m++) {
            for (U32 item_n = 1; item_n <= 8; item_n++) {
                if (item_m * item_n == 1) {
                    continue;
                }
                CHECK_STATUS(set_gemm_tn_opt_mali(item_m, item_n, false, false, false,
                    ACTIVATION_NULL, DT_F32, kernelName, &kernelOpt));
                CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
                gs[0] = (n + item_n - 1) / item_n;
                gs[1] = (m + item_m - 1) / item_m;
                gs[2] = batch;
                CHECK_STATUS(gcl_set_kernelArgs(kernel, m_align, n_align, k, a_str, b_str, c_str,
                    a_off, b_off, c_off, n, n, m, batch, gs[0], gs[1], matrixAMem, matrixBMem,
                    matrixC->mem));
                gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
                U32 be = kernelVecTunning.size() - 1;
                U32 end = kernelVecTunning.size();
                gcl_run_kernelVec_timing(handle, be, end);
                if (minTime > handle->t_execute) {
                    minTime = handle->t_execute;
                    best_m = item_m;
                    best_n = item_n;
                }
            }
        }
        CHECK_STATUS(gcl_clean_kernelVec(handle));
        CHECK_STATUS(gcl_off_queue_profiling(handle));
        handle->kernelVec = &kernelVec;
        CHECK_STATUS(gcl_finish(handle));

        /*set best gemm config*/
        CHECK_STATUS(set_gemm_tn_opt_mali(
            best_m, best_n, false, false, false, ACTIVATION_NULL, DT_F32, kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        gs[0] = (n + best_n - 1) / best_n;
        gs[1] = (m + best_m - 1) / best_m;
        gs[2] = batch;
        CHECK_STATUS(gcl_set_kernelArgs(kernel, m_align, n_align, k, a_str, b_str, c_str, a_off,
            b_off, c_off, n, n, m, batch, gs[0], gs[1], matrixAMem, matrixBMem, matrixC->mem));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
        /*kernel local size auto tunning, fine best kernel ls val*/
        kernelIndex[0] = kernelVec.size() - 1;
        CHECK_STATUS(gcl_run_kernelVec_select_ls(handle, kernelIndex));
#if defined _DEBUG
        CHECK_STATUS(gcl_run_kernelVec_timing(handle, kernelVec.size() - 1, kernelVec.size()));
        double gemmTime = handle->t_execute * 0.001;
        double ops = 2.0 * batch * m * n * k + batch * m * n;
        double totalTime = transATime + paddingATime + paddingBTime + gemmTime;
        UNI_INFO_LOG("transATime   = %lf\n", transATime);
        UNI_INFO_LOG("paddingATime = %lf\n", paddingATime);
        UNI_INFO_LOG("paddingBTime = %lf\n", paddingBTime);
        UNI_INFO_LOG("gemmTime     = %lf\n", gemmTime);
        UNI_INFO_LOG("totalTime    = %lf\n", totalTime);
        char buffer[150];
        char params[120];
        sprintf(params, "(%u %u %u)+(%u %u %u)=(%u %u %u)", batch, m, k, batch, k, n, batch, m, n);
        sprintf(buffer, "%20s, %80s", "matmul", params);
        UNI_INFO_LOG("gflops with Total time:\n");
        ut_log(dt, buffer, ops, totalTime);
        UNI_INFO_LOG("gflops only with gemm time:\n");
        ut_log(dt, buffer, ops, gemmTime);
#else
        /*run all kernels*/
        CHECK_STATUS(gcl_run_kernelVec(handle));
#endif
        auto mem = (OclMemory *)matrixCTensor.get_memory();
        out_val = (U8 *)mem->get_mapped_ptr();
        CHECK_STATUS(gcl_finish(handle));
        CHECK_STATUS(gcl_clean_kernelVec(handle));
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return out_val;
};

int matmulF32Test(int argc, char *argv[], DataType dt)
{
    U32 ac, ah, aw;
    U32 bc, bh, bw;

    ArchInfo archInfo;
    archInfo.arch = MALI;

    ac = 4;
    ah = 4;
    aw = 4;

    bc = 4;
    bh = 4;
    bw = 4;

    if (argc == 7) {
        ac = atoi(argv[1]);
        ah = atoi(argv[2]);
        aw = atoi(argv[3]);
        bc = atoi(argv[4]);
        bh = atoi(argv[5]);
        bw = atoi(argv[6]);
    }
    bool transposeA = false;
    bool transposeB = false;

    TensorDesc matrixADesc, matrixBDesc, matrixCDesc;
    TensorDesc matrixCDesc_cpu;

    matrixADesc = tensor4df(dt, DF_NCHW, 1, ac, ah, aw);
    matrixBDesc = tensor4df(dt, DF_NCHW, 1, bc, bh, bw);

    U8 *matrixACpu = ut_input_v(ac * ah * aw, dt, UT_INIT_RANDOM);
    U8 *matrixBCpu = ut_input_v(bc * bh * bw, dt, UT_INIT_RANDOM);
    U8 *matrixC_gpu = NULL;

    U8 *gpu_res = matmulF32Gpu(
        matrixADesc, matrixBDesc, matrixCDesc, transposeA, transposeB, matrixACpu, matrixBCpu, dt);
    U8 *cpu_res = matmulF32Cpu(
        matrixADesc, matrixBDesc, matrixCDesc, transposeA, transposeB, matrixACpu, matrixBCpu, dt);
    ut_check_a(cpu_res, cpu_res, bw * ah * ac, dt);

    free(matrixACpu);
    free(matrixBCpu);
    return 0;
}
#endif

int main(int argc, char **argv)
{
#ifdef _USE_FP32
    matmulF32Test(argc, argv, DT_F32);
#endif
    return 0;
}
