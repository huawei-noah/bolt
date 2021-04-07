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

#ifdef _USE_FP16
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
int matmulTest(int argc, char *argv[], DataType dt)
{
    U32 ac, ah, aw;
    U32 bc, bh, bw;
    U32 transA, transB;

    ArchInfo archInfo;
    archInfo.arch = MALI;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    ac = 4;
    ah = 4;
    aw = 4;

    bc = 4;
    bh = 4;
    bw = 4;

    transA = 1;
    transB = 0;

    if (argc == 9) {
        ac = atoi(argv[1]);
        ah = atoi(argv[2]);
        aw = atoi(argv[3]);
        bc = atoi(argv[4]);
        bh = atoi(argv[5]);
        bw = atoi(argv[6]);
        transA = atoi(argv[7]);
        transB = atoi(argv[8]);
    }
    bool transposeA = (transA) ? true : false;
    bool transposeB = (transB) ? true : false;
    TensorDesc matrixADesc, matrixBDesc, matrixCDesc;
    TensorDesc matrixCDesc_cpu;

    matrixADesc = tensor4df(dt, DF_NCHW, 1, ac, ah, aw);
    matrixBDesc = tensor4df(dt, DF_NCHW, 1, bc, bh, bw);

    U8 *matrixA_cpu = ut_input_v(ac * ah * aw, dt, UT_INIT_RANDOM);
    U8 *matrixB_cpu = ut_input_v(bc * bh * bw, dt, UT_INIT_RANDOM);
    U8 *matrixC_gpu = NULL;

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    Tensor matrixATensor = Tensor(OCLMem);
    Tensor matrixBTensor = Tensor(OCLMem);
    Tensor matrixCTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    matrixATensor.resize(matrixADesc);
    matrixBTensor.resize(matrixBDesc);

    MaliPara maliPara;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)(CONVOLUTION_ALGORITHM_NULL);
    runInfo.best_w[0] = 1;
    runInfo.best_c[0] = 1;
    runInfo.best_k[0] = 1;
    maliPara.handle = handle;
    maliPara.forwardRunInfo = &runInfo;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(matmul_infer_output_size(
        &matrixATensor, transposeA, &matrixBTensor, transposeB, &matrixCTensor, &archInfo));
    CHECK_STATUS(matmul_infer_forward_algorithm(
        matrixATensor, transposeA, matrixBTensor, transposeB, matrixCTensor, &archInfo));

    U32 maxBytes = 0;
    U32 tmpBytes;
    CHECK_STATUS(matmul_infer_forward_tmp_bytes(
        matrixATensor, transposeA, matrixBTensor, transposeB, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    GCLMem_t matrixC = alloc_map(matrixCTensor);
    GCLMem_t matrixA = alloc(matrixATensor);
    GCLMem_t matrixB = alloc(matrixBTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, matrixA));
    CHECK_STATUS(gcl_fill_memory_zero(handle, matrixB));

    tmpBytes = tensorNumBytes(matrixADesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(matrixBDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);
    CHECK_STATUS(ocl_set_input(handle, matrixA, matrixADesc, matrixA_cpu, tmpbuf, true));
    CHECK_STATUS(ocl_set_input(handle, matrixB, matrixBDesc, matrixB_cpu, tmpbuf, true));

    CHECK_STATUS(matmul(
        matrixATensor, transposeA, matrixBTensor, transposeB, tmpTensor, matrixCTensor, &archInfo));

    /*warp up*/
    UNI_INFO_LOG("warm up gpu:\n")
    for (U32 i = 0; i < 2; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
    }

#ifdef _DEBUG
    std::vector<U32> kernelIndex;
    for (U32 i = 0; i < handle->kernelVec->size(); i++) {
        kernelIndex.push_back(i);
    }
    CHECK_STATUS(gcl_run_kernelVec_select_ls(handle, kernelIndex));
    CHECK_STATUS(gcl_finish(handle));
    double time = 0;
    double min_time = DBL_MAX;
    double max_time = 0;
    U32 loop = 16;
    for (U32 i = 0; i < loop; i++) {
        CHECK_STATUS(gcl_run_kernelVec_timing(handle, 0, handle->kernelVec->size()));
        double t = handle->t_execute * 0.001;
        if (t < min_time)
            min_time = t;
        if (t > max_time)
            max_time = t;
        time += t;
    }
    time = (time - min_time - max_time) / (loop - 2);
    UNI_INFO_LOG("min_time = %lf\n", min_time);
    UNI_INFO_LOG("max_time = %lf\n", max_time);
    UNI_INFO_LOG("avg_time = %lf\n", time);
    time = min_time;
#else
    CHECK_STATUS(gcl_run_kernelVec(handle));
#endif
    matrixCDesc = matrixCTensor.get_desc();
    CHECK_STATUS(ocl_get_output(handle, matrixC, matrixCDesc, true));
    matrixC_gpu = matrixC->mapPtrArray.back();
    U32 cc, ch, cw;
    tensorSelectGet(matrixCDesc, NULL, NULL, NULL, &cc, &ch, &cw);
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u)+(%u %u %u)=(%u %u %u)", ac, ah, aw, bc, bh, bw, cc, ch, cw);
    sprintf(buffer, "%20s, %80s", "matmul", params);
#ifdef _DEBUG
    U32 k = 0;
    if (transA) {
        k = ah;
    } else {
        k = aw;
    }
    double ops = 2.0 * cc * cw * ch * k + cc * cw * ch;
    ut_log(dt, buffer, ops, time);
#endif
    Tensor matrixATensorCpu;
    matrixATensorCpu.resize(matrixADesc);
    matrixATensorCpu.alloc();
    memcpy(get_ptr_from_tensor(matrixATensorCpu, UT_ARCH), matrixA_cpu, tensorNumBytes(matrixADesc));

    Tensor matrixBTensorCpu;
    matrixBTensorCpu.resize(matrixBDesc);
    matrixBTensorCpu.alloc();
    memcpy(get_ptr_from_tensor(matrixBTensorCpu, UT_ARCH), matrixB_cpu, tensorNumBytes(matrixBDesc));

    Tensor matrixCTensorCpu;
    CHECK_STATUS(matmul_infer_output_size(&matrixATensorCpu, transposeA, &matrixBTensorCpu,
        transposeB, &matrixCTensorCpu, &archInfo_org));
    matrixCTensorCpu.alloc();

    Tensor tmpTensorCpu;
    CHECK_STATUS(matmul_infer_forward_tmp_bytes(
        matrixATensorCpu, transposeA, matrixBTensorCpu, transposeB, &tmpBytes, &archInfo_org));
    tmpTensorCpu.resize(tensor1d(DT_F16, tmpBytes / bytesOf(DT_F16)));
    tmpTensorCpu.alloc();

    CHECK_STATUS(matmul(matrixATensorCpu, transposeA, matrixBTensorCpu, transposeB, tmpTensorCpu,
        matrixCTensorCpu, &archInfo_org));
    ut_check_a(matrixC_gpu, get_ptr_from_tensor(matrixCTensorCpu, UT_ARCH), cc * ch * cw, dt);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(matrixA_cpu);
    free(matrixB_cpu);
    return 0;
}
#endif

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    matmulTest(argc, argv, DT_F16);
#endif
    return 0;
}
