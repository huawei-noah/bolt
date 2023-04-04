// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "tensor_computing.h"
#include "ut_util_ocl.h"

int matmulTest(int argc, char *argv[], DataType dt)
{
    U32 an, ac, ah, aw;
    U32 bn, bc, bh, bw;
    U32 transA, transB;

    ArchInfo archInfo;
    archInfo.arch = MALI;
    if (gcl_check_device_qualcomm(OCLContext::getInstance().handle.get())) {
        archInfo.arch = QUALCOMM;
    }

    an = 1;
    ac = 4;
    ah = 4;
    aw = 4;

    bn = 1;
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
    if (argc == 11) {
        an = atoi(argv[1]);
        ac = atoi(argv[2]);
        ah = atoi(argv[3]);
        aw = atoi(argv[4]);
        bn = atoi(argv[5]);
        bc = atoi(argv[6]);
        bh = atoi(argv[7]);
        bw = atoi(argv[8]);
        transA = atoi(argv[9]);
        transB = atoi(argv[10]);
    }
    bool transposeA = (transA) ? true : false;
    bool transposeB = (transB) ? true : false;
    TensorDesc matrixADesc, matrixBDesc, matrixCDesc;
    TensorDesc matrixCDesc_cpu;

    matrixADesc = tensor4df(dt, DF_NCHWC4, an, ac, ah, aw);
    matrixBDesc = tensor4df(dt, DF_NCHWC4, bn, bc, bh, bw);

    U8 *matrixA_cpu = ut_input_v(ac * ah * aw, dt, UT_INIT_RANDOM);
    U8 *matrixB_cpu = ut_input_v(bc * bh * bw, dt, UT_INIT_RANDOM);

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    MemoryType memType = OCLMem;
    if (archInfo.arch == QUALCOMM) {
        memType = OCLMemImg;
    }
    Tensor matrixATensor = Tensor(memType);
    Tensor matrixBTensor = Tensor(memType);
    Tensor matrixCTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    Tensor tmpTensorImgA = Tensor(OCLMemImg);
    Tensor tmpTensorImgB = Tensor(OCLMemImg);
    matrixATensor.resize(matrixADesc);
    matrixBTensor.resize(matrixBDesc);

    MaliPara maliPara;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)(CONVOLUTION_ALGORITHM_NULL);
    runInfo.best_h[0] = 1;
    runInfo.best_c[0] = 1;
    runInfo.best_k[0] = 1;
    maliPara.handle = handle;
    maliPara.forwardRunInfo = &runInfo;
    archInfo.archPara = &maliPara;

    CHECK_STATUS(matmul_infer_output_size(
        &matrixATensor, transposeA, &matrixBTensor, transposeB, &matrixCTensor, &archInfo));
    CHECK_STATUS(matmul_infer_forward_algorithm(
        matrixATensor, transposeA, matrixBTensor, transposeB, matrixCTensor, &archInfo));

    U32 maxBytes[7] = {0};
    CHECK_STATUS(matmul_infer_forward_tmp_bytes(
        matrixATensor, transposeA, matrixBTensor, transposeB, matrixCTensor, maxBytes, &archInfo));

    GCLMem_t matrixC = alloc(matrixCTensor);
    GCLMem_t matrixA = alloc(matrixATensor);
    GCLMem_t matrixB = alloc(matrixBTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, matrixA));
    CHECK_STATUS(gcl_fill_memory_zero(handle, matrixB));

    U32 tmpBytes;
    matrixCDesc = matrixCTensor.get_desc();
    tmpBytes = tensorNumBytes(matrixADesc);
    maxBytes[0] = (tmpBytes > maxBytes[0]) ? tmpBytes : maxBytes[0];
    tmpBytes = tensorNumBytes(matrixBDesc);
    maxBytes[0] = (tmpBytes > maxBytes[0]) ? tmpBytes : maxBytes[0];
    tmpBytes = tensorNumBytes(matrixCDesc);
    maxBytes[0] = (tmpBytes > maxBytes[0]) ? tmpBytes : maxBytes[0];

    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes[0]);
    alloc_img(tmpTensorImgA, maxBytes + 1);
    alloc_img(tmpTensorImgB, maxBytes + 4);
    CHECK_STATUS(ocl_set_input(handle, matrixA, matrixADesc, matrixA_cpu, tmpbuf, true));
    CHECK_STATUS(ocl_set_input(handle, matrixB, matrixBDesc, matrixB_cpu, tmpbuf, true));

    std::vector<Tensor> tmpTensors(3);
    tmpTensors[0] = tmpTensor;
    tmpTensors[1] = tmpTensorImgA;
    tmpTensors[2] = tmpTensorImgB;
    Tensor biasTensor = Tensor(OCLMem);
    CHECK_STATUS(matmul(matrixATensor, transposeA, matrixBTensor, transposeB, biasTensor,
        tmpTensors, matrixCTensor, &archInfo));

    for (U32 i = 0; i < UT_WARMUP; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
    }
        CHECK_STATUS(gcl_finish(handle));
    std::vector<U32> kernelIndex;
    for (U32 i = 0; i < handle->kernelVec->size(); i++) {
        kernelIndex.push_back(i);
    }
    CHECK_STATUS(gcl_run_kernelVec_select_ls(handle, kernelIndex));
    CHECK_STATUS(gcl_finish(handle));
    double time = 0;
#ifdef _DEBUG
    for (I32 i = 0; i < UT_LOOPS; i++) {
        CHECK_STATUS(gcl_run_kernelVec_timing(handle, 0, handle->kernelVec->size()));
        time += handle->t_execute * 0.001;
    }
#else
    double start = ut_time_ms();
    for (I32 i = 0; i < UT_LOOPS; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
        CHECK_STATUS(gcl_finish(handle));
    }
    double end = ut_time_ms();
    time = (end - start);
#endif
    time /= UT_LOOPS;

    U32 cc, ch, cw;
    tensorSelectGet(matrixCDesc, NULL, NULL, NULL, &cc, &ch, &cw);
    U8 *matrixC_gpu = ut_input_v(cc * ch * cw, dt, UT_INIT_RANDOM);
    CHECK_STATUS(ocl_get_output(handle, matrixC, matrixCDesc, matrixC_gpu, tmpbuf, true));
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u)+(%u %u %u)=(%u %u %u)", ac, ah, aw, bc, bh, bw, cc, ch, cw);
    sprintf(buffer, "%20s, %80s", "matmul", params);
    U32 k = 0;
    if (transA) {
        k = ah;
    } else {
        k = aw;
    }
    double ops = 2.0 * cc * cw * ch * k + cc * cw * ch;
    ut_log(dt, buffer, ops, time);

    matrixADesc.df = DF_NCHW;
    matrixBDesc.df = DF_NCHW;
    Tensor matrixATensorCpu;
    matrixATensorCpu.resize(matrixADesc);
    matrixATensorCpu.alloc();
    UNI_MEMCPY(get_ptr_from_tensor(matrixATensorCpu, CPU_GENERAL), matrixA_cpu,
        tensorNumBytes(matrixADesc));

    Tensor matrixBTensorCpu;
    matrixBTensorCpu.resize(matrixBDesc);
    matrixBTensorCpu.alloc();
    UNI_MEMCPY(get_ptr_from_tensor(matrixBTensorCpu, CPU_GENERAL), matrixB_cpu,
        tensorNumBytes(matrixBDesc));

    Tensor matrixCTensorCpu;
    CHECK_STATUS(matmul_infer_output_size(&matrixATensorCpu, transposeA, &matrixBTensorCpu,
        transposeB, &matrixCTensorCpu, &UT_SERIAL_ARCHINFO));
    matrixCTensorCpu.alloc();

    Tensor tmpTensorCpu;
    CHECK_STATUS(matmul_infer_forward_tmp_bytes(matrixATensorCpu, transposeA, matrixBTensorCpu,
        transposeB, matrixCTensorCpu, &tmpBytes, &UT_SERIAL_ARCHINFO));
    tmpTensorCpu.resize(tensor1d(DT_F16, tmpBytes / bytesOf(DT_F16)));
    tmpTensorCpu.alloc();
    std::vector<Tensor> tmpTensorsCpu(1, tmpTensorCpu);

    CHECK_STATUS(matmul(matrixATensorCpu, transposeA, matrixBTensorCpu, transposeB, biasTensor,
        tmpTensorsCpu, matrixCTensorCpu, &UT_SERIAL_ARCHINFO));
    ut_check_v(matrixC_gpu, get_ptr_from_tensor(matrixCTensorCpu, CPU_GENERAL), cc * ch * cw, dt, 0.3);

    CHECK_STATUS(gcl_finish(handle));
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    free(matrixA_cpu);
    free(matrixB_cpu);
    free(matrixC_gpu);
    return 0;
}

int main(int argc, char **argv)
{
    matmulTest(argc, argv, DT_F16);
    return 0;
}
