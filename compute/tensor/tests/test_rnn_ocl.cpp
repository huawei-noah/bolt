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

int rnncellTest(int argc, char **argv, DataType dt, RNNMode mode)
{
    U32 xDim, hDim, numProjection;
    xDim = atoi(argv[1]);
    hDim = atoi(argv[2]);
    if (argc == 4) {
        numProjection = atoi(argv[3]);
    } else {
        numProjection = 0;
    }
    ArchInfo archInfo;
    archInfo.arch = MALI;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    RNNParamSpec rnnParamSpec;
    rnnParamSpec.mode = RNN_LSTM;
    rnnParamSpec.numOutput = hDim;
    rnnParamSpec.numProjection = numProjection;
    rnnParamSpec.forgetBias = 1.0;
    rnnParamSpec.zoneoutCell = 0;
    rnnParamSpec.zoneoutOutput = 0;
    rnnParamSpec.steps = -1;
    rnnParamSpec.biDirection = false;
    rnnParamSpec.activationMode = ACTIVATION_TANH;

    U32 col = (numProjection > 0) ? numProjection : hDim;
    TensorDesc inputDesc = tensor2df(dt, DF_NORMAL, 1, xDim);
    TensorDesc stateDesc = tensor2df(dt, DF_NORMAL, 1, col + hDim);

    std::vector<TensorDesc> biasDesc(2);
    std::vector<TensorDesc> filterDesc(2);
    filterDesc[0] = tensor2df(dt, DF_NK, 4 * col, xDim + hDim);
    filterDesc[1] = tensor2df(dt, DF_NK, hDim, numProjection);
    biasDesc[0] = tensor1d(dt, 4 * col);
    biasDesc[1] = tensor1d(dt, hDim);

    Tensor inputTensorCpu;
    inputTensorCpu.resize(inputDesc);
    inputTensorCpu.alloc();
    Tensor stateTensorCpu;
    stateTensorCpu.resize(stateDesc);
    stateTensorCpu.alloc();

    std::vector<Tensor> filterTensorCpu(2);
    std::vector<Tensor> biasTensorCpu(2);
    std::vector<Tensor> ftmTensorCpu(2);
    for (U32 i = 0; i < 2; i++) {
        filterTensorCpu[i].resize(filterDesc[i]);
        filterTensorCpu[i].alloc();
        biasTensorCpu[i].resize(biasDesc[i]);
        biasTensorCpu[i].alloc();
    }
    Tensor outputTensorCpu;
    U32 inputLen = tensorNumElements(inputDesc);
    U32 stateLen = tensorNumElements(stateDesc);
    U8 *input_cpu = ut_input_v(inputLen, dt, UT_INIT_RANDOM);
    memcpy(get_ptr_from_tensor(inputTensorCpu, UT_ARCH), input_cpu, inputLen * bytesOf(dt));

    U8 *state_cpu = ut_input_v(stateLen, dt, UT_INIT_RANDOM);
    memcpy(get_ptr_from_tensor(stateTensorCpu, UT_ARCH), state_cpu, stateLen * bytesOf(dt));
    U8 *state_gpu_host = ut_input_v(stateLen, dt, UT_INIT_ZERO);

    std::vector<U8 *> bias_cpu(2);
    std::vector<U8 *> filter_cpu(2);
    for (U32 i = 0; i < 2; i++) {
        U32 len = tensorNumElements(biasDesc[i]);
        bias_cpu[i] = ut_input_v(len, dt, UT_INIT_RANDOM);
        memcpy(get_ptr_from_tensor(biasTensorCpu[i], UT_ARCH), bias_cpu[i], len * bytesOf(dt));

        len = tensorNumElements(filterDesc[i]);
        filter_cpu[i] = ut_input_v(len, dt, UT_INIT_RANDOM);
        memcpy(get_ptr_from_tensor(filterTensorCpu[i], UT_ARCH), filter_cpu[i], len * bytesOf(dt));
    }

    std::vector<Tensor *> inputTensorPtrCpu(2);
    inputTensorPtrCpu[0] = &inputTensorCpu;
    inputTensorPtrCpu[1] = &stateTensorCpu;
    CHECK_STATUS(
        rnncell_infer_output_size(inputTensorPtrCpu, rnnParamSpec, &outputTensorCpu, &archInfo_org));
    outputTensorCpu.alloc();

    U32 tmpBytes;
    CHECK_STATUS(rnncell_infer_forward_tmp_bytes(inputTensorCpu, filterTensorCpu[0],
        outputTensorCpu, rnnParamSpec, &tmpBytes, &archInfo_org));
    Tensor tmpTensorCpu;
    TensorDesc tmpDesc = tensor1d(DT_U8, tmpBytes);
    tmpTensorCpu.resize(tmpDesc);
    tmpTensorCpu.alloc();
    memset(get_ptr_from_tensor(tmpTensorCpu, UT_ARCH), 0, tmpBytes);

    std::vector<U32> ftmBytes(2);
    CHECK_STATUS(
        rnn_transform_filter_bytes(filterTensorCpu, rnnParamSpec, ftmBytes.data(), &archInfo_org));

    std::vector<Tensor *> ftmTensorPtrCpu(2);
    for (U32 i = 0; i < 2; i++) {
        tmpDesc = tensor1d(DT_U8, ftmBytes[i]);
        ftmTensorCpu[i].resize(tmpDesc);
        ftmTensorCpu[i].alloc();
        ftmTensorPtrCpu[i] = &ftmTensorCpu[i];
    }

    CHECK_STATUS(
        rnn_transform_filter(filterTensorCpu, rnnParamSpec, ftmTensorPtrCpu, &archInfo_org));

    CHECK_STATUS(rnncell(inputTensorCpu, ftmTensorCpu, biasTensorCpu, stateTensorCpu, rnnParamSpec,
        xDim, col, 0, tmpTensorCpu, outputTensorCpu, &archInfo_org));

    /*************GPU*************/
    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;

    Tensor inputTensor = Tensor(OCLMem);
    inputTensor.resize(inputDesc);
    inputTensor.alloc();
    Tensor stateTensor = Tensor(OCLMem);
    stateTensor.resize(stateDesc);
    stateTensor.alloc();

    std::vector<Tensor> filterTensor(2);
    std::vector<Tensor> biasTensor(2);
    std::vector<Tensor> ftmTensor(2);
    for (U32 i = 0; i < 2; i++) {
        filterTensor[i] = Tensor(OCLMem);
        filterTensor[i].resize(filterDesc[i]);
        biasTensor[i] = Tensor(OCLMem);
        biasTensor[i].resize(biasDesc[i]);
        ftmTensor[i] = Tensor(OCLMem);
    }
    Tensor outputTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    U32 stride[3] = {0, 0, 0};
    U32 offset[3] = {0, 0, 0};
    GCLMemDesc desc = gcl_mem_desc(stride, offset, DT_U8, DF_NCHW);
    ocl_set_desc(&inputTensor, desc);
    ocl_set_desc(&stateTensor, desc);
    MaliPara maliPara;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)(CONVOLUTION_ALGORITHM_NULL);
    maliPara.handle = handle;
    maliPara.forwardRunInfo = &runInfo;
    archInfo.archPara = &maliPara;

    std::vector<Tensor *> inputTensorPtr(2);
    inputTensorPtr[0] = &inputTensor;
    inputTensorPtr[1] = &stateTensor;
    CHECK_STATUS(rnncell_infer_output_size(inputTensorPtr, rnnParamSpec, &outputTensor, &archInfo));
    CHECK_STATUS(rnncell_infer_forward_algorithm(inputTensor, filterTensor[0], biasTensor[0],
        stateTensor, rnnParamSpec, xDim, col, outputTensor, &archInfo));

    U32 maxBytes = 0;
    CHECK_STATUS(rnncell_infer_forward_tmp_bytes(
        inputTensor, filterTensor[0], outputTensor, rnnParamSpec, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    GCLMemDesc ftmMemDesc[2];
    ftmMemDesc[0] = gclmem_build_desc();
    ftmMemDesc[1] = gclmem_build_desc();
    maliPara.gclmemFilterDesc = ftmMemDesc;
    CHECK_STATUS(rnn_transform_filter_bytes(filterTensor, rnnParamSpec, ftmBytes.data(), &archInfo));

    GCLMem_t output = alloc_map(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    GCLMem_t state = alloc(stateTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));
    CHECK_STATUS(gcl_fill_memory_zero(handle, state));

    stride[0] = 4 * col;
    stride[1] = 1;
    stride[2] = 1;
    std::vector<Tensor *> ftmTensorPtr(2);
    for (U32 i = 0; i < 2; i++) {
        stride[0] = biasDesc[i].dims[0];
        CHECK_STATUS(gclmem_set_desc_padding(&desc, stride, offset, dt, DF_NCHW, GCL_MEM_BUF,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bias_cpu[i]));
        alloc_desc(biasTensor[i], desc);
        stride[0] = filterDesc[i].dims[0];
        stride[1] = filterDesc[i].dims[1];
        CHECK_STATUS(gclmem_set_desc_padding(&desc, stride, offset, dt, DF_NCHW, GCL_MEM_BUF,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, filter_cpu[i]));
        alloc_desc(filterTensor[i], desc);
        alloc_desc(ftmTensor[i], ftmMemDesc[i]);
        ftmTensorPtr[i] = &ftmTensor[i];
    }
    tmpBytes = tensorNumBytes(inputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(stateDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(rnn_transform_filter(filterTensor, rnnParamSpec, ftmTensorPtr, &archInfo));
    CHECK_STATUS(ocl_set_input(handle, input, inputDesc, input_cpu, tmpbuf, true));
    CHECK_STATUS(ocl_set_input(handle, state, stateDesc, state_cpu, tmpbuf, true));
    CHECK_STATUS(rnncell(inputTensor, ftmTensor, biasTensor, stateTensor, rnnParamSpec, xDim, col,
        0, tmpTensor, outputTensor, &archInfo));
    /*warp up*/
    UNI_INFO_LOG("warm up gpu:\n");
    for (U32 i = 0; i < 0; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
    }
#ifdef _DEBUG
    double time = 0;
    double min_time = DBL_MAX;
    double max_time = 0;
    U32 loop = 1;
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
    TensorDesc outputDesc = outputTensor.get_desc();
    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, true));
    void *output_gpu = output->mapPtrArray.back();
    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "%u (%u %u %u)=(%u %u)", 1, 1, xDim, hDim, 1, hDim);
    sprintf(buffer, "%20s, %80s", "RNN", params);
#ifdef _DEBUG
    double hxDim = hDim + xDim;
    double ops = 1.0 *
        (2.0 * hxDim * col * 4 + col * 4 + rnnParamSpec.numProjection * rnnParamSpec.numOutput);
    ut_log(dt, buffer, ops, time);
#endif
    ut_check_a(output_gpu, get_ptr_from_tensor(outputTensorCpu, UT_ARCH),
        tensorNumElements(outputDesc), outputDesc.dt);
    UNI_INFO_LOG("state:\n");
    U32 stateSize = tensorNumBytes(stateDesc);
    CHECK_STATUS(
        gcl_trans_memory(handle, state, state_gpu_host, &stateSize, DEVICE_BUF_TO_HOST, true));
    ut_check_a(state_gpu_host, get_ptr_from_tensor(stateTensorCpu, UT_ARCH),
        tensorNumElements(stateDesc), stateDesc.dt);

    free(input_cpu);
    free(state_cpu);
    for (U32 i = 0; i < 2; i++) {
        free(filter_cpu[i]);
        free(bias_cpu[i]);
    }
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    rnncellTest(argc, argv, DT_F16, RNN_LSTM);
#endif
    return 0;
}
