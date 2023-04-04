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

int rnncellTest(int argc, char **argv, DataType dt, RNNMode mode)
{
    U32 xDim, hDim, num_projection;
    xDim = atoi(argv[1]);
    hDim = atoi(argv[2]);
    if (argc == 4) {
        num_projection = atoi(argv[3]);
    } else {
        num_projection = 0;
    }
    ArchInfo archInfo;
    archInfo.arch = MALI;

    RNNParamSpec rnnParamSpec;
    rnnParamSpec.mode = RNN_LSTM;
    rnnParamSpec.num_outputs = hDim;
    rnnParamSpec.num_projection = num_projection;
    rnnParamSpec.forget_bias = 1.0;
    rnnParamSpec.zoneout_cell = 0;
    rnnParamSpec.zoneout_output = 0;
    rnnParamSpec.steps = -1;
    rnnParamSpec.bi_direction = false;
    rnnParamSpec.activation_type = ACTIVATION_TANH;

    U32 col = (num_projection > 0) ? num_projection : hDim;
    TensorDesc inputDesc = tensor2df(dt, DF_NORMAL, 1, xDim);
    TensorDesc stateDesc = tensor2df(dt, DF_NORMAL, 1, col + hDim);

    std::vector<TensorDesc> biasDesc(2);
    std::vector<TensorDesc> filterDesc(2);
    filterDesc[0] = tensor2df(dt, DF_NK, 4 * col, xDim + hDim);
    filterDesc[1] = tensor2df(dt, DF_NK, hDim, num_projection);
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
    UNI_MEMCPY(get_ptr_from_tensor(inputTensorCpu, CPU_GENERAL), input_cpu, inputLen * bytesOf(dt));

    U8 *state_cpu = ut_input_v(stateLen, dt, UT_INIT_RANDOM);
    UNI_MEMCPY(get_ptr_from_tensor(stateTensorCpu, CPU_GENERAL), state_cpu, stateLen * bytesOf(dt));
    U8 *state_gpu_host = ut_input_v(stateLen, dt, UT_INIT_ZERO);

    std::vector<U8 *> bias_cpu(2);
    std::vector<U8 *> filter_cpu(2);
    for (U32 i = 0; i < 2; i++) {
        U32 len = tensorNumElements(biasDesc[i]);
        bias_cpu[i] = ut_input_v(len, dt, UT_INIT_RANDOM);
        UNI_MEMCPY(
            get_ptr_from_tensor(biasTensorCpu[i], CPU_GENERAL), bias_cpu[i], len * bytesOf(dt));

        len = tensorNumElements(filterDesc[i]);
        filter_cpu[i] = ut_input_v(len, dt, UT_INIT_RANDOM);
        UNI_MEMCPY(
            get_ptr_from_tensor(filterTensorCpu[i], CPU_GENERAL), filter_cpu[i], len * bytesOf(dt));
    }

    std::vector<Tensor *> inputTensorPtrCpu(2);
    inputTensorPtrCpu[0] = &inputTensorCpu;
    inputTensorPtrCpu[1] = &stateTensorCpu;
    CHECK_STATUS(rnncell_infer_output_size(
        inputTensorPtrCpu, rnnParamSpec, &outputTensorCpu, &UT_SERIAL_ARCHINFO));
    outputTensorCpu.alloc();

    U32 tmpBytes;
    CHECK_STATUS(rnncell_infer_forward_tmp_bytes(inputTensorCpu, filterTensorCpu[0],
        outputTensorCpu, rnnParamSpec, &tmpBytes, &UT_SERIAL_ARCHINFO));
    Tensor tmpTensorCpu;
    TensorDesc tmpDesc = tensor1d(DT_U8, tmpBytes);
    tmpTensorCpu.resize(tmpDesc);
    tmpTensorCpu.alloc();
    UNI_MEMSET(get_ptr_from_tensor(tmpTensorCpu, CPU_GENERAL), 0, tmpBytes);

    std::vector<U32> ftmBytes(2);
    CHECK_STATUS(rnn_transform_filter_bytes(
        filterTensorCpu, rnnParamSpec, ftmBytes.data(), &UT_SERIAL_ARCHINFO));

    std::vector<Tensor *> ftmTensorPtrCpu(2);
    for (U32 i = 0; i < 2; i++) {
        tmpDesc = tensor1d(DT_U8, ftmBytes[i]);
        ftmTensorCpu[i].resize(tmpDesc);
        ftmTensorCpu[i].alloc();
        ftmTensorPtrCpu[i] = &ftmTensorCpu[i];
    }

    CHECK_STATUS(rnn_transform_filter(
        filterTensorCpu, rnnParamSpec, tmpTensorCpu, ftmTensorPtrCpu, &UT_SERIAL_ARCHINFO));

    CHECK_STATUS(rnncell(inputTensorCpu, ftmTensorCpu, biasTensorCpu, stateTensorCpu, rnnParamSpec,
        xDim, col, 0, tmpTensorCpu, outputTensorCpu, nullptr, &UT_SERIAL_ARCHINFO));

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;

    Tensor inputTensor = Tensor(OCLMem);
    Tensor stateTensor = Tensor(OCLMem);
    inputTensor.resize(inputDesc);
    stateTensor.resize(stateDesc);

    std::vector<Tensor> filterTensor(2);
    std::vector<Tensor> biasTensor(2);
    std::vector<Tensor> ftmTensor(2);
    for (U32 i = 0; i < 2; i++) {
        filterTensor[i] = Tensor(OCLMem);
        biasTensor[i] = Tensor(OCLMem);
        ftmTensor[i] = Tensor(OCLMem);
        filterTensor[i].resize(filterDesc[i]);
        biasTensor[i].resize(biasDesc[i]);
    }
    Tensor outputTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
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
    TensorDesc outputDesc = outputTensor.get_desc();
    U8 *output_gpu = ut_input_v(tensorNumElements(outputDesc), dt, UT_INIT_RANDOM);

    CHECK_STATUS(rnncell_infer_forward_algorithm(inputTensor, filterTensor[0], biasTensor[0],
        stateTensor, rnnParamSpec, xDim, col, outputTensor, &archInfo));

    U32 maxBytes = 0;
    CHECK_STATUS(rnncell_infer_forward_tmp_bytes(
        inputTensor, filterTensor[0], outputTensor, rnnParamSpec, &tmpBytes, &archInfo));
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;

    TensorDesc ftmDesc[2];
    CHECK_STATUS(rnncell_transform_filter_bytes(filterTensor, rnnParamSpec, ftmDesc, &archInfo));

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    GCLMem_t state = alloc(stateTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));
    CHECK_STATUS(gcl_fill_memory_zero(handle, state));

    std::vector<Tensor *> ftmTensorPtr(2);
    for (U32 i = 0; i < 2; i++) {
        auto biasMemCpu = biasTensorCpu[i].get_memory();
        auto biasMemGpu = (OclMemory *)biasTensor[i].get_memory();
        biasMemGpu->padding(0, 8, 0, 0);
        biasMemGpu->copy_from(biasMemCpu);
        alloc_host_ptr(filterTensor[i], filter_cpu[i]);
        ftmTensor[i].resize(ftmDesc[i]);
        alloc(ftmTensor[i]);
        ftmTensorPtr[i] = &ftmTensor[i];
    }
    tmpBytes = tensorNumBytes(inputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(stateDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    tmpBytes = tensorNumBytes(outputDesc);
    maxBytes = (tmpBytes > maxBytes) ? tmpBytes : maxBytes;
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes);

    CHECK_STATUS(rnncell_transform_filter(filterTensor, rnnParamSpec, ftmTensorPtr, &archInfo));
    CHECK_STATUS(ocl_set_input(handle, input, inputDesc, input_cpu, tmpbuf, true));
    CHECK_STATUS(ocl_set_input(handle, state, stateDesc, state_cpu, tmpbuf, true));
    CHECK_STATUS(rnncell(inputTensor, ftmTensor, biasTensor, stateTensor, rnnParamSpec, xDim, col,
        0, tmpTensor, outputTensor, nullptr, &archInfo));

    for (U32 i = 0; i < UT_WARMUP; i++) {
        CHECK_STATUS(gcl_run_kernelVec(handle));
    }
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
    char buffer[150];
    char params[120];
    sprintf(params, "%u (%u %u %u)=(%u %u)", 1, 1, xDim, hDim, 1, hDim);
    sprintf(buffer, "%20s, %80s", "RNN", params);
    double hxDim = hDim + xDim;
    double ops = 1.0 *
        (2.0 * hxDim * col * 4 + col * 4 + rnnParamSpec.num_projection * rnnParamSpec.num_outputs);
    ut_log(dt, buffer, ops, time);

    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, output_gpu, tmpbuf, true));
    ut_check_v(output_gpu, get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL),
        tensorNumElements(outputDesc), outputDesc.dt, 0.3);
    U32 stateSize = tensorNumBytes(stateDesc);
    CHECK_STATUS(
        gcl_trans_memory(handle, state, state_gpu_host, &stateSize, DEVICE_BUF_TO_HOST, true));
    ut_check_v(state_gpu_host, get_ptr_from_tensor(stateTensorCpu, CPU_GENERAL),
        tensorNumElements(stateDesc), stateDesc.dt, 0.3);

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
    rnncellTest(argc, argv, DT_F16, RNN_LSTM);
    return 0;
}
