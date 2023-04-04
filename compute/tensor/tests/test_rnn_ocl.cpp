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

int rnnTest(int argc, char **argv, DataType dt, RNNMode mode)
{
    U32 batch, step, xDim, hDim, num_projection, biDir;
    batch = atoi(argv[1]);
    step = atoi(argv[2]);
    xDim = atoi(argv[3]);
    hDim = atoi(argv[4]);
    num_projection = atoi(argv[5]);
    biDir = atoi(argv[6]);
    ArchInfo archInfo;
    archInfo.arch = MALI;
    if (gcl_check_device_qualcomm(OCLContext::getInstance().handle.get())) {
        archInfo.arch = QUALCOMM;
    }

    RNNParamSpec rnnParamSpec;
    rnnParamSpec.mode = RNN_LSTM;
    rnnParamSpec.num_outputs = hDim;
    rnnParamSpec.num_projection = num_projection;
    rnnParamSpec.forget_bias = 1.0;
    rnnParamSpec.zoneout_cell = 0;
    rnnParamSpec.zoneout_output = 0;
    rnnParamSpec.steps = 0;
    rnnParamSpec.bi_direction = (biDir) ? true : false;
    rnnParamSpec.activation_type = ACTIVATION_TANH;

    U32 col = (num_projection > 0) ? num_projection : hDim;
    TensorDesc inputDesc = tensor3df(dt, DF_NORMAL, batch, step, xDim);

    std::vector<TensorDesc> biasDesc(2);
    std::vector<TensorDesc> filterDesc(2);
    filterDesc[0] = tensor2df(dt, DF_NK, 4 * col, xDim + hDim);
    filterDesc[1] = tensor2df(dt, DF_NK, hDim, num_projection);
    biasDesc[0] = tensor1d(dt, 4 * col);
    biasDesc[1] = tensor1d(dt, hDim);

    Tensor inputTensorCpu;
    inputTensorCpu.resize(inputDesc);
    inputTensorCpu.alloc();

    U32 filterNum = (num_projection) ? 2 : 1;
    U32 biDirNum = (biDir) ? 2 : 1;
    std::vector<Tensor> filterTensorCpu(filterNum * biDirNum);
    std::vector<Tensor> biasTensorCpu(filterNum * biDirNum);
    std::vector<Tensor> ftmTensorCpu(filterNum * biDirNum);
    for (U32 i = 0; i < biDirNum; i++) {
        for (U32 j = 0; j < filterNum; j++) {
            filterTensorCpu[i * filterNum + j].resize(filterDesc[j]);
            filterTensorCpu[i * filterNum + j].alloc();
            biasTensorCpu[i * filterNum + j].resize(biasDesc[j]);
            biasTensorCpu[i * filterNum + j].alloc();
        }
    }
    Tensor outputTensorCpu;
    U32 inputLen = tensorNumElements(inputDesc);
    U8 *input_cpu = ut_input_v(inputLen, dt, UT_INIT_RANDOM);
    UNI_MEMCPY(get_ptr_from_tensor(inputTensorCpu, CPU_GENERAL), input_cpu, inputLen * bytesOf(dt));

    std::vector<U8 *> bias_cpu(filterNum * biDirNum);
    std::vector<U8 *> filter_cpu(filterNum * biDirNum);
    for (U32 i = 0; i < biDirNum; i++) {
        for (U32 j = 0; j < filterNum; j++) {
            U32 len = tensorNumElements(biasDesc[j]);
            bias_cpu[i * filterNum + j] = ut_input_v(len, dt, UT_INIT_RANDOM);
            UNI_MEMCPY(get_ptr_from_tensor(biasTensorCpu[i * filterNum + j], CPU_GENERAL),
                bias_cpu[i * filterNum + j], len * bytesOf(dt));

            len = tensorNumElements(filterDesc[j]);
            filter_cpu[i * filterNum + j] = ut_input_v(len, dt, UT_INIT_RANDOM);
            UNI_MEMCPY(get_ptr_from_tensor(filterTensorCpu[i * filterNum + j], CPU_GENERAL),
                filter_cpu[i * filterNum + j], len * bytesOf(dt));
        }
    }

    std::vector<Tensor *> inputTensorPtrCpu(1);
    inputTensorPtrCpu[0] = &inputTensorCpu;
    std::vector<Tensor *> outputTensorPtrCpu(1);
    outputTensorPtrCpu[0] = &outputTensorCpu;
    CHECK_STATUS(rnn_infer_output_size(
        inputTensorPtrCpu, rnnParamSpec, outputTensorPtrCpu, &UT_SERIAL_ARCHINFO));
    outputTensorCpu.alloc();

    U32 tmpBytes;
    CHECK_STATUS(rnn_infer_forward_tmp_bytes(inputTensorCpu, filterTensorCpu[0], outputTensorCpu,
        rnnParamSpec, &tmpBytes, &UT_SERIAL_ARCHINFO));
    Tensor tmpTensorCpu;
    TensorDesc tmpDesc = tensor1d(DT_U8, tmpBytes);
    tmpTensorCpu.resize(tmpDesc);
    tmpTensorCpu.alloc();
    UNI_MEMSET(get_ptr_from_tensor(tmpTensorCpu, CPU_GENERAL), 0, tmpBytes);

    std::vector<U32> ftmBytes(4);
    CHECK_STATUS(rnn_transform_filter_bytes(
        filterTensorCpu, rnnParamSpec, ftmBytes.data(), &UT_SERIAL_ARCHINFO));

    std::vector<Tensor *> ftmTensorPtrCpu(filterNum * biDirNum);
    for (U32 i = 0; i < biDirNum; i++) {
        for (U32 j = 0; j < filterNum; j++) {
            tmpDesc = tensor1d(DT_U8, ftmBytes[j]);
            ftmTensorCpu[i * filterNum + j].resize(tmpDesc);
            ftmTensorCpu[i * filterNum + j].alloc();
            ftmTensorPtrCpu[i * filterNum + j] = &ftmTensorCpu[i * filterNum + j];
        }
    }

    CHECK_STATUS(rnn_transform_filter(
        filterTensorCpu, rnnParamSpec, tmpTensorCpu, ftmTensorPtrCpu, &UT_SERIAL_ARCHINFO));

    std::vector<Tensor> inputTensorVecCpu(1, inputTensorCpu);
    std::vector<Tensor> outputTensorVecCpu(1, outputTensorCpu);
    std::vector<Tensor> tmpTensorVecCpu(1, tmpTensorCpu);
    CHECK_STATUS(rnn(inputTensorVecCpu, ftmTensorCpu, biasTensorCpu, rnnParamSpec, tmpTensorVecCpu,
        outputTensorVecCpu, nullptr, &UT_SERIAL_ARCHINFO));

    /*************GPU*************/
    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;

    Tensor inputTensor = Tensor(OCLMem);
    inputTensor.resize(inputDesc);

    std::vector<Tensor> filterTensor(filterNum * biDirNum);
    std::vector<Tensor> biasTensor(filterNum * biDirNum);
    std::vector<Tensor> ftmTensor((filterNum + 1) * biDirNum);
    for (U32 i = 0; i < biDirNum; i++) {
        for (U32 j = 0; j < filterNum; j++) {
            filterTensor[i * filterNum + j] = Tensor(OCLMem);
            filterTensor[i * filterNum + j].resize(filterDesc[j]);
            biasTensor[i * filterNum + j] = Tensor(OCLMem);
            biasTensor[i * filterNum + j].resize(biasDesc[j]);
        }
    }
    for (U32 i = 0; i < biDirNum; i++) {
        for (U32 j = 0; j < filterNum + 1; j++) {
            if (j == 0 && archInfo.arch == QUALCOMM) {
                ftmTensor[i * (filterNum + 1) + j] = Tensor(OCLMemImg);
            } else {
                ftmTensor[i * (filterNum + 1) + j] = Tensor(OCLMem);
            }
        }
    }
    Tensor outputTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    Tensor tmpTensorImg = Tensor(OCLMemImg);
    MaliPara maliPara;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)(CONVOLUTION_ALGORITHM_NULL);
    maliPara.handle = handle;
    maliPara.forwardRunInfo = &runInfo;
    archInfo.archPara = &maliPara;

    std::vector<Tensor *> inputTensorPtr(1);
    inputTensorPtr[0] = &inputTensor;
    std::vector<Tensor *> outputTensorPtr(1);
    outputTensorPtr[0] = &outputTensor;
    CHECK_STATUS(rnn_infer_output_size(inputTensorPtr, rnnParamSpec, outputTensorPtr, &archInfo));
    TensorDesc outputDesc = outputTensor.get_desc();
    U8 *output_gpu = ut_input_v(tensorNumElements(outputDesc), dt, UT_INIT_RANDOM);

    CHECK_STATUS(rnn_infer_forward_algorithm(
        inputTensor, filterTensor, biasTensor, rnnParamSpec, outputTensor, &archInfo));
    runInfo.algorithm = (I32)(CONVOLUTION_ALGORITHM_GEMM);

    U32 maxBytes[4] = {0};
    CHECK_STATUS(rnn_infer_forward_tmp_bytes(
        inputTensor, filterTensor[0], outputTensor, rnnParamSpec, maxBytes, &archInfo));

    TensorDesc ftmDesc[3];
    CHECK_STATUS(rnn_transform_filter_bytes(filterTensor, rnnParamSpec, ftmDesc, &archInfo));

    GCLMem_t output = alloc(outputTensor);
    GCLMem_t input = alloc(inputTensor);
    CHECK_STATUS(gcl_fill_memory_zero(handle, input));

    std::vector<Tensor *> ftmTensorPtr((filterNum + 1) * biDirNum);
    for (U32 i = 0; i < biDirNum; i++) {
        for (U32 j = 0; j < filterNum; j++) {
            auto biasMemCpu = biasTensorCpu[i * filterNum + j].get_memory();
            auto biasMemGpu = (OclMemory *)biasTensor[i * filterNum + j].get_memory();
            biasMemGpu->padding(0, 8, 0, 0);
            biasMemGpu->copy_from(biasMemCpu);
            alloc_host_ptr(filterTensor[i * filterNum + j], filter_cpu[i * filterNum + j]);
        }
    }
    for (U32 i = 0; i < biDirNum; i++) {
        for (U32 j = 0; j < filterNum + 1; j++) {
            ftmTensor[i * (filterNum + 1) + j].resize(ftmDesc[j]);
            alloc(ftmTensor[i * (filterNum + 1) + j]);
            ftmTensorPtr[i * (filterNum + 1) + j] = &ftmTensor[i * (filterNum + 1) + j];
        }
    }
    tmpBytes = tensorNumBytes(inputDesc);
    maxBytes[0] = (tmpBytes > maxBytes[0]) ? tmpBytes : maxBytes[0];
    tmpBytes = tensorNumBytes(outputDesc);
    maxBytes[0] = (tmpBytes > maxBytes[0]) ? tmpBytes : maxBytes[0];
    GCLMem_t tmpbuf = alloc_bytes(tmpTensor, maxBytes[0]);
    alloc_img(tmpTensorImg, maxBytes + 1);
    std::vector<Tensor> tmpTensors(2);
    tmpTensors[0] = tmpTensor;
    tmpTensors[1] = tmpTensorImg;

    CHECK_STATUS(
        rnn_transform_filter(filterTensor, rnnParamSpec, tmpTensor, ftmTensorPtr, &archInfo));
    CHECK_STATUS(ocl_set_input(handle, input, inputDesc, input_cpu, tmpbuf, true));
    std::vector<Tensor> inputTensors(1, inputTensor);
    std::vector<Tensor> outputTensors(1, outputTensor);
    CHECK_STATUS(rnn(inputTensors, ftmTensor, biasTensor, rnnParamSpec, tmpTensors, outputTensors,
        nullptr, &archInfo));

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
    sprintf(params, "%u (%u %u %u)=(%u %u)", batch, step, xDim, hDim, batch, hDim);
    sprintf(buffer, "%20s, %80s", "RNN", params);
    double hxDim = hDim + xDim;
    double ops = 1.0 * batch * step *
        (2.0 * hxDim * col * 4 + col * 4 + rnnParamSpec.num_projection * rnnParamSpec.num_outputs);
    ut_log(dt, buffer, ops, time);

    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, output_gpu, tmpbuf, true));
    ut_check_v(output_gpu, get_ptr_from_tensor(outputTensorCpu, CPU_GENERAL),
        tensorNumElements(outputDesc), outputDesc.dt, 0.3);

    free(output_gpu);
    free(input_cpu);
    for (U32 i = 0; i < filterNum * biDirNum; i++) {
        free(filter_cpu[i]);
        free(bias_cpu[i]);
    }
    return 0;
}

int main(int argc, char **argv)
{
    rnnTest(argc, argv, DT_F16, RNN_LSTM);
    return 0;
}
