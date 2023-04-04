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
#include "ut_util.h"

int rnnTest(int argc, char **argv, DataType dt, RNNMode mode)
{
    CHECK_REQUIREMENT(argc == 5);
    U32 batch = atoi(argv[1]);
    U32 step = atoi(argv[2]);
    U32 xDim = atoi(argv[3]);
    U32 hDim = atoi(argv[4]);

    RNNParamSpec rnnParamSpec;
    rnnParamSpec.mode = mode;
    rnnParamSpec.steps = step;
    rnnParamSpec.bi_direction = false;
    rnnParamSpec.num_outputs = hDim;
    rnnParamSpec.num_projection = 0;
    rnnParamSpec.forget_bias = 1.0;
    rnnParamSpec.activation_type = ACTIVATION_TANH;
    rnnParamSpec.zoneout_cell = 0;
    rnnParamSpec.zoneout_output = 0;

    U32 weightNum = 1;
    U32 biasNum = 1;
    int factor = 0;
    switch (mode) {
        case RNN_LSTM:
            rnnParamSpec.num_projection = 1024;
            factor = 4;
            break;
        case RNN_GRU:
            factor = 3;
            break;
        case RNN_GRU_LBR:
            factor = 3;
            biasNum++;
            break;
        default:
            return 1;
    }
    F32 threshold = 10;
    if (rnnParamSpec.num_projection > 0) {
        weightNum++;
        biasNum++;
        threshold = 40;
    }

    if (rnnParamSpec.mode != RNN_LSTM) {
        rnnParamSpec.num_projection = 0;
        rnnParamSpec.forget_bias = 0;
    }

    U32 column = (rnnParamSpec.num_projection > 0) ? rnnParamSpec.num_projection
                                                   : rnnParamSpec.num_outputs;
    TensorDesc inputDesc = tensor3df(dt, DF_MTK, batch, step, xDim);
    Tensor inputTensor;
    inputTensor.resize(inputDesc);
    inputTensor.alloc();
    U32 inputLength = batch * step * xDim;
    U8 *input = ut_input_v(inputLength, dt, UT_INIT_RANDOM);
    UNI_MEMCPY(get_ptr_from_tensor(inputTensor, CPU_GENERAL), input, tensorNumBytes(inputDesc));

    U32 tmpBytes;
    std::vector<TensorDesc> filterDesc(2), biasDesc(2);
    filterDesc[0] = tensor2df(dt, DF_NK, factor * column, xDim + hDim);
    filterDesc[1] = tensor2df(dt, DF_NK, rnnParamSpec.num_outputs, rnnParamSpec.num_projection);
    biasDesc[0] = tensor1d(dt, column * factor);
    biasDesc[1] = tensor1d(dt, rnnParamSpec.num_outputs);
    std::vector<Tensor> filterTensor(weightNum), biasTensor(biasNum);
    for (U32 i = 0; i < weightNum; i++) {
        filterTensor[i].resize(filterDesc[i]);
        filterTensor[i].alloc();
        U8 *filter = ut_input_v(tensorNumBytes(filterDesc[i]) / bytesOf(dt), dt, UT_INIT_RANDOM);
        UNI_MEMCPY(get_ptr_from_tensor(filterTensor[i], CPU_GENERAL), filter,
            tensorNumBytes(filterDesc[i]));
        free(filter);
    }

    for (U32 i = 0; i < biasNum; i++) {
        biasTensor[i].resize(biasDesc[i]);
        biasTensor[i].alloc();
        U8 *bias = ut_input_v(tensorNumBytes(biasDesc[i]) / bytesOf(dt), dt, UT_INIT_RANDOM);
        UNI_MEMCPY(
            get_ptr_from_tensor(biasTensor[i], CPU_GENERAL), bias, tensorNumBytes(biasDesc[i]));
        free(bias);
    }

    // set output
    Tensor outputTensor, outputTensorRef;
    std::vector<Tensor *> inputTensors(1, &inputTensor);
    std::vector<Tensor *> outputTensors(1, &outputTensor);
    CHECK_STATUS(rnn_infer_output_size(inputTensors, rnnParamSpec, outputTensors, &UT_CPU_ARCHINFO));
    outputTensor.alloc();
    U32 outputLength = outputTensor.length();

    TensorDesc outputDesc_ref = outputTensor.get_desc();
    outputTensorRef.resize(outputDesc_ref);
    outputTensorRef.alloc();

    CHECK_STATUS(rnn_infer_forward_tmp_bytes(
        inputTensor, filterTensor[0], outputTensor, rnnParamSpec, &tmpBytes, &UT_CPU_ARCHINFO));
    std::vector<U32> ftmBytes(weightNum);
    CHECK_STATUS(
        rnn_transform_filter_bytes(filterTensor, rnnParamSpec, ftmBytes.data(), &UT_CPU_ARCHINFO));
    std::vector<Tensor> ftmTensor(weightNum), ftmTensorRef(weightNum);
    std::vector<Tensor *> ftmTensorPtr(weightNum), ftmTensorPtrRef(weightNum);
    for (U32 i = 0; i < weightNum; i++) {
        ftmTensor[i].resize(tensor1d(DT_U8, ftmBytes[i]));
        ftmTensor[i].alloc();
        ftmTensorPtr[i] = &ftmTensor[i];

        ftmTensorRef[i].resize(tensor1d(DT_U8, ftmBytes[i]));
        ftmTensorRef[i].alloc();
        ftmTensorPtrRef[i] = &ftmTensorRef[i];
    }

    Tensor tmpTensor;
    tmpTensor.resize(tensor1d(DT_U8, tmpBytes));
    tmpTensor.alloc();

    CHECK_STATUS(
        rnn_transform_filter(filterTensor, rnnParamSpec, tmpTensor, ftmTensorPtr, &UT_CPU_ARCHINFO));
    CHECK_STATUS(rnn_transform_filter(
        filterTensor, rnnParamSpec, tmpTensor, ftmTensorPtrRef, &UT_SERIAL_ARCHINFO));

    std::vector<Tensor> inputTensorVec(1, inputTensor);
    std::vector<Tensor> outputTensorVec(1, outputTensor);
    std::vector<Tensor> outputTensorRefVec(1, outputTensorRef);
    std::vector<Tensor> tmpTensorVec(1, tmpTensor);
    if (UT_CHECK) {
        UNI_MEMSET(get_ptr_from_tensor(tmpTensor, UT_CPU_ARCHINFO.arch), 0, tmpBytes);
        CHECK_STATUS(rnn(inputTensorVec, ftmTensor, biasTensor, rnnParamSpec, tmpTensorVec,
            outputTensorVec, nullptr, &UT_CPU_ARCHINFO));

        // naive implement
        UNI_MEMSET(get_ptr_from_tensor(tmpTensor, UT_CPU_ARCHINFO.arch), 0, tmpBytes);
        CHECK_STATUS(rnn(inputTensorVec, ftmTensorRef, biasTensor, rnnParamSpec, tmpTensorVec,
            outputTensorRefVec, nullptr, &UT_SERIAL_ARCHINFO));

        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, CPU_GENERAL),
            get_ptr_from_tensor(outputTensorRef, CPU_GENERAL), outputLength, dt, threshold);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(rnn(inputTensorVec, ftmTensor, biasTensor, rnnParamSpec, tmpTensorVec,
            outputTensorVec, nullptr, &UT_CPU_ARCHINFO));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "%u (%u %u %u)=(%u %u)", batch, step, xDim, hDim, batch, hDim);
    sprintf(buffer, "%20s, %80s", "RNN", params);
    double hxDim = hDim + xDim;
    double ops = 1.0 * batch * step *
        (2.0 * hxDim * column * factor + column * factor +
            rnnParamSpec.num_projection * rnnParamSpec.num_outputs);
    ut_log(dt, buffer, ops, time);

    free(input);
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    rnnTest(argc, argv, DT_F16, RNN_LSTM);
    rnnTest(argc, argv, DT_F16, RNN_GRU);
    rnnTest(argc, argv, DT_F16, RNN_GRU_LBR);
#endif
#ifdef _USE_FP32
    rnnTest(argc, argv, DT_F32, RNN_LSTM);
    rnnTest(argc, argv, DT_F32, RNN_GRU);
    rnnTest(argc, argv, DT_F32, RNN_GRU_LBR);
#endif
    return 0;
}
