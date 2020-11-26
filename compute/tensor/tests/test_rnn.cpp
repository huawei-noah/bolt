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

int rnnTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 5);
    U32 batch = atoi(argv[1]);
    U32 step = atoi(argv[2]);
    U32 xDim = atoi(argv[3]);
    U32 hDim = atoi(argv[4]);
    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    RNNParamSpec rnnParamSpec;
    rnnParamSpec.mode = RNN_LSTM;
    rnnParamSpec.biDirection = false;
    rnnParamSpec.numOutput = hDim;
    rnnParamSpec.numProjection = 1024;
    rnnParamSpec.forgetBias = 1.0;
    rnnParamSpec.activationMode = ACTIVATION_TANH;
    rnnParamSpec.zoneoutCell = 0;
    rnnParamSpec.zoneoutOutput = 0;
    F32 threshold = 10;
    if (rnnParamSpec.numProjection > 0) {
        threshold = 40;
    }

    U32 column = (rnnParamSpec.numProjection > 0) ? rnnParamSpec.numProjection
                                                  : rnnParamSpec.numOutput;
    U32 num2 = (rnnParamSpec.numProjection > 0) ? 2 : 1;
    TensorDesc inputDesc = tensor3df(dt, DF_MTK, batch, step, xDim);
    Tensor inputTensor;
    inputTensor.resize(inputDesc);
    inputTensor.alloc();
    U32 inputLength = batch * step * xDim;
    U8 *input = ut_input_v(inputLength, dt, UT_INIT_RANDOM);
    memcpy(get_ptr_from_tensor(inputTensor, UT_ARCH), input, tensorNumBytes(inputDesc));

    U32 tmpBytes;
    std::vector<TensorDesc> filterDesc(2), biasDesc(2);
    filterDesc[0] = tensor2df(dt, DF_NK, 4 * column, xDim + hDim);
    filterDesc[1] = tensor2df(dt, DF_NK, rnnParamSpec.numOutput, rnnParamSpec.numProjection);
    biasDesc[0] = tensor1d(dt, column * 4);
    biasDesc[1] = tensor1d(dt, rnnParamSpec.numOutput);
    std::vector<Tensor> filterTensor(num2), biasTensor(num2);
    for (U32 i = 0; i < num2; i++) {
        filterTensor[i].resize(filterDesc[i]);
        filterTensor[i].alloc();
        U8 *filter = ut_input_v(tensorNumBytes(filterDesc[i]) / bytesOf(dt), dt, UT_INIT_RANDOM);
        memcpy(get_ptr_from_tensor(filterTensor[i], UT_ARCH), filter, tensorNumBytes(filterDesc[i]));
        free(filter);

        biasTensor[i].resize(biasDesc[i]);
        biasTensor[i].alloc();
        U8 *bias = ut_input_v(tensorNumBytes(biasDesc[i]) / bytesOf(dt), dt, UT_INIT_RANDOM);
        memcpy(get_ptr_from_tensor(biasTensor[i], UT_ARCH), bias, tensorNumBytes(biasDesc[i]));
        free(bias);
    }

    // set output
    Tensor outputTensor, outputTensorRef;
    CHECK_STATUS(rnn_infer_output_size(&inputTensor, rnnParamSpec, &outputTensor, &archInfo));
    outputTensor.alloc();
    U32 outputLength = outputTensor.length();

    TensorDesc outputDesc_ref = outputTensor.get_desc();
    outputTensorRef.resize(outputDesc_ref);
    outputTensorRef.alloc();

    CHECK_STATUS(rnn_infer_forward_tmp_bytes(
        inputTensor, filterTensor[0], outputTensor, rnnParamSpec, &tmpBytes, &archInfo));
    std::vector<U32> ftmBytes(num2);
    CHECK_STATUS(rnn_transform_filter_bytes(filterTensor, rnnParamSpec, ftmBytes.data(), &archInfo));
    std::vector<Tensor> ftmTensor(num2);
    std::vector<Tensor *> ftmTensorPtr(num2);
    for (U32 i = 0; i < num2; i++) {
        ftmTensor[i].resize(tensor1d(DT_U8, ftmBytes[i]));
        ftmTensor[i].alloc();
        ftmTensorPtr[i] = &ftmTensor[i];
    }

    Tensor tmpTensor;
    tmpTensor.resize(tensor1d(DT_U8, tmpBytes));
    tmpTensor.alloc();

    CHECK_STATUS(rnn_transform_filter(filterTensor, rnnParamSpec, ftmTensorPtr, &archInfo));

    if (UT_CHECK) {
        CHECK_STATUS(rnn(
            inputTensor, ftmTensor, biasTensor, rnnParamSpec, tmpTensor, outputTensor, &archInfo));

        // naive implement
        CHECK_STATUS(rnn(inputTensor, ftmTensor, biasTensor, rnnParamSpec, tmpTensor,
            outputTensorRef, &archInfo_org));

        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, UT_ARCH),
            get_ptr_from_tensor(outputTensorRef, UT_ARCH), outputLength, dt, threshold, __FILE__,
            __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(rnn(
            inputTensor, ftmTensor, biasTensor, rnnParamSpec, tmpTensor, outputTensor, &archInfo));
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
        (2.0 * hxDim * column * 4 + column * 4 + rnnParamSpec.numProjection * rnnParamSpec.numOutput);
    ut_log(dt, buffer, ops, time);

    free(input);
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    rnnTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    rnnTest(argc, argv, DT_F32);
#endif
    return 0;
}
