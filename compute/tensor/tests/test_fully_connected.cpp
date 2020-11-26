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

int fullyConnectedTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 4);
    U32 m = atoi(argv[1]);
    U32 k = atoi(argv[2]);
    U32 n = atoi(argv[3]);

    DataFormat df = DF_TRANSPOSE;
    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    TensorDesc inputDesc = tensor4df(dt, DF_NCHW, m, 1, 1, k);
    TensorDesc filterDesc = tensor2df(dt, df, n, k);
    TensorDesc biasDesc = tensor1d(dt, n);

    Tensor inputTensor, filterTensor, biasTensor;
    inputTensor.resize(inputDesc);
    inputTensor.alloc();
    U8 *input = ut_input_v(m * k, dt, UT_INIT_RANDOM);
    memcpy(get_ptr_from_tensor(inputTensor, UT_ARCH), input, tensorNumBytes(inputDesc));

    filterTensor.resize(filterDesc);
    filterTensor.alloc();
    U8 *filter = ut_input_v(k * n, dt, UT_INIT_RANDOM);
    memcpy(get_ptr_from_tensor(filterTensor, UT_ARCH), filter, tensorNumBytes(filterDesc));

    biasTensor.resize(biasDesc);
    biasTensor.alloc();
    U8 *bias = ut_input_v(n, dt, UT_INIT_RANDOM);
    memcpy(get_ptr_from_tensor(biasTensor, UT_ARCH), bias, tensorNumBytes(biasDesc));
    // set output
    Tensor outputTensor, outputTensorRef;
    CHECK_STATUS(
        fully_connected_infer_output_size(&inputTensor, filterTensor, &outputTensor, &archInfo));
    outputTensor.alloc();
    TensorDesc outputDesc_ref = outputTensor.get_desc();
    outputTensorRef.resize(outputDesc_ref);
    outputTensorRef.alloc();
    // setup tmp
    Tensor tmpTensor;
    U32 tmpBytes;
    CHECK_STATUS(
        fully_connected_infer_forward_tmp_bytes(inputTensor, filterTensor, &tmpBytes, &archInfo));
    tmpTensor.resize(tensor1d(DT_U8, tmpBytes));
    tmpTensor.alloc();
    // setup filter trans
    U32 ftmBytes;
    CHECK_STATUS(fully_connected_transform_filter_bytes(filterTensor, &ftmBytes, &archInfo));
    // trans filter
    Tensor ftmTensor;
    ftmTensor.resize(tensor1d(DT_U8, ftmBytes));
    ftmTensor.alloc();
    CHECK_STATUS(fully_connected_transform_filter(inputTensor, filterTensor, &ftmTensor, &archInfo));

    //U32 bytes = 0;
    if (UT_CHECK) {
        CHECK_STATUS(
            fully_connected(inputTensor, ftmTensor, biasTensor, tmpTensor, outputTensor, &archInfo));

        // naive implement
        CHECK_STATUS(fully_connected(
            inputTensor, ftmTensor, biasTensor, tmpTensor, outputTensorRef, &archInfo_org));

        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, UT_ARCH),
            get_ptr_from_tensor(outputTensorRef, UT_ARCH), m * n, dt, 1, __FILE__, __LINE__);
    }
    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(
            fully_connected(inputTensor, ftmTensor, biasTensor, tmpTensor, outputTensor, &archInfo));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u)+(%u %u)=(%u %u)", m, k, k, n, m, n);
    sprintf(buffer, "%20s, %80s", "InnerProduct", params);
    double ops = 2.0 * m * n * k + 1.0 * m * n;
    ut_log(dt, buffer, ops, time);

    free(input);
    free(filter);
    free(bias);

    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    fullyConnectedTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    fullyConnectedTest(argc, argv, DT_F32);
#endif
    return 0;
}
