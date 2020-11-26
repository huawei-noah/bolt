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

int preluTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 5);
    // in data
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);

    CHECK_REQUIREMENT(ic % 8 == 0);
    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    PReLUParamSpec prelu_desc;
    prelu_desc.propagate_down = 0;
    TensorDesc inputDesc = tensor4df(dt, DF_NCHWC8, in, ic, ih, iw);
    TensorDesc weightDesc = tensor1d(dt, ic);
    U32 input_len = tensorNumElements(inputDesc);
    U8 *input = ut_input_v(input_len, dt, UT_INIT_RANDOM);
    U8 *weight = ut_input_v(ic, dt, UT_INIT_RANDOM);

    Tensor inputTensor = Tensor::alloc_sized<CPUMem>(inputDesc);
    Tensor weightTensor = Tensor::alloc_sized<CPUMem>(weightDesc);
    memcpy(get_ptr_from_tensor(inputTensor, UT_ARCH), input, tensorNumBytes(inputDesc));
    memcpy(get_ptr_from_tensor(weightTensor, UT_ARCH), weight, tensorNumBytes(weightDesc));

    // set output
    Tensor outputTensor;
    CHECK_STATUS(prelu_infer_output_size(&inputTensor, &outputTensor, &archInfo));
    outputTensor.alloc();
    Tensor outputTensorRef = Tensor::alloc_sized<CPUMem>(outputTensor.get_desc());
    U32 output_len = outputTensor.length();
    CHECK_REQUIREMENT(input_len == in * ic * ih * iw && output_len == in * ic * ih * iw);

    if (UT_CHECK) {
        CHECK_STATUS(prelu(inputTensor, weightTensor, prelu_desc, outputTensor, &archInfo));

        CHECK_STATUS(prelu(inputTensor, weightTensor, prelu_desc, outputTensorRef, &archInfo_org));
        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, UT_ARCH),
            get_ptr_from_tensor(outputTensorRef, UT_ARCH), output_len, dt, 0.05, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(prelu(inputTensor, weightTensor, prelu_desc, outputTensor, &archInfo));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)*(%u)=(%u %u %u %u)", in, ic, ih, iw, ic, in, ic, ih, iw);
    sprintf(buffer, "%20s, %80s", "Prelu", params);
    double ops = 2.0 * in * ic * ih * iw + 1.0 * in;
    ut_log(dt, buffer, ops, time);

    free(input);
    free(weight);
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    preluTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    preluTest(argc, argv, DT_F32);
#endif
    return 0;
}
