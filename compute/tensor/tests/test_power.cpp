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

int powerTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 5);
    U32 len = atoi(argv[1]);
    PowerParamSpec p;
    p.scale = atof(argv[2]);
    p.shift = atof(argv[3]);
    p.power = atof(argv[4]);
    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    Tensor inputTensor;
    TensorDesc inputDesc = tensor1d(dt, len);
    inputTensor.resize(inputDesc);
    inputTensor.alloc();
    U8 *input = ut_input_v(len, dt, UT_INIT_RANDOM);
    memcpy(get_ptr_from_tensor(inputTensor, UT_ARCH), input, tensorNumBytes(inputDesc));
    // set output
    Tensor outputTensor, outputTensorRef;
    CHECK_STATUS(power_infer_output_size(&inputTensor, &outputTensor, &archInfo));
    outputTensor.alloc();
    TensorDesc outputDesc_ref = outputTensor.get_desc();
    outputTensorRef.resize(outputDesc_ref);
    outputTensorRef.alloc();

    if (UT_CHECK) {
        CHECK_STATUS(power(inputTensor, p, outputTensor, &archInfo));

        // naive implement
        CHECK_STATUS(power(inputTensor, p, outputTensorRef, &archInfo_org));

        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, UT_ARCH),
            get_ptr_from_tensor(outputTensorRef, UT_ARCH), len, dt, 0.1, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(power(inputTensor, p, outputTensor, &archInfo));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u)=(%u)", len, len);
    sprintf(buffer, "%20s, %80s", "Power", params);
    double ops = 2.0 * len;
    ut_log(dt, buffer, ops, time / UT_LOOPS);

    free(input);

    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    powerTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    powerTest(argc, argv, DT_F32);
#endif
    return 0;
}
