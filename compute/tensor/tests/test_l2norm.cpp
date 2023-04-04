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

int l2normTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 4);
    U32 ic = atoi(argv[1]);
    U32 ih = atoi(argv[2]);
    U32 iw = atoi(argv[3]);

    DataFormat df = DF_MTK;
    TensorDesc inputDesc = tensor3df(dt, df, ic, ih, iw);
    U32 input_len = tensorNumElements(inputDesc);
    U8 *input = ut_input_v(input_len, dt, UT_INIT_RANDOM);

    Tensor inputTensor;
    inputTensor.resize(inputDesc);
    inputTensor.alloc();
    UNI_MEMCPY(get_ptr_from_tensor(inputTensor, CPU_GENERAL), input, tensorNumBytes(inputDesc));

    // set output
    Tensor outputTensor, outputTensorRef;
    CHECK_STATUS(l2norm_infer_output_size(&inputTensor, &outputTensor, &UT_CPU_ARCHINFO));
    outputTensor.alloc();
    TensorDesc outputDesc_ref = outputTensor.get_desc();
    outputTensorRef.resize(outputDesc_ref);
    outputTensorRef.alloc();

    U32 output_len = outputTensor.length();
    CHECK_REQUIREMENT(input_len == ic * ih * iw && output_len == ic * ih * iw);

    if (UT_CHECK) {
        CHECK_STATUS(l2norm(inputTensor, outputTensor, &UT_CPU_ARCHINFO));

        // naive implement
        CHECK_STATUS(l2norm(inputTensor, outputTensorRef, &UT_SERIAL_ARCHINFO));

        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, CPU_GENERAL),
            get_ptr_from_tensor(outputTensorRef, CPU_GENERAL), output_len, dt, 0.05);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(l2norm(inputTensor, outputTensor, &UT_CPU_ARCHINFO));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    //general benchmark
    time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(l2norm(inputTensor, outputTensorRef, &UT_SERIAL_ARCHINFO));
    }
    time_end = ut_time_ms();
    double general_implement_time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char general_buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u)=(%u %u %u)", ic, ih, iw, ic, ih, iw);
    sprintf(buffer, "%20s, %80s", "L2Normalization", params);
    sprintf(general_buffer, "%20s, %80s", "General L2Normalization", params);
    double ops = input_len;
    ut_log(dt, buffer, ops, time);
    ut_log(dt, general_buffer, ops, general_implement_time);

    free(input);

    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    l2normTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    l2normTest(argc, argv, DT_F32);
#endif
    return 0;
}
