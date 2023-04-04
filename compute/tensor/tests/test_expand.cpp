// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <vector>

#include "tensor_computing.h"
#include "ut_util.h"

int expandTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 9);
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);
    U32 on = atoi(argv[5]);
    U32 oc = atoi(argv[6]);
    U32 oh = atoi(argv[7]);
    U32 ow = atoi(argv[8]);
    ExpandParamSpec p;
    p.num_shape = 4;
    p.shape[0] = on;
    p.shape[1] = oc;
    p.shape[2] = oh;
    p.shape[3] = ow;

    DataFormat df = DF_NCHW;
    TensorDesc inDesc = tensor4df(dt, df, in, ic, ih, iw);
    U32 len = tensorNumElements(inDesc);
    U8 *input = ut_input_v(len, dt, UT_INIT_RANDOM);
    Tensor inputTensor;
    inputTensor.resize(inDesc);
    inputTensor.alloc();
    UNI_MEMCPY(get_ptr_from_tensor(inputTensor, CPU_GENERAL), input, tensorNumBytes(inDesc));

    Tensor outputTensor;
    CHECK_STATUS(expand_infer_output_size(&inputTensor, p, &outputTensor, &UT_CPU_ARCHINFO));
    outputTensor.alloc();
    Tensor blankTensor;

    if (UT_CHECK) {
        CHECK_STATUS(expand(inputTensor, p, blankTensor, outputTensor, &UT_CPU_ARCHINFO));

        double sum1 = 0;
        F32 *data = (F32 *)input;
        for (U32 i = 0; i < len; ++i) {
            sum1 += data[i];
        }
        double sum2 = 0;
        data = (F32 *)get_ptr_from_tensor(outputTensor, CPU_GENERAL);
        U32 outLen = tensorNumElements(outputTensor.get_desc());
        for (U32 i = 0; i < outLen; ++i) {
            sum2 += data[i];
        }

        // check
        ut_check_s(sum1 * outLen / len, sum2, 0, 0);
    }

    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(expand(inputTensor, p, blankTensor, outputTensor, &UT_CPU_ARCHINFO));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    CHECK_STATUS(tensor4dGet(outputTensor.get_desc(), &dt, &df, &on, &oc, &oh, &ow));
    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)=(%u %u %u %u)", in, ic, ih, iw, on, oc, oh, ow);
    sprintf(buffer, "%20s, %80s", "Transpose", params);
    double ops = tensorNumElements(outputTensor.get_desc());
    ut_log(dt, buffer, ops, time);

    free(input);

    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    expandTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    expandTest(argc, argv, DT_F32);
#endif
    return 0;
}
