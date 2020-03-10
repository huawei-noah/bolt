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

int axisMeanTest(int argc, char** argv, DataType dt) {
    CHECK_REQUIREMENT(argc == 6);
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);
    I32 axis = atoi(argv[5]);

    DataFormat df = DF_NCHW;

    TensorDesc inDesc = tensor4df(dt, df, in, ic, ih, iw);
    U8* input = ut_input_v(tensorNumElements(inDesc), dt, UT_INIT_RANDOM);
    TensorDesc outDesc;
    CHECK_STATUS(axis_mean_infer_output_size(inDesc, axis, &outDesc));
    U8* output    = ut_input_v(tensorNumElements(outDesc), dt, UT_INIT_ZERO);
    U8* outputRef = ut_input_v(tensorNumElements(outDesc), dt, UT_INIT_ZERO);

    if (UT_CHECK) {
        CHECK_STATUS(axis_mean(inDesc, input, axis, outDesc, output, UT_ARCH));

        // naive implement
        CHECK_STATUS(axis_mean(inDesc, input, axis, outDesc, outputRef, CPU_GENERAL));

        // check
        ut_check_v(output, outputRef, tensorNumElements(outDesc), dt, 1, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(axis_mean(inDesc, input, axis, outDesc, output, UT_ARCH));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    U32 on = 1, oh = 1, ow = 1;
    CHECK_STATUS(tensor3dGet(outDesc, &dt, &df, &on, &oh, &ow));
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u) %d =(%u %u %u)",
                    in, ic, ih, iw, axis,
                    on, oh, ow);
    sprintf(buffer, "%20s, %80s", "AxisMean", params);
    double ops = 1.0 * in * ic * ih * iw;
    ut_log(dt, buffer, ops, time/UT_LOOPS);

    free(output);
    free(outputRef);

    return 0;
}


int main(int argc, char** argv) {
#ifdef _USE_FP16
    axisMeanTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    axisMeanTest(argc, argv, DT_F32);
#endif
    return 0;
}
