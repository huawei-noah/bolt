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
#include <string.h>

#include "tensor_computing.h"
#include "ut_util.h"

int eltwiseTest(int argc, char** argv, DataType dt) {
    CHECK_REQUIREMENT(argc == 6);
    U32 num = atoi(argv[1]);
    U32 in = atoi(argv[2]);
    U32 ic = atoi(argv[3]);
    U32 ih = atoi(argv[4]);
    U32 iw = atoi(argv[5]);

    U32 len = in * ic * ih * iw;
    EltwiseMode eltwiseMode = ELTWISE_MAX;

    std::vector<TensorDesc> inputDesc(num);
    std::vector<void*> input(num);
    for (U32 i = 0; i < num; i++) {
        inputDesc[i] = tensor4df(dt, DF_NCHWC8, in, ic, ih, iw);
        input[i] = (void*)ut_input_v(len, dt, UT_INIT_RANDOM);
    }
    TensorDesc outputDesc;
    CHECK_STATUS(eltwise_infer_output_size(inputDesc, &outputDesc, UT_ARCH));
    CHECK_REQUIREMENT(len == tensorNumElements(outputDesc));
    U8 *output = ut_input_v(len, dt, UT_INIT_ZERO);
    U8 *output_ref = ut_input_v(len, dt, UT_INIT_ZERO);

    if (UT_CHECK) {
        CHECK_STATUS(eltwise(inputDesc, input, outputDesc, output, eltwiseMode, UT_ARCH));

        CHECK_STATUS(eltwise(inputDesc, input, outputDesc, output_ref, eltwiseMode, CPU_GENERAL));

        // check
        ut_check_v(output, output_ref, len, dt, 1, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(eltwise(inputDesc, input, outputDesc, output, eltwiseMode, UT_ARCH));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "%u (%u %u %u %u)=(%u %u %u %u)",
                    num, in, ic, ih, iw, 
                    in, ic, ih, iw);
    sprintf(buffer, "%20s, %80s", "Eltwise", params);
    double ops = 1.0 * num * in * ic * ih * iw;
    ut_log(dt, buffer, ops, time);

    for(U32 i=0; i<num; i++){
        free(input[i]);
    }
    free(output);
    free(output_ref);

    return 0;
}


int main(int argc, char** argv) {
#ifdef _USE_FP16
    eltwiseTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    eltwiseTest(argc, argv, DT_F32);
#endif
    return 0;
}
