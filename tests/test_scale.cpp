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

int scaleTest(int argc, char** argv, DataType dt){
    CHECK_REQUIREMENT(argc == 5);
    U32 in = atoi(argv[1]);
    U32 ic = atoi(argv[2]);
    U32 ih = atoi(argv[3]);
    U32 iw = atoi(argv[4]);

    DataFormat df = DF_NCHWC8;

    TensorDesc data_desc = tensor4df(dt, df, in, ic, ih, iw); 
    U32 len = tensorNumElements(data_desc);

    U8* alpha = ut_input_v(ic, dt, UT_INIT_RANDOM);
    U8* beta = ut_input_v(ic, dt, UT_INIT_RANDOM);
    U8* data = ut_input_v(len, dt, UT_INIT_RANDOM);
    U8* data_ref = ut_input_v(len, dt, UT_INIT_ZERO);
    memcpy(data_ref, data, len*bytesOf(dt));

    if (UT_CHECK) {
        CHECK_STATUS(scale(alpha, beta, data_desc, data, data_desc, NULL, UT_ARCH));

        // naive implement
        CHECK_STATUS(scale(alpha, beta, data_desc, data_ref, data_desc, NULL, CPU_GENERAL));

        // check
        ut_check_v(data, data_ref, len, dt, 1.0, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(scale(alpha, beta, data_desc, data_ref, data_desc, NULL, UT_ARCH));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u %u %u)=(%u %u %u %u)",
                    in, ic, ih, iw,
                    in, ic, ih, iw);
    sprintf(buffer, "%20s, %80s", "Scale", params);
    double ops = 2.0 * in * ic * ih * iw;
    ut_log(dt, buffer, ops, time/UT_LOOPS);

    free(data);
    free(data_ref);

    return 0;
}


int main(int argc, char** argv) {
#ifdef _USE_FP16
    scaleTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    scaleTest(argc, argv, DT_F32);
#endif
    return 0;
}
