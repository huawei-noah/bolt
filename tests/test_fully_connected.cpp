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

int fullyConnectedTest(int argc, char** argv, DataType dt) {
    CHECK_REQUIREMENT(argc == 4);
    U32 m = atoi(argv[1]);
    U32 k = atoi(argv[2]);
    U32 n = atoi(argv[3]);

    DataFormat df = DF_NORMAL;

    TensorDesc T_i = tensor4df(dt, DF_NCHW, m, 1, 1, k);
    TensorDesc T_w = tensor2df(dt, df, n, k);
    TensorDesc biasDesc = tensor1d(dt, n);
    TensorDesc T_o;
    CHECK_STATUS(fully_connected_infer_output_size(T_i, T_w, &T_o, UT_ARCH));

    U8* input  = ut_input_v(m * k, dt, UT_INIT_RANDOM);
    U8* weight = ut_input_v(k * n, dt, UT_INIT_RANDOM);
    U8* bias   = ut_input_v(n, dt, UT_INIT_RANDOM);
    U8* output = ut_input_v(m * n, dt, UT_INIT_ZERO);
    U8* output_ref = ut_input_v(m * n, dt, UT_INIT_ZERO);
    // setup tmp
    U32 tmpBytes;
    CHECK_STATUS(fully_connected_infer_forward_tmp_bytes(T_i, T_w, &tmpBytes, UT_ARCH));
    U8 *tmp     = ut_input_v(tmpBytes/bytesOf(dt), dt, UT_INIT_ZERO);
    // setup filter trans
    U32 ftmBytes;
    CHECK_STATUS(fully_connected_transform_filter_bytes(T_w, &ftmBytes, UT_ARCH));
    U8 *ftm     = ut_input_v(ftmBytes/bytesOf(dt), dt, UT_INIT_ZERO);
    // trans filter
    TensorDesc ftmDesc;
    CHECK_STATUS(fully_connected_transform_filter(T_i, T_w, weight, &ftmDesc, ftm, UT_ARCH));

    U32 bytes = 0;
    if (UT_CHECK) {
        CHECK_STATUS(fully_connected(T_i, input, ftmDesc, ftm, tmp, bytes, T_o, output, biasDesc, bias, UT_ARCH));

        // naive implement
        CHECK_STATUS(fully_connected(T_i, input, ftmDesc, ftm, tmp, bytes, T_o, output_ref, biasDesc, bias, CPU_GENERAL));

        // check
        ut_check_v(output, output_ref, m*n, dt, 1, __FILE__, __LINE__);
    }
    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(fully_connected(T_i, input, ftmDesc, ftm, tmp, bytes, T_o, output, biasDesc, bias, UT_ARCH));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u)+(%u %u)=(%u %u)",
                    m, k, k, n, m, n);
    sprintf(buffer, "%20s, %80s", "InnerProduct", params);
    double ops = 2.0 * m * n * k + 1.0 * m * n;
    ut_log(dt, buffer, ops, time);

    free(input);
    free(bias);
    free(output);
    free(output_ref);
    free(tmp);
    free(ftm);

    return 0;
}


int main(int argc, char** argv) {
#ifdef _USE_FP16
    fullyConnectedTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    fullyConnectedTest(argc, argv, DT_F32);
#endif
    return 0;
}
