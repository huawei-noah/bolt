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
#include "blas_enhance.h"
#include "ut_util.h"

int axpbyTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 4);
    U32 len = atoi(argv[1]);
    F32 a = atof(argv[2]);
    F32 b = atof(argv[3]);

    TensorDesc xDesc = tensor1d(dt, len);
    TensorDesc yDesc = tensor1d(dt, len);

    U8 *x = ut_input_v(len, dt, UT_INIT_RANDOM);
    U8 *y = ut_input_v(len, dt, UT_INIT_RANDOM);
    U8 *y_ref = ut_input_v(len, dt, UT_INIT_ZERO);

    memcpy(y_ref, y, tensorNumBytes(yDesc));
    // check
    if (UT_CHECK) {
        CHECK_STATUS(vector_vector_axpby(a, xDesc, x, b, yDesc, y, UT_ARCH));

        // naive implement
        CHECK_STATUS(vector_vector_axpby(a, xDesc, x, b, yDesc, y_ref, CPU_GENERAL));

        ut_check_v(y, y_ref, len, dt, 0.01, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        vector_vector_axpby(a, xDesc, x, b, yDesc, y, UT_ARCH);
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%.2f * %u) + (%.2f * %u) = (%u)", a, len, b, len, len);
    sprintf(buffer, "%20s, %80s", "VectorVectoraXpbY", params);
    double ops = 3.0 * len;
    ut_log(dt, buffer, ops, time);

    free(x);
    free(y);
    free(y_ref);

    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    axpbyTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    axpbyTest(argc, argv, DT_F32);
#endif
    return 0;
}
