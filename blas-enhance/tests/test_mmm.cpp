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
#include "blas-enhance.h"
#include "utils.h"

int main(int argc, char** argv) {
    assert(argc == 4);
    U32 m = atoi(argv[1]);
    U32 k = atoi(argv[2]);
    U32 n = atoi(argv[3]);

    DataType dt = DT_F16;
    TensorDesc A_desc = tensor2df(dt, DF_NORMAL, m, k);
    TensorDesc B_desc = tensor2df(dt, DF_NORMAL, k, n);
    TensorDesc C_desc = tensor2df(dt, DF_NORMAL, m, n);

    U32 bytes = 0;
    F16* A = ut_input_v<F16>(m * k, UT_INIT_RANDOM);
    F16* B = ut_input_v<F16>(k * n, UT_INIT_RANDOM);
    F16* C = ut_input_v<F16>(m * n, UT_INIT_ZERO);
    F16* C_ref = ut_input_v<F16>(m * n, UT_INIT_ZERO);
    CHECK_STATUS(matrix_matrix_multiply_tmp_bytes(A_desc, B_desc, &bytes, UT_ARCH));
    F16* tmp = (F16 *)malloc(bytes);
    memset(tmp, 0, bytes);

    if (UT_CHECK) {
        CHECK_STATUS(matrix_matrix_multiply(A_desc, A, B_desc, B, bytes, tmp, C_desc, C, UT_ARCH));

        // naive implement
        CHECK_STATUS(matrix_matrix_multiply(A_desc, A, B_desc, B, bytes, tmp, C_desc, C_ref, CPU_GENERAL));

        // check
        ut_check_v<F16>(C, C_ref, m*n, F16(10), __FILE__, __LINE__);
    }

    // benchmark 
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        matrix_matrix_multiply(A_desc, A, B_desc, B, bytes, tmp, C_desc, C, UT_ARCH);
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u)+(%u %u)=(%u %u)",
                    m, k, k, n, m, n);
    sprintf(buffer, "%20s, %80s", "MatrixMultiply", params);
    double ops = 2.0 * m * n * k + 1.0 * m * n;
    ut_log<F16>(buffer, ops, time);

    free(A);
    free(B);
    free(C);
    free(C_ref);
    free(tmp);
  
    return 0;
}
