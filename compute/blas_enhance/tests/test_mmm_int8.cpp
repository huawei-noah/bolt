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
#include "thread_affinity.h"

//#define COVER_TEST

int testMMM(U32 m, U32 k, U32 n)
{
#ifdef _USE_INT8
    DataType dt = DT_I8;
    DataType odt = DT_I32;
    TensorDesc A_desc = tensor2df(dt, DF_NORMAL, m, k);
    TensorDesc B_desc = tensor2df(dt, DF_NORMAL, k, n);
    TensorDesc tranDescB;
    TensorDesc C_desc = tensor2df(odt, DF_NORMAL, m, n);

    U32 bytes = 0;
    U32 k8 = k;
    U32 n8 = n;
    if (k8 % 8 != 0) {
        k8 = (k8 / 8) * 8 + 8;
    }
    if (n8 % 16 != 0) {
        n8 = (n8 / 16) * 16 + 16;
    }
    INT8 *A = (INT8 *)ut_input_v(m * k, DT_I8, UT_INIT_RANDOM);
    INT8 *A_ref = (INT8 *)ut_input_v(m * k, DT_I8, UT_INIT_RANDOM);
    UNI_MEMCPY(A_ref, A, m * k);
    INT8 *B = (INT8 *)ut_input_v(k * n, DT_I8, UT_INIT_RANDOM);
    INT8 *B_tran = (INT8 *)ut_input_v(k8 * n8 + 64 + n8 * 4, DT_I8, UT_INIT_ZERO);
    I32 *C = (I32 *)ut_input_v(m * n, DT_I32, UT_INIT_ZERO);
    I32 *C_ref = (I32 *)ut_input_v(m * n, DT_I32, UT_INIT_ZERO);
    CHECK_STATUS(matrix_matrix_multiply_tmp_bytes(A_desc, B_desc, &bytes, UT_ARCH));
    bytes += m * n;
    INT8 *tmp = (INT8 *)ut_input_v(bytes, DT_I8, UT_INIT_ZERO);

    matrix_matrix_multiply_transform_rhs(B_desc, B, &tranDescB, B_tran, UT_ARCH);

#ifdef _USE_X86
    UINT8 *uA = (UINT8 *)A;
    for (U32 i = 0; i < m * k; ++i) {
        uA[i] = (UINT8)((I32)A[i] + 128);
    }
    UNI_MEMCPY(tmp, B_tran + n8 * k8, n * bytesOf(DT_I32));
#endif

    if (UT_CHECK) {
        CHECK_STATUS(
            matrix_matrix_multiply(A_desc, A, tranDescB, B_tran, bytes, tmp, C_desc, C, nullptr, UT_ARCH));

        CHECK_STATUS(
            matrix_matrix_multiply(A_desc, A_ref, B_desc, B, bytes, tmp, C_desc, C_ref, nullptr, CPU_GENERAL));

        // check
        ut_check_v(C, C_ref, m * n, DT_I32, 1, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        matrix_matrix_multiply(A_desc, A, tranDescB, B_tran, bytes, tmp, C_desc, C, nullptr, UT_ARCH);
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u)+(%u %u)=(%u %u)", m, k, k, n, m, n);
    sprintf(buffer, "%20s, %80s", "MatrixMultiply", params);
    double ops = 2.0 * m * n * k + 1.0 * m * n;
    ut_log(DT_I8, buffer, ops, time);

    free(A);
    free(B);
    free(B_tran);
    free(C);
    free(C_ref);
    free(tmp);
#endif
    return 0;
}

int main(int argc, char **argv)
{
#ifdef COVER_TEST
    int ret = 0;
    for (U32 m = 1; m < 48; ++m) {
        for (U32 k = 1; k < 48; ++k) {
            for (U32 n = 1; n < 48; ++n) {
                ret = testMMM(m, k, n);
            }
        }
    }
    return ret;
#else
    CHECK_REQUIREMENT(argc == 4);
    return testMMM(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
#endif
}
