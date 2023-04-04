// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "blas_enhance.h"
#include "ut_util.h"

void mmmTestKernel(U32 m,
    U32 k,
    U32 n,
    DataType adt,
    DataType bdt,
    DataType odt,
    bool transform,
    bool at,
    bool bt,
    bool log = true)
{
    float threshold = 0.0001;
    if (odt == DT_F16) {
        threshold = 1;
    }

    TensorDesc A_desc, B_desc;
    if (at) {
        A_desc = tensor2df(adt, DF_TRANSPOSE, k, m);
    } else {
        A_desc = tensor2df(adt, DF_NORMAL, m, k);
    }
    TensorDesc A_ref_desc = A_desc;
    if (bt) {
        B_desc = tensor2df(bdt, DF_TRANSPOSE, n, k);
    } else {
        B_desc = tensor2df(bdt, DF_NORMAL, k, n);
    }
    TensorDesc C_desc = tensor2df(odt, DF_NORMAL, m, n);

    U8 *A = ut_input_v(m * k, adt, UT_INIT_RANDOM);
    U8 *A_ref = ut_input_v(m * k, adt, UT_INIT_ZERO);
    UNI_MEMCPY(A_ref, A, m * k * bytesOf(adt));
    U8 *B = ut_input_v(k * n, bdt, UT_INIT_RANDOM);
    U8 *C = ut_input_v(m * n, odt, UT_INIT_RANDOM);
    U8 *C_ref = ut_input_v(m * n, odt, UT_INIT_ZERO);
    UNI_MEMCPY(C_ref, C, m * n * bytesOf(odt));

#ifdef _USE_X86
    if (adt == DT_U8_Q) {
        A_ref_desc.dt = DT_I8;
        UINT8 *p = (UINT8 *)A;
        for (U32 i = 0; i < m * k; ++i) {
            p[i] = (UINT8)((I32)A[i] + 128);
        }
    }
#endif

    TensorDesc trans_desc = B_desc;
    U8 *mat_trans = B;
    U32 mat_trans_bytes = 0;
    U32 offset = 0;
    if (transform) {
        CHECK_STATUS(
            matrix_matrix_multiply_transform_rhs_bytes(B_desc, &mat_trans_bytes, &offset, UT_ARCH));
        mat_trans = ut_input_v(mat_trans_bytes, DT_I8, UT_INIT_ZERO);
        CHECK_STATUS(
            matrix_matrix_multiply_transform_rhs(B_desc, B, &trans_desc, mat_trans, UT_ARCH));
    }

    U32 bytes = 0;
    CHECK_STATUS(matrix_matrix_multiply_tmp_bytes(A_desc, trans_desc, &bytes, UT_ARCH));
    U8 *tmp = ut_input_v(bytes, DT_I8, UT_INIT_ZERO);

#ifdef _USE_X86
    if (adt == DT_U8_Q || bdt == DT_U8_Q) {
        if (transform) {
            UNI_MEMCPY(tmp, mat_trans + offset, n * bytesOf(DT_I32));
        }
    }
#endif

    if (UT_CHECK) {
        CHECK_STATUS(matrix_matrix_multiply(
            A_desc, A, trans_desc, mat_trans, bytes, tmp, C_desc, C, nullptr, UT_ARCH));

        // naive implement
        CHECK_STATUS(matrix_matrix_multiply(
            A_ref_desc, A_ref, B_desc, B, bytes, tmp, C_desc, C_ref, nullptr, CPU_GENERAL));

        // check
        ut_check_v(C, C_ref, m * n, odt, threshold);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(matrix_matrix_multiply(
            A_desc, A, trans_desc, mat_trans, bytes, tmp, C_desc, C, nullptr, UT_ARCH));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    if (log) {
        char buffer[150];
        char params[120];
        char NT[2] = {'N', 'T'};
        const char *trans[2] = {"", " transform"};
        sprintf(params, "%c(%u %u)+%c(%u %u)=(%u %u)%s", NT[at], m, k, NT[bt], k, n, m, n,
            trans[transform]);
        sprintf(buffer, "%20s, %80s", "MatrixMultiply", params);
        double ops = 2.0 * m * n * k + 1.0 * m * n;
        ut_log(bdt, buffer, ops, time);
    }
    free(A);
    free(A_ref);
    free(B);
    if (transform) {
        free(mat_trans);
    }
    free(C);
    free(C_ref);
    free(tmp);
}

void mmmTest(U32 m, U32 k, U32 n, bool log = true)
{
    for (int transform = 0; transform <= 1; transform++) {
        for (int at = 0; at <= 1; at++) {
            for (int bt = 0; bt <= 1; bt++) {
#ifdef _USE_INT8
#ifdef _USE_X86
                mmmTestKernel(m, k, n, DT_U8_Q, DT_I8, DT_I32, transform, at, bt, log);
#else
                mmmTestKernel(m, k, n, DT_I8, DT_I8, DT_I32, transform, at, bt, log);
#endif
#endif
#ifdef _USE_FP16
                mmmTestKernel(m, k, n, DT_F16, DT_F16, DT_F16, transform, at, bt, log);
#endif
#ifdef _USE_FP32
                mmmTestKernel(m, k, n, DT_F32, DT_F32, DT_F32, transform, at, bt, log);
#endif
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc == 4) {
        U32 m = atoi(argv[1]);
        U32 k = atoi(argv[2]);
        U32 n = atoi(argv[3]);
        mmmTest(m, k, n);
    } else {
        UNI_INFO_LOG("running matrix matrix multiply cover test...\n");
        for (U32 m = 1; m <= 33; ++m) {
            for (U32 k = 1; k <= 33; ++k) {
                for (U32 n = 1; n <= 33; ++n) {
                    mmmTest(m, k, n, false);
                }
            }
        }
    }
    return 0;
}
