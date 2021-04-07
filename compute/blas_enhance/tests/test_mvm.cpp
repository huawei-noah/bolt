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

// #define COVERTEST

int mvmTestKernel(
    U32 m, U32 k, DataType dt, bool weightTransformed = false, bool matrixTransposed = false)
{
    float threshold = 0.0001;
    if (dt == DT_F16) {
        threshold = 0.3;  // 1024x1024 within 0.08; 2048x2048 has a wider gap
    }
    DataFormat df;
    U32 vc, rc;
    if (matrixTransposed) {
        df = DF_TRANSPOSE;
        vc = m;
        rc = k;
    } else {
        df = DF_NORMAL;
        vc = k;
        rc = m;
    }

    TensorDesc mat_desc = tensor2df(dt, df, rc, vc);
    TensorDesc tranDesc = mat_desc;
    TensorDesc vec_desc = tensor1d(dt, k);
    TensorDesc res_desc = tensor1d(dt, m);

    U8 *mat = ut_input_v(m * k, dt, UT_INIT_RANDOM);
    U8 *matTran = mat;
    U8 *vec = ut_input_v(k, dt, UT_INIT_RANDOM);
    U8 *res = ut_input_v(m, dt, UT_INIT_ZERO);
    U8 *res_ref = ut_input_v(m, dt, UT_INIT_ZERO);

    U32 bytes = 0;
    CHECK_STATUS(matrix_vector_multiply_tmp_bytes(mat_desc, vec_desc, &bytes, UT_ARCH));
    U8 *tmp = ut_input_v(bytes / bytesOf(dt), dt, UT_INIT_ZERO);

    if (weightTransformed) {
        matTran = ut_input_v(m * k + 64, dt, UT_INIT_RANDOM);
        matrix_vector_multiply_transform_weight(mat_desc, mat, &tranDesc, matTran, UT_ARCH);
    }

    // check
    if (UT_CHECK) {
        CHECK_STATUS(matrix_vector_multiply(
            tranDesc, matTran, vec_desc, vec, bytes, tmp, res_desc, res, UT_ARCH));

        // naive implement
        CHECK_STATUS(matrix_vector_multiply(
            mat_desc, mat, vec_desc, vec, bytes, tmp, res_desc, res_ref, CPU_GENERAL));

        ut_check_v(res, res_ref, m, dt, threshold, __FILE__, __LINE__);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        matrix_vector_multiply(tranDesc, matTran, vec_desc, vec, bytes, tmp, res_desc, res, UT_ARCH);
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u)+(%u)=(%u), weightTransformed=(%d), matrixTransposed=(%d)", m, k, vc,
        rc, weightTransformed, matrixTransposed);
    sprintf(buffer, "%20s, %80s", "MatrixVectorMultiply", params);
    double ops = 2.0 * m * k;
    ut_log(dt, buffer, ops, time);

    free(mat);
    if (weightTransformed) {
        free(matTran);
    }
    free(vec);
    free(tmp);
    free(res);
    free(res_ref);

    return 0;
}

int mvmTest(
    int argc, char **argv, DataType dt, bool weightTransformed = false, bool matrixTransposed = false)
{
#ifndef COVERTEST
    CHECK_REQUIREMENT(argc == 3);
    U32 m = atoi(argv[1]);
    U32 k = atoi(argv[2]);
    return mvmTestKernel(m, k, dt, weightTransformed, matrixTransposed);
#else
    U32 mmin = 1, mmax = 100;
    U32 kmin = 1, kmax = 100;
    int ret = 0;
    for (U32 m = mmin; m <= mmax; ++m) {
        for (U32 k = kmin; k <= kmax; ++k) {
            if (ret = mvmTestKernel(m, k, dt, weightTransformed, matrixTransposed)) {
                return ret;
            }
        }
    }
    return 0;
#endif
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    mvmTest(argc, argv, DT_F16, true, true);
    mvmTest(argc, argv, DT_F16, true, false);
    mvmTest(argc, argv, DT_F16, false, true);
    mvmTest(argc, argv, DT_F16, false, false);
#endif
#ifdef _USE_FP32
    mvmTest(argc, argv, DT_F32, true, false);
    mvmTest(argc, argv, DT_F32, false, false);
    mvmTest(argc, argv, DT_F32, true, true);
    mvmTest(argc, argv, DT_F32, false, true);
#endif
    return 0;
}
