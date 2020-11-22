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

int main(int argc, char **argv)
{
#ifdef _USE_INT8
    CHECK_REQUIREMENT(argc == 3);
    U32 m = atoi(argv[1]);
    U32 k = atoi(argv[2]);

    DataFormat df = DF_NORMAL;
    DataType dt = DT_I8;
    DataType odt = DT_I32;
    U32 vc, rc;
    if (df == DF_NORMAL) {
        vc = k;
        rc = m;
    } else {
        vc = m;
        rc = k;
    }

    TensorDesc mat_desc = tensor2df(dt, df, m, k);
    TensorDesc tranDesc;
    TensorDesc vec_desc = tensor1d(dt, vc);
    TensorDesc res_desc = tensor1d(odt, rc);

    U32 k4 = k;
    if (k4 % 4 != 0) {
        k4 = (k4 / 4) * 4 + 4;
    }

    INT8 *mat = (INT8 *)ut_input_v(m * k, DT_I8, UT_INIT_RANDOM);
    INT8 *matTran = (INT8 *)ut_input_v(m * k4, DT_I8, UT_INIT_ZERO);
    INT8 *vec = (INT8 *)ut_input_v(vc, DT_I8, UT_INIT_RANDOM);
    I32 *res = (I32 *)ut_input_v(rc, DT_I32, UT_INIT_ZERO);
    I32 *res_ref = (I32 *)ut_input_v(rc, DT_I32, UT_INIT_ZERO);

    matrix_vector_multiply_transform_weight(mat_desc, mat, &tranDesc, matTran, UT_ARCH);

    U32 bytes;
    CHECK_STATUS(matrix_vector_multiply_tmp_bytes(mat_desc, vec_desc, &bytes, UT_ARCH));
    I32 *tmp = (I32 *)ut_input_v(bytes / bytesOf(DT_I32), DT_I32, UT_INIT_ZERO);
    // check
    if (UT_CHECK) {
        CHECK_STATUS(matrix_vector_multiply(
            tranDesc, matTran, vec_desc, vec, bytes, tmp, res_desc, res, UT_ARCH));

        // naive implement
        CHECK_STATUS(matrix_vector_multiply(
            mat_desc, mat, vec_desc, vec, bytes, tmp, res_desc, res_ref, CPU_GENERAL));

        ut_check_v(res, res_ref, rc, DT_I32, 1, __FILE__, __LINE__);
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
    sprintf(params, "(%u %u)+(%u)=(%u)", m, k, vc, rc);
    sprintf(buffer, "%20s, %80s", "MatrixVectorMultiply", params);
    double ops = 2.0 * m * k;
    ut_log(DT_I8, buffer, ops, time);

    free(mat);
    free(vec);
    free(res);
    free(res_ref);
#endif
    return 0;
}
