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
#include "ut_util.h"


int mvmTest(int argc, char** argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 3);
    U32 m = atoi(argv[1]);
    U32 k = atoi(argv[2]);

    DataFormat df = DF_NORMAL;
    U32 vc, rc;
    if (df == DF_NORMAL) {
        vc = k;
        rc = m;
    }
    else {
        vc = m;
        rc = k;
    }

    TensorDesc mat_desc = tensor2df(dt, df, m, k);
    TensorDesc vec_desc = tensor1d(dt, vc);
    TensorDesc res_desc = tensor1d(dt, rc);

    U8* mat = ut_input_v(m * k, dt, UT_INIT_RANDOM);
    U8* vec = ut_input_v(vc, dt, UT_INIT_RANDOM);
    U8* res = ut_input_v(rc, dt, UT_INIT_ZERO);
    U8* res_ref = ut_input_v(rc, dt, UT_INIT_ZERO);

    U32 bytes = 0;
    CHECK_STATUS(matrix_vector_multiply_tmp_bytes(mat_desc, vec_desc, &bytes, UT_ARCH));
    U8* tmp = ut_input_v(bytes/bytesOf(dt), dt, UT_INIT_ZERO);
    // check
    if (UT_CHECK) {
        CHECK_STATUS(matrix_vector_multiply(mat_desc, mat, vec_desc, vec, bytes, tmp, res_desc, res, UT_ARCH));

        // naive implement
        CHECK_STATUS(matrix_vector_multiply(mat_desc, mat, vec_desc, vec, bytes, tmp, res_desc, res_ref, CPU_GENERAL));

        ut_check_v(res, res_ref, rc, dt, 1, __FILE__, __LINE__);
    }

    // benchmark 
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        matrix_vector_multiply(mat_desc, mat, vec_desc, vec, bytes, tmp, res_desc, res, UT_ARCH);
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u)+(%u)=(%u)",
                    m, k, vc, rc);
    sprintf(buffer, "%20s, %80s", "MatrixVectorMultiply", params);
    double ops = 2.0 * m * k;
    ut_log(dt, buffer, ops, time);

    free(mat);
    free(vec);
    free(tmp);
    free(res);
    free(res_ref);
  
    return 0;
}

int main(int argc, char** argv)
{
#ifdef _USE_FP16
    mvmTest(argc, argv, DT_F16);
#endif
#ifdef _USE_FP32
    mvmTest(argc, argv, DT_F32);
#endif
    return 0;
}
