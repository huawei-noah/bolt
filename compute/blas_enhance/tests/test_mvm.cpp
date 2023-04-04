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

void mvmTestKernel(U32 m,
    U32 k,
    DataType mdt,
    DataType vdt,
    DataType odt,
    bool transform = false,
    bool transpose = false,
    bool log = true)
{
    float threshold = 0.0001;
    // 1024x1024 within 0.08; 2048x2048 has a wider gap
    if (vdt == DT_F16) {
        threshold = 0.3;
    }
    DataFormat df;
    TensorDesc mat_desc;
    if (transpose) {
        mat_desc = tensor2df(mdt, DF_TRANSPOSE, k, m);
    } else {
        mat_desc = tensor2df(mdt, DF_NORMAL, m, k);
    }
    TensorDesc vec_desc = tensor1d(vdt, k);
    TensorDesc res_desc = tensor1d(odt, m);
    TensorDesc mat_ref_desc = mat_desc;
    TensorDesc vec_ref_desc = vec_desc;

    U8 *mat = ut_input_v(m * k, mat_ref_desc.dt, UT_INIT_RANDOM);
    U8 *mat_ref = ut_input_v(m * k, mat_ref_desc.dt, UT_INIT_ZERO);
    UNI_MEMCPY(mat_ref, mat, m * k * bytesOf(mat_ref_desc.dt));
    U8 *vec = ut_input_v(k, vec_ref_desc.dt, UT_INIT_RANDOM);
    U8 *vec_ref = ut_input_v(k, vec_ref_desc.dt, UT_INIT_ZERO);
    UNI_MEMCPY(vec_ref, vec, k * bytesOf(vec_ref_desc.dt));
    U8 *res = ut_input_v(m, odt, UT_INIT_ZERO);
    U8 *res_ref = ut_input_v(m, odt, UT_INIT_ZERO);
    UNI_MEMCPY(res_ref, res, m * bytesOf(odt));
#ifdef _USE_X86
    if (mdt == DT_U8_Q) {
        mat_ref_desc.dt = DT_I8;
        UINT8 *p = (UINT8 *)mat;
        for (U32 i = 0; i < m * k; ++i) {
            p[i] = (UINT8)((I32)mat[i] + 128);
        }
    }
    if (vdt == DT_U8_Q) {
        vec_ref_desc.dt = DT_I8;
        UINT8 *p = (UINT8 *)vec;
        for (U32 i = 0; i < k; ++i) {
            p[i] = (UINT8)((I32)vec[i] + 128);
        }
    }
#endif

    TensorDesc trans_desc = mat_desc;
    U8 *mat_trans = mat;
    if (transform) {
        U32 bytes = 0;
        CHECK_STATUS(matrix_vector_multiply_transform_weight_bytes(mat_desc, &bytes, UT_ARCH));
        mat_trans = ut_input_v(bytes, DT_I8, UT_INIT_ZERO);
        CHECK_STATUS(
            matrix_vector_multiply_transform_weight(mat_desc, mat, &trans_desc, mat_trans, UT_ARCH));
    }

    U32 bytes = 0;
    CHECK_STATUS(matrix_vector_multiply_tmp_bytes(trans_desc, vec_desc, &bytes, UT_ARCH));
    U8 *tmp = ut_input_v(bytes, DT_I8, UT_INIT_ZERO);

#ifdef _USE_X86
    if (transform && (vdt == DT_U8_Q)) {
        UNI_MEMCPY(tmp, mat_trans + UNI_ALIGN(m, 16) * UNI_ALIGN(k, 8), m * bytesOf(DT_I32));
    }
#endif

    // check
    if (UT_CHECK) {
        CHECK_STATUS(matrix_vector_multiply(
            trans_desc, mat_trans, vec_desc, vec, bytes, tmp, res_desc, res, nullptr, UT_ARCH));

        // naive implement
        CHECK_STATUS(matrix_vector_multiply(mat_ref_desc, mat_ref, vec_ref_desc, vec_ref, bytes,
            tmp, res_desc, res_ref, nullptr, CPU_GENERAL));

        ut_check_v(res, res_ref, m, odt, threshold);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        matrix_vector_multiply(
            trans_desc, mat_trans, vec_desc, vec, bytes, tmp, res_desc, res, nullptr, UT_ARCH);
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    if (log) {
        char buffer[150];
        char params[120];
        char NT[2] = {'N', 'T'};
        const char *trans[2] = {"", " transform"};
        sprintf(params, "%c(%u %u)+(%u)=(%u)%s", NT[transpose], m, k, k, m, trans[transform]);
        sprintf(buffer, "%20s, %80s", "MatrixVectorMultiply", params);
        double ops = 2.0 * m * k;
        ut_log(vdt, buffer, ops, time);
    }

    free(mat);
    free(mat_ref);
    if (transform) {
        free(mat_trans);
    }
    free(vec);
    free(vec_ref);
    free(tmp);
    free(res);
    free(res_ref);
}

void mvmTest(U32 m, U32 k, bool log = true)
{
    for (int transform = 0; transform <= 1; transform++) {
        for (int transpose = 0; transpose <= 1; transpose++) {
#ifdef _USE_INT8
#ifdef _USE_X86
            if (!transform) {
                mvmTestKernel(m, k, DT_U8_Q, DT_I8, DT_I32, transform, transpose, log);
            }
            mvmTestKernel(m, k, DT_I8, DT_U8_Q, DT_I32, transform, transpose, log);
#else
            mvmTestKernel(m, k, DT_I8, DT_I8, DT_I32, transform, transpose, log);
#endif
#endif
#ifdef _USE_FP16
            mvmTestKernel(m, k, DT_F16, DT_F16, DT_F16, transform, transpose, log);
#endif
#ifdef _USE_FP32
            mvmTestKernel(m, k, DT_F32, DT_F32, DT_F32, transform, transpose, log);
#endif
        }
    }
}

int main(int argc, char **argv)
{
    if (argc == 3) {
        U32 m = atoi(argv[1]);
        U32 k = atoi(argv[2]);
        mvmTest(m, k);
    } else {
        UNI_INFO_LOG("running matrix vector multiply cover test...\n");
        for (U32 m = 1; m <= 65; ++m) {
            for (U32 k = 1; k <= 65; ++k) {
                mvmTest(m, k, false);
            }
        }
    }
    return 0;
}
