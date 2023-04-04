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

int MatmulTest(int argc, char **argv, DataType dt, DataType filterDataType)
{
    CHECK_REQUIREMENT(argc == 4);
    U32 m = atoi(argv[1]);
    U32 k = atoi(argv[2]);
    U32 n = atoi(argv[3]);

    TensorDesc matrixADesc = tensor2df(DT_I8, DF_NORMAL, m, k);
    TensorDesc matrixADescRef = tensor2df(DT_I8, DF_NORMAL, m, k);
    TensorDesc matrixBDesc = tensor2df(DT_I8, DF_TRANSPOSE, n, k);
    TensorDesc matrixBDescRef = tensor2df(DT_I8, DF_TRANSPOSE, n, k);

    Tensor matrixATensor, matrixATensorRef, matrixBTensor, matrixBTensorRef;
    matrixATensor.resize(matrixADesc);
    matrixATensor.alloc();
    matrixATensorRef.resize(matrixADescRef);
    matrixATensorRef.alloc();
    matrixBTensor.resize(matrixBDesc);
    matrixBTensor.alloc();
    matrixBTensorRef.resize(matrixBDescRef);
    matrixBTensorRef.alloc();
    U8 *A = ut_input_v(m * k, DT_I8, UT_INIT_RANDOM);
    U8 *ARef = ut_input_v(m * k, DT_I8, UT_INIT_RANDOM);
    matrixATensor.set_scale(1);
    matrixATensorRef.set_scale(1);
    U8 *B = ut_input_v(n * k, DT_I8, UT_INIT_RANDOM);
    U8 *BRef = ut_input_v(n * k, DT_I8, UT_INIT_RANDOM);
    if (CPU_GENERAL == X86_AVX512) {
        matrixADesc.dt = DT_U8_Q;
        matrixBDesc.dt = DT_U8_Q;
        matrixBTensor.resize(matrixBDesc);
        matrixATensor.resize(matrixADesc);
        for (U32 i = 0; i < n * k; ++i) {
            B[i] = BRef[i] + 128;
        }
        for (U32 i = 0; i < m * k; ++i) {
            A[i] = ARef[i] + 128;
        }
    }
    matrixBTensor.set_scale(1);
    matrixBTensorRef.set_scale(1);
    UNI_MEMCPY(get_ptr_from_tensor(matrixATensor, CPU_GENERAL), A, tensorNumBytes(matrixADesc));
    UNI_MEMCPY(
        get_ptr_from_tensor(matrixATensorRef, CPU_GENERAL), ARef, tensorNumBytes(matrixADescRef));
    UNI_MEMCPY(get_ptr_from_tensor(matrixBTensor, CPU_GENERAL), B, tensorNumBytes(matrixBDesc));
    UNI_MEMCPY(
        get_ptr_from_tensor(matrixBTensorRef, CPU_GENERAL), BRef, tensorNumBytes(matrixBDescRef));

    bool transposeA = (matrixADesc.df == DF_TRANSPOSE);
    bool transposeB = (matrixBDesc.df == DF_TRANSPOSE);

    // set output
    Tensor matrixCTensor, matrixCTensorRef;
    CHECK_STATUS(matmul_infer_output_size(
        &matrixATensor, transposeA, &matrixBTensor, transposeB, &matrixCTensor, &UT_CPU_ARCHINFO));
    TensorDesc matrixCDesc = matrixCTensor.get_desc();
    matrixCDesc.dt = DT_F32;
    matrixCTensor.resize(matrixCDesc);
    matrixCTensor.alloc();
    matrixCTensorRef.resize(matrixCDesc);
    matrixCTensorRef.alloc();

    // setup tmp
    Tensor tmpTensor;
    U32 tmpBytes;
    CHECK_STATUS(matmul_infer_forward_tmp_bytes(matrixATensor, transposeA, matrixBTensor,
        transposeB, matrixCTensor, &tmpBytes, &UT_CPU_ARCHINFO));
    tmpTensor.resize(tensor1d(DT_U8, tmpBytes));
    tmpTensor.alloc();
    std::vector<Tensor> tmpTensors(1, tmpTensor);

    Tensor biasTensor;
    if (UT_CHECK) {
        CHECK_STATUS(matmul(matrixATensor, transposeA, matrixBTensor, transposeB, biasTensor,
            tmpTensors, matrixCTensor, &UT_CPU_ARCHINFO));

        // naive implement
        CHECK_STATUS(matmul(matrixATensorRef, transposeA, matrixBTensorRef, transposeB, biasTensor,
            tmpTensors, matrixCTensorRef, &UT_SERIAL_ARCHINFO));

        // check
        ut_check_v(get_ptr_from_tensor(matrixCTensor, CPU_GENERAL),
            get_ptr_from_tensor(matrixCTensorRef, CPU_GENERAL), m * n, dt, 0.0);
    }

    // benchmark
    double time_start = ut_time_ms();
    for (int iter = 0; iter < UT_LOOPS; iter++) {
        CHECK_STATUS(matmul(matrixATensor, transposeA, matrixBTensor, transposeB, biasTensor,
            tmpTensors, matrixCTensor, &UT_CPU_ARCHINFO));
    }
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;

    // log performance data
    char buffer[150];
    char params[120];
    sprintf(params, "(%u %u)+(%u %u)=(%u %u)", m, k, k, n, m, n);
    sprintf(buffer, "%20s, %80s", "MatMul", params);
    double ops = 2.0 * m * n * k + 1.0 * m * n;
    ut_log(dt, buffer, ops, time);

    free(A);
    free(B);
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_INT8
#ifdef _USE_FP16
    MatmulTest(argc, argv, DT_F32, DT_F32);
#else
    MatmulTest(argc, argv, DT_F16, DT_F16);
#endif
#endif
    return 0;
}
