// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <arm_neon.h>
#include "type.h"
#include "blas_fp32.h"


inline void mvm_col_tail(U32 N, U32 K, F32* matrix, F32* vector, F32* result)
{
    float32x4_t tmp, res, mat;
    U32 NTail = N % 4;
    U32 NInner = N - NTail;
    
    for (U32 i = 0; i < K; i++) {
        for (U32 j = 0; j < NInner; j += 4) {
            tmp = vld1q_f32(result + j);
            mat = vld1q_f32(&matrix[j + N * i]);
            res = vfmaq_n_f32(tmp, mat, vector[i]);
            vst1q_f32(result + j, res);
        }
        if (NTail != 0) {
            for (U32 p = 0; p < NTail; p++) {
                result[NInner + p] += vector[i] * matrix[NInner + N * i + p];
            }
        }
    }
}

void mvm_col_kernel(U32 N, U32 K, F32* matrix, F32* vector, F32* result)
{
    float32x4_t mat[4] = {0};
   
    F32* w0 = matrix;
    F32* w1 = matrix + K * N;
    F32* w2 = matrix + 2 * K * N;
    F32* w3 = matrix + 3 * K * N;
    
    U32 N_tail = N % 4;
    U32 N_inner = N - N_tail;
    
    for(U32 i = 0; i < K; i++) {
        for(U32 j = 0; j < N_inner; j += 4) {

            float32x4_t res[4] = {0};

            res[3] = vld1q_f32(result + j);
            mat[0] = vld1q_f32(w0);
            mat[1] = vld1q_f32(w1);
            mat[2] = vld1q_f32(w2);
            mat[3] = vld1q_f32(w3);

            res[0] = vfmaq_n_f32(res[3], mat[0], vector[i]);
            res[1] = vfmaq_n_f32(res[0], mat[1], vector[K + i]);
            res[2] = vfmaq_n_f32(res[1], mat[2], vector[2 * K + i]);
            res[3] = vfmaq_n_f32(res[2], mat[3], vector[3 * K + i]);

            w0 += 4;
            w1 += 4;
            w2 += 4;
            w3 += 4;
            vst1q_f32(result + j, res[3]);
         }
         if (N_tail != 0) {
             for(U32 p = 0; p < N_tail; p++) {
                 result[N_inner + p] += vector[i] * *w0++;
                 result[N_inner + p] += vector[i + K] * *w1++;
                 result[N_inner + p] += vector[i + 2 * K] * *w2++;
                 result[N_inner + p] += vector[i + 3 * K] * *w3++;
             }
         }
     }
}

void mvm_col_V8(U32 numRows, U32 numColumns, F32* matrix, F32* vector, F32* result)
{
    //Actual layout is KN, and vector is K
    U32 N = numRows;
    U32 K = numColumns;
    U32 KInner = K / 4;
    U32 KTail = K % 4;
    mvm_col_kernel(N, KInner, matrix, vector, result);
    if (KTail != 0) {
        mvm_col_tail(N, KTail, matrix + (K - KTail) * N, vector + (K - KTail), result);
    }
}
