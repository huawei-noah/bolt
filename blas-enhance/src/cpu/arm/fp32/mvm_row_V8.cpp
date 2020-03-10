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
#include "blas_fp32.h"


void mvm_row_tail(U32 N, U32 K, F32* matrix, F32* vector, F32* result)
{
    float32x4_t vec, res, mat;
    U32 KTail = K % 4;
    U32 KInner = K - KTail;
    
    for (U32 i = 0; i < N; i++) {
        res = vdupq_n_f32(0);

        for (U32 j = 0; j < KInner; j += 4) {
            vec = vld1q_f32(&vector[j]);
            mat = vld1q_f32(&matrix[j + K * i]);
            res = vfmaq_f32(res, vec, mat);
        }
        result[i] += vaddvq_f32(res);
        
        if (KTail != 0) {
            for (U32 p = 0; p < KTail; p++) {
                result[i] += vector[p + KInner] * matrix[KInner + p + K * i];
            }
        }
        
    }
}

void mvm_row_kernel(U32 N, U32 K, F32* matrix, F32* vector, F32* result)
{
    I32 KTail = K % 4;
    I32 KInner = K - KTail;
    F32* w0 = matrix;
    F32* w1 = matrix + K * N;
    F32* w2 = matrix + K * 2 * N;
    F32* w3 = matrix + K * 3 * N;
    asm volatile(
        "mov x19, %5\n"
        "ld1 {v18.s}[0], [x19]\n"
        "add x19, x19, %8\n"
        "ld1 {v18.s}[1], [x19]\n"
        "add x19, x19, %8\n"
        "ld1 {v18.s}[2], [x19]\n"
        "add x19, x19, %8\n"
        "ld1 {v18.s}[3], [x19]\n"

        "movi v17.4s, #0x0\n"
        "movi v16.4s, #0x0\n"
        "movi v9.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "movi v11.4s, #0x0\n"
        "movi v12.4s, #0x0\n"
        "mov x20, %6\n"
        "cmp x20, #0x0\n"
        "beq 3f\n"
        "0:\n"

        "ld1 {v0.4s}, [%0], #16\n"
        "ld1 {v1.4s}, [%1], #16\n"
        "ld1 {v2.4s}, [%2], #16\n"
        "ld1 {v3.4s}, [%3], #16\n"
        "ld1 {v4.4s}, [%4], #16\n"

        "fmla v9.4s,  v1.4s, v0.4s\n"
        "fmla v10.4s,  v2.4s, v0.4s\n"
        "fmla v11.4s,  v3.4s, v0.4s\n"
        "fmla v12.4s,  v4.4s, v0.4s\n"

        "subs x20, x20, #4\n"
        "bne 0b\n"

        "faddp v13.4s,  v9.4s, v10.4s\n"
        "faddp v14.4s, v11.4s, v12.4s\n"
        "faddp v17.4s, v13.4s, v14.4s\n"
        "3:\n"
        "mov x16, %7\n"
        "cmp x16, #0x0\n"
        "beq 2f\n"

        "1:\n"
        "ld1 {v8.s}[0], [%0], #4\n"

        "ld1 {v1.s}[0], [%1], #4\n"
        "ld1 {v1.s}[1], [%2], #4\n"
        "ld1 {v1.s}[2], [%3], #4\n"
        "ld1 {v1.s}[3], [%4], #4\n"
        "fmla v16.4s,  v1.4s, v8.s[0]\n"

        "subs x16, x16, 0x1\n"
        "bne 1b\n"

        "fadd v17.4s, v17.4s, v16.4s\n"

        "2:\n"

        "fadd v17.4s, v17.4s, v18.4s\n"
        "mov x19, %5\n"
        "st1 {v17.s}[0], [x19]\n"
        "add x19, x19, %8\n"
        "st1 {v17.s}[1], [x19]\n"
        "add x19, x19, %8\n"
        "st1 {v17.s}[2], [x19]\n"
        "add x19, x19, %8\n"
        "st1 {v17.s}[3], [x19]\n"
        :"+r" (vector),
         "+r" (w0),
         "+r" (w1),
         "+r" (w2),
         "+r" (w3),
         "+r" (result)
        :"r" ((I64)KInner),
         "r" ((I64)KTail),
         "r" ((I64)N*4)
        :"memory", "cc", "x19", "x20", "x21", "x22", "x23", "x24", "x15", "x16",
            "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v16", "v17", "v18"
    );
}

void mvm_row_V8(U32 numRows, U32 numColumns, F32* matrix, F32* vector, F32* result)
{
    //Actual layout is NK, and vector is K
    U32 N = numRows;
    U32 K = numColumns;
    U32 NTail = N % 4;
    U32 NInner = N / 4;
    for (U32 i = 0; i < NInner; i++) {
        mvm_row_kernel(NInner, K, matrix + i * K, vector, result + i);
    }
    if (NTail != 0) {
        mvm_row_tail(NTail, K, matrix + (N - NTail) * K, vector, result + (N - NTail));
    }
}
