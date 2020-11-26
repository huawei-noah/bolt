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

#include "mvm_common.h"
#include "mvm.h"

inline void mvm_row_kernel_A76(U32 N, U32 K, F16 *matrix, F16 *vector, F16 *result)
{
    U32 KTail = K % 8;
    U32 KInner = K - KTail;
    F16 *w0 = matrix;
    F16 *w1 = matrix + K * N / 2;
    F16 *w2 = matrix + K * 2 * N / 2;
    F16 *w3 = matrix + K * 3 * N / 2;
    asm volatile("mov x19, %5\n"
                 "ld1 {v18.h}[0], [x19]\n"
                 "add x19, x19, %8\n"
                 "ld1 {v18.h}[1], [x19]\n"
                 "add x19, x19, %8\n"
                 "ld1 {v18.h}[2], [x19]\n"
                 "add x19, x19, %8\n"
                 "ld1 {v18.h}[3], [x19]\n"

                 "movi v17.8h, #0x0\n"
                 "movi v16.8h, #0x0\n"
                 "movi v9.8h, #0x0\n"
                 "movi v10.8h, #0x0\n"
                 "movi v11.8h, #0x0\n"
                 "movi v12.8h, #0x0\n"
                 "mov x20, %6\n"
                 "cmp x20, #0x0\n"
                 "beq 3f\n"
                 "0:\n"

                 "ld1 {v0.8h}, [%0], #16\n"
                 "ld1 {v1.8h}, [%1], #16\n"
                 "ld1 {v2.8h}, [%2], #16\n"
                 "ld1 {v3.8h}, [%3], #16\n"
                 "ld1 {v4.8h}, [%4], #16\n"

                 "fmla v9.8h,  v1.8h, v0.8h\n"
                 "fmla v10.8h,  v2.8h, v0.8h\n"
                 "fmla v11.8h,  v3.8h, v0.8h\n"
                 "fmla v12.8h,  v4.8h, v0.8h\n"

                 "subs x20, x20, 0x8\n"
                 "bne 0b\n"

                 "faddp v13.8h,  v9.8h, v10.8h\n"
                 "faddp v14.8h, v11.8h, v12.8h\n"
                 "faddp v15.8h, v13.8h, v14.8h\n"
                 "faddp v17.8h, v15.8h, v15.8h\n"
                 "3:\n"
                 "mov x16, %7\n"
                 "cmp x16, #0x0\n"
                 "beq 2f\n"

                 "1:\n"
                 "ld1 {v8.h}[0], [%0], #2\n"

                 "ld1 {v1.h}[0], [%1], #2\n"
                 "ld1 {v1.h}[1], [%2], #2\n"
                 "ld1 {v1.h}[2], [%3], #2\n"
                 "ld1 {v1.h}[3], [%4], #2\n"
                 "fmla v16.8h,  v1.8h, v8.h[0]\n"

                 "subs x16, x16, 0x1\n"
                 "bne 1b\n"

                 "fadd v17.8h, v17.8h, v16.8h\n"

                 "2:\n"

                 "fadd v17.8h, v17.8h, v18.8h\n"
                 "mov x19, %5\n"
                 "st1 {v17.h}[0], [x19]\n"
                 "add x19, x19, %8\n"
                 "st1 {v17.h}[1], [x19]\n"
                 "add x19, x19, %8\n"
                 "st1 {v17.h}[2], [x19]\n"
                 "add x19, x19, %8\n"
                 "st1 {v17.h}[3], [x19]\n"
                 : "+r"(vector), "+r"(w0), "+r"(w1), "+r"(w2), "+r"(w3), "+r"(result)
                 : "r"((I64)KInner), "r"((I64)KTail), "r"((I64)N)
                 : "memory", "cc", "x19", "x20", "x21", "x22", "x23", "x24", "x15", "x16", "v0",
                 "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                 "v16", "v17", "v18");
}

inline void mvm_row_A76(U32 numRows, U32 numColumns, F16 *matrix, F16 *vector, F16 *result)
{
    // Actual layout is NK, and vector is K
    U32 N = numRows;
    U32 K = numColumns;
    U32 NTail = N % 4;
    U32 NInner = N / 4;
    for (U32 i = 0; i < NInner; i++) {
        mvm_row_kernel_A76(NInner * 2, K, matrix + i * K, vector, result + i);
    }
    if (NTail != 0) {
        mvm_row_tail(NTail, K, matrix + (N - NTail) * K, vector, result + (N - NTail));
    }
}

void mvm_A76(U32 row, U32 col, bool transpose, F16 *matrix, F16 *vector, F16 *result)
{
    if (transpose) {
        mvm_col(row, col, matrix, vector, result);
    } else {
        mvm_row_A76(row, col, matrix, vector, result);
    }
}
