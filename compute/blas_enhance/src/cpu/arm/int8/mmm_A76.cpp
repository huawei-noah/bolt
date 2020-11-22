// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_INT8
#include <arm_neon.h>
#include <math.h>
#include "cpu/arm/blas_arm.h"
#include "cpu/arm/int8/mmm_common.h"
#include "cpu/arm/int8/mmm.h"

inline void mmm_4x4_A76(U32 offset, U32 K, INT8 *in, INT8 *w, I32 *out)
{
    asm volatile(
        // init in- > v1, w- > v0
        "ldr q1, [%0]\n"

        "ldr q0, [%1]\n"

        // give in address to x3
        "mov x3, %0\n"

        // give w address to x0
        "mov x0, %1\n"

        // K- > x2
        "mov x2, %3\n"

        // give out address to x26
        "mov x26, %2\n"

        // load in bias
        "ldr q5, [x26]\n"
        "add x26, x26, %4\n"

        "ldr q7, [x26]\n"
        "add x26, x26, %4\n"

        "ldr q9, [x26]\n"
        "add x26, x26, %4\n"

        "ldr q11, [x26]\n"

        // Computation loop
        "0:\n"

        "ldr q3, [x3, 16]!\n"
        "ldr q29, [x0, 16]!\n"
        "sdot v5.4s, v0.16b, v1.4b[0]\n"
        "sdot v7.4s, v0.16b, v1.4b[1]\n"
        "subs x2, x2, #4\n"
        "sdot v9.4s, v0.16b, v1.4b[2]\n"
        "sdot v11.4s, v0.16b, v1.4b[3]\n"
        "mov	v1.16b, v3.16b\n"
        "mov	v0.16b, v29.16b\n"
        "bne 0b\n"

        "1:\n"

        // give out address to x26
        "mov x26, %2\n"

        "str q5, [x26]\n"
        "add x26, x26, %4\n"

        "str q7, [x26]\n"
        "add x26, x26, %4\n"

        "str q9, [x26]\n"
        "add x26, x26, %4\n"

        "str q11, [x26]\n"

        : "+r"(in), "+r"(w), "+r"(out)
        : "r"((I64)K), "r"((I64)offset)
        : "memory", "cc", "v30", "v29", "v11", "v9", "v7", "v5", "v3", "v1", "v0", "x26", "x3",
        "x2", "x0");
}

inline void mmm_8x4_A76(U32 offset, U32 K, INT8 *in, INT8 *w, I32 *out)
{
    asm volatile(
        // init in- > v1, w- > v0
        "ldr q1, [%0]\n"

        "ldr q0, [%1]\n"

        // give in address to x3
        "mov x3, %0\n"

        // give w address to x0
        "mov x0, %1\n"

        // K- > x2
        "mov x2, %3\n"

        // give out address to x26
        "mov x26, %2\n"

        // load in bias
        "ldr q5, [x26]\n"
        "add x26, x26, %4\n"

        "ldr q7, [x26]\n"
        "add x26, x26, %4\n"

        "ldr q9, [x26]\n"
        "add x26, x26, %4\n"

        "ldr q11, [x26]\n"
        "add x26, x26, %4\n"

        "ldr q13, [x26]\n"
        "add x26, x26, %4\n"

        "ldr q15, [x26]\n"
        "add x26, x26, %4\n"

        "ldr q17, [x26]\n"
        "add x26, x26, %4\n"

        "ldr q19, [x26]\n"

        // Computation loop
        "0:\n"

        "ldr q3, [x3, 16]\n"
        "ldr q29, [x0, 16]!\n"
        "sdot v5.4s, v0.16b, v1.4b[0]\n"
        "sdot v7.4s, v0.16b, v1.4b[1]\n"
        "sdot v9.4s, v0.16b, v1.4b[2]\n"
        "sdot v11.4s, v0.16b, v1.4b[3]\n"

        "subs x2, x2, #4\n"
        "sdot v13.4s, v0.16b, v3.4b[0]\n"
        "sdot v15.4s, v0.16b, v3.4b[1]\n"
        "ldr q1, [x3, 32]!\n"
        "sdot v17.4s, v0.16b, v3.4b[2]\n"
        "sdot v19.4s, v0.16b, v3.4b[3]\n"
        "mov v0.16b, v29.16b\n"
        "bne 0b\n"

        // give out address to x26
        "mov x26, %2\n"

        "str q5, [x26]\n"
        "add x26, x26, %4\n"

        "str q7, [x26]\n"
        "add x26, x26, %4\n"

        "str q9, [x26]\n"
        "add x26, x26, %4\n"

        "str q11, [x26]\n"
        "add x26, x26, %4\n"

        "str q13, [x26]\n"
        "add x26, x26, %4\n"

        "str q15, [x26]\n"
        "add x26, x26, %4\n"

        "str q17, [x26]\n"
        "add x26, x26, %4\n"

        "str q19, [x26]\n"

        : "+r"(in), "+r"(w), "+r"(out)
        : "r"((I64)K), "r"((I64)offset)
        : "memory", "cc", "v30", "v29", "v19", "v17", "v15", "v13", "v11", "v9", "v7", "v5", "v3",
        "v1", "v0", "x26", "x3", "x2", "x0");
}

inline void mmm_4x8_A76(U32 offset, U32 K, INT8 *in, INT8 *w, I32 *out)
{
    asm volatile(
        // init in- > v1, w- > v0
        "ldr q1, [%0]\n"

        "ldr q0, [%1]\n"

        // give in address to x3
        "mov x3, %0\n"

        // give w address to x0
        "mov x0, %1\n"

        // K- > x2
        "mov x2, %3\n"

        // give out address to x26
        "mov x26, %2\n"

        "ld1 {v5.4s, v6.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v7.4s, v8.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v9.4s, v10.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v11.4s, v12.4s}, [x26]\n"

        /* Layout
         * 5   6
         * 7   8
         * 9   10
         * 11  12
         */

        // Computation loop
        "0:\n"

        "ldr q29, [x0, 16]\n"
        "sdot v5.4s, v0.16b, v1.4b[0]\n"
        "ldr q3, [x3, 16]!\n"
        "sdot v7.4s, v0.16b, v1.4b[1]\n"
        "subs x2, x2, #4\n"
        "sdot v9.4s, v0.16b, v1.4b[2]\n"
        "sdot v11.4s, v0.16b, v1.4b[3]\n"

        "sdot v6.4s, v29.16b, v1.4b[0]\n"
        "ldr q0, [x0, 32]!\n"
        "sdot v8.4s, v29.16b, v1.4b[1]\n"
        "sdot v10.4s, v29.16b, v1.4b[2]\n"
        "sdot v12.4s, v29.16b, v1.4b[3]\n"
        "mov	v1.16b, v3.16b\n"
        "bne 0b\n"

        // give out address to x26
        "mov x26, %2\n"

        "st1 {v5.4s, v6.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v7.4s, v8.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v9.4s, v10.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v11.4s, v12.4s}, [x26]\n"

        : "+r"(in), "+r"(w), "+r"(out)
        : "r"((I64)K), "r"((I64)offset)
        : "memory", "cc", "v29", "v12", "v11", "v10", "v9", "v8", "v7", "v6", "v5", "v3", "v1",
        "v0", "x26", "x3", "x2", "x0");
}

inline void mmm_8x8_A76(U32 offset, U32 K, INT8 *in, INT8 *w, I32 *out)
{
    asm volatile(
        // init in- > v1, w- > v0
        "ldr q1, [%0]\n"

        "ldr q0, [%1]\n"

        // give in address to x3
        "mov x3, %0\n"

        // give w address to x0
        "mov x0, %1\n"

        // K- > x2
        "mov x2, %3\n"

        // give out address to x26
        "mov x26, %2\n"

        // load in bias
        "ld1 {v5.4s, v6.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v7.4s, v8.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v9.4s, v10.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v11.4s, v12.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v13.4s, v14.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v15.4s, v16.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v17.4s, v18.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v19.4s, v20.4s}, [x26]\n"

        /* Layout
           5   6
           7   8
           9   10
           11  12

           13  14
           15  16
           17  18
           19  20
         */

        // Computation loop
        "0:\n"

        "sdot v5.4s, v0.16b, v1.4b[0]\n"
        "ldr q3, [x3, 16]!\n"
        "sdot v7.4s, v0.16b, v1.4b[1]\n"
        "ldr q29, [x0, 16]\n"
        "sdot v9.4s, v0.16b, v1.4b[2]\n"
        "sdot v11.4s, v0.16b, v1.4b[3]\n"

        "sdot v13.4s, v0.16b, v3.4b[0]\n"
        "subs x2, x2, #4\n"
        "sdot v15.4s, v0.16b, v3.4b[1]\n"
        "sdot v17.4s, v0.16b, v3.4b[2]\n"
        "sdot v19.4s, v0.16b, v3.4b[3]\n"

        "sdot v6.4s, v29.16b, v1.4b[0]\n"
        "sdot v8.4s, v29.16b, v1.4b[1]\n"
        "ldr  q0, [x0, 32]!\n"
        "sdot v10.4s, v29.16b, v1.4b[2]\n"
        "sdot v12.4s, v29.16b, v1.4b[3]\n"

        "sdot v14.4s, v29.16b, v3.4b[0]\n"
        "sdot v16.4s, v29.16b, v3.4b[1]\n"
        "ldr q1, [x3, 16]!\n"
        "sdot v18.4s, v29.16b, v3.4b[2]\n"
        "sdot v20.4s, v29.16b, v3.4b[3]\n"

        "bne 0b\n"

        // give out address to x26
        "mov x26, %2\n"

        "st1 {v5.4s, v6.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v7.4s, v8.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v9.4s, v10.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v11.4s, v12.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v13.4s, v14.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v15.4s, v16.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v17.4s, v18.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v19.4s, v20.4s}, [x26]\n"

        : "+r"(in), "+r"(w), "+r"(out)
        : "r"((I64)K), "r"((I64)offset)
        : "memory", "cc", "v29", "v20", "v19", "v18", "v17", "v16", "v15", "v14", "v13", "v12",
        "v11", "v10", "v9", "v8", "v7", "v6", "v5", "v3", "v1", "v0", "x26", "x3", "x2", "x0");
}

inline void mmm_4x12_A76(U32 offset, U32 K, INT8 *in, INT8 *w, I32 *out)
{
    asm volatile(
        // init in->v1, w->v0
        "ldr q1, [%0]\n"

        "ldr q0, [%1]\n"

        "ldr q29, [%1, 16]\n"  // prefetch one more w

        // give in address to x3
        "mov x3, %0\n"

        // give w address to x0
        "mov x0, %1\n"

        // K->x2
        "mov x2, %3\n"

        // give out address to x26
        "mov x26, %2\n"

        // load in bias
        "ld1 {v5.4s, v6.4s, v7.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v8.4s, v9.4s, v10.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v11.4s, v12.4s, v13.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v14.4s, v15.4s, v16.4s}, [x26]\n"

        /* Layout
           5   6   7
           8   9   10
           11  12  13
           14  15  16
         */

        // Computation loop
        "0:\n"
        // in(x3): v1
        // w(x0):  v0 v29 v30

        "sdot v5.4s, v0.16b, v1.4b[0]\n"
        "ldr q30, [x0, 32]\n"
        "sdot v8.4s, v0.16b, v1.4b[1]\n"
        "sdot v11.4s, v0.16b, v1.4b[2]\n"
        "ldr q2, [x3, 16]!\n"  // input of next round
        "sdot v14.4s, v0.16b, v1.4b[3]\n"

        "sdot v6.4s, v29.16b, v1.4b[0]\n"
        "sdot v9.4s, v29.16b, v1.4b[1]\n"
        "ldr q0, [x0, 48]!\n"  // first w of next round
        "sdot v12.4s, v29.16b, v1.4b[2]\n"
        "sdot v15.4s, v29.16b, v1.4b[3]\n"

        "sdot v7.4s, v30.16b, v1.4b[0]\n"
        "ldr q29, [x0, 16]\n"
        "sdot v10.4s, v30.16b, v1.4b[1]\n"
        "sdot v13.4s, v30.16b, v1.4b[2]\n"
        "subs x2, x2, #4\n"
        "sdot v16.4s, v30.16b, v1.4b[3]\n"

        "mov v1.16b, v2.16b\n"
        "bne 0b\n"

        // give out address to x26
        "mov x26, %2\n"

        "st1 {v5.4s, v6.4s, v7.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v8.4s, v9.4s, v10.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v11.4s, v12.4s, v13.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v14.4s, v15.4s, v16.4s}, [x26]\n"

        : "+r"(in), "+r"(w), "+r"(out)
        : "r"((I64)K), "r"((I64)offset)
        : "memory", "cc", "v30", "v29", "v16", "v15", "v14", "v13", "v12", "v11", "v10", "v9", "v8",
        "v7", "v6", "v5", "v3", "v2", "v1", "v0", "x26", "x19", "x3", "x2", "x0");
}

inline void mmm_8x12_A76(U32 offset, U32 K, INT8 *in, INT8 *w, I32 *out)
{
    asm volatile(
        // init in->v1, w->v0
        "ldr q1, [%0]\n"

        "ldr q0, [%1]\n"

        "ldr q29, [%1, 16]\n"  // prefetch one more w

        // give in address to x3
        "mov x3, %0\n"

        // give w address to x0
        "mov x0, %1\n"

        // K->x2
        "mov x2, %3\n"

        // give out address to x26
        "mov x26, %2\n"

        // load in bias
        "ld1 {v5.4s, v6.4s, v7.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v8.4s, v9.4s, v10.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v11.4s, v12.4s, v13.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v14.4s, v15.4s, v16.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v17.4s, v18.4s, v19.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v20.4s, v21.4s, v22.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v23.4s, v24.4s, v25.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v26.4s, v27.4s, v28.4s}, [x26]\n"

        /* Layout
           5   6   7
           8   9   10
           11  12  13
           14  15  16

           17  18  19
           20  21  22
           23  24  25
           26  27  28
         */

        // Computation loop
        "0:\n"
        // in(x3): v1 v2
        // w(x0):  v0 v29 v30

        "sdot v5.4s, v0.16b, v1.4b[0]\n"
        "ldr q30, [x0, 32]\n"
        "sdot v8.4s, v0.16b, v1.4b[1]\n"
        "sdot v11.4s, v0.16b, v1.4b[2]\n"
        "ldr q2, [x3, 16]\n"
        "sdot v14.4s, v0.16b, v1.4b[3]\n"

        "sdot v6.4s, v29.16b, v1.4b[0]\n"
        "sdot v9.4s, v29.16b, v1.4b[1]\n"
        "ldr q3, [x0, 48]!\n"  // first w of next round
        "sdot v12.4s, v29.16b, v1.4b[2]\n"
        "sdot v15.4s, v29.16b, v1.4b[3]\n"

        "sdot v7.4s, v30.16b, v1.4b[0]\n"
        "subs x2, x2, #4\n"
        "sdot v10.4s, v30.16b, v1.4b[1]\n"
        "sdot v13.4s, v30.16b, v1.4b[2]\n"
        "sdot v16.4s, v30.16b, v1.4b[3]\n"

        "sdot v17.4s, v0.16b, v2.4b[0]\n"
        "ldr q1, [x3, 32]!\n"
        "sdot v20.4s, v0.16b, v2.4b[1]\n"
        "sdot v23.4s, v0.16b, v2.4b[2]\n"
        "sdot v26.4s, v0.16b, v2.4b[3]\n"

        "sdot v18.4s, v29.16b, v2.4b[0]\n"
        "mov v0.16b, v3.16b\n"
        "sdot v21.4s, v29.16b, v2.4b[1]\n"
        "sdot v24.4s, v29.16b, v2.4b[2]\n"
        "sdot v27.4s, v29.16b, v2.4b[3]\n"

        "sdot v19.4s, v30.16b, v2.4b[0]\n"
        "ldr q29, [x0, 16]\n"
        "sdot v22.4s, v30.16b, v2.4b[1]\n"
        "sdot v25.4s, v30.16b, v2.4b[2]\n"
        "sdot v28.4s, v30.16b, v2.4b[3]\n"

        "bne 0b\n"

        // give out address to x26
        "mov x26, %2\n"

        "st1 {v5.4s, v6.4s, v7.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v8.4s, v9.4s, v10.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v11.4s, v12.4s, v13.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v14.4s, v15.4s, v16.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v17.4s, v18.4s, v19.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v20.4s, v21.4s, v22.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v23.4s, v24.4s, v25.4s}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v26.4s, v27.4s, v28.4s}, [x26]\n"

        : "+r"(in), "+r"(w), "+r"(out)
        : "r"((I64)K), "r"((I64)offset)
        : "memory", "cc", "v30", "v29", "v28", "v27", "v26", "v25", "v24", "v23", "v22", "v21",
        "v20", "v19", "v18", "v17", "v16", "v15", "v14", "v13", "v12", "v11", "v10", "v9", "v8",
        "v7", "v6", "v5", "v3", "v2", "v1", "v0", "x26", "x3", "x2", "x0");
}

void mmm_A76(
    int M, int N, int K, bool transposeA, INT8 *matrix1, INT8 *matrix2, INT8 *tmp, I32 *result)
{
    int blockK = K;
    U32 K4 = pad_to_4_multiple(K);
    int blockM = 96;
    INT8 *matrix1Trans = tmp;
    I32 *resultCurrent = result;

    int KInner, MInner, m, n;
    for (int k = 0; k < K; k += blockK) {
        KInner = UNI_MIN(blockK, K - k);  // K for this inner iteration
        for (int i = 0; i < M; i += blockM) {
            MInner = UNI_MIN(blockM, M - i);  // M for this inner iteration
            for (n = 0; n <= N - 8; n += 8) {
                if (i == 0) {
                    if (transposeA) {
                        matrix2_trans_int8(8, KInner, N, matrix1 + n, matrix1Trans + n * K4);
                    } else {
                        matrix1_trans_n8(KInner, K, matrix1 + n * K + k, matrix1Trans + n * K4);
                    }
                }

                for (m = 0; m <= (MInner - 12); m += 12) {
                    resultCurrent = result + n * M + m + i;
                    mmm_8x12_A76(
                        M * 4, K4, matrix1Trans + n * K4, matrix2 + (i + m) * K4, resultCurrent);
                }
                for (; m <= (MInner - 8); m += 8) {
                    resultCurrent = result + n * M + m + i;
                    mmm_8x8_A76(
                        M * 4, K4, matrix1Trans + n * K4, matrix2 + (i + m) * K4, resultCurrent);
                }

                if ((MInner - m) >= 4) {
                    resultCurrent = result + n * M + m + i;
                    mmm_8x4_A76(
                        M * 4, K4, matrix1Trans + n * K4, matrix2 + (i + m) * K4, resultCurrent);
                    m += 4;
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_N8_MTail(MInner - m, M, K4, matrix1Trans + n * K4, matrix2 + (i + m) * K4,
                        resultCurrent);
                }
            }

            if ((N - n) >= 4) {
                if (i == 0) {
                    if (transposeA) {
                        matrix2_trans_int8(4, KInner, N, matrix1 + n, matrix1Trans + n * K4);
                    } else {
                        matrix1_trans_int8(4, KInner, K, matrix1 + n * K + k, matrix1Trans + n * K4);
                    }
                }

                for (m = 0; m <= (MInner - 12); m += 12) {
                    resultCurrent = result + n * M + m + i;
                    mmm_4x12_A76(
                        M * 4, K4, matrix1Trans + n * K4, matrix2 + (i + m) * K4, resultCurrent);
                }

                for (; m <= (MInner - 8); m += 8) {
                    resultCurrent = result + n * M + m + i;
                    mmm_4x8_A76(
                        M * 4, K4, matrix1Trans + n * K4, matrix2 + (i + m) * K4, resultCurrent);
                }

                if ((MInner - m) >= 4) {
                    resultCurrent = result + n * M + m + i;
                    mmm_4x4_A76(
                        M * 4, K4, matrix1Trans + n * K4, matrix2 + (i + m) * K4, resultCurrent);
                    m += 4;
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_N4_MTail(MInner - m, M, K4, matrix1Trans + n * K4, matrix2 + (i + m) * K4,
                        resultCurrent);
                }
                n += 4;
            }

            if (N - n) {
                if (i == 0) {
                    if (transposeA) {
                        matrix2_trans_int8(N - n, KInner, N, matrix1 + n, matrix1Trans + n * K4);
                    } else {
                        matrix1_trans_int8(
                            N - n, KInner, K, matrix1 + n * K + k, matrix1Trans + n * K4);
                    }
                }

                for (m = 0; m <= (MInner - 12); m += 12) {
                    resultCurrent = result + n * M + m + i;
                    mmm_NTail_M12(
                        M, N - n, K4, matrix1Trans + n * K4, matrix2 + (i + m) * K4, resultCurrent);
                }

                for (; m <= (MInner - 8); m += 8) {
                    resultCurrent = result + n * M + m + i;
                    mmm_NTail_M8(
                        M, N - n, K4, matrix1Trans + n * K4, matrix2 + (i + m) * K4, resultCurrent);
                }

                if ((MInner - m) >= 4) {
                    resultCurrent = result + n * M + m + i;
                    mmm_NTail_M4(
                        M, N - n, K4, matrix1Trans + n * K4, matrix2 + (i + m) * K4, resultCurrent);
                    m += 4;
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_NTail_M(MInner - m, M, N - n, K4, matrix1Trans + n * K4,
                        matrix2 + (i + m) * K4, resultCurrent);
                }
            }
        }
    }
}
#endif
