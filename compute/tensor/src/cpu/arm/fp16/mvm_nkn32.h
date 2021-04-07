// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MVM_NKN32
#define _H_MVM_NKN32
#include "tensor_desc.h"
#include "thread_affinity.h"

inline void mvm_nkn32(U32 fn, U32 fk, const F16 *filterArray, F16 *input, F16 *output)
{
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (U32 n = 0; n < fn; n++) {
        F16 *in = input;
        const F16 *f = filterArray + n * fk * 32;
        F16 *out = output + n * 32;
        __asm__ __volatile__("ldr s0, [%[in]]\n"
                             "ldr q1, [%[out]]\n"
                             "ldr q2, [%[out], #16]\n"
                             "ldr q3, [%[out], #32]\n"
                             "ldr q4, [%[out], #48]\n"
                             "mov x0, %[k]\n"
                             "ldr q5, [%[f]]\n"
                             "ldr q6, [%[f], #16]\n"
                             "ldr q7, [%[f], #32]\n"
                             "ldr q8, [%[f], #48]\n"
                             "0:\n"
                             "prfm pldl2strm, [%[f], #4096]\n"
                             "prfm pldl1strm, [%[f], #1024]\n"
                             "ldr d9, [%[f], #64]\n"
                             "fmla v1.8h, v5.8h, v0.h[0]\n"
                             "ldr x9, [%[f], #72]\n"
                             "ins v9.d[1], x9\n"
                             "ldr d10, [%[f], #80]\n"
                             "fmla v2.8h, v6.8h, v0.h[0]\n"
                             "ldr x10, [%[f], #88]\n"
                             "ins v10.d[1], x10\n"
                             "ldr d11, [%[f], #96]\n"
                             "fmla v3.8h, v7.8h, v0.h[0]\n"
                             "ldr x11, [%[f], #104]\n"
                             "ins v11.d[1], x11\n"
                             "ldr d12, [%[f], #112]\n"
                             "fmla v4.8h, v8.8h, v0.h[0]\n"
                             "ldr x12, [%[f], #120]\n"
                             "ins v12.d[1], x12\n"

                             "ldr d5, [%[f], #128]\n"
                             "fmla v1.8h, v9.8h, v0.h[1]\n"
                             "ldr x5, [%[f], #136]\n"
                             "ins v5.d[1], x5\n"
                             "ldr d6, [%[f], #144]\n"
                             "fmla v2.8h, v10.8h, v0.h[1]\n"
                             "ldr x6, [%[f], #152]\n"
                             "ins v6.d[1], x6\n"
                             "ldr d7, [%[f], #160]\n"
                             "fmla v3.8h, v11.8h, v0.h[1]\n"
                             "ldr x7, [%[f], #168]\n"
                             "ins v7.d[1], x7\n"
                             "ldr d8, [%[f], #176]\n"
                             "fmla v4.8h, v12.8h, v0.h[1]\n"
                             "ldr x8, [%[f], #184]\n"
                             "add %[in], %[in], #4\n"
                             "ins v8.d[1], x8\n"
                             "add %[f], %[f], #128\n"
                             "ldr s0, [%[in]]\n"
                             "sub x0, x0, #2\n"

                             "cmp x0, #3\n"
                             "bgt 0b\n"
                             "ldr  q9, [%[f], #64]\n"
                             "ldr q10, [%[f], #80]\n"
                             "ldr q11, [%[f], #96]\n"
                             "ldr q12, [%[f], #112]\n"
                             "fmla v1.8h,  v5.8h, v0.h[0]\n"
                             "fmla v2.8h,  v6.8h, v0.h[0]\n"
                             "fmla v3.8h,  v7.8h, v0.h[0]\n"
                             "fmla v4.8h,  v8.8h, v0.h[0]\n"
                             "fmla v1.8h,  v9.8h, v0.h[1]\n"
                             "fmla v2.8h, v10.8h, v0.h[1]\n"
                             "fmla v3.8h, v11.8h, v0.h[1]\n"
                             "fmla v4.8h, v12.8h, v0.h[1]\n"
                             "cmp x0, #3\n"
                             "bne 1f\n"
                             "ldr h0, [%[in], #4]\n"
                             "ldr q5, [%[f], #128]\n"
                             "ldr q6, [%[f], #144]\n"
                             "ldr q7, [%[f], #160]\n"
                             "ldr q8, [%[f], #176]\n"
                             "fmla v1.8h,  v5.8h, v0.h[0]\n"
                             "fmla v2.8h,  v6.8h, v0.h[0]\n"
                             "fmla v3.8h,  v7.8h, v0.h[0]\n"
                             "fmla v4.8h,  v8.8h, v0.h[0]\n"

                             "1:\n"
                             "str q1, [%[out]]\n"
                             "str q2, [%[out], #16]\n"
                             "str q3, [%[out], #32]\n"
                             "str q4, [%[out], #48]\n"
                             : [out] "+r"(out), [f] "+r"(f), [in] "+r"(in)
                             : [k] "r"((I64)fk)
                             : "memory", "cc", "x0", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
                             "x12", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                             "v10", "v11", "v12");
    }
}
#endif
