// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_ARM_FUNCTIONS_INT32
#define _H_ARM_FUNCTIONS_INT32

#include "cpu/cpu_functions_template.h"
#include "arm_neon_expand.h"

inline I32 array_sum_i32(const I32 *data, I32 len)
{
    if (len <= 0) {
        return 0;
    }

    I32 i = 0;
    I32 sum_s = 0;
    int32x4_t sum_v = vdupq_n_s32(0);
#pragma unroll(4)
    for (i = 0; i < len - 3; i += 4) {
        int32x4_t in = vld1q_s32(data + i);
        sum_v = vaddq_s32(sum_v, in);
    }
    sum_s += vaddvq_s32(sum_v);
    for (; i < len; i++) {
        sum_s += data[i];
    }
    return sum_s;
}

inline void array_scale_i32(const I32 *input, I32 *output, I32 len, F32 alpha, F32 beta)
{
    int32x4_t alpha_v = vdupq_n_s32(alpha);
    int32x4_t beta_v = vdupq_n_s32(beta);
    I32 i = 0;
#pragma unroll(4)
    for (; i < len - 3; i += 4) {
        int32x4_t in = vld1q_s32(input + i);
        int32x4_t tmp_v = vmlaq_s32(beta_v, alpha_v, in);
        vst1q_s32(output + i, tmp_v);
    }
    for (; i < len; i++) {
        output[i] = alpha * input[i] + beta;
    }
}

#ifdef __aarch64__
// num_v is the number of q-form vectors (I32) = length / 4
// factor = scale * 16777216
inline void quantize_I32(I32 num_v, const I32 *out_d, I32 factor, INT8 *out_q)
{
    I64 i28 = num_v / 28;
    if (i28 > 0) {
        __asm__ __volatile__("ld4 {v1.4s, v2.4s, v3.4s, v4.4s}, [%[out_d]], #64\n"
                             "ldr s0, [%[factor]]\n"
                             "ld4 {v5.4s, v6.4s, v7.4s, v8.4s}, [%[out_d]], #64\n"
                             "mov x1, %[i]\n"
                             "ld4 {v9.4s, v10.4s, v11.4s, v12.4s}, [%[out_d]], #64\n"
                             "dup v0.4s, v0.s[0]\n"
                             "ld4 {v13.4s, v14.4s, v15.4s, v16.4s}, [%[out_d]], #64\n"
                             "ld4 {v17.4s, v18.4s, v19.4s, v20.4s}, [%[out_d]], #64\n"
                             "ld4 {v21.4s, v22.4s, v23.4s, v24.4s}, [%[out_d]], #64\n"

                             "0:\n"
                             "ld4 {v25.4s, v26.4s, v27.4s, v28.4s}, [%[out_d]], #64\n"
                             "subs x1, x1, #1\n"

                             "mul v4.4s, v4.4s, v0.4s\n"
                             "mul v3.4s, v3.4s, v0.4s\n"
                             "mul v2.4s, v2.4s, v0.4s\n"
                             "mul v1.4s, v1.4s, v0.4s\n"

                             "mul v8.4s, v8.4s, v0.4s\n"
                             "sri v4.4s, v3.4s, #8\n"
                             "mul v7.4s, v7.4s, v0.4s\n"
                             "sri v2.4s, v1.4s, #8\n"
                             "mul v6.4s, v6.4s, v0.4s\n"
                             "mul v5.4s, v5.4s, v0.4s\n"
                             "sri v4.4s, v2.4s, #16\n"

                             "mul v12.4s, v12.4s, v0.4s\n"
                             "sri v8.4s, v7.4s, #8\n"
                             "mul v11.4s, v11.4s, v0.4s\n"
                             "sri v6.4s, v5.4s, #8\n"
                             "mul v10.4s, v10.4s, v0.4s\n"
                             "str q4, [%[out_q]], #16\n"
                             "ld4 {v1.4s, v2.4s, v3.4s, v4.4s}, [%[out_d]], #64\n"
                             "mul v9.4s, v9.4s, v0.4s\n"
                             "sri v8.4s, v6.4s, #16\n"

                             "mul v16.4s, v16.4s, v0.4s\n"
                             "sri v12.4s, v11.4s, #8\n"
                             "mul v15.4s, v15.4s, v0.4s\n"
                             "sri v10.4s, v9.4s, #8\n"
                             "mul v14.4s, v14.4s, v0.4s\n"
                             "str q8, [%[out_q]], #16\n"
                             "ld4 {v5.4s, v6.4s, v7.4s, v8.4s}, [%[out_d]], #64\n"
                             "mul v13.4s, v13.4s, v0.4s\n"
                             "sri v12.4s, v10.4s, #16\n"

                             "mul v20.4s, v20.4s, v0.4s\n"
                             "sri v16.4s, v15.4s, #8\n"
                             "mul v19.4s, v19.4s, v0.4s\n"
                             "sri v14.4s, v13.4s, #8\n"
                             "mul v18.4s, v18.4s, v0.4s\n"
                             "str q12, [%[out_q]], #16\n"
                             "ld4 {v9.4s, v10.4s, v11.4s, v12.4s}, [%[out_d]], #64\n"
                             "mul v17.4s, v17.4s, v0.4s\n"
                             "sri v16.4s, v14.4s, #16\n"

                             "mul v24.4s, v24.4s, v0.4s\n"
                             "sri v20.4s, v19.4s, #8\n"
                             "mul v23.4s, v23.4s, v0.4s\n"
                             "sri v18.4s, v17.4s, #8\n"
                             "mul v22.4s, v22.4s, v0.4s\n"
                             "str q16, [%[out_q]], #16\n"
                             "ld4 {v13.4s, v14.4s, v15.4s, v16.4s}, [%[out_d]], #64\n"
                             "mul v21.4s, v21.4s, v0.4s\n"
                             "sri v20.4s, v18.4s, #16\n"

                             "mul v28.4s, v28.4s, v0.4s\n"
                             "sri v24.4s, v23.4s, #8\n"
                             "mul v27.4s, v27.4s, v0.4s\n"
                             "sri v22.4s, v21.4s, #8\n"
                             "mul v26.4s, v26.4s, v0.4s\n"
                             "str q20, [%[out_q]], #16\n"
                             "ld4 {v17.4s, v18.4s, v19.4s, v20.4s}, [%[out_d]], #64\n"
                             "mul v25.4s, v25.4s, v0.4s\n"
                             "sri v24.4s, v22.4s, #16\n"

                             "sri v28.4s, v27.4s, #8\n"
                             "sri v26.4s, v25.4s, #8\n"
                             "str q24, [%[out_q]], #16\n"
                             "sri v28.4s, v26.4s, #16\n"
                             "ld4 {v21.4s, v22.4s, v23.4s, v24.4s}, [%[out_d]], #64\n"
                             "str q28, [%[out_q]], #16\n"
                             "bne 0b\n"
                             : [out_d] "+r"(out_d), [out_q] "+r"(out_q)
                             : [factor] "r"(&factor), [i] "r"(i28)
                             : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                             "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                             "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                             "x1");
        // Prefetched 24 extra vectors
        out_d -= 96;
    }

    I32 remainder = num_v - i28 * 28;
    if (remainder % 4) {
        for (I32 i = 0; i < 8; i++) {
            out_q[i] = (out_d[i] * factor) >> 8;
        }
        out_d += 8;
        out_q += 8;
        remainder -= 2;
    }
    switch (remainder) {
        case 24: {
            __asm__ __volatile__("ldr s0, [%[factor]]\n"
                                 "dup v0.4s, v0.s[0]\n"

                                 "ld4 {v1.4s, v2.4s, v3.4s, v4.4s}, [%[out_d]], #64\n"
                                 "ld4 {v5.4s, v6.4s, v7.4s, v8.4s}, [%[out_d]], #64\n"
                                 "ld4 {v9.4s, v10.4s, v11.4s, v12.4s}, [%[out_d]], #64\n"
                                 "ld4 {v13.4s, v14.4s, v15.4s, v16.4s}, [%[out_d]], #64\n"
                                 "ld4 {v17.4s, v18.4s, v19.4s, v20.4s}, [%[out_d]], #64\n"
                                 "ld4 {v21.4s, v22.4s, v23.4s, v24.4s}, [%[out_d]], #64\n"

                                 "mul v4.4s, v4.4s, v0.4s\n"
                                 "mul v3.4s, v3.4s, v0.4s\n"
                                 "mul v2.4s, v2.4s, v0.4s\n"
                                 "mul v1.4s, v1.4s, v0.4s\n"

                                 "mul v8.4s, v8.4s, v0.4s\n"
                                 "sri v4.4s, v3.4s, #8\n"
                                 "mul v7.4s, v7.4s, v0.4s\n"
                                 "sri v2.4s, v1.4s, #8\n"
                                 "mul v6.4s, v6.4s, v0.4s\n"
                                 "mul v5.4s, v5.4s, v0.4s\n"
                                 "sri v4.4s, v2.4s, #16\n"

                                 "mul v12.4s, v12.4s, v0.4s\n"
                                 "sri v8.4s, v7.4s, #8\n"
                                 "mul v11.4s, v11.4s, v0.4s\n"
                                 "sri v6.4s, v5.4s, #8\n"
                                 "mul v10.4s, v10.4s, v0.4s\n"
                                 "str q4, [%[out_q]], #16\n"
                                 "mul v9.4s, v9.4s, v0.4s\n"
                                 "sri v8.4s, v6.4s, #16\n"

                                 "mul v16.4s, v16.4s, v0.4s\n"
                                 "sri v12.4s, v11.4s, #8\n"
                                 "mul v15.4s, v15.4s, v0.4s\n"
                                 "sri v10.4s, v9.4s, #8\n"
                                 "mul v14.4s, v14.4s, v0.4s\n"
                                 "str q8, [%[out_q]], #16\n"
                                 "mul v13.4s, v13.4s, v0.4s\n"
                                 "sri v12.4s, v10.4s, #16\n"

                                 "mul v20.4s, v20.4s, v0.4s\n"
                                 "sri v16.4s, v15.4s, #8\n"
                                 "mul v19.4s, v19.4s, v0.4s\n"
                                 "sri v14.4s, v13.4s, #8\n"
                                 "mul v18.4s, v18.4s, v0.4s\n"
                                 "str q12, [%[out_q]], #16\n"
                                 "mul v17.4s, v17.4s, v0.4s\n"
                                 "sri v16.4s, v14.4s, #16\n"

                                 "mul v24.4s, v24.4s, v0.4s\n"
                                 "sri v20.4s, v19.4s, #8\n"
                                 "mul v23.4s, v23.4s, v0.4s\n"
                                 "sri v18.4s, v17.4s, #8\n"
                                 "mul v22.4s, v22.4s, v0.4s\n"
                                 "str q16, [%[out_q]], #16\n"
                                 "mul v21.4s, v21.4s, v0.4s\n"
                                 "sri v20.4s, v18.4s, #16\n"

                                 "sri v24.4s, v23.4s, #8\n"
                                 "sri v22.4s, v21.4s, #8\n"
                                 "str q20, [%[out_q]], #16\n"
                                 "sri v24.4s, v22.4s, #16\n"

                                 "str q24, [%[out_q]], #16\n"
                                 : [out_d] "+r"(out_d), [out_q] "+r"(out_q)
                                 : [factor] "r"(&factor)
                                 : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                                 "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                                 "v18", "v19", "v20", "v21", "v22", "v23", "v24", "x1");
            break;
        }
        case 20: {
            __asm__ __volatile__("ldr s0, [%[factor]]\n"
                                 "dup v0.4s, v0.s[0]\n"

                                 "ld4 {v1.4s, v2.4s, v3.4s, v4.4s}, [%[out_d]], #64\n"
                                 "ld4 {v5.4s, v6.4s, v7.4s, v8.4s}, [%[out_d]], #64\n"
                                 "ld4 {v9.4s, v10.4s, v11.4s, v12.4s}, [%[out_d]], #64\n"
                                 "ld4 {v13.4s, v14.4s, v15.4s, v16.4s}, [%[out_d]], #64\n"
                                 "ld4 {v17.4s, v18.4s, v19.4s, v20.4s}, [%[out_d]], #64\n"

                                 "mul v4.4s, v4.4s, v0.4s\n"
                                 "mul v3.4s, v3.4s, v0.4s\n"
                                 "mul v2.4s, v2.4s, v0.4s\n"
                                 "mul v1.4s, v1.4s, v0.4s\n"

                                 "mul v8.4s, v8.4s, v0.4s\n"
                                 "sri v4.4s, v3.4s, #8\n"
                                 "mul v7.4s, v7.4s, v0.4s\n"
                                 "sri v2.4s, v1.4s, #8\n"
                                 "mul v6.4s, v6.4s, v0.4s\n"
                                 "mul v5.4s, v5.4s, v0.4s\n"
                                 "sri v4.4s, v2.4s, #16\n"

                                 "mul v12.4s, v12.4s, v0.4s\n"
                                 "sri v8.4s, v7.4s, #8\n"
                                 "mul v11.4s, v11.4s, v0.4s\n"
                                 "sri v6.4s, v5.4s, #8\n"
                                 "mul v10.4s, v10.4s, v0.4s\n"
                                 "str q4, [%[out_q]], #16\n"
                                 "mul v9.4s, v9.4s, v0.4s\n"
                                 "sri v8.4s, v6.4s, #16\n"

                                 "mul v16.4s, v16.4s, v0.4s\n"
                                 "sri v12.4s, v11.4s, #8\n"
                                 "mul v15.4s, v15.4s, v0.4s\n"
                                 "sri v10.4s, v9.4s, #8\n"
                                 "mul v14.4s, v14.4s, v0.4s\n"
                                 "str q8, [%[out_q]], #16\n"
                                 "mul v13.4s, v13.4s, v0.4s\n"
                                 "sri v12.4s, v10.4s, #16\n"

                                 "mul v20.4s, v20.4s, v0.4s\n"
                                 "sri v16.4s, v15.4s, #8\n"
                                 "mul v19.4s, v19.4s, v0.4s\n"
                                 "sri v14.4s, v13.4s, #8\n"
                                 "mul v18.4s, v18.4s, v0.4s\n"
                                 "str q12, [%[out_q]], #16\n"
                                 "mul v17.4s, v17.4s, v0.4s\n"
                                 "sri v16.4s, v14.4s, #16\n"

                                 "sri v20.4s, v19.4s, #8\n"
                                 "sri v18.4s, v17.4s, #8\n"
                                 "str q16, [%[out_q]], #16\n"
                                 "sri v20.4s, v18.4s, #16\n"

                                 "str q20, [%[out_q]], #16\n"
                                 : [out_d] "+r"(out_d), [out_q] "+r"(out_q)
                                 : [factor] "r"(&factor)
                                 : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                                 "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                                 "v18", "v19", "v20", "x1");
            break;
        }
        case 16: {
            __asm__ __volatile__("ldr s0, [%[factor]]\n"
                                 "dup v0.4s, v0.s[0]\n"

                                 "ld4 {v1.4s, v2.4s, v3.4s, v4.4s}, [%[out_d]], #64\n"
                                 "ld4 {v5.4s, v6.4s, v7.4s, v8.4s}, [%[out_d]], #64\n"
                                 "ld4 {v9.4s, v10.4s, v11.4s, v12.4s}, [%[out_d]], #64\n"
                                 "ld4 {v13.4s, v14.4s, v15.4s, v16.4s}, [%[out_d]], #64\n"

                                 "mul v4.4s, v4.4s, v0.4s\n"
                                 "mul v3.4s, v3.4s, v0.4s\n"
                                 "mul v2.4s, v2.4s, v0.4s\n"
                                 "mul v1.4s, v1.4s, v0.4s\n"

                                 "mul v8.4s, v8.4s, v0.4s\n"
                                 "sri v4.4s, v3.4s, #8\n"
                                 "mul v7.4s, v7.4s, v0.4s\n"
                                 "sri v2.4s, v1.4s, #8\n"
                                 "mul v6.4s, v6.4s, v0.4s\n"
                                 "mul v5.4s, v5.4s, v0.4s\n"
                                 "sri v4.4s, v2.4s, #16\n"

                                 "mul v12.4s, v12.4s, v0.4s\n"
                                 "sri v8.4s, v7.4s, #8\n"
                                 "mul v11.4s, v11.4s, v0.4s\n"
                                 "sri v6.4s, v5.4s, #8\n"
                                 "mul v10.4s, v10.4s, v0.4s\n"
                                 "str q4, [%[out_q]], #16\n"
                                 "mul v9.4s, v9.4s, v0.4s\n"
                                 "sri v8.4s, v6.4s, #16\n"

                                 "mul v16.4s, v16.4s, v0.4s\n"
                                 "sri v12.4s, v11.4s, #8\n"
                                 "mul v15.4s, v15.4s, v0.4s\n"
                                 "sri v10.4s, v9.4s, #8\n"
                                 "mul v14.4s, v14.4s, v0.4s\n"
                                 "str q8, [%[out_q]], #16\n"
                                 "mul v13.4s, v13.4s, v0.4s\n"
                                 "sri v12.4s, v10.4s, #16\n"

                                 "sri v16.4s, v15.4s, #8\n"
                                 "sri v14.4s, v13.4s, #8\n"
                                 "str q12, [%[out_q]], #16\n"
                                 "sri v16.4s, v14.4s, #16\n"

                                 "str q16, [%[out_q]], #16\n"
                                 : [out_d] "+r"(out_d), [out_q] "+r"(out_q)
                                 : [factor] "r"(&factor)
                                 : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                                 "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "x1");
            break;
        }
        case 12: {
            __asm__ __volatile__("ldr s0, [%[factor]]\n"
                                 "dup v0.4s, v0.s[0]\n"

                                 "ld4 {v1.4s, v2.4s, v3.4s, v4.4s}, [%[out_d]], #64\n"
                                 "ld4 {v5.4s, v6.4s, v7.4s, v8.4s}, [%[out_d]], #64\n"
                                 "ld4 {v9.4s, v10.4s, v11.4s, v12.4s}, [%[out_d]], #64\n"

                                 "mul v4.4s, v4.4s, v0.4s\n"
                                 "mul v3.4s, v3.4s, v0.4s\n"
                                 "mul v2.4s, v2.4s, v0.4s\n"
                                 "mul v1.4s, v1.4s, v0.4s\n"

                                 "mul v8.4s, v8.4s, v0.4s\n"
                                 "sri v4.4s, v3.4s, #8\n"
                                 "mul v7.4s, v7.4s, v0.4s\n"
                                 "sri v2.4s, v1.4s, #8\n"
                                 "mul v6.4s, v6.4s, v0.4s\n"
                                 "mul v5.4s, v5.4s, v0.4s\n"
                                 "sri v4.4s, v2.4s, #16\n"

                                 "mul v12.4s, v12.4s, v0.4s\n"
                                 "sri v8.4s, v7.4s, #8\n"
                                 "mul v11.4s, v11.4s, v0.4s\n"
                                 "sri v6.4s, v5.4s, #8\n"
                                 "mul v10.4s, v10.4s, v0.4s\n"
                                 "str q4, [%[out_q]], #16\n"
                                 "mul v9.4s, v9.4s, v0.4s\n"
                                 "sri v8.4s, v6.4s, #16\n"

                                 "sri v12.4s, v11.4s, #8\n"
                                 "sri v10.4s, v9.4s, #8\n"
                                 "str q8, [%[out_q]], #16\n"
                                 "sri v12.4s, v10.4s, #16\n"

                                 "str q12, [%[out_q]], #16\n"
                                 : [out_d] "+r"(out_d), [out_q] "+r"(out_q)
                                 : [factor] "r"(&factor)
                                 : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                                 "v8", "v9", "v10", "v11", "v12", "x1");
            break;
        }
        case 8: {
            __asm__ __volatile__(
                "ldr s0, [%[factor]]\n"
                "dup v0.4s, v0.s[0]\n"

                "ld4 {v1.4s, v2.4s, v3.4s, v4.4s}, [%[out_d]], #64\n"
                "ld4 {v5.4s, v6.4s, v7.4s, v8.4s}, [%[out_d]], #64\n"

                "mul v4.4s, v4.4s, v0.4s\n"
                "mul v3.4s, v3.4s, v0.4s\n"
                "mul v2.4s, v2.4s, v0.4s\n"
                "mul v1.4s, v1.4s, v0.4s\n"

                "mul v8.4s, v8.4s, v0.4s\n"
                "sri v4.4s, v3.4s, #8\n"
                "mul v7.4s, v7.4s, v0.4s\n"
                "sri v2.4s, v1.4s, #8\n"
                "mul v6.4s, v6.4s, v0.4s\n"
                "mul v5.4s, v5.4s, v0.4s\n"
                "sri v4.4s, v2.4s, #16\n"

                "sri v8.4s, v7.4s, #8\n"
                "sri v6.4s, v5.4s, #8\n"
                "str q4, [%[out_q]], #16\n"
                "sri v8.4s, v6.4s, #16\n"

                "str q8, [%[out_q]], #16\n"
                : [out_d] "+r"(out_d), [out_q] "+r"(out_q)
                : [factor] "r"(&factor)
                : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "x1");
            break;
        }
        case 4: {
            __asm__ __volatile__("ldr s0, [%[factor]]\n"
                                 "dup v0.4s, v0.s[0]\n"

                                 "ld4 {v1.4s, v2.4s, v3.4s, v4.4s}, [%[out_d]], #64\n"

                                 "mul v4.4s, v4.4s, v0.4s\n"
                                 "mul v3.4s, v3.4s, v0.4s\n"
                                 "mul v2.4s, v2.4s, v0.4s\n"
                                 "mul v1.4s, v1.4s, v0.4s\n"

                                 "sri v4.4s, v3.4s, #8\n"
                                 "sri v2.4s, v1.4s, #8\n"
                                 "sri v4.4s, v2.4s, #16\n"

                                 "str q4, [%[out_q]], #16\n"
                                 : [out_d] "+r"(out_d), [out_q] "+r"(out_q)
                                 : [factor] "r"(&factor)
                                 : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "x1");
            break;
        }
        case 0: {
            break;
        }
        default: {
            CHECK_STATUS(NOT_MATCH);
        }
    }
}
#endif

inline void array_scale_round_i32_naive(
    const I32 *input, INT8 *output, I32 len, F32 alpha, bool clamp)
{
    float32x4_t alpha_v = vdupq_n_f32(alpha);
    I32 *p = (I32 *)output;
    I32 i = 0;
#pragma unroll(4)
    for (; i < len - 15; i += 16, p += 4) {
        int32x4x4_t g = vld4q_s32(input + i);
        g.val[0] = vcvtq_s32_f32(vmulq_f32(alpha_v, vcvtq_f32_s32(g.val[0])));
        g.val[1] = vcvtq_s32_f32(vmulq_f32(alpha_v, vcvtq_f32_s32(g.val[1])));
        g.val[2] = vcvtq_s32_f32(vmulq_f32(alpha_v, vcvtq_f32_s32(g.val[2])));
        g.val[3] = vcvtq_s32_f32(vmulq_f32(alpha_v, vcvtq_f32_s32(g.val[3])));
        int32x4_t p0 = vsliq_n_s32(g.val[0], g.val[1], 8);
        int32x4_t p1 = vsliq_n_s32(g.val[2], g.val[3], 8);
        int32x4_t p2 = vsliq_n_s32(p0, p1, 16);
        vst1q_s32(p, p2);
    }
    for (; i < len; i++) {
        output[i] = alpha * input[i];
    }
}

inline void array_scale_round_i32(const I32 *input, INT8 *output, I32 len, F32 _alpha, bool clamp)
{
    I32 alpha = _alpha * 16777216;
    int32x4_t alpha_v = vdupq_n_s32(alpha);
    I32 *p = (I32 *)output;
    I32 i = 0;
#pragma unroll(4)
    for (; i < len - 15; i += 16, p += 4) {
        int32x4x4_t g = vld4q_s32(input + i);
        g.val[0] = vmulq_s32(alpha_v, g.val[0]);
        g.val[1] = vmulq_s32(alpha_v, g.val[1]);
        g.val[2] = vmulq_s32(alpha_v, g.val[2]);
        g.val[3] = vmulq_s32(alpha_v, g.val[3]);
        int32x4_t p0 = vsriq_n_s32(g.val[1], g.val[0], 8);
        int32x4_t p1 = vsriq_n_s32(g.val[3], g.val[2], 8);
        int32x4_t p2 = vsriq_n_s32(p1, p0, 16);
        vst1q_s32(p, p2);
    }
    for (; i < len; i++) {
        output[i] = (alpha * input[i]) >> 24;
    }
}
#endif
