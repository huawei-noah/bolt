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
#include <math.h>

#include "error.h"
#include "cpu/arm/fp16/mmm_common.h"
#include "cpu/arm/fp16/mmm.h"

#define MMM_FMA_4x8_V5V14s3_V1xV0    \
    "fmla  v5.8h,  v1.8h, v0.h[0]\n" \
    "fmla  v8.8h,  v1.8h, v0.h[1]\n" \
    "fmla v11.8h,  v1.8h, v0.h[2]\n" \
    "fmla v14.8h,  v1.8h, v0.h[3]\n"
#define MMM_FMA_4x8_V17V26s3_V1xV0   \
    "fmla v17.8h,  v1.8h, v0.h[4]\n" \
    "fmla v20.8h,  v1.8h, v0.h[5]\n" \
    "fmla v23.8h,  v1.8h, v0.h[6]\n" \
    "fmla v26.8h,  v1.8h, v0.h[7]\n"
#define MMM_FMA_4x8_V6V15s3_V2xV0    \
    "fmla  v6.8h,  v2.8h, v0.h[0]\n" \
    "fmla  v9.8h,  v2.8h, v0.h[1]\n" \
    "fmla v12.8h,  v2.8h, v0.h[2]\n" \
    "fmla v15.8h,  v2.8h, v0.h[3]\n"
#define MMM_FMA_4x8_V18V27s3_V2xV0   \
    "fmla v18.8h,  v2.8h, v0.h[4]\n" \
    "fmla v21.8h,  v2.8h, v0.h[5]\n" \
    "fmla v24.8h,  v2.8h, v0.h[6]\n" \
    "fmla v27.8h,  v2.8h, v0.h[7]\n"
#define MMM_FMA_4x8_V7V16s3_V3xV0    \
    "fmla  v7.8h,  v3.8h, v0.h[0]\n" \
    "fmla v10.8h,  v3.8h, v0.h[1]\n" \
    "fmla v13.8h,  v3.8h, v0.h[2]\n" \
    "fmla v16.8h,  v3.8h, v0.h[3]\n"
#define MMM_FMA_4x8_V19V28s3_V3xV0   \
    "fmla v19.8h,  v3.8h, v0.h[4]\n" \
    "fmla v22.8h,  v3.8h, v0.h[5]\n" \
    "fmla v25.8h,  v3.8h, v0.h[6]\n" \
    "fmla v28.8h,  v3.8h, v0.h[7]\n"
#define MMM_FMA_4x8_V5V14s3_V29xV4   \
    "fmla  v5.8h, v29.8h, v4.h[0]\n" \
    "fmla  v8.8h, v29.8h, v4.h[1]\n" \
    "fmla v11.8h, v29.8h, v4.h[2]\n" \
    "fmla v14.8h, v29.8h, v4.h[3]\n"
#define MMM_FMA_4x8_V17V26s3_V29xV4  \
    "fmla v17.8h, v29.8h, v4.h[4]\n" \
    "fmla v20.8h, v29.8h, v4.h[5]\n" \
    "fmla v23.8h, v29.8h, v4.h[6]\n" \
    "fmla v26.8h, v29.8h, v4.h[7]\n"
#define MMM_FMA_4x8_V6V15s3_V30xV4   \
    "fmla  v6.8h, v30.8h, v4.h[0]\n" \
    "fmla  v9.8h, v30.8h, v4.h[1]\n" \
    "fmla v12.8h, v30.8h, v4.h[2]\n" \
    "fmla v15.8h, v30.8h, v4.h[3]\n"
#define MMM_FMA_4x8_V18V27s3_V30xV4  \
    "fmla v18.8h, v30.8h, v4.h[4]\n" \
    "fmla v21.8h, v30.8h, v4.h[5]\n" \
    "fmla v24.8h, v30.8h, v4.h[6]\n" \
    "fmla v27.8h, v30.8h, v4.h[7]\n"
#define MMM_FMA_4x8_V7V16s3_V31xV4   \
    "fmla  v7.8h, v31.8h, v4.h[0]\n" \
    "fmla v10.8h, v31.8h, v4.h[1]\n" \
    "fmla v13.8h, v31.8h, v4.h[2]\n" \
    "fmla v16.8h, v31.8h, v4.h[3]\n"
#define MMM_FMA_4x8_V19V28s3_V31xV4  \
    "fmla v19.8h, v31.8h, v4.h[4]\n" \
    "fmla v22.8h, v31.8h, v4.h[5]\n" \
    "fmla v25.8h, v31.8h, v4.h[6]\n" \
    "fmla v28.8h, v31.8h, v4.h[7]\n"

inline void mmm_4x24_A76(U32 M, U32 K, F16 *w, F16 *in, F16 *out)
{
    U32 KTail = K % 2;
    U32 KInner = K - KTail;
    asm volatile(
        // init in0- > v1, w- > v0
        "ld1 {v1.8h}, [%1], #16\n"
        "ld1 {v0.4h}, [%2], #8\n"
        "mov x26, %0\n"
        "ld1 {v5.8h, v6.8h, v7.8h},    [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v8.8h, v9.8h, v10.8h},   [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v11.8h, v12.8h, v13.8h}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v14.8h, v15.8h, v16.8h}, [x26]\n"

        "mov x20, %3\n"
        "cbz x20, 2f\n"

        "0:\n"
        // w- > v4, in0- > v2/v3/v1, out0=v5~v28
        "ld1 {v2.8h}, [%1], #16\n" MMM_FMA_4x8_V5V14s3_V1xV0 "ld1 {v3.8h}, [%1], #16\n"
        "ld1 {v29.8h}, [%1], #16\n" MMM_FMA_4x8_V6V15s3_V2xV0 "ld1 {v4.4h}, [%2], "
        "#8\n" MMM_FMA_4x8_V7V16s3_V3xV0

        // w- > v0, in0- > v2/v3/v1, out0- > v5~v28
        "ld1 {v30.8h}, [%1], #16\n" MMM_FMA_4x8_V5V14s3_V29xV4
        "ld1 {v31.8h}, [%1], #16\n" MMM_FMA_4x8_V6V15s3_V30xV4 "ld1 {v1.8h}, [%1], #16\n"
        "ld1 {v0.4h}, [%2], #8\n" MMM_FMA_4x8_V7V16s3_V31xV4

        "subs x20, x20, #0x2\n"
        "bne 0b\n"

        "2:\n"
        "cbz %5, 1f\n"
        "ld1 {v2.8h}, [%1], #16\n" MMM_FMA_4x8_V5V14s3_V1xV0
        "ld1 {v3.8h}, [%1], #16\n" MMM_FMA_4x8_V6V15s3_V2xV0 MMM_FMA_4x8_V7V16s3_V3xV0

        "1:\n"
        "mov x26, %0\n"
        "st1 {v5.8h, v6.8h, v7.8h},    [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v8.8h, v9.8h, v10.8h},   [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v11.8h, v12.8h, v13.8h}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v14.8h, v15.8h, v16.8h}, [x26]\n"
        : "+r"(out), "+r"(in), "+r"(w)
        : "r"((I64)KInner), "r"((I64)M), "r"((I64)KTail)
        : "memory", "cc", "x20", "x26", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v29", "v30", "v31");
}
inline void mmm_8x4_A76(U32 M, U32 K, F16 *w, F16 *in, F16 *out)
{
    U32 KTail = K % 2;
    U32 KInner = K - KTail;
    asm volatile("ld1 {v1.8h}, [%2], #16\n"
                 "ld1 {v0.4h}, [%1], #8\n"

                 "mov x26, %0\n"
                 "ld1 {v5.h}[0], [x26], #2\n"
                 "ld1 {v8.h}[0], [x26], #2\n"
                 "ld1 {v11.h}[0], [x26], #2\n"
                 "ld1 {v14.h}[0], [x26], #2\n"
                 "sub x26, x26, #8\n"
                 "add x26, x26, %4\n"
                 "ld1 {v5.h}[1], [x26], #2\n"
                 "ld1 {v8.h}[1], [x26], #2\n"
                 "ld1 {v11.h}[1], [x26], #2\n"
                 "ld1 {v14.h}[1], [x26], #2\n"
                 "sub x26, x26, #8\n"
                 "add x26, x26, %4\n"
                 "ld1 {v5.h}[2], [x26], #2\n"
                 "ld1 {v8.h}[2], [x26], #2\n"
                 "ld1 {v11.h}[2], [x26], #2\n"
                 "ld1 {v14.h}[2], [x26], #2\n"
                 "sub x26, x26, #8\n"
                 "add x26, x26, %4\n"
                 "ld1 {v5.h}[3], [x26], #2\n"
                 "ld1 {v8.h}[3], [x26], #2\n"
                 "ld1 {v11.h}[3], [x26], #2\n"
                 "ld1 {v14.h}[3], [x26], #2\n"
                 "sub x26, x26, #8\n"
                 "add x26, x26, %4\n"
                 "ld1 {v5.h}[4], [x26], #2\n"
                 "ld1 {v8.h}[4], [x26], #2\n"
                 "ld1 {v11.h}[4], [x26], #2\n"
                 "ld1 {v14.h}[4], [x26], #2\n"
                 "sub x26, x26, #8\n"
                 "add x26, x26, %4\n"
                 "ld1 {v5.h}[5], [x26], #2\n"
                 "ld1 {v8.h}[5], [x26], #2\n"
                 "ld1 {v11.h}[5], [x26], #2\n"
                 "ld1 {v14.h}[5], [x26], #2\n"
                 "sub x26, x26, #8\n"
                 "add x26, x26, %4\n"
                 "ld1 {v5.h}[6], [x26], #2\n"
                 "ld1 {v8.h}[6], [x26], #2\n"
                 "ld1 {v11.h}[6], [x26], #2\n"
                 "ld1 {v14.h}[6], [x26], #2\n"
                 "sub x26, x26, #8\n"
                 "add x26, x26, %4\n"
                 "ld1 {v5.h}[7], [x26], #2\n"
                 "ld1 {v8.h}[7], [x26], #2\n"
                 "ld1 {v11.h}[7], [x26], #2\n"
                 "ld1 {v14.h}[7], [x26], #2\n"

                 "mov x20, %3\n"
                 "cbz x20, 2f\n"

                 "0:\n"
                 "ld1 {v4.4h}, [%1], #8\n"
                 "ld1 {v29.8h}, [%2], #16\n" MMM_FMA_4x8_V5V14s3_V1xV0 "ld1 {v1.8h}, [%2], #16\n"
                 "ld1 {v0.4h}, [%1], #8\n" MMM_FMA_4x8_V5V14s3_V29xV4

                 "subs x20, x20, 0x2\n"
                 "bne 0b\n"

                 "2:\n"
                 "cbz %5, 1f\n" MMM_FMA_4x8_V5V14s3_V1xV0

                 "1:\n"
                 "mov x26, %0\n"
                 "st1 {v5.h}[0], [x26], #2\n"
                 "st1 {v8.h}[0], [x26], #2\n"
                 "st1 {v11.h}[0], [x26], #2\n"
                 "st1 {v14.h}[0], [x26], #2\n"
                 "sub x26, x26, #8\n"
                 "add x26, x26, %4\n"
                 "st1 {v5.h}[1], [x26], #2\n"
                 "st1 {v8.h}[1], [x26], #2\n"
                 "st1 {v11.h}[1], [x26], #2\n"
                 "st1 {v14.h}[1], [x26], #2\n"
                 "sub x26, x26, #8\n"
                 "add x26, x26, %4\n"
                 "st1 {v5.h}[2], [x26], #2\n"
                 "st1 {v8.h}[2], [x26], #2\n"
                 "st1 {v11.h}[2], [x26], #2\n"
                 "st1 {v14.h}[2], [x26], #2\n"
                 "sub x26, x26, #8\n"
                 "add x26, x26, %4\n"
                 "st1 {v5.h}[3], [x26], #2\n"
                 "st1 {v8.h}[3], [x26], #2\n"
                 "st1 {v11.h}[3], [x26], #2\n"
                 "st1 {v14.h}[3], [x26], #2\n"
                 "sub x26, x26, #8\n"
                 "add x26, x26, %4\n"
                 "st1 {v5.h}[4], [x26], #2\n"
                 "st1 {v8.h}[4], [x26], #2\n"
                 "st1 {v11.h}[4], [x26], #2\n"
                 "st1 {v14.h}[4], [x26], #2\n"
                 "sub x26, x26, #8\n"
                 "add x26, x26, %4\n"
                 "st1 {v5.h}[5], [x26], #2\n"
                 "st1 {v8.h}[5], [x26], #2\n"
                 "st1 {v11.h}[5], [x26], #2\n"
                 "st1 {v14.h}[5], [x26], #2\n"
                 "sub x26, x26, #8\n"
                 "add x26, x26, %4\n"
                 "st1 {v5.h}[6], [x26], #2\n"
                 "st1 {v8.h}[6], [x26], #2\n"
                 "st1 {v11.h}[6], [x26], #2\n"
                 "st1 {v14.h}[6], [x26], #2\n"
                 "sub x26, x26, #8\n"
                 "add x26, x26, %4\n"
                 "st1 {v5.h}[7], [x26], #2\n"
                 "st1 {v8.h}[7], [x26], #2\n"
                 "st1 {v11.h}[7], [x26], #2\n"
                 "st1 {v14.h}[7], [x26], #2\n"
                 : "+r"(out), "+r"(in), "+r"(w)
                 : "r"((I64)KInner), "r"((I64)M), "r"((I64)KTail)
                 : "memory", "cc", "x20", "x26", "v0", "v1", "v4", "v29", "v5", "v8", "v11", "v14");
}

inline void mmm_4x8_A76(U32 M, U32 K, F16 *w, F16 *in, F16 *out)
{
    U32 KTail = K % 2;
    U32 KInner = K - KTail;
    asm volatile("ld1 {v1.8h}, [%1], #16\n"
                 "ld1 {v0.4h}, [%2], #8\n"
                 "mov x26, %0\n"
                 "ld1 {v5.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "ld1 {v8.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "ld1 {v11.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "ld1 {v14.8h}, [x26]\n"

                 "mov x20, %3\n"
                 "cbz x20, 2f\n"

                 "0:\n"
                 "ld1 {v29.8h}, [%1], #16\n"
                 "ld1 {v4.4h}, [%2], #8\n" MMM_FMA_4x8_V5V14s3_V1xV0 "ld1 {v1.8h}, [%1], #16\n"
                 "ld1 {v0.4h}, [%2], #8\n" MMM_FMA_4x8_V5V14s3_V29xV4

                 "subs x20, x20, 0x2\n"
                 "bne 0b\n"

                 "2:\n"
                 "cbz %5, 1f\n" MMM_FMA_4x8_V5V14s3_V1xV0

                 "1:\n"
                 "mov x26, %0\n"
                 "st1 {v5.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "st1 {v8.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "st1 {v11.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "st1 {v14.8h}, [x26]\n"
                 : "+r"(out), "+r"(in), "+r"(w)
                 : "r"((I64)KInner), "r"((I64)M), "r"((I64)KTail)
                 : "memory", "cc", "x20", "x26", "v0", "v1", "v4", "v5", "v8", "v11", "v14", "v29");
}

inline void mmm_4x4_A76(U32 M, U32 K, F16 *w, F16 *in, F16 *out)
{
    U32 KTail = K % 2;
    U32 KInner = K - KTail;
    asm volatile("ld1 {v1.4h}, [%1], #8\n"
                 "ld1 {v0.4h}, [%2], #8\n"
                 "mov x26, %0\n"
                 "ld1 {v5.4h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "ld1 {v8.4h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "ld1 {v11.4h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "ld1 {v14.4h}, [x26]\n"

                 "mov x20, %3\n"
                 "cbz x20, 2f\n"

                 "0:\n"
                 "ld1 {v29.4h}, [%1], #8\n"
                 "ld1 {v4.4h}, [%2], #8\n" MMM_FMA_4x8_V5V14s3_V1xV0 "ld1 {v1.4h}, [%1], #8\n"
                 "ld1 {v0.4h}, [%2], #8\n" MMM_FMA_4x8_V5V14s3_V29xV4

                 "subs x20, x20, 0x2\n"
                 "bne 0b\n"

                 "2:\n"
                 "cbz %5, 1f\n" MMM_FMA_4x8_V5V14s3_V1xV0

                 "1:\n"
                 "mov x26, %0\n"
                 "st1 {v5.4h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "st1 {v8.4h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "st1 {v11.4h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "st1 {v14.4h}, [x26]\n"
                 : "+r"(out), "+r"(in), "+r"(w)
                 : "r"((I64)KInner), "r"((I64)M), "r"((I64)KTail)
                 : "memory", "cc", "x20", "x26", "v0", "v1", "v4", "v29", "v5", "v8", "v11", "v14");
}

inline void mmm_8x8_A76(U32 M, U32 K, F16 *w, F16 *in, F16 *out)
{
    U32 KTail = K % 2;
    U32 KInner = K - KTail;
    asm volatile("mov x26, %0\n"
                 "ld1 {v5.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "ld1 {v8.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "ld1 {v11.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "ld1 {v14.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "ld1 {v17.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "ld1 {v20.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "ld1 {v23.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "ld1 {v26.8h}, [x26]\n"

                 "mov x20, %3\n"

                 "ld1 {v1.8h}, [%1], #16\n"
                 "ld1 {v0.8h}, [%2], #16\n"
                 "cbz x20, 2f\n"

                 "0:\n"
                 "ld1 {v29.8h}, [%1], #16\n"
                 "ld1 {v4.8h}, [%2], #16\n" MMM_FMA_4x8_V5V14s3_V1xV0 MMM_FMA_4x8_V17V26s3_V1xV0

                 "ld1 {v1.8h}, [%1], #16\n"
                 "ld1 {v0.8h}, [%2], #16\n" MMM_FMA_4x8_V5V14s3_V29xV4 MMM_FMA_4x8_V17V26s3_V29xV4

                 "subs x20, x20, 0x2\n"
                 "bne 0b\n"

                 "2:\n"
                 "cbz %5, 1f\n" MMM_FMA_4x8_V5V14s3_V1xV0 MMM_FMA_4x8_V17V26s3_V1xV0

                 "1:\n"
                 "mov x26, %0\n"
                 "st1 {v5.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "st1 {v8.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "st1 {v11.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "st1 {v14.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "st1 {v17.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "st1 {v20.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "st1 {v23.8h}, [x26]\n"
                 "add x26, x26, %4\n"
                 "st1 {v26.8h}, [x26]\n"
                 : "+r"(out), "+r"(in), "+r"(w)
                 : "r"((I64)KInner), "r"((I64)M), "r"((I64)KTail)
                 : "memory", "cc", "x20", "x26", "v1", "v0", "v29", "v4", "v5", "v8", "v11", "v14",
                 "v17", "v20", "v23", "v26");
}

inline void mmm_8x24_A76(U32 M, U32 K, F16 *w, F16 *in, F16 *out)
{
    U32 KTail = K % 2;
    U32 KInner = K - KTail;
    asm volatile(
        // init in0- > v1, w- > v0
        "ld1 {v1.8h}, [%1], #16\n"
        "ld1 {v0.8h}, [%2], #16\n"
        "mov x26, %0\n"
        "ld1 {v5.8h, v6.8h, v7.8h},    [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v8.8h, v9.8h, v10.8h},   [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v11.8h, v12.8h, v13.8h}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v14.8h, v15.8h, v16.8h}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v17.8h, v18.8h, v19.8h}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v20.8h, v21.8h, v22.8h}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v23.8h, v24.8h, v25.8h}, [x26]\n"
        "add x26, x26, %4\n"
        "ld1 {v26.8h, v27.8h, v28.8h}, [x26]\n"

        "mov x20, %3\n"
        "cbz x20, 2f\n"

        "0:\n"
        // w- > v4, in0- > v2/v3/v1, out0=v5~v28
        "ld1 {v2.8h}, [%1], #16\n"
        "ld1 {v3.8h}, [%1], #16\n" MMM_FMA_4x8_V5V14s3_V1xV0 MMM_FMA_4x8_V17V26s3_V1xV0

        "ld1 {v4.8h}, [%2], #16\n" MMM_FMA_4x8_V6V15s3_V2xV0 MMM_FMA_4x8_V18V27s3_V2xV0

        "ld1 {v29.8h}, [%1], #16\n" MMM_FMA_4x8_V7V16s3_V3xV0 MMM_FMA_4x8_V19V28s3_V3xV0

        // w- > v0, in0- > v2/v3/v1, out0- > v5~v28
        "ld1 {v30.8h}, [%1], #16\n"
        "ld1 {v0.8h}, [%2], #16\n" MMM_FMA_4x8_V5V14s3_V29xV4 MMM_FMA_4x8_V17V26s3_V29xV4

        "ld1 {v31.8h}, [%1], #16\n" MMM_FMA_4x8_V6V15s3_V30xV4 MMM_FMA_4x8_V18V27s3_V30xV4

        "ld1 {v1.8h}, [%1], #16\n" MMM_FMA_4x8_V7V16s3_V31xV4 "subs x20, x20, "
        "#0x2\n" MMM_FMA_4x8_V19V28s3_V31xV4

        "bne 0b\n"

        "2:\n"
        "cbz %5, 1f\n"
        "ld1 {v2.8h}, [%1], #16\n"
        "ld1 {v3.8h}, [%1], #16\n" MMM_FMA_4x8_V5V14s3_V1xV0 MMM_FMA_4x8_V17V26s3_V1xV0
            MMM_FMA_4x8_V6V15s3_V2xV0 MMM_FMA_4x8_V18V27s3_V2xV0 MMM_FMA_4x8_V7V16s3_V3xV0
                MMM_FMA_4x8_V19V28s3_V3xV0

        "1:\n"
        "mov x26, %0\n"
        "st1 {v5.8h, v6.8h, v7.8h},    [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v8.8h, v9.8h, v10.8h},   [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v11.8h, v12.8h, v13.8h}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v14.8h, v15.8h, v16.8h}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v17.8h, v18.8h, v19.8h}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v20.8h, v21.8h, v22.8h}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v23.8h, v24.8h, v25.8h}, [x26]\n"
        "add x26, x26, %4\n"
        "st1 {v26.8h, v27.8h, v28.8h}, [x26]\n"
        : "+r"(out), "+r"(in), "+r"(w)
        : "r"((I64)KInner), "r"((I64)M), "r"((I64)KTail)
        : "memory", "cc", "x0", "x20", "x26", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
        "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

void mmm_A76(int M, int N, int K, bool transposeA, F16 *matrix1, F16 *matrix2, F16 *tmp, F16 *result)
{
    int blockK = K;
    int blockM = 192;
    F16 *matrix1Trans = tmp;
    F16 *resultCurrent = result;
    int KInner, MInner, m, n;
    for (int k = 0; k < K; k += blockK) {
        KInner = UNI_MIN(blockK, K - k);
        for (int i = 0; i < M; i += blockM) {
            MInner = UNI_MIN(blockM, M - i);
            for (n = 0; n <= N - 8; n += 8) {
                if (i == 0) {
                    if (transposeA) {
                        matrix2_trans(8, KInner, N, matrix1 + n, matrix1Trans + n * KInner);
                    } else {
                        matrix1_trans(8, KInner, K, matrix1 + n * K + k, matrix1Trans + n * KInner);
                    }
                }
                for (m = 0; m <= (MInner - 24); m += 24) {
                    resultCurrent = result + n * M + m + i;
                    mmm_8x24_A76(M * 2, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                }
                for (; m <= (MInner - 8); m += 8) {
                    resultCurrent = result + n * M + m + i;
                    mmm_8x8_A76(M * 2, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                }

                if ((MInner - m) >= 4) {
                    resultCurrent = result + n * M + m + i;
                    mmm_8x4_A76(M * 2, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                    m += 4;
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_N8_MTail(MInner - m, M, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                }
            }

            if ((N - n) >= 4) {
                if (i == 0) {
                    if (transposeA) {
                        matrix2_trans(4, KInner, N, matrix1 + n, matrix1Trans + n * KInner);
                    } else {
                        matrix1_trans(4, KInner, K, matrix1 + n * K + k, matrix1Trans + n * KInner);
                    }
                }

                for (m = 0; m <= (MInner - 24); m += 24) {
                    resultCurrent = result + n * M + m + i;
                    mmm_4x24_A76(M * 2, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                }

                for (; m <= (MInner - 8); m += 8) {
                    resultCurrent = result + n * M + m + i;
                    mmm_4x8_A76(M * 2, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                }

                if ((MInner - m) >= 4) {
                    resultCurrent = result + n * M + m + i;
                    mmm_4x4_A76(M * 2, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                    m += 4;
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_N4_MTail(MInner - m, M, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                }

                n += 4;
            }

            if (N - n) {
                if (i == 0) {
                    if (transposeA) {
                        matrix2_trans(N - n, KInner, N, matrix1 + n, matrix1Trans + n * KInner);
                    } else {
                        matrix1_trans(
                            N - n, KInner, K, matrix1 + n * K + k, matrix1Trans + n * KInner);
                    }
                }

                for (m = 0; m <= (MInner - 24); m += 24) {
                    resultCurrent = result + n * M + m + i;
                    mmm_NTail_M24(M, N - n, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                }

                for (; m <= (MInner - 8); m += 8) {
                    resultCurrent = result + n * M + m + i;
                    mmm_NTail_M8(M, N - n, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                }

                if ((MInner - m) >= 4) {
                    resultCurrent = result + n * M + m + i;
                    mmm_NTail_M4(M, N - n, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                    m += 4;
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_NTail_M(MInner - m, M, N - n, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                }
            }
        }
    }
}
