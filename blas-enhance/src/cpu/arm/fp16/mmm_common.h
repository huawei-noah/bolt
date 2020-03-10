// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_MMM_COMMON
#define _H_MMM_COMMON
#include <string.h>
#include <arm_neon.h>

#include "type.h"


inline void matrix1_trans(U32 size, U32 blockK, U32 K, F16* src, F16* dst) {
    F16* src1 = src;
    U32 offset;
    for (U32 i = 0; i < blockK; i++) {
        for (U32 j = 0; j < size; j++) {
            src1 = src + j * K;
            offset = 8 * blockK;
            if (i % 32) {
                asm volatile(
                    "prfm pldl2keep, [%0, %1]\n"
                    :"+r" (src1)
                    :"r"((I64)offset)
                    :"memory","cc"
                );
            }
            *dst++ = *(src1 + i);
        }
    }
}

inline void matrix2_trans(U32 size, U32 blockK, U32 M, F16* src, F16* dst) {
    for (U32 i = 0; i < blockK; i++) {
        asm volatile(
            "prfm pldl2keep, [%0, #48]\n"
            :"+r" (src)
            :
            :"memory","cc"
        );
        memcpy(dst, src, size * sizeof (F16));
        dst += size;
        src += M;
    }
}

inline void mmm_NTail_M24(U32 M, U32 N, U32 K, F16* matrix1, F16* matrix2, F16* result) {
    float16x8x3_t mat2, res;
    for (U32 i = 0; i < N; i++) {
        res = vld3q_f16(result + i * M);
        for (U32 q = 0; q < K; q+=1) {
            mat2 = vld3q_f16(matrix2 + q * 24);
            res.val[0] = vfmaq_n_f16(res.val[0], mat2.val[0], matrix1[q * N + i]);
            res.val[1] = vfmaq_n_f16(res.val[1], mat2.val[1], matrix1[q * N + i]);
            res.val[2] = vfmaq_n_f16(res.val[2], mat2.val[2], matrix1[q * N + i]);
        }
        vst3q_f16(result + i * M, res);
    }
}

inline void mmm_NTail_M8(U32 M, U32 N, U32 K, F16* matrix1, F16* matrix2, F16* result) {
    float16x8_t mat2, res;
    for (U32 i = 0; i < N; i++) {
        res = vld1q_f16(result + i * M);
        for (U32 q = 0; q < K; q+=1) {
            mat2 = vld1q_f16(matrix2 + q * 8);
            res = vfmaq_n_f16(res, mat2, matrix1[q * N + i]);
        }
        vst1q_f16(result + i * M, res);
    }
}

inline void mmm_NTail_M4(U32 M, U32 N, U32 K, F16* matrix1, F16* matrix2, F16* result) {
    float16x4_t mat2, res;
    for (U32 i = 0; i < N; i++) {
        res = vld1_f16(result + i * M);
        for (U32 q = 0; q < K; q+=1) {
            mat2 = vld1_f16(matrix2 + q * 4);
            res = vfma_n_f16(res, mat2, matrix1[q * N + i]);
        }
        vst1_f16(result + i * M, res);
    }
}

inline void mmm_NTail_M(U32 MInner, U32 M, U32 N, U32 K, F16* matrix1, F16* matrix2, F16* result) {
    for(U32 i = 0; i < N; i++) {
        for(U32 j = 0; j < MInner; j++) {
            for(U32 k = 0; k < K; k++) {
                result[i * M + j] += *(matrix1 + k * N + i) * *(matrix2 + k * MInner + j);
            }

        }
    }
}

inline void mmm_N8_MTail(U32 MInner, U32 M, U32 K, F16* matrix1, F16* matrix2, F16* result) {
    float16x8_t mat1 = {0}, res[4] = {0};
    F16 tmp[8] = {0};
    CHECK_REQUIREMENT(MInner < 4);

    for(U32 i = 0; i < K; i++) {
        mat1 = vld1q_f16(matrix1 + i * 8);
        for(U32 j = 0; j < MInner; j++) {
            res[j] = vfmaq_n_f16(res[j], mat1, matrix2[j + i * MInner]);
        }
    }
    for(U32 p = 0; p < MInner; p++) {
        vst1q_f16(tmp, res[p]);
        for(U32 q = 0; q < 8; q++) {
            result[q * M + p] += tmp[q];
        }
        res[p] = vdupq_n_f16(0);
    }
}

inline void mmm_N4_MTail(U32 MInner, U32 M, U32 K, F16* matrix1, F16* matrix2, F16* result) {
    float16x4_t mat1 = {0}, res[4] = {0};
    F16 tmp[4] = {0};
    CHECK_REQUIREMENT(MInner < 4);

    for(U32 i = 0; i < K; i++) {
        mat1 = vld1_f16(matrix1 + i * 4);
        for(U32 j = 0; j < MInner; j++) {
            res[j] = vfma_n_f16(res[j], mat1, matrix2[j + i * MInner]);
        }
    }
    for(U32 p = 0; p < MInner; p++) {
        vst1_f16(tmp, res[p]);
        for(U32 q = 0; q < 4; q++) {
            result[q * M + p] += tmp[q];
        }
        res[p] = vdup_n_f16(0);
    }
}
#endif
