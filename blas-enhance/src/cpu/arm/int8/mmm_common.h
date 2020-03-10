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

#ifdef _USE_INT8
#include <string.h> 
#include <arm_neon.h>

#include "type.h"
#include "error.h"
#include "cpu/arm/arm_neon_expand.h"

inline void matrix1_trans_n8(U32 blockK, U32 K, INT8* src, INT8* dst)
{
    // Move k4 as one I32
    I32* dst1 = (I32*)dst;

    I32 *in[8];
    for (U32 i=0; i<8; i++) {
        in[i] = (I32*)(src + i * K);
    }
    U32 k = 0;
    for (; k<blockK-7; k+=8) {
        if(k % 64 == 0){
            asm volatile(
                "prfm pldl2keep, [%[in0], 64]\n"
                "prfm pldl2keep, [%[in1], 64]\n"
                "prfm pldl2keep, [%[in2], 64]\n"
                "prfm pldl2keep, [%[in3], 64]\n"
                "prfm pldl2keep, [%[in4], 64]\n"
                "prfm pldl2keep, [%[in5], 64]\n"
                "prfm pldl2keep, [%[in6], 64]\n"
                "prfm pldl2keep, [%[in7], 64]\n"
                :[in0]"+r"(in[0]),
                [in1]"+r"(in[1]),
                [in2]"+r"(in[2]),
                [in3]"+r"(in[3]),
                [in4]"+r"(in[4]),
                [in5]"+r"(in[5]),
                [in6]"+r"(in[6]),
                [in7]"+r"(in[7])
                :
                :"memory","cc" 
            );
        }
        asm volatile(
            "ldr d0, [%[in0]], 8\n"
            "ldr d1, [%[in1]], 8\n"
            "ldr d2, [%[in2]], 8\n"
            "ldr d3, [%[in3]], 8\n"
            "ldr d4, [%[in4]], 8\n"
            "ldr d5, [%[in5]], 8\n"
            "ldr d6, [%[in6]], 8\n"
            "ldr d7, [%[in7]], 8\n"

            "zip1 v8.2s, v0.2s, v1.2s\n"
            "zip2 v12.2s, v0.2s, v1.2s\n"
            "zip1 v9.2s, v2.2s, v3.2s\n"
            "zip2 v13.2s, v2.2s, v3.2s\n"
            "zip1 v10.2s, v4.2s, v5.2s\n"
            "zip2 v14.2s, v4.2s, v5.2s\n"
            "zip1 v11.2s, v6.2s, v7.2s\n"
            "zip2 v15.2s, v6.2s, v7.2s\n"
            
            "str d8, [%[out]]\n"
            "str d9, [%[out], 8]\n"
            "str d10, [%[out], 16]\n"
            "str d11, [%[out], 24]\n"
            "str d12, [%[out], 32]\n"
            "str d13, [%[out], 40]\n"
            "str d14, [%[out], 48]\n"
            "str d15, [%[out], 56]\n"
            :[in0]"+r"(in[0]),
             [in1]"+r"(in[1]),
             [in2]"+r"(in[2]),
             [in3]"+r"(in[3]),
             [in4]"+r"(in[4]),
             [in5]"+r"(in[5]),
             [in6]"+r"(in[6]),
             [in7]"+r"(in[7])
            :[out]"r"(dst1)
            :"memory","cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
        );
        dst1 += 16;
    }

    if (k != blockK) {
        for (U32 i=0; i<8; i++) {
            dst1[i] = in[i][0];
        }
    }
}

//Trans from NK to NKn(size)k4
inline void matrix1_trans_int8(U32 size, U32 blockK, U32 K, INT8* src, INT8* dst)
{
    // Move k4 as one I32
    I32* src1;
    I32* dst1 = (I32*)dst;
    U32 offset = 64;

    for(U32 i = 0; i < blockK/4; i++){
        for(U32 j = 0; j < size; j++){
            src1 = (I32*)(src + j * K);

            if(i % 16 == 0){
                asm volatile(
                    "prfm pldl2keep, [%0, %1]\n"
                    :"+r"(src1)
                    :"r"((I64)offset)
                    :"memory","cc" 
                );
            }
            *dst1++ = *(src1 + i);
        }
    }
}

inline void matrix2_trans_m12(U32 blockK, U32 M, INT8* src, INT8* dst)
{
    INT8* src1 = src;
    INT8* dst1 = dst;
    U32 offset = 4 * M;

    for (U32 i = 0; i < blockK; i+=4) {
        // Prefetch for the next iteration
        asm volatile(
            "prfm pldl2keep, [%0, %1]\n"
            :"+r"(src1)
            :"r"((I64)offset)
            :"memory","cc" 
        );

        INT8 *in12[4];
        for (U32 j=0; j<4; j++) {
            in12[j] = src1 + j * M;
        }
        src1 += offset;

        asm volatile(
            "ldr d0, [%[in0]]\n"
            "ldr d1, [%[in1]]\n"
            "ldr d2, [%[in2]]\n"
            "ldr d3, [%[in3]]\n"
            "zip1 v4.8b, v0.8b, v1.8b\n"
            "zip2 v5.8b, v0.8b, v1.8b\n"
            "zip1 v6.8b, v2.8b, v3.8b\n"
            "zip2 v7.8b, v2.8b, v3.8b\n"

            "zip1 v0.4h, v4.4h, v6.4h\n"
            "zip2 v1.4h, v4.4h, v6.4h\n"
            "zip1 v2.4h, v5.4h, v7.4h\n"
            "zip2 v3.4h, v5.4h, v7.4h\n"
            "str d0, [%[out]]\n"
            "str d1, [%[out], 8]\n"
            "str d2, [%[out], 16]\n"
            "str d3, [%[out], 24]\n"
            :
            :[in0]"r"(in12[0]),
             [in1]"r"(in12[1]),
             [in2]"r"(in12[2]),
             [in3]"r"(in12[3]),
             [out]"r"(dst1)
            :"memory","cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
        );

        for (U32 j=0; j<4; j++) {
            for (U32 k=0; k<4; k++) {
                dst1[32 + j*4 + k] = in12[k][8+j];
            }
        }

        dst1 += 48;
    }
}

//Trans from KM to MKm(size)k4
inline void matrix2_trans_int8(U32 size, U32 blockK, U32 M, INT8* src, INT8* dst)
{
    INT8* src1 = src;
    INT8* dst1 = dst;
    U32 offset = 4 * M;

    for(U32 i = 0; i < blockK; i+=4){
        src1 = src + i * M;
        asm volatile(
            "prfm pldl2keep, [%0, %1]\n"
            :"+r"(src1)
            :"r"((I64)offset)
            :"memory","cc" 
        );
        for(U32 j = 0; j < size; j++){
            src1 = src + i * M + j;
            for(U32 k=0; k<4; k++){
                *dst1 = *src1;
                dst1++;
                src1 += M;
            }
        }
    }
}

inline void mmm_N8_MTail(U32 MInner, U32 M, U32 K, INT8* matrix1, INT8* matrix2, I32* result)
{
    int8x16_t mat1[2];
    int8x16_t mat2;
    int32x4_t res[4][2] = {{0}};
    I32 tmp[8] = {0};

    CHECK_REQUIREMENT(MInner < 4);

    for(U32 i = 0; i < K; i+=4){
        mat1[0] = vld1q_s8(matrix1 + i * 8);
        mat1[1] = vld1q_s8(matrix1 + i * 8 + 16);

        mat2 = vld1q_s8(matrix2 + i * MInner);

        for(U32 j = 0; j < MInner; j++){
            res[j][0] = vdotq_laneq_s32_builtin(res[j][0], mat1[0], mat2, j);
            res[j][1] = vdotq_laneq_s32_builtin(res[j][1], mat1[1], mat2, j);
        }
    }
    for(U32 p = 0; p < MInner; p++){
        vst1q_s32(tmp, res[p][0]);
        vst1q_s32(tmp+4, res[p][1]);
        for(U32 q = 0; q < 8; q++){
            result[q * M + p] += tmp[q];
        }
        res[p][0] = vdupq_n_s32(0);
        res[p][1] = vdupq_n_s32(0);
    }
}

inline void mmm_N4_MTail(U32 MInner, U32 M, U32 K, INT8* matrix1, INT8* matrix2, I32* result)
{
    int8x16_t mat1 = {0};
    int8x16_t mat2 = {0};
    int32x4_t res[4] = {0};
    I32 tmp[8] = {0};

    CHECK_REQUIREMENT(MInner < 4);

    for(U32 i = 0; i < K; i+=4){
        mat1 = vld1q_s8(matrix1 + i * 8);

        mat2 = vld1q_s8(matrix2 + i * MInner);

        for(U32 j = 0; j < MInner; j++){
            res[j] = vdotq_laneq_s32_builtin(res[j], mat1, mat2, j);
        }
    }
    for(U32 p = 0; p < MInner; p++){
        vst1q_s32(tmp, res[p]);
        for(U32 q = 0; q < 8; q++){
            result[q * M + p] += tmp[q];
        }
        res[p] = vdupq_n_s32(0);
    }
}

inline void mmm_NTail_M12(U32 M, U32 N, U32 K, INT8* matrix1, INT8* matrix2, I32* result) {
    int8x16_t mat1 = {0};
    int8x16_t mat2[3] = {0};
    int32x4_t res[4][3] = {{0}};

    for (U32 i = 0; i < N; i++) {
        res[i][0] = vld1q_s32(result + i*M);
        res[i][1] = vld1q_s32(result + i*M + 4);
        res[i][2] = vld1q_s32(result + i*M + 8);
    }

    for (U32 q=0; q<K; q+=4) {
        mat1 = vld1q_s8(matrix1 + q * N);

        mat2[0] = vld1q_s8(matrix2 + q*12);
        mat2[1] = vld1q_s8(matrix2 + q*12 + 16);
        mat2[2] = vld1q_s8(matrix2 + q*12 + 32);

        for (U32 n=0; n<N; n++) {
            res[n][0] = vdotq_laneq_s32_builtin(res[n][0], mat2[0], mat1, n);
            res[n][1] = vdotq_laneq_s32_builtin(res[n][1], mat2[1], mat1, n);
            res[n][2] = vdotq_laneq_s32_builtin(res[n][2], mat2[2], mat1, n);
        }
    }

    for (U32 i = 0; i < N; i++) {
        vst1q_s32(result + i*M, res[i][0]);
        vst1q_s32(result + i*M + 4, res[i][1]);
        vst1q_s32(result + i*M + 8, res[i][2]);
    }
}

inline void mmm_NTail_M8(U32 M, U32 N, U32 K, INT8* matrix1, INT8* matrix2, I32* result) {
    int8x16_t mat1 = {0};
    int8x16_t mat2[2] = {0};
    int32x4_t res[4][2] = {{0}};

    for (U32 i = 0; i < N; i++) {
        res[i][0] = vld1q_s32(result + i*M);
        res[i][1] = vld1q_s32(result + i*M + 4);
    }

    for (U32 q=0; q<K; q+=4) {
        mat1 = vld1q_s8(matrix1 + q * N);

        mat2[0] = vld1q_s8(matrix2 + q*8);
        mat2[1] = vld1q_s8(matrix2 + q*8 + 16);

        for (U32 n=0; n<N; n++) {
            res[n][0] = vdotq_laneq_s32_builtin(res[n][0], mat2[0], mat1, n);
            res[n][1] = vdotq_laneq_s32_builtin(res[n][1], mat2[1], mat1, n);
        }
    }

    for (U32 i = 0; i < N; i++) {
        vst1q_s32(result + i*M, res[i][0]);
        vst1q_s32(result + i*M + 4, res[i][1]);
    }
}

inline void mmm_NTail_M4(U32 M, U32 N, U32 K, INT8* matrix1, INT8* matrix2, I32* result) {
    int8x16_t mat1 = {0};
    int8x16_t mat2 = {0};
    int32x4_t res[4] = {0};

    for (U32 i = 0; i < N; i++) {
        res[i] = vld1q_s32(result + i*M);
    }

    for (U32 q=0; q<K; q+=4) {
        mat1 = vld1q_s8(matrix1 + q * N);

        mat2 = vld1q_s8(matrix2 + q*4);

        for (U32 n=0; n<N; n++) {
            res[n] = vdotq_laneq_s32_builtin(res[n], mat2, mat1, n);
        }
    }

    for (U32 i = 0; i < N; i++) {
        vst1q_s32(result + i*M, res[i]);
    }
}

inline void mmm_NTail_M(U32 MInner, U32 M, U32 N, U32 K, INT8* matrix1, INT8* matrix2, I32* result) {
    for(U32 i = 0; i < N; i++) {
        for(U32 j = 0; j < MInner; j++) {
            for(U32 k = 0; k < K; k++) {
                result[i * M + j] += matrix1[i*K + k] * matrix2[k*M + j];
            }
        }
    }
}
#endif
#endif
