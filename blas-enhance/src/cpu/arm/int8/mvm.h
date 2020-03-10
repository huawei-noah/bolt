// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_MVM
#define _H_MVM

#ifdef _USE_INT8
#include <arm_neon.h> 
#include <string.h>


inline void mvm_col_tail(U32 N, U32 K, INT8* matrix, INT8* vector, I32* result) {
    for (U32 n = 0; n < N; n++) {
        I32 tmp = 0;
        for (U32 k = 0; k < K; k++) {
            tmp += vector[k] * matrix[k*N + n];
        }
        result[n] += tmp;
    }
}

inline void mvm_row_tail(U32 N, U32 K, INT8* matrix, INT8* vector, I32* result) {
    INT8* cur_row = matrix;
    for (U32 n = 0; n < N; n++) {
        I32 tmp = 0;
        for(U32 k = 0; k < K; k++) {
            tmp += vector[k] * cur_row[k];
        }
        result[n] += tmp;
        cur_row += K;
    }
}

inline void mvm_row_kernel(U32 Nbatch, U32 K, INT8* matrix, INT8* vector, I32* result) {
    U32 N = Nbatch * 4;
    int8x16_t mat[4], v;
    U32 K_tail = K % 16;
    U32 K_inner = K - K_tail;
    for (U32 n = 0; n < N; n+=4) {
        int32x4_t res[4] = {0};
        int32x4_t bias;

        INT8* w0 = matrix + n * K;
        INT8* w1 = w0 + K;
        INT8* w2 = w1 + K;
        INT8* w3 = w2 + K;

        for (U32 k = 0; k < K_inner; k+=16) {
            v = vld1q_s8(vector + k);
            mat[0] = vld1q_s8(w0);
            mat[1] = vld1q_s8(w1);
            mat[2] = vld1q_s8(w2);
            mat[3] = vld1q_s8(w3);

            res[0] = vdotq_s32(res[0], mat[0], v);
            res[1] = vdotq_s32(res[1], mat[1], v);
            res[2] = vdotq_s32(res[2], mat[2], v);
            res[3] = vdotq_s32(res[3], mat[3], v);

            w0 += 16;
            w1 += 16;
            w2 += 16;
            w3 += 16;
        }
        bias = vld1q_s32(result + n);

        res[0] = vpaddq_s32(res[0], res[1]);
        res[2] = vpaddq_s32(res[2], res[3]);
        res[0] = vpaddq_s32(res[0], res[2]);
        res[0] = vaddq_s32(res[0], bias);
        
        vst1q_s32(result + n, res[0]);
            
        if (K_tail != 0) {
            I32 tmp[4] = {0};
            for(U32 p = K_inner; p < K; p++) {
                tmp[0] += vector[p] * *w0++;
                tmp[1] += vector[p] * *w1++;
                tmp[2] += vector[p] * *w2++;
                tmp[3] += vector[p] * *w3++;
            }
            result[n] += tmp[0];
            result[n+1] += tmp[1];
            result[n+2] += tmp[2];
            result[n+3] += tmp[3];
        }
     }
}

inline void mvm_col(U32 numRows, U32 numColumns, INT8* matrix, INT8* vector, I32*tmp, I32* result) {
    //Actual layout is KN, and vector is K
    U32 N = numRows;
    U32 K = numColumns;
    U32 NTail = N % 64;
    U32 NInner = N - NTail;

    for (U32 n = 0; n < NInner; n+=64) {
        memset(tmp, 0, sizeof(I32)*64);
        for (U32 k = 0; k < K; k++) {
            for(U32 i = 0; i < 64; i++) {
                tmp[i] += vector[k] * matrix[k * N + n + i];
            }
        }
        
        for (U32 i = 0; i < 64; i++) {
            result[n + i] += tmp[i];
        }
    }

    memset(tmp, 0, sizeof(I32)*64);
    for (U32 k = 0; k < K; k++) {
        for(U32 i = 0; i < NTail; i++) {
            tmp[i] += vector[k] * matrix[k * N + NInner + i];
        }
        for(U32 i=0; i < NTail; i++) {
            result[NInner + i] += tmp[i];
        }
    }
}

inline void mvm_row(U32 numRows, U32 numColumns, INT8* matrix, INT8* vector, I32* result) {
    //Actual layout is NK, and vector is K
    U32 N = numRows;
    U32 K = numColumns;
    U32 Nbatch = N / 4;
    U32 NTail = N % 4;

    mvm_row_kernel(Nbatch, K, matrix, vector, result);

    if (NTail != 0) {
        mvm_row_tail(NTail, K, matrix + (N - NTail) * K, vector, result + N - NTail);
    }
}
#endif
#endif
