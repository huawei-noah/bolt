// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MVM_INT8
#define _H_MVM_INT8

static const int ALIGN = 32;

inline void mvm_row_pack(U32 Nbatch, U32 K, INT8 *matrix, INT8 *vector, I32 *result)
{
    U32 N = Nbatch * 32;
    int8x16_t mat[16];
    int8x8_t v;
    int32x4_t res[8];
    int K_tail = K % 4;
    int K_inner = K - K_tail;
    int K4 = UNI_ALIGN(K, 4);
    for (U32 n = 0; n < N; n += 32) {
        INT8 *bufMov = matrix + n * K4;
        if (K_inner > 0) {
            for (int i = 0; i < 8; i++) {
                res[i] = vld1q_s32(result + n + i * 4);
            }
            int k = 0;
            for (; k < K_inner - 7; k += 8) {
                v = vld1_s8(vector + k);
                for (int i = 0; i < 8; i++) {
                    mat[i] = vld1q_s8(bufMov + i * 16);
                }
                for (int i = 0; i < 8; i++) {
                    res[i] = vdotq_lane_s32(res[i], mat[i], v, 0);
                }
                for (int i = 8; i < 16; i++) {
                    mat[i] = vld1q_s8(bufMov + i * 16);
                }
                for (int i = 0; i < 8; i++) {
                    res[i] = vdotq_lane_s32(res[i], mat[i + 8], v, 1);
                }
                bufMov += 256;
            }
            if (K_inner == k + 4) {
                v = vld1_s8(vector + k);
                for (int i = 0; i < 8; i++) {
                    mat[i] = vld1q_s8(bufMov + i * 16);
                }
                for (int i = 0; i < 8; i++) {
                    res[i] = vdotq_lane_s32(res[i], mat[i], v, 0);
                }
                bufMov += 128;
            }
            for (int i = 0; i < 8; i++) {
                vst1q_s32(result + n + i * 4, res[i]);
            }
        }
        if (K_tail > 0) {
            for (int i = 0; i < 32; i++) {
                I32 tmp = 0;
                for (int j = 0; j < (int)K_tail; j++) {
                    tmp += vector[K_inner + j] * bufMov[j];
                }
                result[n + i] += tmp;
                bufMov += 4;
            }
        }
    }
}

inline void mvm_row_unpack(U32 Nbatch, U32 K, INT8 *matrix, INT8 *vector, I32 *result)
{
    U32 N = Nbatch * 8;
    int8x16_t mat[8];
    U32 K_tail = K % 16;
    U32 K_inner = K - K_tail;
    INT8 *w[8];
    for (U32 n = 0; n < N; n += 8) {
        for (int i = 0; i < 8; i++) {
            w[i] = matrix + (n + i) * K;
        }

        int32x4_t bias0 = vld1q_s32(result + n);
        int32x4_t bias1 = vld1q_s32(result + n + 4);
        int32x4_t res[8] = {0};
        for (U32 k = 0; k < K_inner; k += 16) {
            int8x16_t v = vld1q_s8(vector + k);
            for (int i = 0; i < 8; i++) {
                mat[i] = vld1q_s8(w[i] + k);
            }
            for (int i = 0; i < 8; i++) {
                res[i] = vdotq_s32(res[i], mat[i], v);
            }
        }
        res[0] = vpaddq_s32(res[0], res[1]);
        res[4] = vpaddq_s32(res[4], res[5]);
        res[2] = vpaddq_s32(res[2], res[3]);
        res[6] = vpaddq_s32(res[6], res[7]);
        res[0] = vpaddq_s32(res[0], res[2]);
        res[4] = vpaddq_s32(res[4], res[6]);
        res[0] = vaddq_s32(res[0], bias0);
        res[4] = vaddq_s32(res[4], bias1);
        vst1q_s32(result + n, res[0]);
        vst1q_s32(result + n + 4, res[4]);

        if (K_tail != 0) {
            for (int i = 0; i < 8; i++) {
                I32 tmp = 0;
                for (U32 p = K_inner; p < K; p++) {
                    tmp += vector[p] * w[i][p];
                }
                result[n + i] += tmp;
            }
        }
    }
}

#endif
