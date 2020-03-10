// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_CONVOLUTION_WINOGRAD_TRANSFORM
#define _H_CONVOLUTION_WINOGRAD_TRANSFORM

#ifdef _USE_INT8
#include <math.h>
#include <string.h>
#include "type.h"
#include "error.h"
#include "cpu/arm/fp16/convolution_winograd_transform.h"

inline void trans_I_int8(short *Iw[36], INT8* const I[36])
{
    short T[6][6][8];

    int8x8_t v_4 = vmov_n_s8(4);
    int8x8_t v_minus_4 = vmov_n_s8(-4);
    int8x8_t v_minus_5 = vmov_n_s8(-5);

    for (U32 i = 0; i < 6; i++) {
        int8x8_t v_I0 = vld1_s8(I[0*6+i]);
        int8x8_t v_I1 = vld1_s8(I[1*6+i]);
        int8x8_t v_I2 = vld1_s8(I[2*6+i]);
        int8x8_t v_I3 = vld1_s8(I[3*6+i]);
        int8x8_t v_I4 = vld1_s8(I[4*6+i]);
        int8x8_t v_I5 = vld1_s8(I[5*6+i]);

        // Reorder to accelerate
        int16x8_t v_t0 = vmull_s8(v_I2, v_minus_4);

        int16x8_t v_t1 = vmull_s8(v_I1, v_minus_4);

        int16x8_t v_t2 = vsubl_s8(v_I4, v_I2);

        int16x8_t v_t3 = vsubl_s8(v_I3, v_I1);

        v_t0 = vaddw_s8(v_t0, v_I4);

        v_t1 = vaddw_s8(v_t1, v_I3);

        v_t3 = vmulq_n_s16(v_t3, 2);

        int16x8_t v_t4 = vmull_s8(v_I0, v_4);

        int16x8_t v_t5 = vmull_s8(v_I1, v_4);

        int16x8_t v_T0 = vmull_s8(v_I2, v_minus_5);

        int16x8_t v_T1 = vaddq_s16(v_t1, v_t0);

        v_t4 = vaddw_s8(v_t4, v_I4);

        v_t5 = vaddw_s8(v_t5, v_I5);

        v_T0 = vaddq_s16(v_T0, v_t4);

        int16x8_t v_T2 = vsubq_s16(v_t0, v_t1);

        int16x8_t v_T3 = vaddq_s16(v_t3, v_t2);

        int16x8_t v_T4 = vsubq_s16(v_t2, v_t3);

        int16x8_t v_T5 = vmull_s8(v_I3, v_minus_5);

        vst1q_s16(T[0][i], v_T0);
        vst1q_s16(T[1][i], v_T1);
        vst1q_s16(T[2][i], v_T2);
        vst1q_s16(T[3][i], v_T3);
        v_T5 = vaddq_s16(v_T5, v_t5);
        vst1q_s16(T[4][i], v_T4);
        vst1q_s16(T[5][i], v_T5);
    }

    for (U32 i = 0; i < 6; i++) {
        int16x8_t v_T0 = vld1q_s16(T[i][0]);
        int16x8_t v_T1 = vld1q_s16(T[i][1]);
        int16x8_t v_T2 = vld1q_s16(T[i][2]);
        int16x8_t v_T3 = vld1q_s16(T[i][3]);
        int16x8_t v_T4 = vld1q_s16(T[i][4]);
        int16x8_t v_T5 = vld1q_s16(T[i][5]);

        int16x8_t v_t0 = vmlaq_n_s16(v_T4, v_T2, -4);
        int16x8_t v_t1 = vmlaq_n_s16(v_T3, v_T1, -4);
        int16x8_t v_t2 = vsubq_s16(v_T4, v_T2);
        int16x8_t v_t3 = vsubq_s16(v_T3, v_T1);
        int16x8_t v_t4 = vmlaq_n_s16(v_T4, v_T0, 4);
        int16x8_t v_t5 = vmlaq_n_s16(v_T5, v_T1, 4);

        v_t3 = vmulq_n_s16(v_t3, 2);

        int16x8_t v_Iw0 = vmlaq_n_s16(v_t4, v_T2, -5);
        int16x8_t v_Iw1 = vaddq_s16(v_t1, v_t0);
        int16x8_t v_Iw2 = vsubq_s16(v_t0, v_t1);
        int16x8_t v_Iw3 = vaddq_s16(v_t3, v_t2);
        int16x8_t v_Iw4 = vsubq_s16(v_t2, v_t3);
        int16x8_t v_Iw5 = vmlaq_n_s16(v_t5, v_T3, -5);

        vst1q_s16(Iw[i*6+0], v_Iw0);
        vst1q_s16(Iw[i*6+1], v_Iw1);
        vst1q_s16(Iw[i*6+2], v_Iw2);
        vst1q_s16(Iw[i*6+3], v_Iw3);
        vst1q_s16(Iw[i*6+4], v_Iw4);
        vst1q_s16(Iw[i*6+5], v_Iw5);
    }
}

inline void trans_O(F16* const Ow[36], F16 *O[16], const F16* bias,
                     U32 h, U32 w, U32 _pad_h_mod_4, U32 _pad_w_mod_4, U32 oh, U32 ow, F16* max, F16* min, ActivationMode am)
{
    F16 T[4][6][8];
    // bias
    float16x8_t v_b = vld1q_f16(bias);

    float16x8_t v_0 = vmovq_n_f16(0);
    float16x8_t v_2 = vmovq_n_f16(2);
    float16x8_t v_4 = vmovq_n_f16(4);
    float16x8_t v_8 = vmovq_n_f16(8);

    for (U32 i = 0; i < 6; i++) {
        float16x8_t v_Ow0 = vld1q_f16(Ow[i]);
        float16x8_t v_Ow1 = vld1q_f16(Ow[1*6+i]);
        float16x8_t v_Ow2 = vld1q_f16(Ow[2*6+i]);
        float16x8_t v_Ow3 = vld1q_f16(Ow[3*6+i]);
        float16x8_t v_Ow4 = vld1q_f16(Ow[4*6+i]);
        float16x8_t v_Ow5 = vld1q_f16(Ow[5*6+i]);

        float16x8_t v_t0 = vaddq_f16(v_Ow1, v_Ow2);
        float16x8_t v_t1 = vaddq_f16(v_Ow3, v_Ow4);
        float16x8_t v_t2 = vsubq_f16(v_Ow1, v_Ow2);
        float16x8_t v_t3 = vsubq_f16(v_Ow3, v_Ow4);

        float16x8_t v_T0 = vaddq_f16(v_t0, v_t1);
        float16x8_t v_T1 = vfmaq_f16(v_t2, v_t3, v_2);
        float16x8_t v_T2 = vfmaq_f16(v_t0, v_t1, v_4);
        float16x8_t v_T3 = vfmaq_f16(v_t2, v_t3, v_8);
        v_T0 = vaddq_f16(v_T0, v_Ow0);
        v_T3 = vaddq_f16(v_T3, v_Ow5);

        vst1q_f16(T[0][i], v_T0);
        vst1q_f16(T[1][i], v_T1);
        vst1q_f16(T[2][i], v_T2);
        vst1q_f16(T[3][i], v_T3);
    }

    float16x8_t max_v = vld1q_f16(max);
    float16x8_t min_v = vld1q_f16(min);

    U32 pad_h_mod_4 = 0, pad_w_mod_4 = 0;
    if (h == oh && w == ow) {
        pad_h_mod_4 = _pad_h_mod_4;
        pad_w_mod_4 = _pad_w_mod_4;
    } else if (h == oh) {
        pad_h_mod_4 = _pad_h_mod_4;
    } else if (w == ow) {
        pad_w_mod_4 = _pad_w_mod_4;
    }

    for (U32 i = 0; i < 4 - pad_h_mod_4; i++) {
        float16x8_t v_T0 = vld1q_f16(T[i][0]);
        float16x8_t v_T1 = vld1q_f16(T[i][1]);
        float16x8_t v_T2 = vld1q_f16(T[i][2]);
        float16x8_t v_T3 = vld1q_f16(T[i][3]);
        float16x8_t v_T4 = vld1q_f16(T[i][4]);
        float16x8_t v_T5 = vld1q_f16(T[i][5]);

        float16x8_t v_t0 = vaddq_f16(v_T1, v_T2);
        float16x8_t v_t1 = vaddq_f16(v_T3, v_T4);
        float16x8_t v_t2 = vsubq_f16(v_T1, v_T2);
        float16x8_t v_t3 = vsubq_f16(v_T3, v_T4);

        float16x8_t v_O0 = vaddq_f16(v_t0, v_t1);
        float16x8_t v_O1 = vfmaq_f16(v_t2, v_t3, v_2);
        float16x8_t v_O2 = vfmaq_f16(v_t0, v_t1, v_4);
        float16x8_t v_O3 = vfmaq_f16(v_t2, v_t3, v_8);
        v_O0 = vaddq_f16(v_O0, v_T0);
        v_O3 = vaddq_f16(v_O3, v_T5);

        float16x8_t temp;

        if (am == ACTIVATION_RELU) {
            if (pad_w_mod_4 == 0) {
                temp = vmaxq_f16(vaddq_f16(v_O0, v_b), v_0);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+0], temp);

                temp = vmaxq_f16(vaddq_f16(v_O1, v_b), v_0);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+1], temp);

                temp = vmaxq_f16(vaddq_f16(v_O2, v_b), v_0);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+2], temp);

                temp = vmaxq_f16(vaddq_f16(v_O3, v_b), v_0);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+3], temp);
            } else if (pad_w_mod_4 == 1) {
                temp = vmaxq_f16(vaddq_f16(v_O0, v_b), v_0);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+0], temp);

                temp = vmaxq_f16(vaddq_f16(v_O1, v_b), v_0);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+1], temp);

                temp = vmaxq_f16(vaddq_f16(v_O2, v_b), v_0);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+2], temp);
            } else if (pad_w_mod_4 == 2) {
                temp = vmaxq_f16(vaddq_f16(v_O0, v_b), v_0);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+0], temp);

                temp = vmaxq_f16(vaddq_f16(v_O1, v_b), v_0);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+1], temp);
            } else if (pad_w_mod_4 == 3) {
                temp = vmaxq_f16(vaddq_f16(v_O0, v_b), v_0);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+0], temp);
            }
        } else {
            if (pad_w_mod_4 == 0) {
                temp = vaddq_f16(v_O0, v_b);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+0], temp);

                temp = vaddq_f16(v_O1, v_b);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+1], temp);

                temp = vaddq_f16(v_O2, v_b);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+2], temp);

                temp = vaddq_f16(v_O3, v_b);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+3], temp);
            } else if (pad_w_mod_4 == 1) {
                temp = vaddq_f16(v_O0, v_b);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+0], temp);

                temp = vaddq_f16(v_O1, v_b);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+1], temp);

                temp = vaddq_f16(v_O2, v_b);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+2], temp);
            } else if (pad_w_mod_4 == 2) {
                temp = vaddq_f16(v_O0, v_b);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+0], temp);

                temp = vaddq_f16(v_O1, v_b);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+1], temp);
            } else if (pad_w_mod_4 == 3) {
                temp = vaddq_f16(v_O0, v_b);
                max_v = vmaxq_f16(max_v, temp);
                min_v = vminq_f16(min_v, temp);
                vst1q_f16(O[i*4+0], temp);
            }
        }
    }

    vst1q_f16(max, max_v);
    vst1q_f16(min, min_v);
}
#endif
#endif
