// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_WINOGRAD_TRANSFORM
#define _H_WINOGRAD_TRANSFORM

#include <math.h>
#include <string.h>
#include "cpu/arm/fp16/arm_functions_fp16.h"

inline void trans_W_4x4_3x3(F16 *Fw[36], F16 *const F[9])
{
    F16 T[6][3][8];

    float16x8_t v_01666 = vmovq_n_f16(0.1666666666666667f);
    float16x8_t v_minus_01666 = vmovq_n_f16(-0.1666666666666667f);
    float16x8_t v_00833 = vmovq_n_f16(0.0833333333333333f);
    float16x8_t v_minus_00833 = vmovq_n_f16(-0.0833333333333333f);
    float16x8_t v_004166 = vmovq_n_f16(0.0416666666666667f);
    float16x8_t v_025 = vmovq_n_f16(0.25f);

    for (U32 i = 0; i < 3; i++) {
        float16x8_t v_F0 = vld1q_f16(F[0 * 3 + i]);
        float16x8_t v_F1 = vld1q_f16(F[1 * 3 + i]);
        float16x8_t v_F2 = vld1q_f16(F[2 * 3 + i]);

        float16x8_t v_t0 = vmulq_f16(v_01666, v_F2);
        float16x8_t v_t1 = vsubq_f16(vmulq_f16(v_minus_01666, v_F0), v_t0);
        float16x8_t v_t2 = vfmaq_f16(v_t0, v_004166, v_F0);

        float16x8_t v_T0 = vmulq_f16(v_025, v_F0);
        float16x8_t v_T1 = vfmaq_f16(v_t1, v_minus_01666, v_F1);
        float16x8_t v_T2 = vfmaq_f16(v_t1, v_01666, v_F1);
        float16x8_t v_T3 = vfmaq_f16(v_t2, v_00833, v_F1);
        float16x8_t v_T4 = vfmaq_f16(v_t2, v_minus_00833, v_F1);

        vst1q_f16(T[0][i], v_T0);
        vst1q_f16(T[1][i], v_T1);
        vst1q_f16(T[2][i], v_T2);
        vst1q_f16(T[3][i], v_T3);
        vst1q_f16(T[4][i], v_T4);
        vst1q_f16(T[5][i], v_F2);
    }
    for (U32 i = 0; i < 6; i++) {
        float16x8_t v_T0 = vld1q_f16(T[i][0]);
        float16x8_t v_T1 = vld1q_f16(T[i][1]);
        float16x8_t v_T2 = vld1q_f16(T[i][2]);

        float16x8_t v_t0 = vmulq_f16(v_01666, v_T2);
        float16x8_t v_t1 = vsubq_f16(vmulq_f16(v_minus_01666, v_T0), v_t0);
        float16x8_t v_t2 = vfmaq_f16(v_t0, v_004166, v_T0);

        float16x8_t v_Fw0 = vmulq_f16(v_025, v_T0);
        float16x8_t v_Fw1 = vfmaq_f16(v_t1, v_minus_01666, v_T1);
        float16x8_t v_Fw2 = vfmaq_f16(v_t1, v_01666, v_T1);
        float16x8_t v_Fw3 = vfmaq_f16(v_t2, v_00833, v_T1);
        float16x8_t v_Fw4 = vfmaq_f16(v_t2, v_minus_00833, v_T1);

        vst1q_f16(Fw[i * 6 + 0], v_Fw0);
        vst1q_f16(Fw[i * 6 + 1], v_Fw1);
        vst1q_f16(Fw[i * 6 + 2], v_Fw2);
        vst1q_f16(Fw[i * 6 + 3], v_Fw3);
        vst1q_f16(Fw[i * 6 + 4], v_Fw4);
        vst1q_f16(Fw[i * 6 + 5], v_T2);
    }
}

inline EE trans_O_4x4_3x3(F16 *const Ow[36],
    F16 *O[16],
    const F16 *bias,
    U32 h,
    U32 w,
    U32 _pad_h_mod_4,
    U32 _pad_w_mod_4,
    U32 oh,
    U32 ow,
    ActivationParamSpec activationDesc)
{
    F16 T[4][6][8];
    // bias
    float16x8_t v_b = vld1q_f16(bias);

    float16x8_t v_0 = vmovq_n_f16(0);
    float16x8_t v_2 = vmovq_n_f16(2);
    float16x8_t v_4 = vmovq_n_f16(4);
    float16x8_t v_6 = vmovq_n_f16(6);
    float16x8_t v_8 = vmovq_n_f16(8);

    for (U32 i = 0; i < 6; i++) {
        float16x8_t v_Ow0 = vld1q_f16(Ow[i]);
        float16x8_t v_Ow1 = vld1q_f16(Ow[1 * 6 + i]);
        float16x8_t v_Ow2 = vld1q_f16(Ow[2 * 6 + i]);
        float16x8_t v_Ow3 = vld1q_f16(Ow[3 * 6 + i]);
        float16x8_t v_Ow4 = vld1q_f16(Ow[4 * 6 + i]);
        float16x8_t v_Ow5 = vld1q_f16(Ow[5 * 6 + i]);

        float16x8_t v_t0 = vaddq_f16(v_Ow1, v_Ow2);
        float16x8_t v_t1 = vaddq_f16(v_Ow3, v_Ow4);
        float16x8_t v_t2 = vsubq_f16(v_Ow1, v_Ow2);
        float16x8_t v_t3 = vsubq_f16(v_Ow3, v_Ow4);

        float16x8_t v_T0 = vaddq_f16(vaddq_f16(v_t0, v_t1), v_Ow0);
        float16x8_t v_T1 = vfmaq_f16(v_t2, v_t3, v_2);
        float16x8_t v_T2 = vfmaq_f16(v_t0, v_t1, v_4);
        float16x8_t v_T3 = vaddq_f16(vfmaq_f16(v_t2, v_t3, v_8), v_Ow5);

        vst1q_f16(T[0][i], v_T0);
        vst1q_f16(T[1][i], v_T1);
        vst1q_f16(T[2][i], v_T2);
        vst1q_f16(T[3][i], v_T3);
    }

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

        float16x8_t v_O0 = vaddq_f16(vaddq_f16(v_t0, v_t1), v_T0);
        float16x8_t v_O1 = vfmaq_f16(v_t2, v_t3, v_2);
        float16x8_t v_O2 = vfmaq_f16(v_t0, v_t1, v_4);
        float16x8_t v_O3 = vaddq_f16(vfmaq_f16(v_t2, v_t3, v_8), v_T5);

        switch (activationDesc.mode) {
            case ACTIVATION_NULL: {
                if (pad_w_mod_4 == 0) {
                    vst1q_f16(O[i * 4 + 0], vaddq_f16(v_O0, v_b));
                    vst1q_f16(O[i * 4 + 1], vaddq_f16(v_O1, v_b));
                    vst1q_f16(O[i * 4 + 2], vaddq_f16(v_O2, v_b));
                    vst1q_f16(O[i * 4 + 3], vaddq_f16(v_O3, v_b));
                } else if (pad_w_mod_4 == 1) {
                    vst1q_f16(O[i * 4 + 0], vaddq_f16(v_O0, v_b));
                    vst1q_f16(O[i * 4 + 1], vaddq_f16(v_O1, v_b));
                    vst1q_f16(O[i * 4 + 2], vaddq_f16(v_O2, v_b));
                } else if (pad_w_mod_4 == 2) {
                    vst1q_f16(O[i * 4 + 0], vaddq_f16(v_O0, v_b));
                    vst1q_f16(O[i * 4 + 1], vaddq_f16(v_O1, v_b));
                } else if (pad_w_mod_4 == 3) {
                    vst1q_f16(O[i * 4 + 0], vaddq_f16(v_O0, v_b));
                }
                break;
            }
            case ACTIVATION_RELU: {
                if (pad_w_mod_4 == 0) {
                    vst1q_f16(O[i * 4 + 0], vmaxq_f16(vaddq_f16(v_O0, v_b), v_0));
                    vst1q_f16(O[i * 4 + 1], vmaxq_f16(vaddq_f16(v_O1, v_b), v_0));
                    vst1q_f16(O[i * 4 + 2], vmaxq_f16(vaddq_f16(v_O2, v_b), v_0));
                    vst1q_f16(O[i * 4 + 3], vmaxq_f16(vaddq_f16(v_O3, v_b), v_0));
                } else if (pad_w_mod_4 == 1) {
                    vst1q_f16(O[i * 4 + 0], vmaxq_f16(vaddq_f16(v_O0, v_b), v_0));
                    vst1q_f16(O[i * 4 + 1], vmaxq_f16(vaddq_f16(v_O1, v_b), v_0));
                    vst1q_f16(O[i * 4 + 2], vmaxq_f16(vaddq_f16(v_O2, v_b), v_0));
                } else if (pad_w_mod_4 == 2) {
                    vst1q_f16(O[i * 4 + 0], vmaxq_f16(vaddq_f16(v_O0, v_b), v_0));
                    vst1q_f16(O[i * 4 + 1], vmaxq_f16(vaddq_f16(v_O1, v_b), v_0));
                } else if (pad_w_mod_4 == 3) {
                    vst1q_f16(O[i * 4 + 0], vmaxq_f16(vaddq_f16(v_O0, v_b), v_0));
                }
                break;
            }
            case ACTIVATION_SIGMOID: {
                if (pad_w_mod_4 == 0) {
                    vst1q_f16(O[i * 4 + 0], vsigmoidq_f16(vaddq_f16(v_O0, v_b)));
                    vst1q_f16(O[i * 4 + 1], vsigmoidq_f16(vaddq_f16(v_O1, v_b)));
                    vst1q_f16(O[i * 4 + 2], vsigmoidq_f16(vaddq_f16(v_O2, v_b)));
                    vst1q_f16(O[i * 4 + 3], vsigmoidq_f16(vaddq_f16(v_O3, v_b)));
                } else if (pad_w_mod_4 == 1) {
                    vst1q_f16(O[i * 4 + 0], vsigmoidq_f16(vaddq_f16(v_O0, v_b)));
                    vst1q_f16(O[i * 4 + 1], vsigmoidq_f16(vaddq_f16(v_O1, v_b)));
                    vst1q_f16(O[i * 4 + 2], vsigmoidq_f16(vaddq_f16(v_O2, v_b)));
                } else if (pad_w_mod_4 == 2) {
                    vst1q_f16(O[i * 4 + 0], vsigmoidq_f16(vaddq_f16(v_O0, v_b)));
                    vst1q_f16(O[i * 4 + 1], vsigmoidq_f16(vaddq_f16(v_O1, v_b)));
                } else if (pad_w_mod_4 == 3) {
                    vst1q_f16(O[i * 4 + 0], vsigmoidq_f16(vaddq_f16(v_O0, v_b)));
                }
                break;
            }
            case ACTIVATION_RELU6: {
                if (pad_w_mod_4 == 0) {
                    vst1q_f16(O[i * 4 + 0], vminq_f16(vmaxq_f16(vaddq_f16(v_O0, v_b), v_0), v_6));
                    vst1q_f16(O[i * 4 + 1], vminq_f16(vmaxq_f16(vaddq_f16(v_O1, v_b), v_0), v_6));
                    vst1q_f16(O[i * 4 + 2], vminq_f16(vmaxq_f16(vaddq_f16(v_O2, v_b), v_0), v_6));
                    vst1q_f16(O[i * 4 + 3], vminq_f16(vmaxq_f16(vaddq_f16(v_O3, v_b), v_0), v_6));
                } else if (pad_w_mod_4 == 1) {                                                   
                    vst1q_f16(O[i * 4 + 0], vminq_f16(vmaxq_f16(vaddq_f16(v_O0, v_b), v_0), v_6));
                    vst1q_f16(O[i * 4 + 1], vminq_f16(vmaxq_f16(vaddq_f16(v_O1, v_b), v_0), v_6));
                    vst1q_f16(O[i * 4 + 2], vminq_f16(vmaxq_f16(vaddq_f16(v_O2, v_b), v_0), v_6));
                } else if (pad_w_mod_4 == 2) {                                                   
                    vst1q_f16(O[i * 4 + 0], vminq_f16(vmaxq_f16(vaddq_f16(v_O0, v_b), v_0), v_6));
                    vst1q_f16(O[i * 4 + 1], vminq_f16(vmaxq_f16(vaddq_f16(v_O1, v_b), v_0), v_6));
                } else if (pad_w_mod_4 == 3) {                                                   
                    vst1q_f16(O[i * 4 + 0], vminq_f16(vmaxq_f16(vaddq_f16(v_O0, v_b), v_0), v_6));
                }
                break;
            }            
            default:
                return NOT_SUPPORTED;
        }
    }
    return SUCCESS;
}

inline void trans_I_4x4_3x3(F16 *Iw[36], F16 *const I[36])
{
    F16 T[6][6][8];

    float16x8_t v_4 = vmovq_n_f16(4);
    float16x8_t v_minus_4 = vmovq_n_f16(-4);
    float16x8_t v_2 = vmovq_n_f16(2);
    float16x8_t v_minus_5 = vmovq_n_f16(-5);

    for (U32 i = 0; i < 6; i++) {
        float16x8_t v_I0 = vld1q_f16(I[0 * 6 + i]);
        float16x8_t v_I1 = vld1q_f16(I[1 * 6 + i]);
        float16x8_t v_I2 = vld1q_f16(I[2 * 6 + i]);
        float16x8_t v_I3 = vld1q_f16(I[3 * 6 + i]);
        float16x8_t v_I4 = vld1q_f16(I[4 * 6 + i]);
        float16x8_t v_I5 = vld1q_f16(I[5 * 6 + i]);

        float16x8_t v_t0 = vfmaq_f16(v_I4, v_I2, v_minus_4);
        float16x8_t v_t1 = vfmaq_f16(v_I3, v_I1, v_minus_4);
        float16x8_t v_t2 = vsubq_f16(v_I4, v_I2);
        float16x8_t v_t3 = vmulq_f16(vsubq_f16(v_I3, v_I1), v_2);
        float16x8_t v_t4 = vfmaq_f16(v_I4, v_I0, v_4);
        float16x8_t v_t5 = vfmaq_f16(v_I5, v_I1, v_4);

        float16x8_t v_T0 = vfmaq_f16(v_t4, v_I2, v_minus_5);
        float16x8_t v_T1 = vaddq_f16(v_t1, v_t0);
        float16x8_t v_T2 = vsubq_f16(v_t0, v_t1);
        float16x8_t v_T3 = vaddq_f16(v_t3, v_t2);
        float16x8_t v_T4 = vsubq_f16(v_t2, v_t3);
        float16x8_t v_T5 = vfmaq_f16(v_t5, v_I3, v_minus_5);

        vst1q_f16(T[0][i], v_T0);
        vst1q_f16(T[1][i], v_T1);
        vst1q_f16(T[2][i], v_T2);
        vst1q_f16(T[3][i], v_T3);
        vst1q_f16(T[4][i], v_T4);
        vst1q_f16(T[5][i], v_T5);
    }

    for (U32 i = 0; i < 6; i++) {
        float16x8_t v_T0 = vld1q_f16(T[i][0]);
        float16x8_t v_T1 = vld1q_f16(T[i][1]);
        float16x8_t v_T2 = vld1q_f16(T[i][2]);
        float16x8_t v_T3 = vld1q_f16(T[i][3]);
        float16x8_t v_T4 = vld1q_f16(T[i][4]);
        float16x8_t v_T5 = vld1q_f16(T[i][5]);

        float16x8_t v_t0 = vfmaq_f16(v_T4, v_T2, v_minus_4);
        float16x8_t v_t1 = vfmaq_f16(v_T3, v_T1, v_minus_4);
        float16x8_t v_t2 = vsubq_f16(v_T4, v_T2);
        float16x8_t v_t3 = vmulq_f16(vsubq_f16(v_T3, v_T1), v_2);
        float16x8_t v_t4 = vfmaq_f16(v_T4, v_T0, v_4);
        float16x8_t v_t5 = vfmaq_f16(v_T5, v_T1, v_4);

        float16x8_t v_Iw0 = vfmaq_f16(v_t4, v_T2, v_minus_5);
        float16x8_t v_Iw1 = vaddq_f16(v_t1, v_t0);
        float16x8_t v_Iw2 = vsubq_f16(v_t0, v_t1);
        float16x8_t v_Iw3 = vaddq_f16(v_t3, v_t2);
        float16x8_t v_Iw4 = vsubq_f16(v_t2, v_t3);
        float16x8_t v_Iw5 = vfmaq_f16(v_t5, v_T3, v_minus_5);

        F16 max = vmaxvq_f16(v_Iw0);
        F16 min = vminvq_f16(v_Iw0);
        if (UNI_ISNAN(max) || UNI_ISINF(max) || UNI_ISNAN(min) || UNI_ISINF(min)) {
            F16 check[8];
            vst1q_f16(check, v_Iw0);
            for (U32 c = 0; c < 8; c++) {
                F16 tmp = check[c];
                if (UNI_ISINF(tmp)) {
                    if (tmp > 0) {
                        check[c] = 65504;  // FMAX for F16
                    } else {
                        check[c] = -65504;
                    }
                } else if (UNI_ISNAN(tmp)) {
                    tmp = (T[i][0][c] - T[i][2][c]) * 4;
                    if (UNI_ISINF(tmp)) {
                        if (tmp > 0) {
                            tmp = 65504;  // FMAX for F16
                        } else {
                            tmp = -65504;
                        }
                    }
                    F16 diff = T[i][4][c] - T[i][2][c];
                    tmp += diff;
                    if (UNI_ISINF(tmp)) {
                        if (diff > 0) {
                            tmp = 65504;
                        } else {
                            tmp = -65504;
                        }
                    }
                    check[c] = tmp;
                }
            }
            memcpy(Iw[i * 6 + 0], check, 8 * bytesOf(DT_F16));
        } else {
            vst1q_f16(Iw[i * 6 + 0], v_Iw0);
        }

        max = vmaxvq_f16(v_Iw1);
        min = vminvq_f16(v_Iw1);
        if (UNI_ISNAN(max) || UNI_ISINF(max) || UNI_ISNAN(min) || UNI_ISINF(min)) {
            F16 check[8];
            vst1q_f16(check, v_Iw1);
            for (U32 c = 0; c < 8; c++) {
                F16 tmp = check[c];
                if (UNI_ISINF(tmp)) {
                    if (tmp > 0) {
                        check[c] = 65504;  // FMAX for F16
                    } else {
                        check[c] = -65504;
                    }
                } else if (UNI_ISNAN(tmp)) {
                    tmp = (T[i][1][c] + T[i][2][c]) * -4;
                    if (UNI_ISINF(tmp)) {
                        if (tmp > 0) {
                            tmp = 65504;  // FMAX for F16
                        } else {
                            tmp = -65504;
                        }
                    }
                    F16 sum = T[i][3][c] + T[i][4][c];
                    tmp += sum;
                    if (UNI_ISINF(tmp)) {
                        if (sum > 0) {
                            tmp = 65504;
                        } else {
                            tmp = -65504;
                        }
                    }
                    check[c] = tmp;
                }
            }
            memcpy(Iw[i * 6 + 1], check, 8 * bytesOf(DT_F16));
        } else {
            vst1q_f16(Iw[i * 6 + 1], v_Iw1);
        }

        max = vmaxvq_f16(v_Iw2);
        min = vminvq_f16(v_Iw2);
        if (UNI_ISNAN(max) || UNI_ISINF(max) || UNI_ISNAN(min) || UNI_ISINF(min)) {
            F16 check[8];
            vst1q_f16(check, v_Iw2);
            for (U32 c = 0; c < 8; c++) {
                F16 tmp = check[c];
                if (UNI_ISINF(tmp)) {
                    if (tmp > 0) {
                        check[c] = 65504;  // FMAX for F16
                    } else {
                        check[c] = -65504;
                    }
                } else if (UNI_ISNAN(tmp)) {
                    tmp = (T[i][1][c] - T[i][2][c]) * 4;
                    if (UNI_ISINF(tmp)) {
                        if (tmp > 0) {
                            tmp = 65504;  // FMAX for F16
                        } else {
                            tmp = -65504;
                        }
                    }
                    F16 diff = T[i][4][c] - T[i][3][c];
                    tmp += diff;
                    if (UNI_ISINF(tmp)) {
                        if (diff > 0) {
                            tmp = 65504;
                        } else {
                            tmp = -65504;
                        }
                    }
                    check[c] = tmp;
                }
            }
            memcpy(Iw[i * 6 + 2], check, 8 * bytesOf(DT_F16));
        } else {
            vst1q_f16(Iw[i * 6 + 2], v_Iw2);
        }

        max = vmaxvq_f16(v_Iw3);
        min = vminvq_f16(v_Iw3);
        if (UNI_ISNAN(max) || UNI_ISINF(max) || UNI_ISNAN(min) || UNI_ISINF(min)) {
            F16 check[8];
            vst1q_f16(check, v_Iw3);
            for (U32 c = 0; c < 8; c++) {
                F16 tmp = check[c];
                if (UNI_ISINF(tmp)) {
                    if (tmp > 0) {
                        check[c] = 65504;  // FMAX for F16
                    } else {
                        check[c] = -65504;
                    }
                } else if (UNI_ISNAN(tmp)) {
                    tmp = (T[i][3][c] - T[i][1][c]) * 2;
                    if (UNI_ISINF(tmp)) {
                        if (tmp > 0) {
                            tmp = 65504;  // FMAX for F16
                        } else {
                            tmp = -65504;
                        }
                    }
                    F16 diff = T[i][4][c] - T[i][2][c];
                    tmp += diff;
                    if (UNI_ISINF(tmp)) {
                        if (diff > 0) {
                            tmp = 65504;
                        } else {
                            tmp = -65504;
                        }
                    }
                    check[c] = tmp;
                }
            }
            memcpy(Iw[i * 6 + 3], check, 8 * bytesOf(DT_F16));
        } else {
            vst1q_f16(Iw[i * 6 + 3], v_Iw3);
        }

        max = vmaxvq_f16(v_Iw4);
        min = vminvq_f16(v_Iw4);
        if (UNI_ISNAN(max) || UNI_ISINF(max) || UNI_ISNAN(min) || UNI_ISINF(min)) {
            F16 check[8];
            vst1q_f16(check, v_Iw4);
            for (U32 c = 0; c < 8; c++) {
                F16 tmp = check[c];
                if (UNI_ISINF(tmp)) {
                    if (tmp > 0) {
                        check[c] = 65504;  // FMAX for F16
                    } else {
                        check[c] = -65504;
                    }
                } else if (UNI_ISNAN(tmp)) {
                    tmp = (T[i][1][c] - T[i][3][c]) * 2;
                    if (UNI_ISINF(tmp)) {
                        if (tmp > 0) {
                            tmp = 65504;  // FMAX for F16
                        } else {
                            tmp = -65504;
                        }
                    }
                    F16 diff = T[i][4][c] - T[i][2][c];
                    tmp += diff;
                    if (UNI_ISINF(tmp)) {
                        if (diff > 0) {
                            tmp = 65504;
                        } else {
                            tmp = -65504;
                        }
                    }
                    check[c] = tmp;
                }
            }
            memcpy(Iw[i * 6 + 4], check, 8 * bytesOf(DT_F16));
        } else {
            vst1q_f16(Iw[i * 6 + 4], v_Iw4);
        }

        max = vmaxvq_f16(v_Iw5);
        min = vminvq_f16(v_Iw5);
        if (UNI_ISNAN(max) || UNI_ISINF(max) || UNI_ISNAN(min) || UNI_ISINF(min)) {
            F16 check[8];
            vst1q_f16(check, v_Iw5);
            for (U32 c = 0; c < 8; c++) {
                F16 tmp = check[c];
                if (UNI_ISINF(tmp)) {
                    if (tmp > 0) {
                        check[c] = 65504;  // FMAX for F16
                    } else {
                        check[c] = -65504;
                    }
                } else if (UNI_ISNAN(tmp)) {
                    tmp = (T[i][1][c] - T[i][3][c]) * 4;
                    if (UNI_ISINF(tmp)) {
                        if (tmp > 0) {
                            tmp = 65504;  // FMAX for F16
                        } else {
                            tmp = -65504;
                        }
                    }
                    F16 diff = T[i][5][c] - T[i][3][c];
                    tmp += diff;
                    if (UNI_ISINF(tmp)) {
                        if (diff > 0) {
                            tmp = 65504;
                        } else {
                            tmp = -65504;
                        }
                    }
                    check[c] = tmp;
                }
            }
            memcpy(Iw[i * 6 + 5], check, 8 * bytesOf(DT_F16));
        } else {
            vst1q_f16(Iw[i * 6 + 5], v_Iw5);
        }
    }
}
#endif
