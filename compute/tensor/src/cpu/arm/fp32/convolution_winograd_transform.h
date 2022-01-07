// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_WINOGRAD_TRANSFORM_FP32
#define _H_WINOGRAD_TRANSFORM_FP32

#ifdef _USE_FP32
#include <math.h>
#include <string.h>
#include "cpu/arm/fp32/arm_functions_fp32.h"

inline void trans_W_4x4_3x3(float *WTM[36], float *W[9])
{
    float T[6][3][4];

    float32x4_t v_01666 = vmovq_n_f32(0.1666666666666667f);
    float32x4_t v_minus_01666 = vmovq_n_f32(-0.1666666666666667f);
    float32x4_t v_00833 = vmovq_n_f32(0.0833333333333333f);
    float32x4_t v_minus_00833 = vmovq_n_f32(-0.0833333333333333f);
    float32x4_t v_004166 = vmovq_n_f32(0.0416666666666667f);
    float32x4_t v_025 = vmovq_n_f32(0.25f);

    for (int i = 0; i < 3; i++) {
        float32x4_t v_W0 = vld1q_f32(W[0 * 3 + i]);
        float32x4_t v_W1 = vld1q_f32(W[1 * 3 + i]);
        float32x4_t v_W2 = vld1q_f32(W[2 * 3 + i]);

        float32x4_t v_t0 = vmulq_f32(v_01666, v_W2);
        float32x4_t v_t1 = vsubq_f32(vmulq_f32(v_minus_01666, v_W0), v_t0);
        float32x4_t v_t2 = vfmaq_f32(v_t0, v_004166, v_W0);

        float32x4_t v_T0 = vmulq_f32(v_025, v_W0);
        float32x4_t v_T1 = vfmaq_f32(v_t1, v_minus_01666, v_W1);
        float32x4_t v_T2 = vfmaq_f32(v_t1, v_01666, v_W1);
        float32x4_t v_T3 = vfmaq_f32(v_t2, v_00833, v_W1);
        float32x4_t v_T4 = vfmaq_f32(v_t2, v_minus_00833, v_W1);

        vst1q_f32(T[0][i], v_T0);
        vst1q_f32(T[1][i], v_T1);
        vst1q_f32(T[2][i], v_T2);
        vst1q_f32(T[3][i], v_T3);
        vst1q_f32(T[4][i], v_T4);
        vst1q_f32(T[5][i], v_W2);
    }
    for (int i = 0; i < 6; i++) {
        float32x4_t v_T0 = vld1q_f32(T[i][0]);
        float32x4_t v_T1 = vld1q_f32(T[i][1]);
        float32x4_t v_T2 = vld1q_f32(T[i][2]);

        float32x4_t v_t0 = vmulq_f32(v_01666, v_T2);
        float32x4_t v_t1 = vsubq_f32(vmulq_f32(v_minus_01666, v_T0), v_t0);
        float32x4_t v_t2 = vfmaq_f32(v_t0, v_004166, v_T0);

        float32x4_t v_WTM0 = vmulq_f32(v_025, v_T0);
        float32x4_t v_WTM1 = vfmaq_f32(v_t1, v_minus_01666, v_T1);
        float32x4_t v_WTM2 = vfmaq_f32(v_t1, v_01666, v_T1);
        float32x4_t v_WTM3 = vfmaq_f32(v_t2, v_00833, v_T1);
        float32x4_t v_WTM4 = vfmaq_f32(v_t2, v_minus_00833, v_T1);

        vst1q_f32(WTM[i * 6 + 0], v_WTM0);
        vst1q_f32(WTM[i * 6 + 1], v_WTM1);
        vst1q_f32(WTM[i * 6 + 2], v_WTM2);
        vst1q_f32(WTM[i * 6 + 3], v_WTM3);
        vst1q_f32(WTM[i * 6 + 4], v_WTM4);
        vst1q_f32(WTM[i * 6 + 5], v_T2);
    }
}

inline EE trans_O_4x4_3x3(float *OTM[36],
    float *O[16],
    const float *bias,
    U32 h,
    U32 w,
    U32 _pad_h_mod_4,
    U32 _pad_w_mod_4,
    U32 oh,
    U32 ow,
    ActivationParamSpec activationDesc)
{
    float T[4][6][4];
    // bias
    float32x4_t v_b = vld1q_f32(bias);

    float32x4_t v_0 = vmovq_n_f32(0);
    float32x4_t v_2 = vmovq_n_f32(2);
    float32x4_t v_4 = vmovq_n_f32(4);
    float32x4_t v_6 = vmovq_n_f32(6);
    float32x4_t v_8 = vmovq_n_f32(8);

    for (int i = 0; i < 6; i++) {
        float32x4_t v_OTM0 = vld1q_f32(OTM[i]);
        float32x4_t v_OTM1 = vld1q_f32(OTM[1 * 6 + i]);
        float32x4_t v_OTM2 = vld1q_f32(OTM[2 * 6 + i]);
        float32x4_t v_OTM3 = vld1q_f32(OTM[3 * 6 + i]);
        float32x4_t v_OTM4 = vld1q_f32(OTM[4 * 6 + i]);
        float32x4_t v_OTM5 = vld1q_f32(OTM[5 * 6 + i]);

        float32x4_t v_t0 = vaddq_f32(v_OTM1, v_OTM2);
        float32x4_t v_t1 = vaddq_f32(v_OTM3, v_OTM4);
        float32x4_t v_t2 = vsubq_f32(v_OTM1, v_OTM2);
        float32x4_t v_t3 = vsubq_f32(v_OTM3, v_OTM4);

        float32x4_t v_T0 = vaddq_f32(vaddq_f32(v_t0, v_t1), v_OTM0);
        float32x4_t v_T1 = vfmaq_f32(v_t2, v_t3, v_2);
        float32x4_t v_T2 = vfmaq_f32(v_t0, v_t1, v_4);
        float32x4_t v_T3 = vaddq_f32(vfmaq_f32(v_t2, v_t3, v_8), v_OTM5);

        vst1q_f32(T[0][i], v_T0);
        vst1q_f32(T[1][i], v_T1);
        vst1q_f32(T[2][i], v_T2);
        vst1q_f32(T[3][i], v_T3);
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
        float32x4_t v_T0 = vld1q_f32(T[i][0]);
        float32x4_t v_T1 = vld1q_f32(T[i][1]);
        float32x4_t v_T2 = vld1q_f32(T[i][2]);
        float32x4_t v_T3 = vld1q_f32(T[i][3]);
        float32x4_t v_T4 = vld1q_f32(T[i][4]);
        float32x4_t v_T5 = vld1q_f32(T[i][5]);

        float32x4_t v_t0 = vaddq_f32(v_T1, v_T2);
        float32x4_t v_t1 = vaddq_f32(v_T3, v_T4);
        float32x4_t v_t2 = vsubq_f32(v_T1, v_T2);
        float32x4_t v_t3 = vsubq_f32(v_T3, v_T4);

        float32x4_t v_O0 = vaddq_f32(vaddq_f32(v_t0, v_t1), v_T0);
        float32x4_t v_O1 = vfmaq_f32(v_t2, v_t3, v_2);
        float32x4_t v_O2 = vfmaq_f32(v_t0, v_t1, v_4);
        float32x4_t v_O3 = vaddq_f32(vfmaq_f32(v_t2, v_t3, v_8), v_T5);

        switch (activationDesc.mode) {
            case ACTIVATION_NULL: {
                if (pad_w_mod_4 == 0) {
                    vst1q_f32(O[i * 4 + 0], vaddq_f32(v_O0, v_b));
                    vst1q_f32(O[i * 4 + 1], vaddq_f32(v_O1, v_b));
                    vst1q_f32(O[i * 4 + 2], vaddq_f32(v_O2, v_b));
                    vst1q_f32(O[i * 4 + 3], vaddq_f32(v_O3, v_b));
                } else if (pad_w_mod_4 == 1) {
                    vst1q_f32(O[i * 4 + 0], vaddq_f32(v_O0, v_b));
                    vst1q_f32(O[i * 4 + 1], vaddq_f32(v_O1, v_b));
                    vst1q_f32(O[i * 4 + 2], vaddq_f32(v_O2, v_b));
                } else if (pad_w_mod_4 == 2) {
                    vst1q_f32(O[i * 4 + 0], vaddq_f32(v_O0, v_b));
                    vst1q_f32(O[i * 4 + 1], vaddq_f32(v_O1, v_b));
                } else if (pad_w_mod_4 == 3) {
                    vst1q_f32(O[i * 4 + 0], vaddq_f32(v_O0, v_b));
                }
                break;
            }
            case ACTIVATION_RELU: {
                if (pad_w_mod_4 == 0) {
                    vst1q_f32(O[i * 4 + 0], vmaxq_f32(vaddq_f32(v_O0, v_b), v_0));
                    vst1q_f32(O[i * 4 + 1], vmaxq_f32(vaddq_f32(v_O1, v_b), v_0));
                    vst1q_f32(O[i * 4 + 2], vmaxq_f32(vaddq_f32(v_O2, v_b), v_0));
                    vst1q_f32(O[i * 4 + 3], vmaxq_f32(vaddq_f32(v_O3, v_b), v_0));
                } else if (pad_w_mod_4 == 1) {
                    vst1q_f32(O[i * 4 + 0], vmaxq_f32(vaddq_f32(v_O0, v_b), v_0));
                    vst1q_f32(O[i * 4 + 1], vmaxq_f32(vaddq_f32(v_O1, v_b), v_0));
                    vst1q_f32(O[i * 4 + 2], vmaxq_f32(vaddq_f32(v_O2, v_b), v_0));
                } else if (pad_w_mod_4 == 2) {
                    vst1q_f32(O[i * 4 + 0], vmaxq_f32(vaddq_f32(v_O0, v_b), v_0));
                    vst1q_f32(O[i * 4 + 1], vmaxq_f32(vaddq_f32(v_O1, v_b), v_0));
                } else if (pad_w_mod_4 == 3) {
                    vst1q_f32(O[i * 4 + 0], vmaxq_f32(vaddq_f32(v_O0, v_b), v_0));
                }
                break;
            }
            case ACTIVATION_SIGMOID: {
                if (pad_w_mod_4 == 0) {
                    vst1q_f32(O[i * 4 + 0], vsigmoidq_f32(vaddq_f32(v_O0, v_b)));
                    vst1q_f32(O[i * 4 + 1], vsigmoidq_f32(vaddq_f32(v_O1, v_b)));
                    vst1q_f32(O[i * 4 + 2], vsigmoidq_f32(vaddq_f32(v_O2, v_b)));
                    vst1q_f32(O[i * 4 + 3], vsigmoidq_f32(vaddq_f32(v_O3, v_b)));
                } else if (pad_w_mod_4 == 1) {
                    vst1q_f32(O[i * 4 + 0], vsigmoidq_f32(vaddq_f32(v_O0, v_b)));
                    vst1q_f32(O[i * 4 + 1], vsigmoidq_f32(vaddq_f32(v_O1, v_b)));
                    vst1q_f32(O[i * 4 + 2], vsigmoidq_f32(vaddq_f32(v_O2, v_b)));
                } else if (pad_w_mod_4 == 2) {
                    vst1q_f32(O[i * 4 + 0], vsigmoidq_f32(vaddq_f32(v_O0, v_b)));
                    vst1q_f32(O[i * 4 + 1], vsigmoidq_f32(vaddq_f32(v_O1, v_b)));
                } else if (pad_w_mod_4 == 3) {
                    vst1q_f32(O[i * 4 + 0], vsigmoidq_f32(vaddq_f32(v_O0, v_b)));
                }
                break;
            }
            case ACTIVATION_RELU6: {
                if (pad_w_mod_4 == 0) {
                    vst1q_f32(O[i * 4 + 0], vminq_f32(vmaxq_f32(vaddq_f32(v_O0, v_b), v_0), v_6));
                    vst1q_f32(O[i * 4 + 1], vminq_f32(vmaxq_f32(vaddq_f32(v_O1, v_b), v_0), v_6));
                    vst1q_f32(O[i * 4 + 2], vminq_f32(vmaxq_f32(vaddq_f32(v_O2, v_b), v_0), v_6));
                    vst1q_f32(O[i * 4 + 3], vminq_f32(vmaxq_f32(vaddq_f32(v_O3, v_b), v_0), v_6));
                } else if (pad_w_mod_4 == 1) {
                    vst1q_f32(O[i * 4 + 0], vminq_f32(vmaxq_f32(vaddq_f32(v_O0, v_b), v_0), v_6));
                    vst1q_f32(O[i * 4 + 1], vminq_f32(vmaxq_f32(vaddq_f32(v_O1, v_b), v_0), v_6));
                    vst1q_f32(O[i * 4 + 2], vminq_f32(vmaxq_f32(vaddq_f32(v_O2, v_b), v_0), v_6));
                } else if (pad_w_mod_4 == 2) {
                    vst1q_f32(O[i * 4 + 0], vminq_f32(vmaxq_f32(vaddq_f32(v_O0, v_b), v_0), v_6));
                    vst1q_f32(O[i * 4 + 1], vminq_f32(vmaxq_f32(vaddq_f32(v_O1, v_b), v_0), v_6));
                } else if (pad_w_mod_4 == 3) {
                    vst1q_f32(O[i * 4 + 0], vminq_f32(vmaxq_f32(vaddq_f32(v_O0, v_b), v_0), v_6));
                }
                break;
            }
            default:
                return NOT_SUPPORTED;
        }
    }
    return SUCCESS;
}

inline void trans_I_4x4_3x3(float *ITM[36], float *I[36])
{
    float T[6][6][4];

    float32x4_t v_4 = vmovq_n_f32(4);
    float32x4_t v_minus_4 = vmovq_n_f32(-4);
    float32x4_t v_2 = vmovq_n_f32(2);
    float32x4_t v_minus_5 = vmovq_n_f32(-5);

    for (int i = 0; i < 6; i++) {
        float32x4_t v_I0 = vld1q_f32(I[0 * 6 + i]);
        float32x4_t v_I1 = vld1q_f32(I[1 * 6 + i]);
        float32x4_t v_I2 = vld1q_f32(I[2 * 6 + i]);
        float32x4_t v_I3 = vld1q_f32(I[3 * 6 + i]);
        float32x4_t v_I4 = vld1q_f32(I[4 * 6 + i]);
        float32x4_t v_I5 = vld1q_f32(I[5 * 6 + i]);

        float32x4_t v_t0 = vfmaq_f32(v_I4, v_I2, v_minus_4);
        float32x4_t v_t1 = vfmaq_f32(v_I3, v_I1, v_minus_4);
        float32x4_t v_t2 = vsubq_f32(v_I4, v_I2);
        float32x4_t v_t3 = vmulq_f32(vsubq_f32(v_I3, v_I1), v_2);
        float32x4_t v_t4 = vfmaq_f32(v_I4, v_I0, v_4);
        float32x4_t v_t5 = vfmaq_f32(v_I5, v_I1, v_4);

        float32x4_t v_T0 = vfmaq_f32(v_t4, v_I2, v_minus_5);
        float32x4_t v_T1 = vaddq_f32(v_t1, v_t0);
        float32x4_t v_T2 = vsubq_f32(v_t0, v_t1);
        float32x4_t v_T3 = vaddq_f32(v_t3, v_t2);
        float32x4_t v_T4 = vsubq_f32(v_t2, v_t3);
        float32x4_t v_T5 = vfmaq_f32(v_t5, v_I3, v_minus_5);

        vst1q_f32(T[0][i], v_T0);
        vst1q_f32(T[1][i], v_T1);
        vst1q_f32(T[2][i], v_T2);
        vst1q_f32(T[3][i], v_T3);
        vst1q_f32(T[4][i], v_T4);
        vst1q_f32(T[5][i], v_T5);
    }

    for (int i = 0; i < 6; i++) {
        float32x4_t v_T0 = vld1q_f32(T[i][0]);
        float32x4_t v_T1 = vld1q_f32(T[i][1]);
        float32x4_t v_T2 = vld1q_f32(T[i][2]);
        float32x4_t v_T3 = vld1q_f32(T[i][3]);
        float32x4_t v_T4 = vld1q_f32(T[i][4]);
        float32x4_t v_T5 = vld1q_f32(T[i][5]);

        float32x4_t v_t0 = vfmaq_f32(v_T4, v_T2, v_minus_4);
        float32x4_t v_t1 = vfmaq_f32(v_T3, v_T1, v_minus_4);
        float32x4_t v_t2 = vsubq_f32(v_T4, v_T2);
        float32x4_t v_t3 = vmulq_f32(vsubq_f32(v_T3, v_T1), v_2);
        float32x4_t v_t4 = vfmaq_f32(v_T4, v_T0, v_4);
        float32x4_t v_t5 = vfmaq_f32(v_T5, v_T1, v_4);

        float32x4_t v_ITM0 = vfmaq_f32(v_t4, v_T2, v_minus_5);
        float32x4_t v_ITM1 = vaddq_f32(v_t1, v_t0);
        float32x4_t v_ITM2 = vsubq_f32(v_t0, v_t1);
        float32x4_t v_ITM3 = vaddq_f32(v_t3, v_t2);
        float32x4_t v_ITM4 = vsubq_f32(v_t2, v_t3);
        float32x4_t v_ITM5 = vfmaq_f32(v_t5, v_T3, v_minus_5);

        vst1q_f32(ITM[i * 6 + 0], v_ITM0);
        vst1q_f32(ITM[i * 6 + 1], v_ITM1);
        vst1q_f32(ITM[i * 6 + 2], v_ITM2);
        vst1q_f32(ITM[i * 6 + 3], v_ITM3);
        vst1q_f32(ITM[i * 6 + 4], v_ITM4);
        vst1q_f32(ITM[i * 6 + 5], v_ITM5);
    }
}
#endif
#endif
