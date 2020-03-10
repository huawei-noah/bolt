// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_ARM_NEON_EXPAND
#define _H_ARM_NEON_EXPAND

#include "error.h"

#ifdef _USE_FP16
inline F32 vaddvq_f16(float16x8_t x)
{
    float32x4_t a = vcvt_f32_f16(vget_high_f16(x));
    float32x4_t b = vcvt_f32_f16(vget_low_f16(x));
    F32 sum = vaddvq_f32(vaddq_f32(a, b));
    return sum;
}

inline void vst1q_lane_f16_builtin(F16* address, float16x8_t vec, const int laneId) {
    switch (laneId) {
        case 0:
            vst1q_lane_f16(address, vec, 0);
            break;
        case 1:
            vst1q_lane_f16(address, vec, 1);
            break;
        case 2:
            vst1q_lane_f16(address, vec, 2);
            break;
        case 3:
            vst1q_lane_f16(address, vec, 3);
            break;
        case 4:
            vst1q_lane_f16(address, vec, 4);
            break;
        case 5:
            vst1q_lane_f16(address, vec, 5);
            break;
        case 6:
            vst1q_lane_f16(address, vec, 6);
            break;
        case 7:
            vst1q_lane_f16(address, vec, 7);
            break;
        default:
            CHECK_REQUIREMENT(0);
    }
}
#endif

#ifdef _USE_INT8
inline int32x4_t vdotq_laneq_s32_builtin(int32x4_t c, int8x16_t a, int8x16_t b, const int laneId) {
    switch (laneId) {
        case 0:
            return vdotq_laneq_s32(c, a, b, 0);
        case 1:
            return vdotq_laneq_s32(c, a, b, 1);
        case 2:
            return vdotq_laneq_s32(c, a, b, 2);
        case 3:
            return vdotq_laneq_s32(c, a, b, 3);
        default:
            CHECK_REQUIREMENT(0);
    }
}
#endif

#endif
