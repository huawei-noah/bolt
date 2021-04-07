// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _TRANSFORM_FUNCTIONS_FP16_H
#define _TRANSFORM_FUNCTIONS_FP16_H
#include "cpu/arm/transform_functions.h"

inline void convolution_nchwc8_input_pack_tile8_fp16(U32 ic,
    U32 it_pad,
    U32 ih_pad,
    U32 iw_pad,
    ConvolutionParamSpec p,
    U32 ft,
    U32 fh,
    U32 fw,
    U32 ot,
    U32 oh,
    U32 ow,
    F16 *src,
    U32 hw,
    F16 *dst)
{
    const int TileSize = 8;
    U32 padding_input_offset[8];
    convolution_padding_input_offset<8, 8>(ih_pad, iw_pad, p, oh, ow, hw, padding_input_offset);
    float16x8_t v[TileSize];
    for (U32 c = 0; c < ic; c++) {
        for (U32 ft_idx = 0; ft_idx < ft; ft_idx++) {
            for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                    F16 *in_hw8c8 = src +
                        (((c * it_pad + ft_idx * p.dilatedRate_t) * ih_pad +
                             fh_idx * p.dilatedRate_h) *
                                iw_pad +
                            p.dilatedRate_w * fw_idx) *
                            8;
                    for (int id = 0; id < TileSize; id++) {
                        F16 *in_0 = in_hw8c8 + padding_input_offset[id];
                        v[id] = vld1q_f16(in_0);
                    }
                    float16x8_t zip1q_04 = vzip1q_f16(v[0], v[4]);
                    float16x8_t zip2q_04 = vzip2q_f16(v[0], v[4]);
                    float16x8_t zip1q_15 = vzip1q_f16(v[1], v[5]);
                    float16x8_t zip2q_15 = vzip2q_f16(v[1], v[5]);
                    float16x8_t zip1q_26 = vzip1q_f16(v[2], v[6]);
                    float16x8_t zip2q_26 = vzip2q_f16(v[2], v[6]);
                    float16x8_t zip1q_37 = vzip1q_f16(v[3], v[7]);
                    float16x8_t zip2q_37 = vzip2q_f16(v[3], v[7]);
                    F16 *in_pack_c8hw8 =
                        dst + (((ft_idx * fh + fh_idx) * fw + fw_idx) * ic + c) * TileSize * 8;
                    vst1q_f16(in_pack_c8hw8,
                        vzip1q_f16(vzip1q_f16(zip1q_04, zip1q_26), vzip1q_f16(zip1q_15, zip1q_37)));
                    vst1q_f16(in_pack_c8hw8 + 8,
                        vzip2q_f16(vzip1q_f16(zip1q_04, zip1q_26), vzip1q_f16(zip1q_15, zip1q_37)));
                    vst1q_f16(in_pack_c8hw8 + 8 * 2,
                        vzip1q_f16(vzip2q_f16(zip1q_04, zip1q_26), vzip2q_f16(zip1q_15, zip1q_37)));
                    vst1q_f16(in_pack_c8hw8 + 8 * 3,
                        vzip2q_f16(vzip2q_f16(zip1q_04, zip1q_26), vzip2q_f16(zip1q_15, zip1q_37)));
                    vst1q_f16(in_pack_c8hw8 + 8 * 4,
                        vzip1q_f16(vzip1q_f16(zip2q_04, zip2q_26), vzip1q_f16(zip2q_15, zip2q_37)));
                    vst1q_f16(in_pack_c8hw8 + 8 * 5,
                        vzip2q_f16(vzip1q_f16(zip2q_04, zip2q_26), vzip1q_f16(zip2q_15, zip2q_37)));
                    vst1q_f16(in_pack_c8hw8 + 8 * 6,
                        vzip1q_f16(vzip2q_f16(zip2q_04, zip2q_26), vzip2q_f16(zip2q_15, zip2q_37)));
                    vst1q_f16(in_pack_c8hw8 + 8 * 7,
                        vzip2q_f16(vzip2q_f16(zip2q_04, zip2q_26), vzip2q_f16(zip2q_15, zip2q_37)));
                }
            }
        }
    }
}

inline void convolution_nchwc8_input_pack_tile4_fp16(U32 ic,
    U32 it_pad,
    U32 ih_pad,
    U32 iw_pad,
    ConvolutionParamSpec p,
    U32 ft,
    U32 fh,
    U32 fw,
    U32 ot,
    U32 oh,
    U32 ow,
    F16 *src,
    U32 hw,
    F16 *dst)
{
    const int TileSize = 4;
    U32 padding_input_offset[4];
    convolution_padding_input_offset<4, 8>(ih_pad, iw_pad, p, oh, ow, hw, padding_input_offset);
    F16* in[TileSize];
    for (U32 c = 0; c < ic; c++) {
        for (U32 ft_idx = 0; ft_idx < ft; ft_idx++) {
            for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                    F16 *in_hw4c8 = src +
                        (((c * it_pad + ft_idx * p.dilatedRate_t) * ih_pad +
                             fh_idx * p.dilatedRate_h) *
                                iw_pad +
                            p.dilatedRate_w * fw_idx) *
                            8;
                    F16 *in_pack_c8hw4 =
                        dst + (((ft_idx * fh + fh_idx) * fw + fw_idx) * ic + c) * TileSize * 8;

                    for (int id = 0; id < TileSize; id++) {
                        in[id] = in_hw4c8 + padding_input_offset[id];
                    }
                    __asm__ __volatile__(
                        "ldr q0, [%[in_0]]\n"
                        "ldr q1, [%[in_1]]\n"
                        "ldr q2, [%[in_2]]\n"
                        "ldr q3, [%[in_3]]\n"
                        "st4 {v0.8h, v1.8h, v2.8h, v3.8h}, [%[in_pack_0]]\n"
                        : [in_pack_0] "+r"(in_pack_c8hw4)
                        : [in_0] "r"(in[0]), [in_1] "r"(in[1]), [in_2] "r"(in[2]), [in_3] "r"(in[3])
                        : "memory", "cc", "v0", "v1", "v2", "v3");
                }
            }
        }
    }
}
#endif
