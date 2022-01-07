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

inline void convolution_nchwc8_input_pack_tile8_fp16(const U32 &ic,
    const U32 &it_pad,
    const U32 &ih_pad,
    const U32 &iw_pad,
    const ConvolutionParamSpec &p,
    const U32 &ft,
    const U32 &fh,
    const U32 &fw,
    F16 *src,
    U32 *padding_input_offset,
    F16 *dst)
{
    const int TileSize = 8;
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
                    F16 *in_pack_c8hw8 =
                        dst + (((ft_idx * fh + fh_idx) * fw + fw_idx) * ic + c) * TileSize * 8;
                   F16 *in_0 = in_hw8c8 + padding_input_offset[0];
                   F16 *in_1 = in_hw8c8 + padding_input_offset[1];
                   F16 *in_2 = in_hw8c8 + padding_input_offset[2];
                   F16 *in_3 = in_hw8c8 + padding_input_offset[3];
                   F16 *in_4 = in_hw8c8 + padding_input_offset[4];
                   F16 *in_5 = in_hw8c8 + padding_input_offset[5];
                   F16 *in_6 = in_hw8c8 + padding_input_offset[6];
                   F16 *in_7 = in_hw8c8 + padding_input_offset[7];
                  __asm__ __volatile__(
                      "ldr q0, [%[in_0]]\n"
                      "ldr q4, [%[in_4]]\n"
                      "ldr q1, [%[in_1]]\n"
                      "ldr q5, [%[in_5]]\n"
                      "ldr q2, [%[in_2]]\n"
                      "ldr q6, [%[in_6]]\n"
                      "ldr q3, [%[in_3]]\n"
                      "ldr q7, [%[in_7]]\n"

                      "zip1  v8.8h,  v0.8h,  v4.8h\n"
                      "zip2  v9.8h,  v0.8h,  v4.8h\n"
                      "zip1 v10.8h,  v1.8h,  v5.8h\n"
                      "zip2 v11.8h,  v1.8h,  v5.8h\n"
                      "zip1 v12.8h,  v2.8h,  v6.8h\n"
                      "zip2 v13.8h,  v2.8h,  v6.8h\n"
                      "zip1 v14.8h,  v3.8h,  v7.8h\n"
                      "zip2 v15.8h,  v3.8h,  v7.8h\n"

                      "zip1  v0.8h,  v8.8h, v12.8h\n"
                      "zip2  v1.8h,  v8.8h, v12.8h\n"
                      "zip1  v2.8h,  v9.8h, v13.8h\n"
                      "zip2  v3.8h,  v9.8h, v13.8h\n"
                      "zip1  v4.8h, v10.8h, v14.8h\n"
                      "zip2  v5.8h, v10.8h, v14.8h\n"
                      "zip1  v6.8h, v11.8h, v15.8h\n"
                      "zip2  v7.8h, v11.8h, v15.8h\n"

                      "zip1  v8.8h,  v0.8h,  v4.8h\n"
                      "zip2  v9.8h,  v0.8h,  v4.8h\n"
                      "str  q8, [%[pack]]\n"
                      "zip1 v10.8h,  v1.8h,  v5.8h\n"
                      "str  q9, [%[pack], #16]\n"
                      "zip2 v11.8h,  v1.8h,  v5.8h\n"
                      "str q10, [%[pack], #32]\n"
                      "zip1 v12.8h,  v2.8h,  v6.8h\n"
                      "str q11, [%[pack], #48]\n"
                      "zip2 v13.8h,  v2.8h,  v6.8h\n"
                      "str q12, [%[pack], #64]\n"
                      "zip1 v14.8h,  v3.8h,  v7.8h\n"
                      "str q13, [%[pack], #80]\n"
                      "zip2 v15.8h,  v3.8h,  v7.8h\n"
                      "str q14, [%[pack], #96]\n"
                      "str q15, [%[pack], #112]\n"
                      :
                      : [pack] "r"(in_pack_c8hw8), [in_0] "r"(in_0), [in_1] "r"(in_1),
                      [in_2] "r"(in_2), [in_3] "r"(in_3), [in_4] "r"(in_4),
                      [in_5] "r"(in_5), [in_6] "r"(in_6), [in_7] "r"(in_7)
                      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                      "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
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
