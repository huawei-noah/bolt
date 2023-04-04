// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/int8/v7/convolution_gemm.h"
#include "cpu/arm/arm_functions.h"
#include "cpu/arm/transform_functions.h"

template <typename OT>
EE convolution_gemm_v7(TensorDesc inputDesc,
    const void *input,
    F32 *inputScale,
    TensorDesc filterDesc,
    const void *filter,
    F32 *filterScale,
    ConvolutionParamSpec p,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    F32 *outputScale,
    ActivationParamSpec activationDesc)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    // still im2col + gemm with a smaller buffer
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, it = 1, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 strideH = p.stride_h;
    U32 strideW = p.stride_w;
    U32 paddingT = p.pad_top;
    U32 paddingB = p.pad_bottom;
    U32 paddingL = p.pad_left;
    U32 paddingR = p.pad_right;
    U32 dilateH = p.dilatedRate_h;
    U32 dilateW = p.dilatedRate_w;

    if (fdf != DF_NHWCN8) {
        return NOT_MATCH;
    }

    I64 conv_relu_bool = (activationDesc.mode == ACTIVATION_RELU) ? 1 : 0;
    I64 out_f32_bool = (odt == DT_F32) ? 1 : 0;
    I64 scale_known_bool = 0;
    if (*outputScale > 0 || ACTIVATION_RELU6 == activationDesc.mode) {
        scale_known_bool = 1;
    }

    INT8 *inArray = (INT8 *)input;
    INT8 *filterArray = (INT8 *)filter;
    F32 *outArray = (F32 *)output;
    F32 *biasArray = (F32 *)bias;
    INT8 *in_pad = (INT8 *)tmp;

    oc /= 8;
    ic /= 8;

    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;
    I32 ohow = oh * ow;
    U32 ihiw = ih_pad * iw_pad;

    I32 *biasScaled = (I32 *)(in_pad + ic * ihiw * 8 + 12 * fh * fw * ic * 8);  // Initialize

    // double start, end;
    I32 max_i32[4] = {0};  // To record max I32 values
    I32 min_i32[4] = {0};  // To record min I32 values

    for (U32 n = 0; n < in; n++) {  // for each batch
        F32 scale_i = 1.0;

        // quantize input if necessary
        if (idt == DT_F32) {
            F32 *in = ((F32 *)input) + n * ic * ih * iw * 8;
            // After the space for padding and packing
            inArray = in_pad + ic * ihiw * 8 + 12 * fh * fw * ic * 8;

            I32 numData = ic * ih * iw * 8;
            if (*inputScale > 0) {
                scale_i = *inputScale;
            } else {
                F32 minmax[2] = {1, -1};
                CHECK_STATUS(array_minmax_value_f32(in, numData, 3, minmax));
                F32 absMax = UNI_MAX(UNI_ABS(minmax[0]), UNI_ABS(minmax[1]));
                scale_i = 127.0 / absMax;
            }
            array_scale_round_f32(in, inArray, numData, scale_i, true);
            *inputScale = scale_i;
        } else {
            scale_i = *inputScale;
        }

        if (1 == scale_known_bool) {
            if (ACTIVATION_RELU6 == activationDesc.mode) {
                *outputScale = 127.0 / 6.0;
            }
            F32 scaleInt = (*outputScale / *inputScale) / *filterScale;
            I32 thresholdP = 127.0 / scaleInt;
            I32 thresholdN = 0;
            if (ACTIVATION_RELU6 != activationDesc.mode) {
                thresholdN = thresholdP * -1;
            }
            for (U32 i = 0; i < 4; i++) {
                max_i32[i] = thresholdP;
                min_i32[i] = thresholdN;
            }
        }

        if (odt == DT_I8) {  // Scale the bias
            if (idt == DT_F32) {
                biasScaled += ic * ih * iw * 8 / bytesOf(DT_I32);  // After the quantized input
            }
            F32 scale = (*inputScale) * (*filterScale);
            array_scale_round_f32_i32(biasArray, biasScaled, oc * 8, scale, true);
        }

        F32 factor_s = 1.0 / ((F32)scale_i) / ((F32)(*filterScale));

        INT8 *inArray_pad = convolution_input_padding_per_channel<INT8, 8>(
            n, ic, it, ih, iw, p, inArray, (INT8 *)tmp);

        for (I32 hw = 0; hw < ohow - 3; hw += 4) {
            F32 *b0 = biasArray;
            I32 *b0_s = biasScaled;
            INT8 *in_pack = ((INT8 *)tmp) + ic * ih_pad * iw_pad * 8;
            U32 padding_input_offset[4];
            convolution_padding_input_offset<4, 8>(
                ih_pad, iw_pad, p, oh, ow, hw, padding_input_offset);
            convolution_input_pack<INT8, 4, 8>(
                ic, 1, ih_pad, iw_pad, p, 1, fh, fw, inArray_pad, padding_input_offset, in_pack);

            // compute
            for (U32 o = 0; o < oc; o++) {
                INT8 *in_hw = in_pack;
                INT8 *f_o = filterArray + o * 8 * fh * fw * ic * 8;
                I32 *out_buf = biasScaled + oc * 8 + ((n * oc + o) * ohow + hw) * 8;
                F32 *out_o0hw0 = outArray + ((n * oc + o) * ohow + hw) * 8;
#if 1
                asm volatile(
                    "cmp %[out_f32], #0\n"
                    "beq 0f\n"
                    "vmov.s32  q4, #0.\n"
                    "vmov.s32  q5, #0.\n"
                    "vmov.s32  q6, #0.\n"
                    "vmov.s32  q7, #0.\n"
                    "vmov.s32  q8, #0.\n"
                    "vmov.s32  q9, #0.\n"
                    "vmov.s32 q10, #0.\n"
                    "vmov.s32 q11, #0.\n"
                    "b 1f\n"

                    "0:\n"
                    "vld1.s32 {d8-d11},  [%[b0_s]]\n"
                    "vld1.s32 {d12-d15}, [%[b0_s]]\n"
                    "vld1.s32 {d16-d19}, [%[b0_s]]\n"
                    "vld1.s32 {d20-d23}, [%[b0_s]]\n"

                    "1:\n"
                    "vld1.s8 {d0[]}, [%[in]]!\n"
                    "vld1.s8 {d1[]}, [%[in]]!\n"
                    "vld1.s8 {d2[]}, [%[in]]!\n"
                    "vld1.s8 {d3[]}, [%[in]]!\n"

                    "vld1.s8  {d4-d5}, [%[w]]!\n"

                    // K- > r2
                    "mov r2, %[K]\n"

                    // Computation loop
                    "2:\n"

                    "vmull.s8 q12, d4, d0\n"
                    "vld1.s8 {d0[]}, [%[in]]!\n"
                    "vmull.s8 q13, d4, d1\n"
                    "vld1.s8 {d1[]}, [%[in]]!\n"
                    "vmull.s8 q14, d4, d2\n"
                    "vld1.s8 {d2[]}, [%[in]]!\n"
                    "vmull.s8 q15, d4, d3\n"
                    "vld1.s8 {d3[]}, [%[in]]!\n"
                    "vld1.s8  {d4}, [%[w]]!\n"

                    "vmlal.s8 q12, d5, d0\n"
                    "vmlal.s8 q13, d5, d1\n"
                    "vld1.s8 {d0[]}, [%[in]]!\n"
                    "vmlal.s8 q14, d5, d2\n"
                    "vld1.s8 {d1[]}, [%[in]]!\n"
                    "vmlal.s8 q15, d5, d3\n"

                    //"vaddw.s16 q4, q4, d24\n"
                    //"vaddw.s16 q5, q5, d25\n"
                    //"vaddw.s16 q6, q6, d26\n"
                    //"vaddw.s16 q7, q7, d27\n"
                    //"vaddw.s16 q8, q8, d28\n"
                    //"vaddw.s16 q9, q9, d29\n"
                    //"vaddw.s16 q10, q10, d30\n"
                    //"vaddw.s16 q11, q11, d31\n"
                    //"vmov.s32 q12, #0\n"
                    //"vmov.s32 q13, #0\n"
                    //"vmov.s32 q14, #0\n"
                    //"vmov.s32 q15, #0\n"

                    "vld1.s8 {d2[]}, [%[in]]!\n"
                    "vmlal.s8 q12, d4, d0\n"
                    "vld1.s8 {d3[]}, [%[in]]!\n"
                    "vld1.s8  {d5}, [%[w]]!\n"
                    "vmlal.s8 q13, d4, d1\n"
                    "vld1.s8 {d0[]}, [%[in]]!\n"
                    "vmlal.s8 q14, d4, d2\n"
                    "vld1.s8 {d1[]}, [%[in]]!\n"
                    "vmlal.s8 q15, d4, d3\n"
                    "vld1.s8 {d2[]}, [%[in]]!\n"

                    "vmlal.s8 q12, d5, d0\n"
                    "vld1.s8 {d3[]}, [%[in]]!\n"
                    "vld1.s8  {d4}, [%[w]]!\n"
                    "vmlal.s8 q13, d5, d1\n"
                    "vld1.s8 {d0[]}, [%[in]]!\n"
                    "vmlal.s8 q14, d5, d2\n"
                    "vld1.s8 {d1[]}, [%[in]]!\n"
                    "vmlal.s8 q15, d5, d3\n"
                    "vld1.s8 {d2[]}, [%[in]]!\n"
                    "vld1.s8 {d3[]}, [%[in]]!\n"
                    "vld1.s8  {d5}, [%[w]]!\n"

                    "subs r2, r2, #4\n"

                    "vaddw.s16 q4, q4, d24\n"
                    "vaddw.s16 q5, q5, d25\n"
                    "vaddw.s16 q6, q6, d26\n"
                    "vaddw.s16 q7, q7, d27\n"
                    "vaddw.s16 q8, q8, d28\n"
                    "vaddw.s16 q9, q9, d29\n"
                    "vaddw.s16 q10, q10, d30\n"
                    "vaddw.s16 q11, q11, d31\n"

                    "bne 2b\n"
                    : [in] "+r"(in_hw), [w] "+r"(f_o)
                    : [K] "r"((I64)(ic * fh * fw * 8)), [b0_s] "r"(b0_s), [out_f32] "r"(out_f32_bool)
                    : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                    "q10", "q11", "q12", "q13", "q14", "q15", "r1", "r2");

                asm volatile("cmp %[out_f32], #0\n"
                             "beq 4f\n"
                             "vdup.32 q0, %[factor]\n"
                             "vld1.f32 {d2-d5}, [%[b0]]\n"
                             "vcvt.f32.s32  q3, q4\n"
                             "vcvt.f32.s32 q12, q5\n"
                             "vcvt.f32.s32 q13, q6\n"
                             "vcvt.f32.s32 q14, q7\n"
                             "vmulq.f32  q3,  q3, q0\n"
                             "vmulq.f32 q12, q12, q0\n"
                             "vmulq.f32 q13, q13, q0\n"
                             "vmulq.f32 q14, q14, q0\n"
                             "vaddq.f32  q4,  q3, q1\n"
                             "vaddq.f32  q5, q12, q2\n"
                             "vaddq.f32  q6, q13, q1\n"
                             "vaddq.f32  q7, q14, q2\n"
                             "vcvt.f32.s32  q3,  q8\n"
                             "vcvt.f32.s32 q12,  q9\n"
                             "vcvt.f32.s32 q13, q10\n"
                             "vcvt.f32.s32 q14, q11\n"
                             "vmulq.f32  q3,  q3, q0\n"
                             "vmulq.f32 q12, q12, q0\n"
                             "vmulq.f32 q13, q13, q0\n"
                             "vmulq.f32 q14, q14, q0\n"
                             "vaddq.f32  q8,  q3, q1\n"
                             "vaddq.f32  q9, q12, q2\n"
                             "vaddq.f32 q10, q13, q1\n"
                             "vaddq.f32 q11, q14, q2\n"
                             "cmp %[conv_relu], #0\n"
                             "beq 3f\n"
                             "vmov.f32 q0, #0.\n"
                             "vmaxq.f32  q4,  q4, q0\n"
                             "vmaxq.f32  q5,  q5, q0\n"
                             "vmaxq.f32  q6,  q6, q0\n"
                             "vmaxq.f32  q7,  q7, q0\n"
                             "vmaxq.f32  q8,  q8, q0\n"
                             "vmaxq.f32  q9,  q9, q0\n"
                             "vmaxq.f32 q10, q10, q0\n"
                             "vmaxq.f32 q11, q11, q0\n"

                             "3:\n"
                             "mov r1, %[out_o0hw0]\n"
                             "vst1.f32  {d8-d11}, [r1]!\n"
                             "vst1.f32 {d12-d15}, [r1]!\n"
                             "vst1.f32 {d16-d19}, [r1]!\n"
                             "vst1.f32 {d20-d23}, [r1]\n"

                             "4:"
                             : [out_o0hw0] "+r"(out_o0hw0)
                             : [factor] "r"(factor_s), [b0] "r"(b0),
                             [conv_relu] "r"(conv_relu_bool), [out_f32] "r"(out_f32_bool)
                             : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
                             "q9", "q10", "q11", "q12", "q13", "q14", "q15", "r1");

                asm volatile(
                    "cmp %[out_f32], #0\n"
                    "bne 8f\n"

                    "4:\n"
                    "cmp %[conv_relu], #0\n"
                    "beq 5f\n"
                    "vmov.s32  q0, #0\n"
                    "vmaxq.s32  q4,  q4, q0\n"
                    "vmaxq.s32  q5,  q5, q0\n"
                    "vmaxq.s32  q6,  q6, q0\n"
                    "vmaxq.s32  q7,  q7, q0\n"
                    "vmaxq.s32  q8,  q8, q0\n"
                    "vmaxq.s32  q9,  q9, q0\n"
                    "vmaxq.s32 q10, q10, q0\n"
                    "vmaxq.s32 q11, q11, q0\n"

                    "5:\n"
                    "vld1.s32 {d0-d1}, [%[max_i32]]\n"
                    "vld1.s32 {d2-d3}, [%[min_i32]]\n"
                    "cmp %[scale_known], #0\n"
                    "beq 6f\n"
                    "vmaxq.s32  q4,  q4, q1\n"
                    "vmaxq.s32  q5,  q5, q1\n"
                    "vmaxq.s32  q6,  q6, q1\n"
                    "vmaxq.s32  q7,  q7, q1\n"
                    "vmaxq.s32  q8,  q8, q1\n"
                    "vmaxq.s32  q9,  q9, q1\n"
                    "vmaxq.s32 q10, q10, q1\n"
                    "vmaxq.s32 q11, q11, q1\n"
                    "vminq.s32  q4,  q4, q0\n"
                    "vminq.s32  q5,  q5, q0\n"
                    "vminq.s32  q6,  q6, q0\n"
                    "vminq.s32  q7,  q7, q0\n"
                    "vminq.s32  q8,  q8, q0\n"
                    "vminq.s32  q9,  q9, q0\n"
                    "vminq.s32 q10, q10, q0\n"
                    "vminq.s32 q11, q11, q0\n"
                    "b 7f\n"

                    "6:\n"
                    "vmaxq.s32 q0,  q4, q0\n"
                    "vmaxq.s32 q0,  q5, q0\n"
                    "vmaxq.s32 q0,  q6, q0\n"
                    "vmaxq.s32 q0,  q7, q0\n"
                    "vmaxq.s32 q0,  q8, q0\n"
                    "vmaxq.s32 q0,  q9, q0\n"
                    "vmaxq.s32 q0, q10, q0\n"
                    "vmaxq.s32 q0, q11, q0\n"
                    "vminq.s32 q1,  q4, q1\n"
                    "vminq.s32 q1,  q5, q1\n"
                    "vminq.s32 q1,  q6, q1\n"
                    "vminq.s32 q1,  q7, q1\n"
                    "vminq.s32 q1,  q8, q1\n"
                    "vminq.s32 q1,  q9, q1\n"
                    "vminq.s32 q1, q10, q1\n"
                    "vminq.s32 q1, q11, q1\n"
                    "vst1.s32 {d0-d1}, [%[max_i32]]\n"
                    "vst1.s32 {d2-d3}, [%[min_i32]]\n"

                    "7:\n"
                    "mov r1, %[out_buf]\n"
                    "vst1.s32  {d8-d11}, [r1]!\n"
                    "vst1.s32 {d12-d15}, [r1]!\n"
                    "vst1.s32 {d16-d19}, [r1]!\n"
                    "vst1.s32 {d20-d23}, [r1]\n"

                    "8:\n"
                    : [out_buf] "+r"(out_buf)
                    : [max_i32] "r"(max_i32), [min_i32] "r"(min_i32), [conv_relu] "r"(conv_relu_bool),
                    [out_f32] "r"(out_f32_bool), [scale_known] "r"(scale_known_bool)
                    : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                    "q10", "q11", "q12", "q13", "q14", "q15", "r1");
#else
                int32x4_t res[4][2] = {0};
                if (out_f32_bool == 0) {
                    for (U32 i = 0; i < 4; i++) {
                        res[i][0] = vld1q_s32(b0_s);
                        res[i][1] = vld1q_s32(b0_s + 4);
                    }
                }

#if 0
                for (U32 c = 0; c < ic * fh * fw * 8; c++, f_o += 8) {
                    int8x8_t b = vld1_s8(f_o);
                    for (U32 i = 0; i < 4; i++, in_hw++) {
                        int8x8_t a = vdup_n_s8(in_hw[0]);
                        int16x8_t r = vmull_s8(a, b);
                        res[i][0] = vaddw_s16(res[i][0], vget_low_s16(r));
                        res[i][1] = vaddw_s16(res[i][1], vget_high_s16(r));
                    }
                }
#else
                for (U32 c = 0; c < ic * fh * fw * 2; c++) {
                    int16x8_t r = vdupq_n_s16(0);
                    for (U32 i = 0; i < 4; i++, in_hw++, f_o += 8) {
                        int8x8_t a = vdup_n_s8(in_hw[0]);
                        int8x8_t b = vld1_s8(f_o);
                        r = vmlal_s8(r, a, b);
                    }
                    res[0] = vaddw_s16(res[0], vget_low_s16(r));
                    res[1] = vaddw_s16(res[1], vget_high_s16(r));
                }
#endif
                if (out_f32_bool == 1) {
                    float32x4_t fac = vdupq_n_f32(factor_s);
                    float32x4_t bias0 = vld1q_f32(b0);
                    float32x4_t bias1 = vld1q_f32(b0 + 4);
                    for (U32 i = 0; i < 4; i++) {
                        float32x4_t resf0 = vcvtq_f32_s32(res[i][0]);
                        float32x4_t resf1 = vcvtq_f32_s32(res[i][1]);
                        resf0 = vmulq_f32(resf0, fac);
                        resf1 = vmulq_f32(resf1, fac);
                        resf0 = vaddq_f32(resf0, bias0);
                        resf1 = vaddq_f32(resf1, bias1);

                        if (conv_relu_bool) {
                            float32x4_t z = vdupq_n_f32(0);
                            resf0 = vmaxq_f32(resf0, z);
                            resf1 = vmaxq_f32(resf1, z);
                        }
                        vst1q_f32(out_o0hw0, resf0);
                        vst1q_f32(out_o0hw0 + 4, resf1);
                        out_o0hw0 += 8;
                    }
                } else {
                    int32x4_t max = vld1q_s32(max_i32);
                    int32x4_t min = vld1q_s32(min_i32);
                    for (U32 i = 0; i < 4; i++) {
                        if (conv_relu_bool) {
                            int32x4_t z = vdupq_n_s32(0);
                            res[i][0] = vmaxq_s32(res[i][0], z);
                            res[i][1] = vmaxq_s32(res[i][1], z);
                        }
                        if (1 == scale_known_bool) {
                            res[i][0] = vmaxq_s32(min, res[i][0]);
                            res[i][1] = vmaxq_s32(min, res[i][1]);
                            res[i][0] = vminq_s32(max, res[i][0]);
                            res[i][1] = vminq_s32(max, res[i][1]);
                        } else {
                            max = vmaxq_s32(max, res[i][0]);
                            min = vminq_s32(min, res[i][0]);
                            max = vmaxq_s32(max, res[i][1]);
                            min = vminq_s32(min, res[i][1]);
                            vst1q_s32(max_i32, max);
                            vst1q_s32(min_i32, min);
                        }
                        vst1q_s32(out_buf, res[i][0]);
                        vst1q_s32(out_buf + 4, res[i][1]);
                        out_buf += 8;
                    }
                }
#endif
                b0 += 8;
                b0_s += 8;
            }
        }

        for (I32 hw = ohow / 4 * 4; hw < ohow; hw++) {
            F32 *b0 = biasArray;
            I32 *b0_s = biasScaled;
            INT8 *in_pack = ((INT8 *)tmp) + ic * ih_pad * iw_pad * 8;
            convolution_nchwc8_input_pack_tile1<INT8>(
                ic, 1, ih_pad, iw_pad, p, 1, fh, fw, 1, oh, ow, inArray_pad, hw, in_pack);

            // compute
            for (U32 o = 0; o < oc; o++) {
                INT8 *in_hw = in_pack;
                INT8 *f_o = filterArray + o * 8 * fh * fw * ic * 8;
                I32 *out_buf = biasScaled + oc * 8 + ((n * oc + o) * ohow + hw) * 8;
                F32 *out_o0hw0 = outArray + ((n * oc + o) * ohow + hw) * 8;

                int32x4_t res[2] = {0};
                if (out_f32_bool == 0) {
                    res[0] = vld1q_s32(b0_s);
                    res[1] = vld1q_s32(b0_s + 4);
                }

#if 0
                for (U32 c = 0; c < ic * fh * fw * 8; c++, in_hw++, f_o += 8) {
                    int8x8_t a = vdup_n_s8(in_hw[0]);
                    int8x8_t b = vld1_s8(f_o);
                    int16x8_t r = vmull_s8(a, b);
                    res[0] = vaddw_s16(res[0], vget_low_s16(r));
                    res[1] = vaddw_s16(res[1], vget_high_s16(r));
                }
#else
                for (U32 c = 0; c < ic * fh * fw * 2; c++) {
                    int16x8_t r = vdupq_n_s16(0);
                    for (U32 i = 0; i < 4; i++, in_hw++, f_o += 8) {
                        int8x8_t a = vdup_n_s8(in_hw[0]);
                        int8x8_t b = vld1_s8(f_o);
                        r = vmlal_s8(r, a, b);
                    }
                    res[0] = vaddw_s16(res[0], vget_low_s16(r));
                    res[1] = vaddw_s16(res[1], vget_high_s16(r));
                }
#endif
                if (out_f32_bool == 1) {
                    float32x4_t fac = vdupq_n_f32(factor_s);
                    float32x4_t bias0 = vld1q_f32(b0);
                    float32x4_t bias1 = vld1q_f32(b0 + 4);
                    float32x4_t resf0 = vcvtq_f32_s32(res[0]);
                    float32x4_t resf1 = vcvtq_f32_s32(res[1]);
                    resf0 = vmulq_f32(resf0, fac);
                    resf1 = vmulq_f32(resf1, fac);
                    resf0 = vaddq_f32(resf0, bias0);
                    resf1 = vaddq_f32(resf1, bias1);

                    if (conv_relu_bool) {
                        float32x4_t z = vdupq_n_f32(0);
                        resf0 = vmaxq_f32(resf0, z);
                        resf1 = vmaxq_f32(resf1, z);
                    }
                    vst1q_f32(out_o0hw0, resf0);
                    vst1q_f32(out_o0hw0 + 4, resf1);
                } else {
                    int32x4_t max = vld1q_s32(max_i32);
                    int32x4_t min = vld1q_s32(min_i32);
                    if (conv_relu_bool) {
                        int32x4_t z = vdupq_n_s32(0);
                        res[0] = vmaxq_s32(res[0], z);
                        res[1] = vmaxq_s32(res[1], z);
                    }
                    if (1 == scale_known_bool) {
                        res[0] = vmaxq_s32(min, res[0]);
                        res[1] = vmaxq_s32(min, res[1]);
                        res[0] = vminq_s32(max, res[0]);
                        res[1] = vminq_s32(max, res[1]);
                    } else {
                        max = vmaxq_s32(max, res[0]);
                        min = vminq_s32(min, res[0]);
                        max = vmaxq_s32(max, res[1]);
                        min = vminq_s32(min, res[1]);
                        vst1q_s32(max_i32, max);
                        vst1q_s32(min_i32, min);
                    }
                    vst1q_s32(out_buf, res[0]);
                    vst1q_s32(out_buf + 4, res[1]);
                }

                b0 += 8;
                b0_s += 8;
            }
        }
    }

    EE ret = SUCCESS;
    if (out_f32_bool == 0) {
        I32 factor;
        F32 scale_o;

        if (1 == scale_known_bool) {
            scale_o = (*outputScale / *inputScale) / *filterScale;
            factor = 127 * 16777216 / max_i32[0];
        } else {
            I32 max = max_i32[0];
            I32 min = min_i32[0];
            for (U32 i = 1; i < 4; i++) {
                if (max < max_i32[i]) {
                    max = max_i32[i];
                }
                if (min > min_i32[i]) {
                    min = min_i32[i];
                }
            }

            if (max == 0 && min == 0) {
                return NOT_SUPPORTED;
            }

            if (max > 0 && min < 0) {
                I32 factor_max = 127 * 16777216 / max;
                I32 factor_min = -127 * 16777216 / min;
                factor = (factor_max < factor_min) ? factor_max : factor_min;
                scale_o = (factor_max < factor_min) ? (127.0 / max) : (-127.0 / min);
            } else if (max > 0) {
                factor = 127 * 16777216 / max;
                scale_o = 127.0 / max;
            } else {
                factor = -127 * 16777216 / min;
                scale_o = -127.0 / min;
            }
            *outputScale = (*inputScale) * (*filterScale) * scale_o;
        }

        I32 *out_buf = biasScaled + oc * 8;
        INT8 *out_q = (INT8 *)output;
        array_scale_round_i32(out_buf, out_q, tensorNumElements(outputDesc), scale_o, true);
    }
    return ret;
}

template EE convolution_gemm_v7<INT8>(TensorDesc inputDesc,
    const void *input,
    F32 *inputScale,
    TensorDesc filterDesc,
    const void *filter,
    F32 *filterScale,
    ConvolutionParamSpec p,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    F32 *outputScale,
    ActivationParamSpec activationDesc);

template EE convolution_gemm_v7<F32>(TensorDesc inputDesc,
    const void *input,
    F32 *inputScale,
    TensorDesc filterDesc,
    const void *filter,
    F32 *filterScale,
    ConvolutionParamSpec p,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    F32 *outputScale,
    ActivationParamSpec activationDesc);
