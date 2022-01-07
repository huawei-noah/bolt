// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/fp32/depthwise_pointwise_convolution.h"

EE depthwise_pointwise_convolution_direct_V7(TensorDesc inputDesc,
    F32 *inArray,
    TensorDesc dwFilterDesc,
    const F32 *dwFilterArray,
    TensorDesc pwFilterDesc,
    const F32 *pwFilterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc dwBiasDesc,
    const F32 *dwBiasArray,
    TensorDesc pwBiasDesc,
    const F32 *pwBiasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F32 *outArray,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec)
{
    UNUSED(dwBiasDesc);
    UNUSED(pwBiasDesc);
    UNUSED(tmpBytes);

    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(dwFilterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingB = convParamSpec.padding_bottom;
    U32 paddingL = convParamSpec.padding_left;
    U32 paddingR = convParamSpec.padding_right;
    U32 dilateH = convParamSpec.dilatedRate_h;
    U32 dilateW = convParamSpec.dilatedRate_w;

    if (dwFilterDesc.df != DF_NCHWC8) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (pwFilterArray != nullptr && pwFilterDesc.df != DF_NHWCN8) {
        CHECK_STATUS(NOT_MATCH);
    }

    oc /= 8;
    ic /= 8;

    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;
    U32 ihiw = ih * iw;
    I32 ohow = oh * ow;
    F32 *pwArray = (F32 *)tmp + ic * ih_pad * iw_pad * 8;

    for (U32 n = 0; n < in; n++) {
        // copy input into a input with padding
        F32 *inArray_pad = (F32 *)tmp;
        F32 *inArray_pad_mov = inArray_pad;
        F32 *inArray_mov = inArray + n * ic * ihiw * 8;
        for (U32 c = 0; c < ic; c++) {
            if (paddingT > 0) {
                memset(inArray_pad_mov, 0, paddingT * iw_pad * 8 * bytesOf(fdt));
                inArray_pad_mov += paddingT * iw_pad * 8;
            }
            for (U32 h = paddingT; h < ih_pad - paddingB; h++) {
                memset(inArray_pad_mov, 0, paddingL * 8 * bytesOf(fdt));
                inArray_pad_mov += paddingL * 8;
                memcpy(inArray_pad_mov, inArray_mov, iw * 8 * bytesOf(fdt));
                inArray_pad_mov += iw * 8;
                inArray_mov += iw * 8;
                memset(inArray_pad_mov, 0, paddingR * 8 * bytesOf(fdt));
                inArray_pad_mov += paddingR * 8;
            }
            if (paddingB > 0) {
                memset(inArray_pad_mov, 0, paddingB * iw_pad * 8 * bytesOf(fdt));
                inArray_pad_mov += paddingB * iw_pad * 8;
            }

            const F32 *b = dwBiasArray + c * 8;
            F32 *in_pad = inArray_pad + c * ih_pad * iw_pad * 8;
            const F32 *f = dwFilterArray + c * fh * fw * 8;
            // ohow / 4
            for (I32 hw = 0; hw < ohow - 3; hw += 4) {
                U32 in_h_0 = hw / ow * strideH;
                U32 in_w_0 = hw % ow * strideW;
                U32 in_h_1 = (hw + 1) / ow * strideH;
                U32 in_w_1 = (hw + 1) % ow * strideW;
                U32 in_h_2 = (hw + 2) / ow * strideH;
                U32 in_w_2 = (hw + 2) % ow * strideW;
                U32 in_h_3 = (hw + 3) / ow * strideH;
                U32 in_w_3 = (hw + 3) % ow * strideW;

                __asm__ __volatile__(
                    "vld1.f32 {d0-d3}, [%[b]]\n"
                    "vmov.f32 q2, q0\n"
                    "vmov.f32 q3, q1\n"
                    "vmov.f32 q4, q0\n"
                    "vmov.f32 q5, q1\n"
                    "vmov.f32 q6, q0\n"
                    "vmov.f32 q7, q1\n"
                    :
                    : [b] "r"(b)
                    : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");

                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        const F32 *f_0 = f + fh_idx * fw * 8 + fw_idx * 8;
                        F32 *in_idx = in_pad + fh_idx * dilateH * iw_pad * 8 + fw_idx * dilateW * 8;
                        F32 *in_0 = in_idx + in_h_0 * iw_pad * 8 + in_w_0 * 8;
                        F32 *in_1 = in_idx + in_h_1 * iw_pad * 8 + in_w_1 * 8;
                        F32 *in_2 = in_idx + in_h_2 * iw_pad * 8 + in_w_2 * 8;
                        F32 *in_3 = in_idx + in_h_3 * iw_pad * 8 + in_w_3 * 8;

                        __asm__ __volatile__("vld1.f32 {d28-d31}, [%[f0]]\n"
                                             "vld1.f32 {d16-d19}, [%[in0]]\n"
                                             "vld1.f32 {d20-d23}, [%[in1]]\n"
                                             "vld1.f32 {d24-d27}, [%[in2]]\n"

                                             "vmla.f32 q0,  q8, q14\n"
                                             "vmla.f32 q1,  q9, q15\n"
                                             "vld1.f32 {d16-d19}, [%[in3]]\n"
                                             "vmla.f32 q2, q10, q14\n"
                                             "vmla.f32 q3, q11, q15\n"
                                             "vmla.f32 q4, q12, q14\n"
                                             "vmla.f32 q5, q13, q15\n"
                                             "vmla.f32 q6,  q8, q14\n"
                                             "vmla.f32 q7,  q9, q15\n"
                                             :
                                             : [in0] "r"(in_0), [in1] "r"(in_1), [in2] "r"(in_2),
                                             [in3] "r"(in_3), [f0] "r"(f_0)
                                             : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5",
                                             "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13",
                                             "q14", "q15");
                    }
                }

                // activation
                switch (depthwiseActivationParamSpec.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("veor q15, q15, q15\n"  // zero
                                             "vmax.f32 q0, q0, q15\n"
                                             "vmax.f32 q1, q1, q15\n"
                                             "vmax.f32 q2, q2, q15\n"
                                             "vmax.f32 q3, q3, q15\n"
                                             "vmax.f32 q4, q4, q15\n"
                                             "vmax.f32 q5, q5, q15\n"
                                             "vmax.f32 q6, q6, q15\n"
                                             "vmax.f32 q7, q7, q15\n"
                                             :
                                             :
                                             : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5",
                                             "q6", "q7", "q15");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("veor q15, q15, q15\n"  // zero
                                             "vmov.f32 q14, #6.0\n"  // six
                                             "vmax.f32 q0, q0, q15\n"
                                             "vmax.f32 q1, q1, q15\n"
                                             "vmax.f32 q2, q2, q15\n"
                                             "vmax.f32 q3, q3, q15\n"
                                             "vmax.f32 q4, q4, q15\n"
                                             "vmax.f32 q5, q5, q15\n"
                                             "vmax.f32 q6, q6, q15\n"
                                             "vmax.f32 q7, q7, q15\n"

                                             "vmin.f32 q0, q0, q14\n"
                                             "vmin.f32 q1, q1, q14\n"
                                             "vmin.f32 q2, q2, q14\n"
                                             "vmin.f32 q3, q3, q14\n"
                                             "vmin.f32 q4, q4, q14\n"
                                             "vmin.f32 q5, q5, q14\n"
                                             "vmin.f32 q6, q6, q14\n"
                                             "vmin.f32 q7, q7, q14\n"
                                             :
                                             :
                                             : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5",
                                             "q6", "q7", "q14", "q15");
                        break;
                    }
                    case ACTIVATION_H_SWISH: {
                        __asm__ __volatile__("vmov.f32 q13, #3.0\n"  // three
                                             "vmov.f32 q14, #6.0\n"  // six
                                             "veor q15, q15, q15\n"  // zero
                                             "vadd.f32  q8, q0, q13\n"
                                             "vadd.f32  q9, q1, q13\n"
                                             "vadd.f32 q10, q2, q13\n"
                                             "vadd.f32 q11, q3, q13\n"
                                             "vmax.f32  q8,  q8, q15\n"
                                             "vmax.f32  q9,  q9, q15\n"
                                             "vmax.f32 q10, q10, q15\n"
                                             "vmax.f32 q11, q11, q15\n"
                                             "vmin.f32  q8,  q8, q14\n"
                                             "vmin.f32  q9,  q9, q14\n"
                                             "vmin.f32 q10, q10, q14\n"
                                             "vmin.f32 q11, q11, q14\n"
                                             "vrecpe.f32 q12, q14\n"
                                             "vrecps.f32 q14, q14, q12\n"
                                             "vmul.f32 q12, q14, q12\n"
                                             "vmul.f32  q8,  q8, q12\n"
                                             "vmul.f32  q9,  q9, q12\n"
                                             "vmul.f32 q10, q10, q12\n"
                                             "vmul.f32 q11, q11, q12\n"
                                             "vmov.f32 q14, #6.0\n"  // six
                                             "vmul.f32 q0, q0,  q8\n"
                                             "vmul.f32 q1, q1,  q9\n"
                                             "vmul.f32 q2, q2, q10\n"
                                             "vmul.f32 q3, q3, q11\n"

                                             "vadd.f32  q8, q4, q13\n"
                                             "vadd.f32  q9, q5, q13\n"
                                             "vadd.f32 q10, q6, q13\n"
                                             "vadd.f32 q11, q7, q13\n"
                                             "vmax.f32  q8,  q8, q15\n"
                                             "vmax.f32  q9,  q9, q15\n"
                                             "vmax.f32 q10, q10, q15\n"
                                             "vmax.f32 q11, q11, q15\n"
                                             "vmin.f32  q8,  q8, q14\n"
                                             "vmin.f32  q9,  q9, q14\n"
                                             "vmin.f32 q10, q10, q14\n"
                                             "vmin.f32 q11, q11, q14\n"
                                             "vrecpe.f32 q12, q14\n"
                                             "vrecps.f32 q14, q14, q12\n"
                                             "vmul.f32 q12, q14, q12\n"
                                             "vmul.f32  q8,  q8, q12\n"
                                             "vmul.f32  q9,  q9, q12\n"
                                             "vmul.f32 q10, q10, q12\n"
                                             "vmul.f32 q11, q11, q12\n"
                                             "vmul.f32 q4, q4,  q8\n"
                                             "vmul.f32 q5, q5,  q9\n"
                                             "vmul.f32 q6, q6, q10\n"
                                             "vmul.f32 q7, q7, q11\n"
                                             :
                                             :
                                             : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5",
                                             "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13",
                                             "q14", "q15");
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }

                if (pwFilterArray != nullptr) {
                    F32 *pw_pack_0 = pwArray + hw * ic * 8 + c * 4 * 8;
                    __asm__ __volatile__(
                        "vzip.32 q0, q4\n"
                        "vzip.32 q2, q6\n"
                        "vzip.32 q1, q5\n"
                        "vzip.32 q3, q7\n"

                        "vzip.32 q0, q2\n"
                        "vzip.32 q4, q6\n"
                        "vzip.32 q1, q3\n"
                        "vzip.32 q5, q7\n"

                        "vst1.f32 {q0}, [%[pw0]]!\n"
                        "vst1.f32 {q2}, [%[pw0]]!\n"
                        "vst1.f32 {q4}, [%[pw0]]!\n"
                        "vst1.f32 {q6}, [%[pw0]]!\n"
                        "vst1.f32 {q1}, [%[pw0]]!\n"
                        "vst1.f32 {q3}, [%[pw0]]!\n"
                        "vst1.f32 {q5}, [%[pw0]]!\n"
                        "vst1.f32 {q7}, [%[pw0]]!\n"
                        : [pw0] "+r"(pw_pack_0)
                        :
                        : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
                } else {
                    F32 *out_ptr = outArray + ((n * ic + c) * ohow + hw) * 8;
                    __asm__ __volatile__(
                        "vstm %[out], {d0-d15}\n"
                        : [out] "+r"(out_ptr)
                        :
                        : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
                }
            }

            // ohow_reminder % 4
            U32 ohow_s = (ohow / 4) * 4;
            for (I32 hw = ohow_s; hw < ohow; hw++) {
                U32 in_h_0 = hw / ow * strideH;
                U32 in_w_0 = hw % ow * strideW;

                __asm__ __volatile__("vld1.f32 {d0-d3}, [%[b]]\n"
                                     :
                                     : [b] "r"(b)
                                     : "memory", "cc", "q0", "q1");

                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        const F32 *f_0 = f + fh_idx * fw * 8 + fw_idx * 8;
                        F32 *in_idx = in_pad + fh_idx * dilateH * iw_pad * 8 + fw_idx * dilateW * 8;
                        F32 *in_0 = in_idx + in_h_0 * iw_pad * 8 + in_w_0 * 8;
                        __asm__ __volatile__(
                            "vld1.f32 {d28-d31}, [%[f0]]\n"
                            "vld1.f32 {d24-d27}, [%[in0]]\n"

                            "vmla.f32 q0, q12, q14\n"
                            "vmla.f32 q1, q13, q15\n"
                            :
                            : [in0] "r"(in_0), [f0] "r"(f_0)
                            : "memory", "cc", "q0", "q1", "q12", "q13", "q14", "q15");
                    }
                }

                // activation
                switch (depthwiseActivationParamSpec.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("veor q15, q15, q15\n"  // zero
                                             "vmax.f32 q0, q0, q15\n"
                                             "vmax.f32 q1, q1, q15\n"
                                             :
                                             :
                                             : "memory", "cc", "q0", "q1", "q15");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("veor q15, q15, q15\n"  // zero
                                             "vmov.f32 q14, #6.0\n"  // six
                                             "vmax.f32 q0, q0, q15\n"
                                             "vmax.f32 q1, q1, q15\n"

                                             "vmin.f32 q0, q0, q14\n"
                                             "vmin.f32 q1, q1, q14\n"
                                             :
                                             :
                                             : "memory", "cc", "q0", "q1", "q14", "q15");
                        break;
                    }
                    case ACTIVATION_H_SWISH: {
                        __asm__ __volatile__(
                            "vmov.f32 q13, #3.0\n"  // three
                            "vmov.f32 q14, #6.0\n"  // six
                            "veor q15, q15, q15\n"  // zero
                            "vadd.f32 q11, q0, q13\n"
                            "vadd.f32 q12, q1, q13\n"

                            "vmax.f32 q11, q11, q15\n"
                            "vmax.f32 q12, q12, q15\n"

                            "vmin.f32 q11, q11, q14\n"
                            "vmin.f32 q12, q12, q14\n"

                            "vrecpe.f32 q13, q14\n"
                            "vrecps.f32 q14, q14, q13\n"
                            "vmul.f32 q14, q14, q13\n"
                            "vmul.f32 q11, q11, q14\n"
                            "vmul.f32 q12, q12, q14\n"

                            "vmul.f32 q0, q0, q11\n"
                            "vmul.f32 q1, q1, q12\n"
                            :
                            :
                            : "memory", "cc", "q0", "q1", "q11", "q12", "q13", "q14", "q15");
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }

                F32 *out_ptr;
                if (pwFilterArray != nullptr) {
                    out_ptr = pwArray + hw * ic * 8 + c * 8;
                } else {
                    out_ptr = outArray + ((n * ic + c) * ohow + hw) * 8;
                }
                __asm__ __volatile__("vst1.f32 {d0-d3}, [%[pw0]]\n"
                                     : [pw0] "+r"(out_ptr)
                                     :
                                     : "memory", "cc", "q0", "q1");
            }
        }

        if (pwFilterArray == nullptr) {
            continue;
        }
        // pw_conv
        // ohow / 4
        for (I32 hw = 0; hw < ohow - 3; hw += 4) {
            const F32 *b0 = pwBiasArray;
            const F32 *b1 = b0 + 4;
            F32 *in_pack = pwArray + hw * ic * 8;
            for (I32 o = 0; o < I32(oc); o++) {
                F32 *in_hw0 = in_pack;
                const F32 *f_o0c0 = pwFilterArray + o * 8 * ic * 8;
                F32 *out_o0hw0 = outArray + n * oc * ohow * 8 + o * ohow * 8 + hw * 8;
                // bias
                const F32 *b_o0 = b0;
                const F32 *b_o1 = b1;
                __asm__ __volatile__("vld1.f32 {d0-d1}, [%[b_0]]\n"
                                     "vld1.f32 {d2-d3}, [%[b_1]]\n"
                                     "vld1.f32  {d12-d13}, [%[in_0]]!\n"
                                     "vld1.f32  {d20-d23}, [%[f_0]]!\n"

                                     "vmov.f32  q2, q0\n"
                                     "vmov.f32  q4, q0\n"
                                     "vmov.f32  q8, q0\n"

                                     "mov  r2, %[ic]\n"

                                     "vmov.f32  q3, q1\n"
                                     "vmov.f32  q5, q1\n"
                                     "vmov.f32  q9, q1\n"

                                     "0:\n"
                                     "vmla.f32  q0, q10, d12[0]\n"
                                     "vmla.f32  q2, q10, d12[1]\n"
                                     "vmla.f32  q4, q10, d13[0]\n"
                                     "vmla.f32  q8, q10, d13[1]\n"

                                     "vld1.f32  {d14-d15}, [%[in_0]]!\n"
                                     "vld1.f32  {d20-d21}, [%[f_0]]!\n"

                                     "vmla.f32  q1, q11, d12[0]\n"
                                     "vmla.f32  q3, q11, d12[1]\n"
                                     "vmla.f32  q5, q11, d13[0]\n"
                                     "vmla.f32  q9, q11, d13[1]\n"

                                     "vld1.f32  {d22-d23}, [%[f_0]]!\n"
                                     "subs r2, r2, #2\n"

                                     "vmla.f32  q0, q10, d14[0]\n"
                                     "vmla.f32  q2, q10, d14[1]\n"
                                     "vmla.f32  q4, q10, d15[0]\n"
                                     "vmla.f32  q8, q10, d15[1]\n"

                                     "vld1.f32  {d12-d13}, [%[in_0]]!\n"
                                     "vld1.f32  {d20-d21}, [%[f_0]]!\n"

                                     "vmla.f32  q1, q11, d14[0]\n"
                                     "vmla.f32  q3, q11, d14[1]\n"
                                     "vmla.f32  q5, q11, d15[0]\n"
                                     "vmla.f32  q9, q11, d15[1]\n"

                                     "vld1.f32  {d22-d23}, [%[f_0]]!\n"
                                     "bne 0b\n"
                                     : [in_0] "+r"(in_hw0), [f_0] "+r"(f_o0c0)
                                     : [ic] "r"((I64)ic * 8), [b_0] "r"(b_o0), [b_1] "r"(b_o1)
                                     : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                                     "q7", "q8", "q9", "q10", "q11", "r2");

                // activation
                switch (pointwiseActivationParamSpec.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("veor q15, q15, q15\n"  // zero
                                             "vmax.f32 q0, q0, q15\n"
                                             "vmax.f32 q1, q1, q15\n"
                                             "vmax.f32 q2, q2, q15\n"
                                             "vmax.f32 q3, q3, q15\n"
                                             "vmax.f32 q4, q4, q15\n"
                                             "vmax.f32 q5, q5, q15\n"
                                             "vmax.f32 q8, q8, q15\n"
                                             "vmax.f32 q9, q9, q15\n"
                                             :
                                             :
                                             : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5",
                                             "q8", "q9", "q15");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("veor q15, q15, q15\n"  // zero
                                             "vmov.f32 q14, #6.0\n"  // six
                                             "vmax.f32 q0, q0, q15\n"
                                             "vmax.f32 q1, q1, q15\n"
                                             "vmax.f32 q2, q2, q15\n"
                                             "vmax.f32 q3, q3, q15\n"
                                             "vmax.f32 q4, q4, q15\n"
                                             "vmax.f32 q5, q5, q15\n"
                                             "vmax.f32 q8, q8, q15\n"
                                             "vmax.f32 q9, q9, q15\n"

                                             "vmin.f32 q0, q0, q14\n"
                                             "vmin.f32 q1, q1, q14\n"
                                             "vmin.f32 q2, q2, q14\n"
                                             "vmin.f32 q3, q3, q14\n"
                                             "vmin.f32 q4, q4, q14\n"
                                             "vmin.f32 q5, q5, q14\n"
                                             "vmin.f32 q8, q8, q14\n"
                                             "vmin.f32 q9, q9, q14\n"
                                             :
                                             :
                                             : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5",
                                             "q8", "q9", "q14", "q15");
                        break;
                    }
                    case ACTIVATION_H_SWISH: {
                        __asm__ __volatile__("vmov.f32  q6, q8\n"
                                             "vmov.f32  q7, q9\n"

                                             "vmov.f32 q13, #3.0\n"  // three
                                             "vmov.f32 q14, #6.0\n"  // six
                                             "veor q15, q15, q15\n"  // zero
                                             "vadd.f32  q8, q0, q13\n"
                                             "vadd.f32  q9, q1, q13\n"
                                             "vadd.f32 q10, q2, q13\n"
                                             "vadd.f32 q11, q3, q13\n"
                                             "vmax.f32  q8,  q8, q15\n"
                                             "vmax.f32  q9,  q9, q15\n"
                                             "vmax.f32 q10, q10, q15\n"
                                             "vmax.f32 q11, q11, q15\n"
                                             "vmin.f32  q8,  q8, q14\n"
                                             "vmin.f32  q9,  q9, q14\n"
                                             "vmin.f32 q10, q10, q14\n"
                                             "vmin.f32 q11, q11, q14\n"
                                             "vrecpe.f32 q12, q14\n"
                                             "vrecps.f32 q14, q14, q12\n"
                                             "vmul.f32 q12, q14, q12\n"
                                             "vmul.f32  q8,  q8, q12\n"
                                             "vmul.f32  q9,  q9, q12\n"
                                             "vmul.f32 q10, q10, q12\n"
                                             "vmul.f32 q11, q11, q12\n"
                                             "vmov.f32 q14, #6.0\n"  // six
                                             "vmul.f32 q0, q0,  q8\n"
                                             "vmul.f32 q1, q1,  q9\n"
                                             "vmul.f32 q2, q2, q10\n"
                                             "vmul.f32 q3, q3, q11\n"

                                             "vadd.f32  q8, q4, q13\n"
                                             "vadd.f32  q9, q5, q13\n"
                                             "vadd.f32 q10, q6, q13\n"
                                             "vadd.f32 q11, q7, q13\n"
                                             "vmax.f32  q8,  q8, q15\n"
                                             "vmax.f32  q9,  q9, q15\n"
                                             "vmax.f32 q10, q10, q15\n"
                                             "vmax.f32 q11, q11, q15\n"
                                             "vmin.f32  q8,  q8, q14\n"
                                             "vmin.f32  q9,  q9, q14\n"
                                             "vmin.f32 q10, q10, q14\n"
                                             "vmin.f32 q11, q11, q14\n"
                                             "vrecpe.f32 q12, q14\n"
                                             "vrecps.f32 q14, q14, q12\n"
                                             "vmul.f32 q12, q14, q12\n"
                                             "vmul.f32  q8,  q8, q12\n"
                                             "vmul.f32  q9,  q9, q12\n"
                                             "vmul.f32 q10, q10, q12\n"
                                             "vmul.f32 q11, q11, q12\n"
                                             "vmul.f32 q4, q4,  q8\n"
                                             "vmul.f32 q5, q5,  q9\n"
                                             "vmul.f32 q6, q6, q10\n"
                                             "vmul.f32 q7, q7, q11\n"

                                             "vmov.f32  q8, q6\n"
                                             "vmov.f32  q9, q7\n"
                                             :
                                             :
                                             : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5",
                                             "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13",
                                             "q14", "q15");
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }

                __asm__ __volatile__(
                    "vst1.f32  {d0-d3}, [%[out_0]]!\n"
                    "vst1.f32  {d4-d7}, [%[out_0]]!\n"
                    "vst1.f32  {d8-d11}, [%[out_0]]!\n"
                    "vst1.f32  {d16-d19}, [%[out_0]]!\n"
                    : [out_0] "+r"(out_o0hw0)
                    :
                    : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q8", "q9");
                b0 += 8;
                b1 += 8;
            }
        }

        // ohow_reminder % 4
        U32 ohow_s = (ohow / 4) * 4;
        for (I32 hw = ohow_s; hw < ohow; hw++) {
            const F32 *b0 = pwBiasArray;
            const F32 *b1 = b0 + 4;
            F32 *in_pack = pwArray + hw * ic * 8;
            for (I32 o = 0; o < I32(oc); o++) {
                F32 *in_hw0 = in_pack;
                const F32 *f_o0c0 = pwFilterArray + o * 8 * ic * 8;
                F32 *out_o0hw0 = outArray + n * oc * ohow * 8 + o * ohow * 8 + hw * 8;
                // bias
                const F32 *b_o0 = b0;
                const F32 *b_o1 = b1;
                __asm__ __volatile__("vld1.f32 {d0-d1}, [%[b_0]]\n"
                                     "vld1.f32 {d2-d3}, [%[b_1]]\n"
                                     "vld1.f32  {d8}, [%[in_0]]!\n"
                                     "vld1.f32  {d4-d7}, [%[f_0]]!\n"
                                     "mov  r2, %[ic]\n"
                                     "0:\n"
                                     "vmla.f32  q0, q2, d8[0]\n"

                                     "vld1.f32  {d4-d5}, [%[f_0]]!\n"

                                     "vmla.f32  q1, q3, d8[0]\n"

                                     "vld1.f32  {d6-d7}, [%[f_0]]!\n"
                                     "subs r2, r2, #2\n"

                                     "vmla.f32  q0, q2, d8[1]\n"

                                     "vld1.f32  {d4-d5}, [%[f_0]]!\n"

                                     "vmla.f32  q1, q3, d8[1]\n"

                                     "vld1.f32  {d8}, [%[in_0]]!\n"
                                     "vld1.f32  {d6-d7}, [%[f_0]]!\n"
                                     "bne 0b\n"
                                     : [in_0] "+r"(in_hw0), [f_0] "+r"(f_o0c0)
                                     : [ic] "r"((I64)ic * 8), [b_0] "r"(b_o0), [b_1] "r"(b_o1)
                                     : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "r2");

                switch (pointwiseActivationParamSpec.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("veor q15, q15, q15\n"  // zero
                                             "vmax.f32 q0, q0, q15\n"
                                             "vmax.f32 q1, q1, q15\n"
                                             :
                                             :
                                             : "memory", "cc", "q0", "q1", "q15");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("veor q15, q15, q15\n"  // zero
                                             "vmov.f32 q14, #6.0\n"  // six
                                             "vmax.f32 q0, q0, q15\n"
                                             "vmax.f32 q1, q1, q15\n"

                                             "vmin.f32 q0, q0, q14\n"
                                             "vmin.f32 q1, q1, q14\n"
                                             :
                                             :
                                             : "memory", "cc", "q0", "q1", "q14", "q15");
                        break;
                    }
                    case ACTIVATION_H_SWISH: {
                        __asm__ __volatile__(
                            "vmov.f32 q13, #3.0\n"  // three
                            "vmov.f32 q14, #6.0\n"  // six
                            "veor q15, q15, q15\n"  // zero
                            "vadd.f32 q11, q0, q13\n"
                            "vadd.f32 q12, q1, q13\n"

                            "vmax.f32 q11, q11, q15\n"
                            "vmax.f32 q12, q12, q15\n"

                            "vmin.f32 q11, q11, q14\n"
                            "vmin.f32 q12, q12, q14\n"

                            "vrecpe.f32 q13, q14\n"
                            "vrecps.f32 q14, q14, q13\n"
                            "vmul.f32 q14, q14, q13\n"
                            "vmul.f32 q11, q11, q14\n"
                            "vmul.f32 q12, q12, q14\n"

                            "vmul.f32 q0, q0, q11\n"
                            "vmul.f32 q1, q1, q12\n"
                            :
                            :
                            : "memory", "cc", "q0", "q1", "q11", "q12", "q13", "q14", "q15");
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }

                __asm__ __volatile__("vst1.f32  {d0-d3}, [%[out_0]]\n"
                                     : [out_0] "+r"(out_o0hw0)
                                     :
                                     : "memory", "cc", "q0", "q1");
                b0 += 8;
                b1 += 8;
            }
        }
    }
    return SUCCESS;
}
