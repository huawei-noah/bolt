// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/int8/depthwise_pointwise_convolution.h"
#include "cpu/arm/arm_functions.h"

EE depthwise_pointwise_convolution_direct(TensorDesc inputDesc,
    INT8 *inArray,
    TensorDesc dwFilterDesc,
    const INT8 *dwFilterArray,
    TensorDesc pwFilterDesc,
    const INT8 *pwFilterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc dwBiasDesc,
    const I32 *dwBiasArray,
    TensorDesc pwBiasDesc,
    const I32 *pwBiasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    I32 *outArray,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec,
    Arch arch)
{
    UNUSED(tmpBytes);
    UNUSED(arch);

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
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingB = convParamSpec.pad_bottom;
    U32 paddingL = convParamSpec.pad_left;
    U32 paddingR = convParamSpec.pad_right;
    U32 dilateH = convParamSpec.dilatedRate_h;
    U32 dilateW = convParamSpec.dilatedRate_w;

    if (dwFilterDesc.df != DF_NCHWC8 || pwFilterDesc.df != DF_NCHWN8C4) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (pwFilterArray == nullptr) {
        return NOT_SUPPORTED;
    }

    oc /= 8;
    ic /= 8;

    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;
    U32 ihiw = ih * iw;
    I32 ohow = oh * ow;
    INT8 *pwArray = (INT8 *)tmp + ic * ih_pad * iw_pad * 8;
    I32 *dw_out = (I32 *)(pwArray + ic * ohow * 8);

    for (U32 n = 0; n < in; n++) {
        // copy input into a input with padding
        INT8 *inArray_pad = (INT8 *)tmp;
        INT8 *inArray_pad_mov = inArray_pad;
        INT8 *inArray_mov = inArray + n * ic * ihiw * 8;
        for (U32 c = 0; c < ic; c++) {
            if (paddingT > 0) {
                UNI_MEMSET(inArray_pad_mov, 0, paddingT * iw_pad * 8 * bytesOf(idt));
                inArray_pad_mov += paddingT * iw_pad * 8;
            }
            for (U32 h = paddingT; h < ih_pad - paddingB; h++) {
                UNI_MEMSET(inArray_pad_mov, 0, paddingL * 8 * bytesOf(idt));
                inArray_pad_mov += paddingL * 8;
                UNI_MEMCPY(inArray_pad_mov, inArray_mov, iw * 8 * bytesOf(idt));
                inArray_pad_mov += iw * 8;
                inArray_mov += iw * 8;
                UNI_MEMSET(inArray_pad_mov, 0, paddingR * 8 * bytesOf(idt));
                inArray_pad_mov += paddingR * 8;
            }
            if (paddingB > 0) {
                UNI_MEMSET(inArray_pad_mov, 0, paddingB * iw_pad * 8 * bytesOf(idt));
                inArray_pad_mov += paddingB * iw_pad * 8;
            }

            const I32 *b = dwBiasArray + c * 8;
            INT8 *in_pad = inArray_pad + c * ih_pad * iw_pad * 8;
            const INT8 *f = dwFilterArray + c * fh * fw * 8;
            for (I32 hw = 0; hw < ohow; hw++) {
                U32 in_h_0 = hw / ow * strideH;
                U32 in_w_0 = hw % ow * strideW;
                __asm__ __volatile__("vld1.s32 {d0-d3}, [%[b]]\n"
                                     :
                                     : [b] "r"(b)
                                     : "memory", "cc", "q0", "q1");

                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        const INT8 *f_0 = f + (fh_idx * fw + fw_idx) * 8;
                        INT8 *in_idx = in_pad + (fh_idx * dilateH * iw_pad + fw_idx * dilateW) * 8;
                        INT8 *in_0 = in_idx + (in_h_0 * iw_pad + in_w_0) * 8;
                        __asm__ __volatile__("vld1.s8 {d4}, [%[f0]]\n"
                                             "vld1.s8 {d6}, [%[in0]]\n"
                                             "vmull.s8 q4, d4, d6\n"
                                             "vaddw.s16 q0, q0, d8\n"
                                             "vaddw.s16 q1, q1, d9\n"
                                             :
                                             : [in0] "r"(in_0), [f0] "r"(f_0)
                                             : "memory", "cc", "q0", "q1", "q2", "q3", "q4");
                    }
                }

                INT8 *pw_in0 = pwArray + (hw * ic + c) * 8;
                switch (depthwiseActivationParamSpec.mode) {
                    case ACTIVATION_NULL: {
                        break;
                    }
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("vmov.s32 q2, #0\n"
                                             "vmax.s32 q0, q0, q2\n"
                                             "vmax.s32 q1, q1, q2\n"
                                             :
                                             :
                                             : "memory", "cc", "q0", "q1", "q2", "q3", "q4");
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }

                if (depthwiseActivationParamSpec.mode != ACTIVATION_RELU6) {
                    __asm__ __volatile__("vst1.s32 {d0-d3}, [%[in0]]\n"
                                         :
                                         : [in0] "r"(pw_in0)
                                         : "memory", "cc", "q0", "q1", "q2", "q3", "q4");
                }
            }
        }

        I32 scale = 1;
        F32 minmax[2];
        CHECK_STATUS(array_minmax_value_arm(DT_I32, dw_out, ohow * ic * 8, 3, minmax));
        if (minmax[1] > 127 && minmax[0] <= -127) {
            I32 factor = 127 / UNI_MAX(UNI_ABS(minmax[0]), UNI_ABS(minmax[1]));
            scale = 1 / factor;
            for (U32 i = 0; i < ohow * ic * 8; i++) {
                pwArray[i] = dw_out[i] * scale;
            }
        }
        I32 scale_v[4] = {scale, scale, scale, scale};

        // pw_conv
        const INT8 *f_base = pwFilterArray;
        for (I32 hw = 0; hw < ohow; hw++) {
            const I32 *b0 = pwBiasArray;
            INT8 *in_pack = pwArray + hw * ic * 8;

            // compute
            for (U32 o = 0; o < oc; o++) {
                INT8 *in_hw = in_pack;
                const INT8 *f_o = f_base + o * 8 * ic * 8;
                I32 *out_o0hw0 = outArray + ((n * oc + o) * ohow + hw) * 8;

                int32x4_t res[2] = {0};
#if 0
                for (U32 c = 0; c < ic * fh * fw * 8; c++, in_hw++, f_o+=8) {
                    int8x8_t a = vdup_n_s8(in_hw[0]);
                    int8x8_t b = vld1_s8(f_o);
                    int16x8_t r = vmull_s8(a, b);
                    res[0] = vaddw_s16(res[0], vget_low_s16(r));
                    res[1] = vaddw_s16(res[1], vget_high_s16(r));
                }
#else
                for (U32 c = 0; c < ic * fh * fw; c++) {
                    int16x8_t r = vdupq_n_s16(0);
                    for (U32 i = 0; i < 8; i++, in_hw++, f_o += 8) {
                        int8x8_t a = vdup_n_s8(in_hw[0]);
                        int8x8_t b = vld1_s8(f_o);
                        r = vmlal_s8(r, a, b);
                    }
                    res[0] = vaddw_s16(res[0], vget_low_s16(r));
                    res[1] = vaddw_s16(res[1], vget_high_s16(r));
                }
#endif
                if (pointwiseActivationParamSpec.mode != ACTIVATION_RELU6 && scale != 1) {  // Scale
                    int32x4_t sc = vld1q_s32(scale_v);
                    res[0] = vmulq_s32(res[0], sc);
                    res[1] = vmulq_s32(res[1], sc);
                }

                int32x4_t bias[2];
                bias[0] = vld1q_s32(b0);
                bias[1] = vld1q_s32(b0 + 4);
                res[0] = vaddq_s32(res[0], bias[0]);
                res[1] = vaddq_s32(res[1], bias[1]);
                switch (pointwiseActivationParamSpec.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        int32x4_t z = vdupq_n_s32(0);
                        res[0] = vmaxq_s32(res[0], z);
                        res[1] = vmaxq_s32(res[1], z);
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }
                vst1q_s32(out_o0hw0, res[0]);
                vst1q_s32(out_o0hw0 + 4, res[1]);
                b0 += 8;
            }
        }
    }
    return SUCCESS;
}
