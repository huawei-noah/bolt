// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_INT8
#include <string.h>
#include "cpu/arm/int8/convolution_gemm.h"

template <typename OT>
EE convolution_gemm_A55(TensorDesc inputDesc,
    const void *input,
    F16 *inputScale,
    TensorDesc filterDesc,
    const void *filter,
    F16 *filterScale,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    F16 *outputScale,
    ActivationParamSpec activationDesc)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    // still im2col + gemm with a smaller buffer
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingB = convParamSpec.padding_bottom;
    U32 paddingL = convParamSpec.padding_left;
    U32 paddingR = convParamSpec.padding_right;
    U32 dilateH = convParamSpec.dilatedRate_h;
    U32 dilateW = convParamSpec.dilatedRate_w;

    if (fdf != DF_NCHWN8C4) {
        return NOT_MATCH;
    }

    I64 conv_relu_bool = (activationDesc.mode == ACTIVATION_RELU) ? 1 : 0;
    I64 out_f16_bool = (odt == DT_F16) ? 1 : 0;
    I64 scale_known_bool = 0;
    if (*outputScale > 0 || ACTIVATION_RELU6 == activationDesc.mode) {
        scale_known_bool = 1;
    }

    INT8 *inArray = (INT8 *)input;  // It will be updated if there is quantization
    INT8 *filterArray = (INT8 *)filter;
    F16 *outArray = (F16 *)output;
    F16 *biasArray = (F16 *)bias;
    INT8 *in_pad = (INT8 *)tmp;

    // both input and output are stored with C8
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
        F16 scale_i = 1.0;

        // quantize input if necessary
        if (idt == DT_F16) {
            // start = get_current_time_int8();
            F16 *in = ((F16 *)input) + n * ic * ih * iw * 8;
            inArray = in_pad + ic * ihiw * 8 +
                12 * fh * fw * ic * 8;  // After the space for padding and packing

            U32 numData = ic * ih * iw * 8;
            if (*inputScale > 0) {
                scale_i = *inputScale;
            } else {
                float16x8_t temp_v = vld1q_f16(in);
                float16x8_t max_v = temp_v;
                float16x8_t min_v = temp_v;

                for (U32 i = 8; i < numData; i += 8) {
                    temp_v = vld1q_f16(in + i);
                    max_v = vmaxq_f16(max_v, temp_v);
                    min_v = vminq_f16(min_v, temp_v);
                }

                F16 max = vmaxvq_f16(max_v);
                F16 min = vminvq_f16(min_v);

                if (max == 0 && min == 0) {
                    return NOT_SUPPORTED;
                }
                if (max > 0 && min < 0) {
                    F16 scale_max = 127.0 / max;
                    F16 scale_min = -127.0 / min;
                    scale_i = (scale_max < scale_min) ? scale_max : scale_min;
                } else if (max < 0) {
                    scale_i = -127.0 / min;
                } else {  // min > 0
                    scale_i = 127.0 / max;
                }
            }
            for (U32 i = 0; i < numData; i++) {
                F32 temp = in[i] * scale_i;
                inArray[i] = round_towards_zero(temp, (*inputScale) != scale_i);
            }
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
            if (idt == DT_F16) {
                biasScaled += ic * ih * iw * 8 / bytesOf(DT_I32);  // After the quantized input
            }
            F32 scale = (*inputScale) * (*filterScale);
            for (U32 i = 0; i < oc * 8; i++) {
                biasScaled[i] = round(scale * biasArray[i]);
            }
        }

        F32 factor_s = 1.0 / ((F32)scale_i) / ((F32)(*filterScale));
        F32 factor_v[4];
        for (U32 i = 0; i < 4; i++) {
            factor_v[i] = factor_s;
        }

        INT8 *inArray_pad;
        if (paddingT == 0 && paddingB == 0 && paddingL == 0 && paddingR == 0) {
            inArray_pad = inArray + n * ic * ih * iw * 8;  // use this batch directly
        } else {
            // copy input into an input with padding
            inArray_pad = (INT8 *)tmp;
            INT8 *inArray_pad_mov = inArray_pad;
            INT8 *inArray_mov = inArray + n * ic * ih * iw * 8;
            for (U32 c = 0; c < ic; c++) {                                    // for each 8 channels
                for (U32 h = 0; h < paddingT; h++) {                          // Upper rows of 0
                    memset(inArray_pad_mov, 0, iw_pad * 8 * bytesOf(DT_I8));  // 8 comes from C8
                    inArray_pad_mov += iw_pad * 8;
                }
                for (U32 h = paddingT; h < ih_pad - paddingB; h++) {  // for each middle-section rows
                    memset(inArray_pad_mov, 0, paddingL * 8 * bytesOf(DT_I8));  // padding on the left
                    inArray_pad_mov += paddingL * 8;                            // 8 comes from C8
                    memcpy(inArray_pad_mov, inArray_mov, iw * 8 * bytesOf(DT_I8));  // Copy input row
                    inArray_pad_mov += iw * 8;
                    inArray_mov += iw * 8;
                    memset(
                        inArray_pad_mov, 0, paddingR * 8 * bytesOf(DT_I8));  // padding on the right
                    inArray_pad_mov += paddingR * 8;
                }
                for (U32 h = ih_pad - paddingB; h < ih_pad; h++) {  // Bottom rows of 0
                    memset(inArray_pad_mov, 0, iw_pad * 8 * bytesOf(DT_I8));
                    inArray_pad_mov += iw_pad * 8;
                }
            }
        }
        // ohow / 12 (12x8)
        for (I32 hw = 0; hw < ohow - 11; hw += 12) {  // Remainder will be handled later
            F16 *b0 = biasArray;
            I32 *b0_s = biasScaled;
            INT8 *in_pack = ((INT8 *)tmp) + ic * ih_pad * iw_pad * 8;  // After the padded input
            // pack input
            // NCHWc8 => NHWChw12c4 + im2col
            U32 in_h[12];
            U32 in_w[12];

            for (U32 i = 0; i < 12; i++) {
                in_h[i] = ((hw + i) / ow) * strideH;
                in_w[i] = ((hw + i) % ow) * strideW;
            }
            for (U32 c = 0; c < ic; c++) {  // for each 8 channels
                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        INT8 *in_hw12c8 = inArray_pad + c * ihiw * 8 +
                            fh_idx * dilateH * iw_pad * 8 + fw_idx * dilateW * 8;

                        INT8 *in_0 = in_hw12c8 + in_h[0] * iw_pad * 8 + in_w[0] * 8;
                        INT8 *in_1 = in_hw12c8 + in_h[1] * iw_pad * 8 + in_w[1] * 8;
                        INT8 *in_2 = in_hw12c8 + in_h[2] * iw_pad * 8 + in_w[2] * 8;
                        INT8 *in_3 = in_hw12c8 + in_h[3] * iw_pad * 8 + in_w[3] * 8;
                        INT8 *in_4 = in_hw12c8 + in_h[4] * iw_pad * 8 + in_w[4] * 8;
                        INT8 *in_5 = in_hw12c8 + in_h[5] * iw_pad * 8 + in_w[5] * 8;
                        INT8 *in_6 = in_hw12c8 + in_h[6] * iw_pad * 8 + in_w[6] * 8;
                        INT8 *in_7 = in_hw12c8 + in_h[7] * iw_pad * 8 + in_w[7] * 8;
                        INT8 *in_8 = in_hw12c8 + in_h[8] * iw_pad * 8 + in_w[8] * 8;
                        INT8 *in_9 = in_hw12c8 + in_h[9] * iw_pad * 8 + in_w[9] * 8;
                        INT8 *in_10 = in_hw12c8 + in_h[10] * iw_pad * 8 + in_w[10] * 8;
                        INT8 *in_11 = in_hw12c8 + in_h[11] * iw_pad * 8 + in_w[11] * 8;

                        // in_pack (tmp) is reused for each tile
                        // NHWChw12c4
                        INT8 *in_pack_0 =
                            in_pack + c * fh * fw * 12 * 8 + fh_idx * fw * 12 * 4 + fw_idx * 12 * 4;
                        INT8 *in_pack_1 = in_pack_0 + fh * fw * 12 * 4;

                        __asm__ __volatile__(
                            "ldr d0, [%[in_0]]\n"
                            "ldr x2, [%[in_2]]\n"

                            "ldr d1, [%[in_1]]\n"
                            "ldr x3, [%[in_3]]\n"

                            "ldr d4, [%[in_4]]\n"
                            "ldr x6, [%[in_6]]\n"

                            "ldr d5, [%[in_5]]\n"
                            "ins v0.d[1], x2\n"

                            "ldr x7, [%[in_7]]\n"
                            "ins v1.d[1], x3\n"

                            "ldr d8, [%[in_8]]\n"
                            "ins v4.d[1], x6\n"

                            "trn1 v20.4s, v0.4s, v1.4s\n"
                            "ins v5.d[1], x7\n"

                            "trn2 v21.4s, v0.4s, v1.4s\n"
                            "ldr x10, [%[in_10]]\n"

                            "ldr d9, [%[in_9]]\n"
                            "trn1 v24.4s, v4.4s, v5.4s\n"

                            "trn2 v25.4s, v4.4s, v5.4s\n"
                            "ldr x11, [%[in_11]]\n"

                            "str   q20, [%[pack_0]]\n"
                            "ins v8.d[1], x10\n"

                            "str   q24, [%[pack_0], #16]\n"
                            "ins v9.d[1], x11\n"

                            "trn1 v28.4s, v8.4s, v9.4s\n"
                            "str   q21, [%[pack_1]]\n"

                            "trn2 v29.4s, v8.4s, v9.4s\n"
                            "str   q25, [%[pack_1], #16]\n"

                            "str   q28, [%[pack_0], #32]\n"
                            "str   q29, [%[pack_1], #32]\n"
                            :
                            : [pack_0] "r"(in_pack_0), [pack_1] "r"(in_pack_1), [in_0] "r"(in_0),
                            [in_1] "r"(in_1), [in_2] "r"(in_2), [in_3] "r"(in_3), [in_4] "r"(in_4),
                            [in_5] "r"(in_5), [in_6] "r"(in_6), [in_7] "r"(in_7), [in_8] "r"(in_8),
                            [in_9] "r"(in_9), [in_10] "r"(in_10), [in_11] "r"(in_11)
                            : "memory", "cc", "v0", "v1", "v4", "v5", "v8", "v9", "v20", "v21",
                            "v24", "v25", "v28", "v29", "x2", "x3", "x6", "x7", "x10", "x11");
                    }
                }
            }

            // compute
            for (U32 o = 0; o < oc; o++) {  // 8 output channels at a time
                INT8 *in_hw0 = in_pack;
                INT8 *f_o0c0 = filterArray + o * 8 * fh * fw * ic * 8;
                I32 *out_buf = biasScaled + oc * 8 + n * oc * ohow * 8 + o * ohow * 8 + hw * 8;
                F16 *out_o0hw0 = outArray + n * oc * ohow * 8 + o * ohow * 8 + hw * 8;
                // bias
                F16 *b_0 = b0;
                I32 *b_0_s = b0_s;
                __asm__ __volatile__(
                    "cbz %[out_f16], 8f\n"
                    "eor v5.16b, v5.16b, v5.16b\n"
                    "ldr  d1, [%[in_0]]\n"  // in_0
                    "eor v6.16b, v6.16b, v6.16b\n"
                    "ldr  x1, [%[in_0], #8]\n"
                    "eor v7.16b, v7.16b, v7.16b\n"
                    "ins  v1.d[1], x1\n"
                    "eor v8.16b, v8.16b, v8.16b\n"
                    "ldr d0, [%[f_0]]\n"  // f_0
                    "eor v9.16b, v9.16b, v9.16b\n"
                    "ldr  x2, [%[f_0], #8]\n"
                    "eor v10.16b, v10.16b, v10.16b\n"
                    "ins v0.d[1], x2\n"
                    "eor v11.16b, v11.16b, v11.16b\n"
                    "ldr  d3, [%[in_0], #16]\n"  // in_1
                    "eor v12.16b, v12.16b, v12.16b\n"
                    "ldr  x3, [%[in_0], #24]\n"
                    "eor v13.16b, v13.16b, v13.16b\n"
                    "ins v3.d[1], x3\n"
                    "eor v14.16b, v14.16b, v14.16b\n"
                    "eor v15.16b, v15.16b, v15.16b\n"
                    "eor v16.16b, v16.16b, v16.16b\n"

                    "eor v17.16b, v17.16b, v17.16b\n"
                    "eor v18.16b, v18.16b, v18.16b\n"
                    "eor v19.16b, v19.16b, v19.16b\n"
                    "eor v20.16b, v20.16b, v20.16b\n"
                    "eor v21.16b, v21.16b, v21.16b\n"
                    "eor v22.16b, v22.16b, v22.16b\n"
                    "eor v23.16b, v23.16b, v23.16b\n"
                    "eor v24.16b, v24.16b, v24.16b\n"
                    "eor v25.16b, v25.16b, v25.16b\n"
                    "eor v26.16b, v26.16b, v26.16b\n"
                    "eor v27.16b, v27.16b, v27.16b\n"
                    "eor v28.16b, v28.16b, v28.16b\n"
                    "b 7f\n"

                    "8:\n"
                    "ldp q29, q30, [%[b_0_s]]\n"
                    "mov v5.16b, v29.16b\n"
                    "ldr  d1, [%[in_0]]\n"  // in_0
                    "mov v7.16b, v29.16b\n"
                    "ldr  x1, [%[in_0], #8]\n"
                    "mov v9.16b, v29.16b\n"
                    "ins  v1.d[1], x1\n"
                    "mov v11.16b, v29.16b\n"
                    "ldr d0, [%[f_0]]\n"  // f_0
                    "mov v13.16b, v29.16b\n"
                    "ldr  x2, [%[f_0], #8]\n"
                    "mov v15.16b, v29.16b\n"
                    "ins v0.d[1], x2\n"
                    "mov v17.16b, v29.16b\n"
                    "ldr  d3, [%[in_0], #16]\n"  // in_1
                    "mov v19.16b, v29.16b\n"
                    "ldr  x3, [%[in_0], #24]\n"
                    "mov v21.16b, v29.16b\n"
                    "ins v3.d[1], x3\n"
                    "mov v23.16b, v29.16b\n"
                    "mov v25.16b, v29.16b\n"
                    "mov v27.16b, v29.16b\n"

                    "mov v6.16b, v30.16b\n"
                    "mov v8.16b, v30.16b\n"
                    "mov v10.16b, v30.16b\n"
                    "mov v12.16b, v30.16b\n"
                    "mov v14.16b, v30.16b\n"
                    "mov v16.16b, v30.16b\n"
                    "mov v18.16b, v30.16b\n"
                    "mov v20.16b, v30.16b\n"
                    "mov v22.16b, v30.16b\n"
                    "mov v24.16b, v30.16b\n"
                    "mov v26.16b, v30.16b\n"
                    "mov v28.16b, v30.16b\n"

                    "7:\n"
                    // give in address to x3
                    "mov x3, %[in_0]\n"

                    // give f address to x0
                    "mov x0, %[f_0]\n"

                    "mov  x2, %[ic]\n"  // ic_blk
                    "0:\n"
                    "sdot v5.4s, v0.16b, v1.4b[0]\n"
                    "ldr d2, [x3, 32]\n"
                    "ldr x16, [x3, 40]\n"
                    "sdot v7.4s, v0.16b, v1.4b[1]\n"
                    "ldr d29, [x0, 16]\n"
                    "ldr x17, [x0, 24]\n"
                    "sdot v9.4s, v0.16b, v1.4b[2]\n"
                    "ins v2.d[1], x16\n"
                    "ldr d30, [x3, 48]!\n"
                    "sdot v11.4s, v0.16b, v1.4b[3]\n"
                    "ins v29.d[1], x17\n"

                    "sdot v13.4s, v0.16b, v3.4b[0]\n"
                    "ldr x16, [x3, 8]\n"
                    "subs x2, x2, #4\n"
                    "sdot v15.4s, v0.16b, v3.4b[1]\n"
                    "sdot v17.4s, v0.16b, v3.4b[2]\n"
                    "ins v30.d[1], x16\n"
                    "sdot v19.4s, v0.16b, v3.4b[3]\n"

                    "sdot v21.4s, v0.16b, v2.4b[0]\n"
                    "sdot v23.4s, v0.16b, v2.4b[1]\n"
                    "sdot v25.4s, v0.16b, v2.4b[2]\n"
                    "sdot v27.4s, v0.16b, v2.4b[3]\n"

                    "sdot v14.4s, v29.16b, v3.4b[0]\n"
                    "sdot v16.4s, v29.16b, v3.4b[1]\n"
                    "ldr d0, [x0, 32]!\n"
                    "ldr x17, [x0, 8]\n"
                    "sdot v18.4s, v29.16b, v3.4b[2]\n"
                    "sdot v20.4s, v29.16b, v3.4b[3]\n"

                    "sdot v6.4s, v29.16b, v1.4b[0]\n"
                    "sdot v8.4s, v29.16b, v1.4b[1]\n"
                    "ldr d3, [x3, 16]\n"
                    "ldr x16, [x3, 24]\n"
                    "sdot v10.4s, v29.16b, v1.4b[2]\n"
                    "sdot v12.4s, v29.16b, v1.4b[3]\n"

                    "ins v0.d[1], x17\n"
                    "ins v3.d[1], x16\n"

                    "sdot v22.4s, v29.16b, v2.4b[0]\n"
                    "mov v1.16b, v30.16b\n"
                    "sdot v24.4s, v29.16b, v2.4b[1]\n"
                    "sdot v26.4s, v29.16b, v2.4b[2]\n"
                    "sdot v28.4s, v29.16b, v2.4b[3]\n"

                    "bne 0b\n"
                    "cbz %[out_f16], 6f\n"

                    "scvtf v5.4s, v5.4s\n"
                    "scvtf v6.4s, v6.4s\n"
                    "ldr d1, [%[factor]]\n"
                    "ldr x1, [%[factor], #8]\n"
                    "scvtf v7.4s, v7.4s\n"
                    "scvtf v8.4s, v8.4s\n"
                    "ins v1.d[1], x1\n"
                    "scvtf v9.4s, v9.4s\n"
                    "ldr d0, [%[b_0]]\n"
                    "ldr x0, [%[b_0], #8]\n"
                    "scvtf v10.4s, v10.4s\n"
                    "scvtf v11.4s, v11.4s\n"
                    "ins v0.d[1], x0\n"
                    "scvtf v12.4s, v12.4s\n"
                    "scvtf v13.4s, v13.4s\n"
                    "scvtf v14.4s, v14.4s\n"
                    "scvtf v15.4s, v15.4s\n"
                    "scvtf v16.4s, v16.4s\n"
                    "scvtf v17.4s, v17.4s\n"
                    "scvtf v18.4s, v18.4s\n"
                    "scvtf v19.4s, v19.4s\n"
                    "scvtf v20.4s, v20.4s\n"
                    "scvtf v21.4s, v21.4s\n"
                    "scvtf v22.4s, v22.4s\n"
                    "scvtf v23.4s, v23.4s\n"
                    "scvtf v24.4s, v24.4s\n"
                    "scvtf v25.4s, v25.4s\n"
                    "scvtf v26.4s, v26.4s\n"
                    "scvtf v27.4s, v27.4s\n"
                    "scvtf v28.4s, v28.4s\n"

                    "fmul v5.4s, v1.4s, v5.4s\n"
                    "fmul v6.4s, v1.4s, v6.4s\n"
                    "fmul v7.4s, v1.4s, v7.4s\n"
                    "fmul v8.4s, v1.4s, v8.4s\n"
                    "fmul v9.4s, v1.4s, v9.4s\n"
                    "fmul v10.4s, v1.4s, v10.4s\n"
                    "fmul v11.4s, v1.4s, v11.4s\n"
                    "fmul v12.4s, v1.4s, v12.4s\n"
                    "fmul v13.4s, v1.4s, v13.4s\n"
                    "fmul v14.4s, v1.4s, v14.4s\n"
                    "fmul v15.4s, v1.4s, v15.4s\n"
                    "fmul v16.4s, v1.4s, v16.4s\n"
                    "fmul v17.4s, v1.4s, v17.4s\n"
                    "fmul v18.4s, v1.4s, v18.4s\n"
                    "fmul v19.4s, v1.4s, v19.4s\n"
                    "fmul v20.4s, v1.4s, v20.4s\n"
                    "fmul v21.4s, v1.4s, v21.4s\n"
                    "fmul v22.4s, v1.4s, v22.4s\n"
                    "fmul v23.4s, v1.4s, v23.4s\n"
                    "fmul v24.4s, v1.4s, v24.4s\n"
                    "fmul v25.4s, v1.4s, v25.4s\n"
                    "fmul v26.4s, v1.4s, v26.4s\n"
                    "fmul v27.4s, v1.4s, v27.4s\n"
                    "fmul v28.4s, v1.4s, v28.4s\n"

                    "fcvtn v5.4h, v5.4s\n"
                    "fcvtn v7.4h, v7.4s\n"
                    "fcvtn v9.4h, v9.4s\n"
                    "fcvtn v11.4h, v11.4s\n"
                    "fcvtn v13.4h, v13.4s\n"
                    "fcvtn v15.4h, v15.4s\n"
                    "fcvtn v17.4h, v17.4s\n"
                    "fcvtn v19.4h, v19.4s\n"
                    "fcvtn v21.4h, v21.4s\n"
                    "fcvtn v23.4h, v23.4s\n"
                    "fcvtn v25.4h, v25.4s\n"
                    "fcvtn v27.4h, v27.4s\n"

                    "fcvtn2 v5.8h, v6.4s\n"
                    "fcvtn2 v7.8h, v8.4s\n"
                    "fcvtn2 v9.8h, v10.4s\n"
                    "fcvtn2 v11.8h, v12.4s\n"
                    "fcvtn2 v13.8h, v14.4s\n"
                    "fcvtn2 v15.8h, v16.4s\n"
                    "fcvtn2 v17.8h, v18.4s\n"
                    "fcvtn2 v19.8h, v20.4s\n"
                    "fcvtn2 v21.8h, v22.4s\n"
                    "fcvtn2 v23.8h, v24.4s\n"
                    "fcvtn2 v25.8h, v26.4s\n"
                    "fcvtn2 v27.8h, v28.4s\n"

                    "fadd v5.8h, v0.8h, v5.8h\n"
                    "fadd v7.8h, v0.8h, v7.8h\n"
                    "fadd v9.8h, v0.8h, v9.8h\n"
                    "fadd v11.8h, v0.8h, v11.8h\n"
                    "fadd v13.8h, v0.8h, v13.8h\n"
                    "fadd v15.8h, v0.8h, v15.8h\n"
                    "fadd v17.8h, v0.8h, v17.8h\n"
                    "fadd v19.8h, v0.8h, v19.8h\n"
                    "fadd v21.8h, v0.8h, v21.8h\n"
                    "fadd v23.8h, v0.8h, v23.8h\n"
                    "fadd v25.8h, v0.8h, v25.8h\n"
                    "fadd v27.8h, v0.8h, v27.8h\n"

                    "cbz %[conv_relu], 1f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"  // zero
                    "fmax  v5.8h,  v5.8h, v1.8h\n"
                    "fmax  v7.8h,  v7.8h, v1.8h\n"
                    "fmax  v9.8h,  v9.8h, v1.8h\n"
                    "fmax  v11.8h,  v11.8h, v1.8h\n"
                    "fmax  v13.8h,  v13.8h, v1.8h\n"
                    "fmax  v15.8h,  v15.8h, v1.8h\n"
                    "fmax  v17.8h,  v17.8h, v1.8h\n"
                    "fmax  v19.8h,  v19.8h, v1.8h\n"
                    "fmax  v21.8h,  v21.8h, v1.8h\n"
                    "fmax  v23.8h,  v23.8h, v1.8h\n"
                    "fmax  v25.8h,  v25.8h, v1.8h\n"
                    "fmax  v27.8h,  v27.8h, v1.8h\n"

                    "1:\n"
                    "str   q5, [%[out_0]]\n"
                    "str   q7, [%[out_0], #16]\n"
                    "str   q9, [%[out_0], #32]\n"
                    "str   q11, [%[out_0], #48]\n"
                    "str   q13, [%[out_0], #64]\n"
                    "str   q15, [%[out_0], #80]\n"
                    "str   q17, [%[out_0], #96]\n"
                    "str   q19, [%[out_0], #112]\n"
                    "str   q21, [%[out_0], #128]\n"
                    "str   q23, [%[out_0], #144]\n"
                    "str   q25, [%[out_0], #160]\n"
                    "str   q27, [%[out_0], #176]\n"
                    "b 5f\n"

                    "6:\n"
                    "ldr q0, [%[min]]\n"
                    "ldr q30, [%[max]]\n"
                    "cbz %[conv_relu], 2f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"  // zero
                    "smax v5.4s, v5.4s, v1.4s\n"
                    "smax v6.4s, v6.4s, v1.4s\n"
                    "smax v7.4s, v7.4s, v1.4s\n"
                    "smax v8.4s, v8.4s, v1.4s\n"
                    "smax v9.4s, v9.4s, v1.4s\n"
                    "smax v10.4s, v10.4s, v1.4s\n"
                    "smax v11.4s, v11.4s, v1.4s\n"
                    "smax v12.4s, v12.4s, v1.4s\n"
                    "smax v13.4s, v13.4s, v1.4s\n"
                    "smax v14.4s, v14.4s, v1.4s\n"
                    "smax v15.4s, v15.4s, v1.4s\n"
                    "smax v16.4s, v16.4s, v1.4s\n"
                    "smax v17.4s, v17.4s, v1.4s\n"
                    "smax v18.4s, v18.4s, v1.4s\n"
                    "smax v19.4s, v19.4s, v1.4s\n"
                    "smax v20.4s, v20.4s, v1.4s\n"
                    "smax v21.4s, v21.4s, v1.4s\n"
                    "smax v22.4s, v22.4s, v1.4s\n"
                    "smax v23.4s, v23.4s, v1.4s\n"
                    "smax v24.4s, v24.4s, v1.4s\n"
                    "smax v25.4s, v25.4s, v1.4s\n"
                    "smax v26.4s, v26.4s, v1.4s\n"
                    "smax v27.4s, v27.4s, v1.4s\n"
                    "smax v28.4s, v28.4s, v1.4s\n"

                    "2:\n"
                    "cbz %[scale_known], 7f\n"
                    "smax v5.4s, v5.4s, v0.4s\n"
                    "smin v5.4s, v5.4s, v30.4s\n"
                    "smax v6.4s, v6.4s, v0.4s\n"
                    "smin v6.4s, v6.4s, v30.4s\n"
                    "smax v7.4s, v7.4s, v0.4s\n"
                    "smin v7.4s, v7.4s, v30.4s\n"
                    "smax v8.4s, v8.4s, v0.4s\n"
                    "smin v8.4s, v8.4s, v30.4s\n"
                    "smax v9.4s, v9.4s, v0.4s\n"
                    "smin v9.4s, v9.4s, v30.4s\n"
                    "smax v10.4s, v10.4s, v0.4s\n"
                    "smin v10.4s, v10.4s, v30.4s\n"
                    "smax v11.4s, v11.4s, v0.4s\n"
                    "smin v11.4s, v11.4s, v30.4s\n"
                    "smax v12.4s, v12.4s, v0.4s\n"
                    "smin v12.4s, v12.4s, v30.4s\n"
                    "smax v13.4s, v13.4s, v0.4s\n"
                    "smin v13.4s, v13.4s, v30.4s\n"
                    "smax v14.4s, v14.4s, v0.4s\n"
                    "smin v14.4s, v14.4s, v30.4s\n"
                    "smax v15.4s, v15.4s, v0.4s\n"
                    "smin v15.4s, v15.4s, v30.4s\n"
                    "smax v16.4s, v16.4s, v0.4s\n"
                    "smin v16.4s, v16.4s, v30.4s\n"
                    "smax v17.4s, v17.4s, v0.4s\n"
                    "smin v17.4s, v17.4s, v30.4s\n"
                    "smax v18.4s, v18.4s, v0.4s\n"
                    "smin v18.4s, v18.4s, v30.4s\n"
                    "smax v19.4s, v19.4s, v0.4s\n"
                    "smin v19.4s, v19.4s, v30.4s\n"
                    "smax v20.4s, v20.4s, v0.4s\n"
                    "smin v20.4s, v20.4s, v30.4s\n"
                    "smax v21.4s, v21.4s, v0.4s\n"
                    "smin v21.4s, v21.4s, v30.4s\n"
                    "smax v22.4s, v22.4s, v0.4s\n"
                    "smin v22.4s, v22.4s, v30.4s\n"
                    "smax v23.4s, v23.4s, v0.4s\n"
                    "smin v23.4s, v23.4s, v30.4s\n"
                    "smax v24.4s, v24.4s, v0.4s\n"
                    "smin v24.4s, v24.4s, v30.4s\n"
                    "smax v25.4s, v25.4s, v0.4s\n"
                    "smin v25.4s, v25.4s, v30.4s\n"
                    "smax v26.4s, v26.4s, v0.4s\n"
                    "smin v26.4s, v26.4s, v30.4s\n"
                    "smax v27.4s, v27.4s, v0.4s\n"
                    "smin v27.4s, v27.4s, v30.4s\n"
                    "smax v28.4s, v28.4s, v0.4s\n"
                    "smin v28.4s, v28.4s, v30.4s\n"

                    "str   q5, [%[out_buf]]\n"
                    "str   q6, [%[out_buf], 16]\n"
                    "str   q7, [%[out_buf], 32]\n"
                    "str   q8, [%[out_buf], 48]\n"
                    "str   q9, [%[out_buf], 64]\n"
                    "str   q10, [%[out_buf], 80]\n"
                    "str   q11, [%[out_buf], 96]\n"
                    "str   q12, [%[out_buf], 112]\n"
                    "str   q13, [%[out_buf], 128]\n"
                    "str   q14, [%[out_buf], 144]\n"
                    "str   q15, [%[out_buf], 160]\n"
                    "str   q16, [%[out_buf], 176]\n"
                    "str   q17, [%[out_buf], 192]\n"
                    "str   q18, [%[out_buf], 208]\n"
                    "str   q19, [%[out_buf], 224]\n"
                    "str   q20, [%[out_buf], 240]\n"
                    "str   q21, [%[out_buf], 256]\n"
                    "str   q22, [%[out_buf], 272]\n"
                    "str   q23, [%[out_buf], 288]\n"
                    "str   q24, [%[out_buf], 304]\n"
                    "str   q25, [%[out_buf], 320]\n"
                    "str   q26, [%[out_buf], 336]\n"
                    "str   q27, [%[out_buf], 352]\n"
                    "str   q28, [%[out_buf], 368]\n"
                    "b 5f\n"

                    "7:\n"
                    "smax v30.4s, v5.4s, v30.4s\n"
                    "smin v0.4s, v5.4s, v0.4s\n"
                    "str   q5, [%[out_buf]]\n"
                    "smax v30.4s, v6.4s, v30.4s\n"
                    "smin v0.4s, v6.4s, v0.4s\n"
                    "str   q6, [%[out_buf], 16]\n"
                    "smax v30.4s, v7.4s, v30.4s\n"
                    "smin v0.4s, v7.4s, v0.4s\n"
                    "str   q7, [%[out_buf], 32]\n"
                    "smax v30.4s, v8.4s, v30.4s\n"
                    "smin v0.4s, v8.4s, v0.4s\n"
                    "str   q8, [%[out_buf], 48]\n"
                    "smax v30.4s, v9.4s, v30.4s\n"
                    "smin v0.4s, v9.4s, v0.4s\n"
                    "str   q9, [%[out_buf], 64]\n"
                    "smax v30.4s, v10.4s, v30.4s\n"
                    "smin v0.4s, v10.4s, v0.4s\n"
                    "str   q10, [%[out_buf], 80]\n"
                    "smax v30.4s, v11.4s, v30.4s\n"
                    "smin v0.4s, v11.4s, v0.4s\n"
                    "str   q11, [%[out_buf], 96]\n"
                    "smax v30.4s, v12.4s, v30.4s\n"
                    "smin v0.4s, v12.4s, v0.4s\n"
                    "str   q12, [%[out_buf], 112]\n"
                    "smax v30.4s, v13.4s, v30.4s\n"
                    "smin v0.4s, v13.4s, v0.4s\n"
                    "str   q13, [%[out_buf], 128]\n"

                    "smax v30.4s, v14.4s, v30.4s\n"
                    "smin v0.4s, v14.4s, v0.4s\n"
                    "str   q14, [%[out_buf], 144]\n"
                    "smax v30.4s, v15.4s, v30.4s\n"
                    "smin v0.4s, v15.4s, v0.4s\n"
                    "str   q15, [%[out_buf], 160]\n"
                    "smax v30.4s, v16.4s, v30.4s\n"
                    "smin v0.4s, v16.4s, v0.4s\n"
                    "str   q16, [%[out_buf], 176]\n"
                    "smax v30.4s, v17.4s, v30.4s\n"
                    "smin v0.4s, v17.4s, v0.4s\n"
                    "str   q17, [%[out_buf], 192]\n"
                    "smax v30.4s, v18.4s, v30.4s\n"
                    "smin v0.4s, v18.4s, v0.4s\n"
                    "str   q18, [%[out_buf], 208]\n"
                    "smax v30.4s, v19.4s, v30.4s\n"
                    "smin v0.4s, v19.4s, v0.4s\n"
                    "str   q19, [%[out_buf], 224]\n"
                    "smax v30.4s, v20.4s, v30.4s\n"
                    "smin v0.4s, v20.4s, v0.4s\n"
                    "str   q20, [%[out_buf], 240]\n"
                    "smax v30.4s, v21.4s, v30.4s\n"
                    "smin v0.4s, v21.4s, v0.4s\n"
                    "str   q21, [%[out_buf], 256]\n"
                    "smax v30.4s, v22.4s, v30.4s\n"
                    "smin v0.4s, v22.4s, v0.4s\n"
                    "str   q22, [%[out_buf], 272]\n"
                    "smax v30.4s, v23.4s, v30.4s\n"
                    "smin v0.4s, v23.4s, v0.4s\n"
                    "str   q23, [%[out_buf], 288]\n"
                    "smax v30.4s, v24.4s, v30.4s\n"
                    "smin v0.4s, v24.4s, v0.4s\n"
                    "str   q24, [%[out_buf], 304]\n"
                    "smax v30.4s, v25.4s, v30.4s\n"
                    "smin v0.4s, v25.4s, v0.4s\n"
                    "str   q25, [%[out_buf], 320]\n"
                    "smax v30.4s, v26.4s, v30.4s\n"
                    "smin v0.4s, v26.4s, v0.4s\n"
                    "str   q26, [%[out_buf], 336]\n"
                    "smax v30.4s, v27.4s, v30.4s\n"
                    "smin v0.4s, v27.4s, v0.4s\n"
                    "str   q27, [%[out_buf], 352]\n"
                    "smax v30.4s, v28.4s, v30.4s\n"
                    "smin v0.4s, v28.4s, v0.4s\n"
                    "str   q28, [%[out_buf], 368]\n"

                    "str   q30, [%[max]]\n"
                    "str   q0, [%[min]]\n"

                    "5:\n"
                    :
                    : [out_0] "r"(out_o0hw0), [out_buf] "r"(out_buf), [in_0] "r"(in_hw0),
                    [f_0] "r"(f_o0c0), [ic] "r"((I64)ic * 8 * fh * fw), [b_0] "r"(b_0),
                    [b_0_s] "r"(b_0_s), [factor] "r"(factor_v), [max] "r"(max_i32),
                    [min] "r"(min_i32), [conv_relu] "r"(conv_relu_bool),
                    [out_f16] "r"(out_f16_bool), [scale_known] "r"(scale_known_bool)
                    : "memory", "cc", "v0", "v1", "v2", "v3", "v5", "v6", "v7", "v8", "v9", "v10",
                    "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
                    "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "x0", "x1", "x2",
                    "x3", "x17", "x16");
                b0 += 8;
                b0_s += 8;
            }
        }

        // ohow_reminder % 12 / 8
        I32 ohow_s = (ohow / 12) * 12;
        I32 ohow_tail = ohow - ohow_s;

        if (ohow_tail >= 8) {
            I32 hw = ohow_s;
            F16 *b0 = biasArray;
            I32 *b0_s = biasScaled;
            INT8 *in_pack = ((INT8 *)tmp) + ic * ih_pad * iw_pad * 8;
            // pack input
            // NCHWc8 => NHWChw8c4 + im2col
            U32 in_h[8];
            U32 in_w[8];

            for (U32 i = 0; i < 8; i++) {
                in_h[i] = ((hw + i) / ow) * strideH;
                in_w[i] = ((hw + i) % ow) * strideW;
            }
            for (U32 c = 0; c < ic; c++) {
                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        INT8 *in_hw8c8 = inArray_pad + c * ihiw * 8 +
                            fh_idx * dilateH * iw_pad * 8 + fw_idx * dilateW * 8;
                        INT8 *in_0 = in_hw8c8 + in_h[0] * iw_pad * 8 + in_w[0] * 8;
                        INT8 *in_1 = in_hw8c8 + in_h[1] * iw_pad * 8 + in_w[1] * 8;
                        INT8 *in_2 = in_hw8c8 + in_h[2] * iw_pad * 8 + in_w[2] * 8;
                        INT8 *in_3 = in_hw8c8 + in_h[3] * iw_pad * 8 + in_w[3] * 8;
                        INT8 *in_4 = in_hw8c8 + in_h[4] * iw_pad * 8 + in_w[4] * 8;
                        INT8 *in_5 = in_hw8c8 + in_h[5] * iw_pad * 8 + in_w[5] * 8;
                        INT8 *in_6 = in_hw8c8 + in_h[6] * iw_pad * 8 + in_w[6] * 8;
                        INT8 *in_7 = in_hw8c8 + in_h[7] * iw_pad * 8 + in_w[7] * 8;
                        INT8 *in_pack_0 =
                            in_pack + c * fh * fw * 8 * 8 + fh_idx * fw * 8 * 4 + fw_idx * 8 * 4;
                        INT8 *in_pack_1 = in_pack_0 + fh * fw * 8 * 4;

                        __asm__ __volatile__("ldr d0, [%[in_0]]\n"
                                             "ldr x2, [%[in_2]]\n"
                                             "ldr d1, [%[in_1]]\n"
                                             "ldr x3, [%[in_3]]\n"
                                             "ins v0.d[1], x2\n"
                                             "ins v1.d[1], x3\n"
                                             "ldr d4, [%[in_4]]\n"
                                             "ldr x6, [%[in_6]]\n"
                                             "trn1 v20.4s, v0.4s, v1.4s\n"
                                             "trn2 v21.4s, v0.4s, v1.4s\n"

                                             "ldr d5, [%[in_5]]\n"
                                             "ldr x7, [%[in_7]]\n"
                                             "ins v4.d[1], x6\n"
                                             "ins v5.d[1], x7\n"

                                             "str   q20, [%[pack_0]]\n"
                                             "trn1 v24.4s, v4.4s, v5.4s\n"
                                             "trn2 v25.4s, v4.4s, v5.4s\n"
                                             "str   q21, [%[pack_1]]\n"
                                             "str   q24, [%[pack_0], #16]\n"
                                             "str   q25, [%[pack_1], #16]\n"
                                             :
                                             : [pack_0] "r"(in_pack_0), [pack_1] "r"(in_pack_1),
                                             [in_0] "r"(in_0), [in_1] "r"(in_1), [in_2] "r"(in_2),
                                             [in_3] "r"(in_3), [in_4] "r"(in_4), [in_5] "r"(in_5),
                                             [in_6] "r"(in_6), [in_7] "r"(in_7)
                                             : "memory", "cc", "v0", "v1", "v4", "v5", "v20", "v21",
                                             "v24", "v25", "x2", "x3", "x6", "x7");
                    }
                }
            }

            // compute
            for (U32 o = 0; o < oc; o++) {
                INT8 *in_hw0 = in_pack;
                INT8 *f_o0c0 = filterArray + o * 8 * fh * fw * ic * 8;
                I32 *out_buf = biasScaled + oc * 8 + n * oc * ohow * 8 + o * ohow * 8 + hw * 8;
                F16 *out_o0hw0 = outArray + n * oc * ohow * 8 + o * ohow * 8 + hw * 8;
                // bias
                F16 *b_0 = b0;
                I32 *b_0_s = b0_s;
                __asm__ __volatile__(
                    "cbz %[out_f16], 8f\n"
                    "eor v5.16b, v5.16b, v5.16b\n"
                    "ldr  d1, [%[in_0]]\n"  // in_0
                    "eor v6.16b, v6.16b, v6.16b\n"
                    "ldr  x1, [%[in_0], #8]\n"
                    "eor v7.16b, v7.16b, v7.16b\n"
                    "ins  v1.d[1], x1\n"
                    "eor v8.16b, v8.16b, v8.16b\n"
                    "ldr d0, [%[f_0]]\n"  // f_0
                    "eor v9.16b, v9.16b, v9.16b\n"
                    "ldr  x2, [%[f_0], #8]\n"
                    "eor v10.16b, v10.16b, v10.16b\n"
                    "ins v0.d[1], x2\n"
                    "eor v11.16b, v11.16b, v11.16b\n"
                    "eor v12.16b, v12.16b, v12.16b\n"
                    "eor v13.16b, v13.16b, v13.16b\n"
                    "eor v14.16b, v14.16b, v14.16b\n"
                    "eor v15.16b, v15.16b, v15.16b\n"
                    "eor v16.16b, v16.16b, v16.16b\n"
                    "eor v17.16b, v17.16b, v17.16b\n"
                    "eor v18.16b, v18.16b, v18.16b\n"
                    "eor v19.16b, v19.16b, v19.16b\n"
                    "eor v20.16b, v20.16b, v20.16b\n"
                    "b 7f\n"

                    "8:\n"
                    "ldp q29, q30, [%[b_0_s]]\n"
                    "mov v5.16b, v29.16b\n"
                    "ldr  d1, [%[in_0]]\n"  // in_0
                    "mov v7.16b, v29.16b\n"
                    "ldr  x1, [%[in_0], #8]\n"
                    "mov v9.16b, v29.16b\n"
                    "ins  v1.d[1], x1\n"
                    "mov v11.16b, v29.16b\n"
                    "ldr d0, [%[f_0]]\n"  // f_0
                    "mov v13.16b, v29.16b\n"
                    "ldr  x2, [%[f_0], #8]\n"
                    "mov v15.16b, v29.16b\n"
                    "ins v0.d[1], x2\n"
                    "mov v17.16b, v29.16b\n"
                    "mov v19.16b, v29.16b\n"

                    "mov v6.16b, v30.16b\n"
                    "mov v8.16b, v30.16b\n"
                    "mov v10.16b, v30.16b\n"
                    "mov v12.16b, v30.16b\n"
                    "mov v14.16b, v30.16b\n"
                    "mov v16.16b, v30.16b\n"
                    "mov v18.16b, v30.16b\n"
                    "mov v20.16b, v30.16b\n"

                    "7:\n"

                    // give in address to x3
                    "mov x3, %[in_0]\n"

                    // give f address to x0
                    "mov x0, %[f_0]\n"

                    "mov  x2, %[ic]\n"  // ic_blk
                    "0:\n"
                    "sdot v5.4s, v0.16b, v1.4b[0]\n"
                    "ldr d3, [x3, 16]!\n"
                    "ldr x16, [x3, 8]\n"
                    "sdot v7.4s, v0.16b, v1.4b[1]\n"
                    "ldr d29, [x0, 16]\n"
                    "ldr x17, [x0, 24]\n"
                    "sdot v9.4s, v0.16b, v1.4b[2]\n"
                    "ins v3.d[1], x16\n"
                    "ldr d30, [x3, 16]!\n"
                    "sdot v11.4s, v0.16b, v1.4b[3]\n"
                    "ins v29.d[1], x17\n"

                    "sdot v13.4s, v0.16b, v3.4b[0]\n"
                    "ldr x16, [x3, 8]\n"
                    "subs x2, x2, #4\n"
                    "sdot v15.4s, v0.16b, v3.4b[1]\n"
                    "sdot v17.4s, v0.16b, v3.4b[2]\n"
                    "ins v30.d[1], x16\n"
                    "sdot v19.4s, v0.16b, v3.4b[3]\n"

                    "sdot v6.4s, v29.16b, v1.4b[0]\n"
                    "sdot v8.4s, v29.16b, v1.4b[1]\n"
                    "ldr	d0, [x0, 32]!\n"
                    "ldr x17, [x0, 8]\n"
                    "sdot v10.4s, v29.16b, v1.4b[2]\n"
                    "sdot v12.4s, v29.16b, v1.4b[3]\n"

                    "sdot v14.4s, v29.16b, v3.4b[0]\n"
                    "ins v0.d[1], x17\n"
                    "mov	v1.16b, v30.16b\n"
                    "sdot v16.4s, v29.16b, v3.4b[1]\n"
                    "sdot v18.4s, v29.16b, v3.4b[2]\n"
                    "sdot v20.4s, v29.16b, v3.4b[3]\n"

                    "bne 0b\n"
                    "cbz %[out_f16], 6f\n"
                    "scvtf v5.4s, v5.4s\n"
                    "scvtf v6.4s, v6.4s\n"
                    "ldr d1, [%[factor]]\n"
                    "ldr x1, [%[factor], #8]\n"
                    "scvtf v7.4s, v7.4s\n"
                    "scvtf v8.4s, v8.4s\n"
                    "ins v1.d[1], x1\n"
                    "scvtf v9.4s, v9.4s\n"
                    "ldr d0, [%[b_0]]\n"
                    "ldr x0, [%[b_0], #8]\n"
                    "scvtf v10.4s, v10.4s\n"
                    "scvtf v11.4s, v11.4s\n"
                    "ins v0.d[1], x0\n"
                    "scvtf v12.4s, v12.4s\n"
                    "scvtf v13.4s, v13.4s\n"
                    "scvtf v14.4s, v14.4s\n"
                    "scvtf v15.4s, v15.4s\n"
                    "scvtf v16.4s, v16.4s\n"
                    "scvtf v17.4s, v17.4s\n"
                    "scvtf v18.4s, v18.4s\n"
                    "scvtf v19.4s, v19.4s\n"
                    "scvtf v20.4s, v20.4s\n"

                    "fmul v5.4s, v1.4s, v5.4s\n"
                    "fmul v6.4s, v1.4s, v6.4s\n"
                    "fmul v7.4s, v1.4s, v7.4s\n"
                    "fmul v8.4s, v1.4s, v8.4s\n"
                    "fmul v9.4s, v1.4s, v9.4s\n"
                    "fmul v10.4s, v1.4s, v10.4s\n"
                    "fmul v11.4s, v1.4s, v11.4s\n"
                    "fmul v12.4s, v1.4s, v12.4s\n"
                    "fmul v13.4s, v1.4s, v13.4s\n"
                    "fmul v14.4s, v1.4s, v14.4s\n"
                    "fmul v15.4s, v1.4s, v15.4s\n"
                    "fmul v16.4s, v1.4s, v16.4s\n"
                    "fmul v17.4s, v1.4s, v17.4s\n"
                    "fmul v18.4s, v1.4s, v18.4s\n"
                    "fmul v19.4s, v1.4s, v19.4s\n"
                    "fmul v20.4s, v1.4s, v20.4s\n"

                    "fcvtn v5.4h, v5.4s\n"
                    "fcvtn v7.4h, v7.4s\n"
                    "fcvtn v9.4h, v9.4s\n"
                    "fcvtn v11.4h, v11.4s\n"
                    "fcvtn v13.4h, v13.4s\n"
                    "fcvtn v15.4h, v15.4s\n"
                    "fcvtn v17.4h, v17.4s\n"
                    "fcvtn v19.4h, v19.4s\n"

                    "fcvtn2 v5.8h, v6.4s\n"
                    "fcvtn2 v7.8h, v8.4s\n"
                    "fcvtn2 v9.8h, v10.4s\n"
                    "fcvtn2 v11.8h, v12.4s\n"
                    "fcvtn2 v13.8h, v14.4s\n"
                    "fcvtn2 v15.8h, v16.4s\n"
                    "fcvtn2 v17.8h, v18.4s\n"
                    "fcvtn2 v19.8h, v20.4s\n"

                    "fadd v5.8h, v0.8h, v5.8h\n"
                    "fadd v7.8h, v0.8h, v7.8h\n"
                    "fadd v9.8h, v0.8h, v9.8h\n"
                    "fadd v11.8h, v0.8h, v11.8h\n"
                    "fadd v13.8h, v0.8h, v13.8h\n"
                    "fadd v15.8h, v0.8h, v15.8h\n"
                    "fadd v17.8h, v0.8h, v17.8h\n"
                    "fadd v19.8h, v0.8h, v19.8h\n"

                    "cbz %[conv_relu], 1f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"  // zero
                    "fmax  v5.8h,  v5.8h, v1.8h\n"
                    "fmax  v7.8h,  v7.8h, v1.8h\n"
                    "fmax  v9.8h,  v9.8h, v1.8h\n"
                    "fmax  v11.8h,  v11.8h, v1.8h\n"
                    "fmax  v13.8h,  v13.8h, v1.8h\n"
                    "fmax  v15.8h,  v15.8h, v1.8h\n"
                    "fmax  v17.8h,  v17.8h, v1.8h\n"
                    "fmax  v19.8h,  v19.8h, v1.8h\n"

                    "1:\n"
                    "str   q5, [%[out_0]]\n"
                    "str   q7, [%[out_0], #16]\n"
                    "str   q9, [%[out_0], #32]\n"
                    "str   q11, [%[out_0], #48]\n"
                    "str   q13, [%[out_0], #64]\n"
                    "str   q15, [%[out_0], #80]\n"
                    "str   q17, [%[out_0], #96]\n"
                    "str   q19, [%[out_0], #112]\n"
                    "b 5f\n"

                    "6:\n"
                    "ldr q0, [%[min]]\n"
                    "ldr q30, [%[max]]\n"
                    "cbz %[conv_relu], 2f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"  // zero
                    "smax v5.4s, v5.4s, v1.4s\n"
                    "smax v6.4s, v6.4s, v1.4s\n"
                    "smax v7.4s, v7.4s, v1.4s\n"
                    "smax v8.4s, v8.4s, v1.4s\n"
                    "smax v9.4s, v9.4s, v1.4s\n"
                    "smax v10.4s, v10.4s, v1.4s\n"
                    "smax v11.4s, v11.4s, v1.4s\n"
                    "smax v12.4s, v12.4s, v1.4s\n"
                    "smax v13.4s, v13.4s, v1.4s\n"
                    "smax v14.4s, v14.4s, v1.4s\n"
                    "smax v15.4s, v15.4s, v1.4s\n"
                    "smax v16.4s, v16.4s, v1.4s\n"
                    "smax v17.4s, v17.4s, v1.4s\n"
                    "smax v18.4s, v18.4s, v1.4s\n"
                    "smax v19.4s, v19.4s, v1.4s\n"
                    "smax v20.4s, v20.4s, v1.4s\n"

                    "2:\n"
                    "cbz %[scale_known], 7f\n"
                    "smax v5.4s, v5.4s, v0.4s\n"
                    "smin v5.4s, v5.4s, v30.4s\n"
                    "smax v6.4s, v6.4s, v0.4s\n"
                    "smin v6.4s, v6.4s, v30.4s\n"
                    "smax v7.4s, v7.4s, v0.4s\n"
                    "smin v7.4s, v7.4s, v30.4s\n"
                    "smax v8.4s, v8.4s, v0.4s\n"
                    "smin v8.4s, v8.4s, v30.4s\n"
                    "smax v9.4s, v9.4s, v0.4s\n"
                    "smin v9.4s, v9.4s, v30.4s\n"
                    "smax v10.4s, v10.4s, v0.4s\n"
                    "smin v10.4s, v10.4s, v30.4s\n"
                    "smax v11.4s, v11.4s, v0.4s\n"
                    "smin v11.4s, v11.4s, v30.4s\n"
                    "smax v12.4s, v12.4s, v0.4s\n"
                    "smin v12.4s, v12.4s, v30.4s\n"
                    "smax v13.4s, v13.4s, v0.4s\n"
                    "smin v13.4s, v13.4s, v30.4s\n"
                    "smax v14.4s, v14.4s, v0.4s\n"
                    "smin v14.4s, v14.4s, v30.4s\n"
                    "smax v15.4s, v15.4s, v0.4s\n"
                    "smin v15.4s, v15.4s, v30.4s\n"
                    "smax v16.4s, v16.4s, v0.4s\n"
                    "smin v16.4s, v16.4s, v30.4s\n"
                    "smax v17.4s, v17.4s, v0.4s\n"
                    "smin v17.4s, v17.4s, v30.4s\n"
                    "smax v18.4s, v18.4s, v0.4s\n"
                    "smin v18.4s, v18.4s, v30.4s\n"
                    "smax v19.4s, v19.4s, v0.4s\n"
                    "smin v19.4s, v19.4s, v30.4s\n"
                    "smax v20.4s, v20.4s, v0.4s\n"
                    "smin v20.4s, v20.4s, v30.4s\n"

                    "str   q5, [%[out_buf]]\n"
                    "str   q6, [%[out_buf], 16]\n"
                    "str   q7, [%[out_buf], 32]\n"
                    "str   q8, [%[out_buf], 48]\n"
                    "str   q9, [%[out_buf], 64]\n"
                    "str   q10, [%[out_buf], 80]\n"
                    "str   q11, [%[out_buf], 96]\n"
                    "str   q12, [%[out_buf], 112]\n"
                    "str   q13, [%[out_buf], 128]\n"
                    "str   q14, [%[out_buf], 144]\n"
                    "str   q15, [%[out_buf], 160]\n"
                    "str   q16, [%[out_buf], 176]\n"
                    "str   q17, [%[out_buf], 192]\n"
                    "str   q18, [%[out_buf], 208]\n"
                    "str   q19, [%[out_buf], 224]\n"
                    "str   q20, [%[out_buf], 240]\n"
                    "b 5f\n"

                    "7:\n"
                    "smax v30.4s, v5.4s, v30.4s\n"
                    "smin v0.4s, v5.4s, v0.4s\n"
                    "str   q5, [%[out_buf]]\n"
                    "smax v30.4s, v6.4s, v30.4s\n"
                    "smin v0.4s, v6.4s, v0.4s\n"
                    "str   q6, [%[out_buf], 16]\n"
                    "smax v30.4s, v7.4s, v30.4s\n"
                    "smin v0.4s, v7.4s, v0.4s\n"
                    "str   q7, [%[out_buf], 32]\n"
                    "smax v30.4s, v8.4s, v30.4s\n"
                    "smin v0.4s, v8.4s, v0.4s\n"
                    "str   q8, [%[out_buf], 48]\n"
                    "smax v30.4s, v9.4s, v30.4s\n"
                    "smin v0.4s, v9.4s, v0.4s\n"
                    "str   q9, [%[out_buf], 64]\n"
                    "smax v30.4s, v10.4s, v30.4s\n"
                    "smin v0.4s, v10.4s, v0.4s\n"
                    "str   q10, [%[out_buf], 80]\n"
                    "smax v30.4s, v11.4s, v30.4s\n"
                    "smin v0.4s, v11.4s, v0.4s\n"
                    "str   q11, [%[out_buf], 96]\n"
                    "smax v30.4s, v12.4s, v30.4s\n"
                    "smin v0.4s, v12.4s, v0.4s\n"
                    "str   q12, [%[out_buf], 112]\n"
                    "smax v30.4s, v13.4s, v30.4s\n"
                    "smin v0.4s, v13.4s, v0.4s\n"
                    "str   q13, [%[out_buf], 128]\n"

                    "smax v30.4s, v14.4s, v30.4s\n"
                    "smin v0.4s, v14.4s, v0.4s\n"
                    "str   q14, [%[out_buf], 144]\n"
                    "smax v30.4s, v15.4s, v30.4s\n"
                    "smin v0.4s, v15.4s, v0.4s\n"
                    "str   q15, [%[out_buf], 160]\n"
                    "smax v30.4s, v16.4s, v30.4s\n"
                    "smin v0.4s, v16.4s, v0.4s\n"
                    "str   q16, [%[out_buf], 176]\n"
                    "smax v30.4s, v17.4s, v30.4s\n"
                    "smin v0.4s, v17.4s, v0.4s\n"
                    "str   q17, [%[out_buf], 192]\n"
                    "smax v30.4s, v18.4s, v30.4s\n"
                    "smin v0.4s, v18.4s, v0.4s\n"
                    "str   q18, [%[out_buf], 208]\n"
                    "smax v30.4s, v19.4s, v30.4s\n"
                    "smin v0.4s, v19.4s, v0.4s\n"
                    "str   q19, [%[out_buf], 224]\n"
                    "smax v30.4s, v20.4s, v30.4s\n"
                    "smin v0.4s, v20.4s, v0.4s\n"
                    "str   q20, [%[out_buf], 240]\n"

                    "str   q30, [%[max]]\n"
                    "str   q0, [%[min]]\n"
                    "5:\n"
                    :
                    : [out_0] "r"(out_o0hw0), [out_buf] "r"(out_buf), [in_0] "r"(in_hw0),
                    [f_0] "r"(f_o0c0), [ic] "r"((I64)ic * 8 * fh * fw), [b_0] "r"(b_0),
                    [b_0_s] "r"(b_0_s), [factor] "r"(factor_v), [max] "r"(max_i32),
                    [min] "r"(min_i32), [conv_relu] "r"(conv_relu_bool),
                    [out_f16] "r"(out_f16_bool), [scale_known] "r"(scale_known_bool)
                    : "memory", "cc", "v0", "v1", "v3", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                    "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v29", "v30",
                    "x0", "x1", "x2", "x3", "x17", "x16");
                b0 += 8;
                b0_s += 8;
            }
            ohow_s += 8;
            ohow_tail -= 8;
        }

        if (ohow_tail >= 4) {
            I32 hw = ohow_s;
            F16 *b0 = biasArray;
            I32 *b0_s = biasScaled;
            INT8 *in_pack = ((INT8 *)tmp) + ic * ih_pad * iw_pad * 8;
            // pack input
            // NCHWc8 => NHWChw4c4 + im2col
            U32 in_h[4];
            U32 in_w[4];

            for (U32 i = 0; i < 4; i++) {
                in_h[i] = ((hw + i) / ow) * strideH;
                in_w[i] = ((hw + i) % ow) * strideW;
            }
            for (U32 c = 0; c < ic; c++) {
                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        INT8 *in_hw4c8 = inArray_pad + c * ihiw * 8 +
                            fh_idx * dilateH * iw_pad * 8 + fw_idx * dilateW * 8;
                        INT8 *in_0 = in_hw4c8 + in_h[0] * iw_pad * 8 + in_w[0] * 8;
                        INT8 *in_1 = in_hw4c8 + in_h[1] * iw_pad * 8 + in_w[1] * 8;
                        INT8 *in_2 = in_hw4c8 + in_h[2] * iw_pad * 8 + in_w[2] * 8;
                        INT8 *in_3 = in_hw4c8 + in_h[3] * iw_pad * 8 + in_w[3] * 8;
                        INT8 *in_pack_0 =
                            in_pack + c * fh * fw * 4 * 8 + fh_idx * fw * 4 * 4 + fw_idx * 4 * 4;
                        INT8 *in_pack_1 = in_pack_0 + fh * fw * 4 * 4;

                        __asm__ __volatile__(
                            "ldr d0, [%[in_0]]\n"
                            "ldr x2, [%[in_2]]\n"
                            "ldr d1, [%[in_1]]\n"
                            "ldr x3, [%[in_3]]\n"
                            "ins v0.d[1], x2\n"
                            "ins v1.d[1], x3\n"
                            "trn1 v20.4s, v0.4s, v1.4s\n"
                            "trn2 v21.4s, v0.4s, v1.4s\n"
                            "str   q20, [%[pack_0]]\n"
                            "str   q21, [%[pack_1]]\n"
                            :
                            : [pack_0] "r"(in_pack_0), [pack_1] "r"(in_pack_1), [in_0] "r"(in_0),
                            [in_1] "r"(in_1), [in_2] "r"(in_2), [in_3] "r"(in_3)
                            : "memory", "cc", "v0", "v1", "v20", "v21", "x2", "x3");
                    }
                }
            }

            // compute
            for (U32 o = 0; o < oc; o++) {
                INT8 *in_hw0 = in_pack;
                INT8 *f_o0c0 = filterArray + o * 8 * fh * fw * ic * 8;
                I32 *out_buf = biasScaled + oc * 8 + n * oc * ohow * 8 + o * ohow * 8 + hw * 8;
                F16 *out_o0hw0 = outArray + n * oc * ohow * 8 + o * ohow * 8 + hw * 8;
                // bias
                F16 *b_0 = b0;
                I32 *b_0_s = b0_s;
                __asm__ __volatile__(
                    "cbz %[out_f16], 8f\n"
                    "eor v5.16b, v5.16b, v5.16b\n"
                    "ldr  d1, [%[in_0]]\n"  // in_0
                    "eor v6.16b, v6.16b, v6.16b\n"
                    "ldr  x1, [%[in_0], #8]\n"
                    "eor v7.16b, v7.16b, v7.16b\n"
                    "ins  v1.d[1], x1\n"
                    "eor v8.16b, v8.16b, v8.16b\n"
                    "ldr d0, [%[f_0]]\n"  // f_0

                    "eor v9.16b, v9.16b, v9.16b\n"
                    "ldr  x2, [%[f_0], #8]\n"
                    "eor v10.16b, v10.16b, v10.16b\n"
                    "ins v0.d[1], x2\n"
                    "eor v11.16b, v11.16b, v11.16b\n"
                    "eor v12.16b, v12.16b, v12.16b\n"
                    "b 7f\n"

                    "8:\n"
                    "ldp q29, q30, [%[b_0_s]]\n"
                    "ldr  d1, [%[in_0]]\n"  // in_0
                    "mov v5.16b, v29.16b\n"
                    "ldr  x1, [%[in_0], #8]\n"
                    "mov v7.16b, v29.16b\n"
                    "ins  v1.d[1], x1\n"
                    "mov v9.16b, v29.16b\n"
                    "ldr d0, [%[f_0]]\n"  // f_0
                    "mov v11.16b, v29.16b\n"
                    "ldr  x2, [%[f_0], #8]\n"

                    "mov v6.16b, v30.16b\n"
                    "ins v0.d[1], x2\n"
                    "mov v8.16b, v30.16b\n"
                    "mov v10.16b, v30.16b\n"
                    "mov v12.16b, v30.16b\n"

                    "7:\n"

                    // give in address to x3
                    "mov x3, %[in_0]\n"

                    // give f address to x0
                    "mov x0, %[f_0]\n"

                    "mov  x2, %[ic]\n"  // ic_blk
                    "0:\n"
                    "ldr d29, [x0, 16]\n"
                    "ldr x17, [x0, 24]\n"
                    "sdot v5.4s, v0.16b, v1.4b[0]\n"
                    "ldr d3, [x3, 16]!\n"
                    "ldr x16, [x3, 8]\n"
                    "sdot v7.4s, v0.16b, v1.4b[1]\n"
                    "ins v29.d[1], x17\n"
                    "subs x2, x2, #4\n"
                    "sdot v9.4s, v0.16b, v1.4b[2]\n"
                    "ins v3.d[1], x16\n"
                    "sdot v11.4s, v0.16b, v1.4b[3]\n"

                    "sdot v6.4s, v29.16b, v1.4b[0]\n"
                    "ldr d0, [x0, 32]!\n"
                    "ldr x17, [x0, 8]\n"
                    "sdot v8.4s, v29.16b, v1.4b[1]\n"
                    "sdot v10.4s, v29.16b, v1.4b[2]\n"
                    "ins v0.d[1], x17\n"
                    "sdot v12.4s, v29.16b, v1.4b[3]\n"
                    "mov	v1.16b, v3.16b\n"

                    "bne 0b\n"
                    "cbz %[out_f16], 6f\n"

                    "scvtf v5.4s, v5.4s\n"
                    "scvtf v6.4s, v6.4s\n"
                    "ldr d1, [%[factor]]\n"
                    "ldr x1, [%[factor], #8]\n"
                    "scvtf v7.4s, v7.4s\n"
                    "scvtf v8.4s, v8.4s\n"
                    "ins v1.d[1], x1\n"
                    "scvtf v9.4s, v9.4s\n"
                    "ldr d0, [%[b_0]]\n"
                    "ldr x0, [%[b_0], #8]\n"
                    "scvtf v10.4s, v10.4s\n"
                    "scvtf v11.4s, v11.4s\n"
                    "ins v0.d[1], x0\n"
                    "scvtf v12.4s, v12.4s\n"

                    "fmul v5.4s, v1.4s, v5.4s\n"
                    "fmul v6.4s, v1.4s, v6.4s\n"
                    "fmul v7.4s, v1.4s, v7.4s\n"
                    "fmul v8.4s, v1.4s, v8.4s\n"
                    "fmul v9.4s, v1.4s, v9.4s\n"
                    "fmul v10.4s, v1.4s, v10.4s\n"
                    "fmul v11.4s, v1.4s, v11.4s\n"
                    "fmul v12.4s, v1.4s, v12.4s\n"

                    "fcvtn v5.4h, v5.4s\n"
                    "fcvtn v7.4h, v7.4s\n"
                    "fcvtn v9.4h, v9.4s\n"
                    "fcvtn v11.4h, v11.4s\n"

                    "fcvtn2 v5.8h, v6.4s\n"
                    "fcvtn2 v7.8h, v8.4s\n"
                    "fcvtn2 v9.8h, v10.4s\n"
                    "fcvtn2 v11.8h, v12.4s\n"

                    "fadd v5.8h, v0.8h, v5.8h\n"
                    "fadd v7.8h, v0.8h, v7.8h\n"
                    "fadd v9.8h, v0.8h, v9.8h\n"
                    "fadd v11.8h, v0.8h, v11.8h\n"

                    "cbz %[conv_relu], 1f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"  // zero
                    "fmax  v5.8h,  v5.8h, v1.8h\n"
                    "fmax  v7.8h,  v7.8h, v1.8h\n"
                    "fmax  v9.8h,  v9.8h, v1.8h\n"
                    "fmax  v11.8h,  v11.8h, v1.8h\n"

                    "1:\n"
                    "str   q5, [%[out_0]]\n"
                    "str   q7, [%[out_0], #16]\n"
                    "str   q9, [%[out_0], #32]\n"
                    "str   q11, [%[out_0], #48]\n"
                    "b 5f\n"

                    "6:\n"
                    "ldr q0, [%[min]]\n"
                    "ldr q30, [%[max]]\n"
                    "cbz %[conv_relu], 2f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"  // zero
                    "smax v5.4s, v5.4s, v1.4s\n"
                    "smax v6.4s, v6.4s, v1.4s\n"
                    "smax v7.4s, v7.4s, v1.4s\n"
                    "smax v8.4s, v8.4s, v1.4s\n"
                    "smax v9.4s, v9.4s, v1.4s\n"
                    "smax v10.4s, v10.4s, v1.4s\n"
                    "smax v11.4s, v11.4s, v1.4s\n"
                    "smax v12.4s, v12.4s, v1.4s\n"

                    "2:\n"
                    "cbz %[scale_known], 7f\n"
                    "smax v5.4s, v5.4s, v0.4s\n"
                    "smin v5.4s, v5.4s, v30.4s\n"
                    "smax v6.4s, v6.4s, v0.4s\n"
                    "smin v6.4s, v6.4s, v30.4s\n"
                    "smax v7.4s, v7.4s, v0.4s\n"
                    "smin v7.4s, v7.4s, v30.4s\n"
                    "smax v8.4s, v8.4s, v0.4s\n"
                    "smin v8.4s, v8.4s, v30.4s\n"
                    "smax v9.4s, v9.4s, v0.4s\n"
                    "smin v9.4s, v9.4s, v30.4s\n"
                    "smax v10.4s, v10.4s, v0.4s\n"
                    "smin v10.4s, v10.4s, v30.4s\n"
                    "smax v11.4s, v11.4s, v0.4s\n"
                    "smin v11.4s, v11.4s, v30.4s\n"
                    "smax v12.4s, v12.4s, v0.4s\n"
                    "smin v12.4s, v12.4s, v30.4s\n"

                    "str   q5, [%[out_buf]]\n"
                    "str   q6, [%[out_buf], 16]\n"
                    "str   q7, [%[out_buf], 32]\n"
                    "str   q8, [%[out_buf], 48]\n"
                    "str   q9, [%[out_buf], 64]\n"
                    "str   q10, [%[out_buf], 80]\n"
                    "str   q11, [%[out_buf], 96]\n"
                    "str   q12, [%[out_buf], 112]\n"
                    "b 5f\n"

                    "7:\n"
                    "smax v30.4s, v5.4s, v30.4s\n"
                    "smin v0.4s, v5.4s, v0.4s\n"
                    "str   q5, [%[out_buf]]\n"
                    "smax v30.4s, v6.4s, v30.4s\n"
                    "smin v0.4s, v6.4s, v0.4s\n"
                    "str   q6, [%[out_buf], 16]\n"
                    "smax v30.4s, v7.4s, v30.4s\n"
                    "smin v0.4s, v7.4s, v0.4s\n"
                    "str   q7, [%[out_buf], 32]\n"
                    "smax v30.4s, v8.4s, v30.4s\n"
                    "smin v0.4s, v8.4s, v0.4s\n"
                    "str   q8, [%[out_buf], 48]\n"
                    "smax v30.4s, v9.4s, v30.4s\n"
                    "smin v0.4s, v9.4s, v0.4s\n"
                    "str   q9, [%[out_buf], 64]\n"
                    "smax v30.4s, v10.4s, v30.4s\n"
                    "smin v0.4s, v10.4s, v0.4s\n"
                    "str   q10, [%[out_buf], 80]\n"
                    "smax v30.4s, v11.4s, v30.4s\n"
                    "smin v0.4s, v11.4s, v0.4s\n"
                    "str   q11, [%[out_buf], 96]\n"
                    "smax v30.4s, v12.4s, v30.4s\n"
                    "smin v0.4s, v12.4s, v0.4s\n"
                    "str   q12, [%[out_buf], 112]\n"

                    "str   q30, [%[max]]\n"
                    "str   q0, [%[min]]\n"
                    "5:\n"
                    :
                    : [out_0] "r"(out_o0hw0), [out_buf] "r"(out_buf), [in_0] "r"(in_hw0),
                    [f_0] "r"(f_o0c0), [ic] "r"((I64)ic * 8 * fh * fw), [b_0] "r"(b_0),
                    [b_0_s] "r"(b_0_s), [factor] "r"(factor_v), [max] "r"(max_i32),
                    [min] "r"(min_i32), [conv_relu] "r"(conv_relu_bool),
                    [out_f16] "r"(out_f16_bool), [scale_known] "r"(scale_known_bool)
                    : "memory", "cc", "v0", "v1", "v2", "v3", "v5", "v6", "v7", "v8", "v9", "v10",
                    "v11", "v12", "v29", "x0", "x1", "x2", "x3", "x17", "x16");
                b0 += 8;
                b0_s += 8;
            }
            ohow_s += 4;
        }

        for (I32 hw = ohow_s; hw < ohow; hw++) {
            F16 *b0 = biasArray;
            I32 *b0_s = biasScaled;
            INT8 *in_pack = ((INT8 *)tmp) + ic * ih_pad * iw_pad * 8;
            // pack input
            // NCHWc8 => NHWChw1c4 + im2col
            U32 in_h_0 = (hw / ow) * strideH;
            U32 in_w_0 = (hw % ow) * strideW;
            for (U32 c = 0; c < ic; c++) {
                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        INT8 *in_hw1c8 = inArray_pad + c * ihiw * 8 +
                            fh_idx * dilateH * iw_pad * 8 + fw_idx * dilateW * 8;
                        INT8 *in_0 = in_hw1c8 + in_h_0 * iw_pad * 8 + in_w_0 * 8;
                        INT8 *in_pack_0 = in_pack + c * fh * fw * 8 + fh_idx * fw * 4 + fw_idx * 4;
                        INT8 *in_pack_1 = in_pack_0 + fh * fw * 4;

                        memcpy(in_pack_0, in_0, 4 * bytesOf(DT_I8));
                        memcpy(in_pack_1, in_0 + 4, 4 * bytesOf(DT_I8));
                    }
                }
            }

            // compute
            for (U32 o = 0; o < oc; o++) {
                INT8 *in_hw = in_pack;
                INT8 *f_o = filterArray + o * 8 * fh * fw * ic * 8;
                I32 *out_buf = biasScaled + oc * 8 + n * oc * ohow * 8 + o * ohow * 8 + hw * 8;
                F16 *out_o0hw0 = outArray + n * oc * ohow * 8 + o * ohow * 8 + hw * 8;

                int32x4_t res[2] = {0};
                if (out_f16_bool == 0) {
                    res[0] = vld1q_s32(b0_s);
                    res[1] = vld1q_s32(b0_s + 4);
                }

                for (U32 c = 0; c < ic * fh * fw; c++) {
                    int8x8_t in_2 = vld1_s8(in_hw);
                    in_hw += 8;
                    int8x16_t f_8o[4];
                    f_8o[0] = vld1q_s8(f_o);
                    f_8o[1] = vld1q_s8(f_o + 16);
                    res[0] = vdotq_lane_s32(res[0], f_8o[0], in_2, 0);
                    res[1] = vdotq_lane_s32(res[1], f_8o[1], in_2, 0);

                    f_8o[2] = vld1q_s8(f_o + 32);
                    f_8o[3] = vld1q_s8(f_o + 48);
                    f_o += 64;
                    res[0] = vdotq_lane_s32(res[0], f_8o[2], in_2, 1);
                    res[1] = vdotq_lane_s32(res[1], f_8o[3], in_2, 1);
                }
                if (out_f16_bool == 1) {
                    float32x4_t fac = vld1q_f32(factor_v);
                    float32x4_t resf0 = vcvtq_f32_s32(res[0]);
                    float32x4_t resf1 = vcvtq_f32_s32(res[1]);
                    resf0 = vmulq_f32(resf0, fac);
                    resf1 = vmulq_f32(resf1, fac);

                    float16x4_t bias0 = vld1_f16(b0);
                    float16x4_t bias1 = vld1_f16(b0 + 4);
                    float16x4_t resh0 = vcvt_f16_f32(resf0);
                    float16x4_t resh1 = vcvt_f16_f32(resf1);
                    resh0 = vadd_f16(resh0, bias0);
                    resh1 = vadd_f16(resh1, bias1);

                    if (conv_relu_bool) {
                        float16x4_t z = vdup_n_f16(0);
                        resh0 = vmax_f16(resh0, z);
                        resh1 = vmax_f16(resh1, z);
                    }
                    vst1_f16(out_o0hw0, resh0);
                    vst1_f16(out_o0hw0 + 4, resh1);
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
    if (out_f16_bool == 0) {
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

        U32 num_v = oc * ohow * 2;  // Number of q-form vectors
        I32 *out_buf = biasScaled + oc * 8;
        INT8 *out_q = (INT8 *)output;

        ret = quantize_I32(num_v, out_buf, factor, scale_o, out_q);
    }
    return ret;
}

template EE convolution_gemm_A55<INT8>(TensorDesc inputDesc,
    const void *input,
    F16 *inputScale,
    TensorDesc filterDesc,
    const void *filter,
    F16 *filterScale,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    F16 *outputScale,
    ActivationParamSpec activationDesc);

template EE convolution_gemm_A55<F16>(TensorDesc inputDesc,
    const void *input,
    F16 *inputScale,
    TensorDesc filterDesc,
    const void *filter,
    F16 *filterScale,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    F16 *outputScale,
    ActivationParamSpec activationDesc);
#endif
