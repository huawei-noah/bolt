// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/fp32/convolution_winograd_transform.h"
#include "cpu/arm/fp32/tensor_computing_fp32.h"
#ifdef _USE_OPENMP
#include <omp.h>
#endif

EE convolution_winograd_V8(TensorDesc inputDesc,
    F32 *inArray,
    TensorDesc filterDesc,
    const F32 *filterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const F32 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F32 *outArray,
    ActivationParamSpec activationDesc)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);

    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingB = convParamSpec.padding_bottom;
    U32 paddingL = convParamSpec.padding_left;
    U32 paddingR = convParamSpec.padding_right;

    if (fdf != DF_HWNCN8) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (!(fh == 6 && fw == 6)) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    oc /= 8;
    ic /= 8;

    U32 tile_h = (oh + 3) / 4;
    U32 tile_w = (ow + 3) / 4;
    // num of 6x6 tiles
    I32 tiles = tile_h * tile_w;
    U32 pad_left = paddingL;
    U32 pad_right = paddingR + (tile_w * 4 - ow);
    U32 pad_w_mod_4 = tile_w * 4 - ow;
    U32 pad_top = paddingT;
    U32 pad_bottom = paddingB + (tile_h * 4 - oh);
    U32 pad_h_mod_4 = tile_h * 4 - oh;
    U32 ih_pad = ih + pad_top + pad_bottom;
    U32 iw_pad = iw + pad_left + pad_right;
    // tmp = in_pad + itm + otm
    // in_pad: ic*ih_pad*iw_pad*8
    // itm: 6*6*ic*12*8
    // otm: 6*6*12*8
    F32 *inArray_pad = (F32 *)tmp;
    F32 *itmArray = inArray_pad + ic * ih_pad * iw_pad * 8;
    F32 *otmArray = itmArray + 6 * 6 * ic * 12 * 8 * OMP_NUM_THREADS;

    EE ret = SUCCESS;
    // copy input into a input with padding
    for (U32 n = 0; n < in; n++) {
        F32 *inArray_pad_mov = inArray_pad;
        F32 *inArray_mov = inArray + n * ic * ih * iw * 8;
        for (U32 c = 0; c < ic; c++) {
            memset(inArray_pad_mov, 0, pad_top * iw_pad * 8 * bytesOf(idt));
            inArray_pad_mov += pad_top * iw_pad * 8;
            for (U32 h = pad_top; h < ih_pad - pad_bottom; h++) {
                memset(inArray_pad_mov, 0, pad_left * 8 * bytesOf(idt));
                inArray_pad_mov += pad_left * 8;
                memcpy(inArray_pad_mov, inArray_mov, iw * 8 * bytesOf(idt));
                inArray_pad_mov += iw * 8;
                inArray_mov += iw * 8;
                memset(inArray_pad_mov, 0, pad_right * 8 * bytesOf(idt));
                inArray_pad_mov += pad_right * 8;
            }
            memset(inArray_pad_mov, 0, pad_bottom * iw_pad * 8 * bytesOf(idt));
            inArray_pad_mov += pad_bottom * iw_pad * 8;
        }

        // tiles / 12
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (I32 hw = 0; hw < tiles - 11; hw += 12) {
#ifdef _USE_OPENMP
            F32 *thread_itmArray = itmArray + ic * 6 * 6 * 12 * 8 * omp_get_thread_num();
            F32 *thread_otmArray = otmArray + 8 * 6 * 6 * 12 * omp_get_thread_num();
#else
            F32 *thread_itmArray = itmArray;
            F32 *thread_otmArray = otmArray;
#endif
            // in trans
            // NCHWc8 => (6*6)*C*c8*hw12
            for (U32 c = 0; c < ic; c++) {
                F32 *inArray_pad_mov = inArray_pad + c * ih_pad * iw_pad * 8;
                F32 *Iw_ptr0[36];
                F32 *Iw_ptr1[36];
                F32 Iw[12][36][8];
                F32 *I0[12][36];
                F32 *I1[12][36];
                U32 h[12];
                U32 w[12];
                for (U32 index = 0; index < 12; index++) {
                    h[index] = ((hw + index) / tile_w) * 4;
                    w[index] = ((hw + index) % tile_w) * 4;
                }
                for (U32 i = 0; i < 6; i++) {
                    for (U32 j = 0; j < 6; j++) {
                        for (U32 index = 0; index < 12; index++) {
                            I0[index][i * 6 + j] =
                                inArray_pad_mov + (h[index] + i) * iw_pad * 8 + (w[index] + j) * 8;
                            I1[index][i * 6 + j] = inArray_pad_mov + (h[index] + i) * iw_pad * 8 +
                                (w[index] + j) * 8 + 4;
                        }
                    }
                }
                for (U32 index = 0; index < 12; index++) {
                    for (U32 i = 0; i < 36; i++) {
                        Iw_ptr0[i] = Iw[index][i];
                        Iw_ptr1[i] = Iw_ptr0[i] + 4;
                    }
                    trans_I_4x4_3x3(Iw_ptr0, I0[index]);
                    trans_I_4x4_3x3(Iw_ptr1, I1[index]);
                }
                for (U32 i = 0; i < 36; i++) {
                    F32 *itm = thread_itmArray + (i * ic + c) * 8 * 12;

                    __asm__ __volatile__(
                        "ldp q0, q1, [%[in_0]]\n"
                        "ldp q2, q3, [%[in_1]]\n"
                        "ldp q4, q5, [%[in_2]]\n"
                        "ldp q6, q7, [%[in_3]]\n"

                        "ldp q8, q9, [%[in_4]]\n"
                        "ldp q10, q11, [%[in_5]]\n"
                        "ldp q12, q13, [%[in_6]]\n"
                        "ldp q14, q15, [%[in_7]]\n"

                        "ldp q16, q17, [%[in_8]]\n"
                        "ldp q18, q19, [%[in_9]]\n"
                        "ldp q20, q21, [%[in_10]]\n"
                        "ldp q22, q23, [%[in_11]]\n"

                        "zip1 v24.4s, v0.4s, v2.4s\n"
                        "zip2 v25.4s, v0.4s, v2.4s\n"
                        "zip1 v26.4s, v4.4s, v6.4s\n"
                        "zip2 v27.4s, v4.4s, v6.4s\n"

                        "zip1 v0.2d, v24.2d, v26.2d\n"
                        "zip2 v2.2d, v24.2d, v26.2d\n"
                        "zip1 v4.2d, v25.2d, v27.2d\n"
                        "zip2 v6.2d, v25.2d, v27.2d\n"

                        "zip1 v24.4s, v8.4s, v10.4s\n"
                        "zip2 v25.4s, v8.4s, v10.4s\n"
                        "zip1 v26.4s, v12.4s, v14.4s\n"
                        "zip2 v27.4s, v12.4s, v14.4s\n"

                        "zip1 v8.2d, v24.2d, v26.2d\n"
                        "zip2 v10.2d, v24.2d, v26.2d\n"
                        "zip1 v12.2d, v25.2d, v27.2d\n"
                        "zip2 v14.2d, v25.2d, v27.2d\n"

                        "zip1 v24.4s, v16.4s, v18.4s\n"
                        "zip2 v25.4s, v16.4s, v18.4s\n"
                        "zip1 v26.4s, v20.4s, v22.4s\n"
                        "zip2 v27.4s, v20.4s, v22.4s\n"

                        "zip1 v16.2d, v24.2d, v26.2d\n"
                        "zip2 v18.2d, v24.2d, v26.2d\n"
                        "zip1 v20.2d, v25.2d, v27.2d\n"
                        "zip2 v22.2d, v25.2d, v27.2d\n"

                        "stp q0, q8, [%[pack]]\n"
                        "str q16, [%[pack], #32]\n"
                        "stp q2, q10, [%[pack], 48]\n"
                        "str q18, [%[pack], #80]\n"
                        "stp q4, q12, [%[pack], #96]\n"
                        "str q20, [%[pack], #128]\n"
                        "stp q6, q14, [%[pack], #144]\n"
                        "str q22, [%[pack], #176]\n"

                        "zip1 v24.4s, v1.4s, v3.4s\n"
                        "zip2 v25.4s, v1.4s, v3.4s\n"
                        "zip1 v26.4s, v5.4s, v7.4s\n"
                        "zip2 v27.4s, v5.4s, v7.4s\n"

                        "zip1 v1.2d, v24.2d, v26.2d\n"
                        "zip2 v3.2d, v24.2d, v26.2d\n"
                        "zip1 v5.2d, v25.2d, v27.2d\n"
                        "zip2 v7.2d, v25.2d, v27.2d\n"

                        "zip1 v24.4s, v9.4s, v11.4s\n"
                        "zip2 v25.4s, v9.4s, v11.4s\n"
                        "zip1 v26.4s, v13.4s, v15.4s\n"
                        "zip2 v27.4s, v13.4s, v15.4s\n"

                        "zip1 v9.2d, v24.2d, v26.2d\n"
                        "zip2 v11.2d, v24.2d, v26.2d\n"
                        "zip1 v13.2d, v25.2d, v27.2d\n"
                        "zip2 v15.2d, v25.2d, v27.2d\n"

                        "zip1 v24.4s, v17.4s, v19.4s\n"
                        "zip2 v25.4s, v17.4s, v19.4s\n"
                        "zip1 v26.4s, v21.4s, v23.4s\n"
                        "zip2 v27.4s, v21.4s, v23.4s\n"

                        "zip1 v17.2d, v24.2d, v26.2d\n"
                        "zip2 v19.2d, v24.2d, v26.2d\n"
                        "zip1 v21.2d, v25.2d, v27.2d\n"
                        "zip2 v23.2d, v25.2d, v27.2d\n"

                        "stp q1, q9, [%[pack], #192]\n"
                        "str q17, [%[pack], #224]\n"
                        "stp q3, q11, [%[pack], 240]\n"
                        "str q19, [%[pack], #272]\n"
                        "stp q5, q13, [%[pack], 288]\n"
                        "str q21, [%[pack], #320]\n"
                        "stp q7, q15, [%[pack], 336]\n"
                        "str q23, [%[pack], #368]\n"
                        :
                        : [pack] "r"(itm), [in_0] "r"(Iw[0][i]), [in_1] "r"(Iw[1][i]),
                        [in_2] "r"(Iw[2][i]), [in_3] "r"(Iw[3][i]), [in_4] "r"(Iw[4][i]),
                        [in_5] "r"(Iw[5][i]), [in_6] "r"(Iw[6][i]), [in_7] "r"(Iw[7][i]),
                        [in_8] "r"(Iw[8][i]), [in_9] "r"(Iw[9][i]), [in_10] "r"(Iw[10][i]),
                        [in_11] "r"(Iw[11][i])
                        : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                        "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
                        "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
                }
            }
            for (I32 o = 0; o < I32(oc); o++) {
                const F32 *b_0 = biasArray + o * 8;
                const F32 *b_1 = b_0 + 4;
                // dot prod
                // (6*6)*C*c8*hw12 times O*(6*6)*C*c8*o8 = O*(6*6)*hw12*o8
                for (U32 idx = 0; idx < 36; idx++) {
                    F32 *itm_0 = thread_itmArray + idx * 12 * ic * 8;
                    const F32 *f_o0c0 = filterArray + o * 8 * 36 * ic * 8 + idx * 8 * ic * 8;
                    F32 *out_o0hw0 = thread_otmArray + idx * 12 * 8;
                    __asm__ __volatile__(
                        // give in address to x3
                        "mov x3, %[in_0]\n"

                        // give f address to x0
                        "mov x0, %[f_0]\n"

                        "mov  x2, %[ic]\n"

                        "eor  v5.16b, v5.16b, v5.16b\n"
                        "ldr  q1, [%[in_0]]\n"  // in_hw0
                        "eor  v6.16b, v6.16b, v6.16b\n"
                        "eor  v7.16b, v7.16b, v7.16b\n"
                        "eor  v8.16b, v8.16b, v8.16b\n"
                        "ldr q0, [%[f_0]]\n"  // f_o0c0
                        "eor  v9.16b, v9.16b, v9.16b\n"
                        "eor  v10.16b, v10.16b, v10.16b\n"
                        "eor  v11.16b, v11.16b, v11.16b\n"
                        "ldr q3, [%[in_0], #16]\n"
                        "eor  v12.16b, v12.16b, v12.16b\n"
                        "eor  v13.16b, v13.16b, v13.16b\n"
                        "eor  v14.16b, v14.16b, v14.16b\n"
                        "eor  v15.16b, v15.16b, v15.16b\n"

                        "eor  v16.16b, v16.16b, v16.16b\n"
                        "eor  v17.16b, v17.16b, v17.16b\n"
                        "eor  v18.16b, v18.16b, v18.16b\n"
                        "eor  v19.16b, v19.16b, v19.16b\n"
                        "eor  v20.16b, v20.16b, v20.16b\n"
                        "eor  v21.16b, v21.16b, v21.16b\n"
                        "eor  v22.16b, v22.16b, v22.16b\n"
                        "eor  v23.16b, v23.16b, v23.16b\n"
                        "eor  v24.16b, v24.16b, v24.16b\n"
                        "eor  v25.16b, v25.16b, v25.16b\n"
                        "eor  v26.16b, v26.16b, v26.16b\n"
                        "eor  v27.16b, v27.16b, v27.16b\n"
                        "eor  v28.16b, v28.16b, v28.16b\n"
                        "0:\n"
                        "fmla  v5.4s, v0.4s, v1.s[0]\n"
                        "fmla  v7.4s, v0.4s, v1.s[1]\n"
                        "ldr q2, [x3, 32]\n"
                        "ldr q29, [x0, 16]\n"
                        "fmla  v9.4s, v0.4s, v1.s[2]\n"
                        "fmla v11.4s, v0.4s, v1.s[3]\n"

                        "fmla v13.4s, v0.4s, v3.s[0]\n"
                        "fmla v15.4s, v0.4s, v3.s[1]\n"
                        "fmla v17.4s, v0.4s, v3.s[2]\n"
                        "fmla v19.4s, v0.4s, v3.s[3]\n"

                        "fmla v21.4s, v0.4s, v2.s[0]\n"
                        "fmla v23.4s, v0.4s, v2.s[1]\n"
                        "fmla v25.4s, v0.4s, v2.s[2]\n"
                        "fmla v27.4s, v0.4s, v2.s[3]\n"

                        "fmla  v6.4s, v29.4s, v1.s[0]\n"
                        "fmla  v8.4s, v29.4s, v1.s[1]\n"
                        "fmla v10.4s, v29.4s, v1.s[2]\n"
                        "fmla v12.4s, v29.4s, v1.s[3]\n"

                        "fmla v14.4s, v29.4s, v3.s[0]\n"
                        "fmla v16.4s, v29.4s, v3.s[1]\n"
                        "ldr q1, [x3, 48]!\n"
                        "ldr q0, [x0, 32]!\n"
                        "fmla v18.4s, v29.4s, v3.s[2]\n"
                        "fmla v20.4s, v29.4s, v3.s[3]\n"

                        "fmla v22.4s, v29.4s, v2.s[0]\n"
                        "fmla v24.4s, v29.4s, v2.s[1]\n"
                        "ldr q3, [x3, 16]\n"
                        "subs x2, x2, #1\n"
                        "fmla v26.4s, v29.4s, v2.s[2]\n"
                        "fmla v28.4s, v29.4s, v2.s[3]\n"
                        "bne 0b\n"
                        "str   q5, [%[out_0]]\n"
                        "str   q6, [%[out_0], #16]\n"
                        "str   q7, [%[out_0], #32]\n"
                        "str   q8, [%[out_0], #48]\n"
                        "str   q9, [%[out_0], #64]\n"
                        "str   q10, [%[out_0], #80]\n"
                        "str   q11, [%[out_0], #96]\n"
                        "str   q12, [%[out_0], #112]\n"
                        "str   q13, [%[out_0], #128]\n"
                        "str   q14, [%[out_0], #144]\n"
                        "str   q15, [%[out_0], #160]\n"
                        "str   q16, [%[out_0], #176]\n"
                        "str   q17, [%[out_0], #192]\n"
                        "str   q18, [%[out_0], #208]\n"
                        "str   q19, [%[out_0], #224]\n"
                        "str   q20, [%[out_0], #240]\n"
                        "str   q21, [%[out_0], #256]\n"
                        "str   q22, [%[out_0], #272]\n"
                        "str   q23, [%[out_0], #288]\n"
                        "str   q24, [%[out_0], #304]\n"
                        "str   q25, [%[out_0], #320]\n"
                        "str   q26, [%[out_0], #336]\n"
                        "str   q27, [%[out_0], #352]\n"
                        "str   q28, [%[out_0], #368]\n"
                        : [out_0] "+r"(out_o0hw0), [in_0] "+r"(itm_0), [f_0] "+r"(f_o0c0)
                        : [ic] "r"((I64)ic * 8)
                        : "memory", "cc", "v0", "v1", "v2", "v3", "v5", "v6", "v7", "v8", "v9",
                        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "x0",
                        "x1", "x2", "x3");
                }
                // out trans
                // O*(6*6)*hw12*o8 => NOHWo8
                for (U32 hw12 = 0; hw12 < 12; hw12++) {
                    U32 h = (hw + hw12) / tile_w;
                    U32 w = (hw + hw12) % tile_w;
                    F32 *out_0 = outArray + n * oc * oh * ow * 8 + o * oh * ow * 8 +
                        h * 4 * ow * 8 + w * 4 * 8;

                    F32 *Ow_0[36];
                    F32 *Ow_1[36];
                    F32 *O_0[16];
                    F32 *O_1[16];
                    for (U32 idx = 0; idx < 36; idx++) {
                        Ow_0[idx] = thread_otmArray + idx * 12 * 8 + hw12 * 8;
                        Ow_1[idx] = Ow_0[idx] + 4;
                    }
                    for (U32 i = 0; i < 4; ++i) {
                        for (U32 j = 0; j < 4; ++j) {
                            O_0[i * 4 + j] = out_0 + i * ow * 8 + j * 8;
                            O_1[i * 4 + j] = O_0[i * 4 + j] + 4;
                        }
                    }
                    CHECK_STATUS(trans_O_4x4_3x3(Ow_0, O_0, b_0, h, w, pad_h_mod_4, pad_w_mod_4,
                        tile_h - 1, tile_w - 1, activationDesc));
                    CHECK_STATUS(trans_O_4x4_3x3(Ow_1, O_1, b_1, h, w, pad_h_mod_4, pad_w_mod_4,
                        tile_h - 1, tile_w - 1, activationDesc));
                }
            }
        }

        // tiles_reminder % 12 / 8
        I32 tiles_s = (tiles / 12) * 12;
        I32 tiles_tail = tiles - tiles_s;

        if (tiles_tail >= 8) {
            I32 hw = tiles_s;
            // in trans
            // NCHWc8 => (6*6)*C*c8*hw8
            for (U32 c = 0; c < ic; c++) {
                F32 *inArray_pad_mov = inArray_pad + c * ih_pad * iw_pad * 8;
                F32 *itmArray_mov = itmArray + c * 8 * 8;
                F32 *Iw_ptr0[36];
                F32 *Iw_ptr1[36];
                F32 Iw[8][36][8];
                F32 *I0[8][36];
                F32 *I1[8][36];
                U32 h[8];
                U32 w[8];
                for (U32 index = 0; index < 8; index++) {
                    h[index] = ((hw + index) / tile_w) * 4;
                    w[index] = ((hw + index) % tile_w) * 4;
                }
                for (U32 i = 0; i < 6; i++) {
                    for (U32 j = 0; j < 6; j++) {
                        for (U32 index = 0; index < 8; index++) {
                            I0[index][i * 6 + j] =
                                inArray_pad_mov + (h[index] + i) * iw_pad * 8 + (w[index] + j) * 8;
                            I1[index][i * 6 + j] = inArray_pad_mov + (h[index] + i) * iw_pad * 8 +
                                (w[index] + j) * 8 + 4;
                        }
                    }
                }
                for (U32 index = 0; index < 8; index++) {
                    for (U32 i = 0; i < 36; i++) {
                        Iw_ptr0[i] = Iw[index][i];
                        Iw_ptr1[i] = Iw_ptr0[i] + 4;
                    }
                    trans_I_4x4_3x3(Iw_ptr0, I0[index]);
                    trans_I_4x4_3x3(Iw_ptr1, I1[index]);
                }
                for (U32 i = 0; i < 36; i++) {
                    F32 *itm = itmArray_mov + i * ic * 8 * 8;

                    __asm__ __volatile__(
                        "ldp q0, q1, [%[in_0]]\n"
                        "ldp q2, q3, [%[in_1]]\n"
                        "ldp q4, q5, [%[in_2]]\n"
                        "ldp q6, q7, [%[in_3]]\n"

                        "ldp q8, q9, [%[in_4]]\n"
                        "ldp q10, q11, [%[in_5]]\n"
                        "ldp q12, q13, [%[in_6]]\n"
                        "ldp q14, q15, [%[in_7]]\n"

                        "zip1 v24.4s, v0.4s, v2.4s\n"
                        "zip2 v25.4s, v0.4s, v2.4s\n"
                        "zip1 v26.4s, v4.4s, v6.4s\n"
                        "zip2 v27.4s, v4.4s, v6.4s\n"

                        "zip1 v0.2d, v24.2d, v26.2d\n"
                        "zip2 v2.2d, v24.2d, v26.2d\n"
                        "zip1 v4.2d, v25.2d, v27.2d\n"
                        "zip2 v6.2d, v25.2d, v27.2d\n"

                        "zip1 v24.4s, v8.4s, v10.4s\n"
                        "zip2 v25.4s, v8.4s, v10.4s\n"
                        "zip1 v26.4s, v12.4s, v14.4s\n"
                        "zip2 v27.4s, v12.4s, v14.4s\n"

                        "zip1 v8.2d, v24.2d, v26.2d\n"
                        "zip2 v10.2d, v24.2d, v26.2d\n"
                        "zip1 v12.2d, v25.2d, v27.2d\n"
                        "zip2 v14.2d, v25.2d, v27.2d\n"

                        "stp q0, q8, [%[pack]]\n"
                        "stp q2, q10, [%[pack], #32]\n"
                        "stp q4, q12, [%[pack], #64]\n"
                        "stp q6, q14, [%[pack], #96]\n"

                        "zip1 v24.4s, v1.4s, v3.4s\n"
                        "zip2 v25.4s, v1.4s, v3.4s\n"
                        "zip1 v26.4s, v5.4s, v7.4s\n"
                        "zip2 v27.4s, v5.4s, v7.4s\n"

                        "zip1 v1.2d, v24.2d, v26.2d\n"
                        "zip2 v3.2d, v24.2d, v26.2d\n"
                        "zip1 v5.2d, v25.2d, v27.2d\n"
                        "zip2 v7.2d, v25.2d, v27.2d\n"

                        "zip1 v24.4s, v9.4s, v11.4s\n"
                        "zip2 v25.4s, v9.4s, v11.4s\n"
                        "zip1 v26.4s, v13.4s, v15.4s\n"
                        "zip2 v27.4s, v13.4s, v15.4s\n"

                        "zip1 v9.2d, v24.2d, v26.2d\n"
                        "zip2 v11.2d, v24.2d, v26.2d\n"
                        "zip1 v13.2d, v25.2d, v27.2d\n"
                        "zip2 v15.2d, v25.2d, v27.2d\n"

                        "stp q1, q9, [%[pack], #128]\n"
                        "stp q3, q11, [%[pack], #160]\n"
                        "stp q5, q13, [%[pack], #192]\n"
                        "stp q7, q15, [%[pack], #224]\n"
                        :
                        : [pack] "r"(itm), [in_0] "r"(Iw[0][i]), [in_1] "r"(Iw[1][i]),
                        [in_2] "r"(Iw[2][i]), [in_3] "r"(Iw[3][i]), [in_4] "r"(Iw[4][i]),
                        [in_5] "r"(Iw[5][i]), [in_6] "r"(Iw[6][i]), [in_7] "r"(Iw[7][i])
                        : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                        "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v24", "v25", "v26", "v27");
                }
            }
            for (I32 o = 0; o < I32(oc); o++) {
                const F32 *b_0 = biasArray + o * 8;
                const F32 *b_1 = b_0 + 4;
                // dot prod
                // (6*6)*C*c8*hw8 times O*(6*6)*C*c8*o8 = O*(6*6)*hw8*o8
                for (U32 idx = 0; idx < 36; idx++) {
                    F32 *itm_0 = itmArray + idx * 8 * ic * 8;
                    const F32 *f_o0c0 = filterArray + o * 8 * 36 * ic * 8 + idx * 8 * ic * 8;
                    F32 *out_o0hw0 = otmArray + idx * 8 * 8;
                    __asm__ __volatile__(
                        // give in address to x3
                        "mov x3, %[in_0]\n"

                        // give f address to x0
                        "mov x0, %[f_0]\n"

                        "mov  x2, %[ic]\n"

                        "eor  v5.16b, v5.16b, v5.16b\n"
                        "ldr  q1, [%[in_0]]\n"  // in_hw0
                        "eor  v6.16b, v6.16b, v6.16b\n"
                        "eor  v7.16b, v7.16b, v7.16b\n"
                        "eor  v8.16b, v8.16b, v8.16b\n"
                        "ldr q0, [%[f_0]]\n"  // f_o0c0
                        "eor  v9.16b, v9.16b, v9.16b\n"
                        "eor  v10.16b, v10.16b, v10.16b\n"
                        "eor  v11.16b, v11.16b, v11.16b\n"
                        "eor  v12.16b, v12.16b, v12.16b\n"
                        "eor  v13.16b, v13.16b, v13.16b\n"
                        "eor  v14.16b, v14.16b, v14.16b\n"
                        "eor  v15.16b, v15.16b, v15.16b\n"

                        "eor  v16.16b, v16.16b, v16.16b\n"
                        "eor  v17.16b, v17.16b, v17.16b\n"
                        "eor  v18.16b, v18.16b, v18.16b\n"
                        "eor  v19.16b, v19.16b, v19.16b\n"
                        "eor  v20.16b, v20.16b, v20.16b\n"
                        "0:\n"
                        "ldr q3, [x3, 16]!\n"
                        "ldr q29, [x0, 16]\n"
                        "fmla  v5.4s, v0.4s, v1.s[0]\n"
                        "fmla  v7.4s, v0.4s, v1.s[1]\n"
                        "fmla  v9.4s, v0.4s, v1.s[2]\n"
                        "fmla v11.4s, v0.4s, v1.s[3]\n"

                        "fmla v13.4s, v0.4s, v3.s[0]\n"
                        "fmla v15.4s, v0.4s, v3.s[1]\n"
                        "fmla v17.4s, v0.4s, v3.s[2]\n"
                        "fmla v19.4s, v0.4s, v3.s[3]\n"

                        "fmla  v6.4s, v29.4s, v1.s[0]\n"
                        "fmla  v8.4s, v29.4s, v1.s[1]\n"
                        "fmla v10.4s, v29.4s, v1.s[2]\n"
                        "fmla v12.4s, v29.4s, v1.s[3]\n"

                        "fmla v14.4s, v29.4s, v3.s[0]\n"
                        "fmla v16.4s, v29.4s, v3.s[1]\n"
                        "ldr q1, [x3, 16]!\n"
                        "ldr q0, [x0, 32]!\n"
                        "subs x2, x2, #1\n"
                        "fmla v18.4s, v29.4s, v3.s[2]\n"
                        "fmla v20.4s, v29.4s, v3.s[3]\n"
                        "bne 0b\n"
                        "str   q5, [%[out_0]]\n"
                        "str   q6, [%[out_0], #16]\n"
                        "str   q7, [%[out_0], #32]\n"
                        "str   q8, [%[out_0], #48]\n"
                        "str   q9, [%[out_0], #64]\n"
                        "str   q10, [%[out_0], #80]\n"
                        "str   q11, [%[out_0], #96]\n"
                        "str   q12, [%[out_0], #112]\n"
                        "str   q13, [%[out_0], #128]\n"
                        "str   q14, [%[out_0], #144]\n"
                        "str   q15, [%[out_0], #160]\n"
                        "str   q16, [%[out_0], #176]\n"
                        "str   q17, [%[out_0], #192]\n"
                        "str   q18, [%[out_0], #208]\n"
                        "str   q19, [%[out_0], #224]\n"
                        "str   q20, [%[out_0], #240]\n"
                        : [out_0] "+r"(out_o0hw0), [in_0] "+r"(itm_0), [f_0] "+r"(f_o0c0)
                        : [ic] "r"((I64)ic * 8)
                        : "memory", "cc", "v0", "v1", "v3", "v5", "v6", "v7", "v8", "v9", "v10",
                        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v27",
                        "v28", "v29", "x0", "x1", "x2", "x3");
                }
                // out trans
                // O*(6*6)*hw8*o8 => NOHWo8
                for (U32 hw8 = 0; hw8 < 8; hw8++) {
                    U32 h = (hw + hw8) / tile_w;
                    U32 w = (hw + hw8) % tile_w;
                    F32 *out_0 = outArray + n * oc * oh * ow * 8 + o * oh * ow * 8 +
                        h * 4 * ow * 8 + w * 4 * 8;

                    F32 *Ow_0[36];
                    F32 *Ow_1[36];
                    F32 *O_0[16];
                    F32 *O_1[16];
                    for (U32 idx = 0; idx < 36; idx++) {
                        Ow_0[idx] = otmArray + idx * 8 * 8 + hw8 * 8;
                        Ow_1[idx] = Ow_0[idx] + 4;
                    }
                    for (U32 i = 0; i < 4; ++i) {
                        for (U32 j = 0; j < 4; ++j) {
                            O_0[i * 4 + j] = out_0 + i * ow * 8 + j * 8;
                            O_1[i * 4 + j] = O_0[i * 4 + j] + 4;
                        }
                    }
                    CHECK_STATUS(trans_O_4x4_3x3(Ow_0, O_0, b_0, h, w, pad_h_mod_4, pad_w_mod_4,
                        tile_h - 1, tile_w - 1, activationDesc));
                    CHECK_STATUS(trans_O_4x4_3x3(Ow_1, O_1, b_1, h, w, pad_h_mod_4, pad_w_mod_4,
                        tile_h - 1, tile_w - 1, activationDesc));
                }
            }
            tiles_s += 8;
            tiles_tail -= 8;
        }

        if (tiles_tail >= 4) {
            I32 hw = tiles_s;
            // in trans
            // NCHWc8 => (6*6)*C*c8*hw4
            for (U32 c = 0; c < ic; c++) {
                F32 *inArray_pad_mov = inArray_pad + c * ih_pad * iw_pad * 8;
                F32 *itmArray_mov = itmArray + c * 8 * 4;
                F32 *Iw_ptr0[36];
                F32 *Iw_ptr1[36];
                F32 Iw[4][36][8];
                F32 *I0[4][36];
                F32 *I1[4][36];
                U32 h[4];
                U32 w[4];
                for (U32 index = 0; index < 4; index++) {
                    h[index] = ((hw + index) / tile_w) * 4;
                    w[index] = ((hw + index) % tile_w) * 4;
                }
                for (U32 i = 0; i < 6; i++) {
                    for (U32 j = 0; j < 6; j++) {
                        for (U32 index = 0; index < 4; index++) {
                            I0[index][i * 6 + j] =
                                inArray_pad_mov + (h[index] + i) * iw_pad * 8 + (w[index] + j) * 8;
                            I1[index][i * 6 + j] = inArray_pad_mov + (h[index] + i) * iw_pad * 8 +
                                (w[index] + j) * 8 + 4;
                        }
                    }
                }
                for (U32 index = 0; index < 4; index++) {
                    for (U32 i = 0; i < 36; i++) {
                        Iw_ptr0[i] = Iw[index][i];
                        Iw_ptr1[i] = Iw_ptr0[i] + 4;
                    }
                    trans_I_4x4_3x3(Iw_ptr0, I0[index]);
                    trans_I_4x4_3x3(Iw_ptr1, I1[index]);
                }
                for (U32 i = 0; i < 36; i++) {
                    F32 *itm = itmArray_mov + i * ic * 8 * 4;

                    __asm__ __volatile__(
                        "ldp q0, q4, [%[in_0]]\n"
                        "ldp q1, q5, [%[in_1]]\n"
                        "ldp q2, q6, [%[in_2]]\n"
                        "ldp q3, q7, [%[in_3]]\n"

                        "st4 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[pack]], #64\n"
                        "st4 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[pack]]\n"
                        :
                        : [pack] "r"(itm), [in_0] "r"(Iw[0][i]), [in_1] "r"(Iw[1][i]),
                        [in_2] "r"(Iw[2][i]), [in_3] "r"(Iw[3][i])
                        : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
                }
            }
            for (I32 o = 0; o < I32(oc); o++) {
                const F32 *b_0 = biasArray + o * 8;
                const F32 *b_1 = b_0 + 4;
                // dot prod
                // (6*6)*C*c8*hw4 times O*(6*6)*C*c8*o8 = O*(6*6)*hw4*o8
                for (U32 idx = 0; idx < 36; idx++) {
                    F32 *itm_0 = itmArray + idx * 4 * ic * 8;
                    const F32 *f_o0c0 = filterArray + o * 8 * 36 * ic * 8 + idx * 8 * ic * 8;
                    F32 *out_o0hw0 = otmArray + idx * 4 * 8;
                    __asm__ __volatile__(
                        // give in address to x3
                        "mov x3, %[in_0]\n"

                        // give f address to x0
                        "mov x0, %[f_0]\n"

                        "mov  x2, %[ic]\n"

                        "eor  v5.16b, v5.16b, v5.16b\n"
                        "ldr  q1, [%[in_0]]\n"  // in_hw0
                        "eor  v6.16b, v6.16b, v6.16b\n"
                        "eor  v7.16b, v7.16b, v7.16b\n"
                        "eor  v8.16b, v8.16b, v8.16b\n"
                        "ldr q0, [%[f_0]]\n"  // f_o0c0
                        "eor  v9.16b, v9.16b, v9.16b\n"
                        "eor  v10.16b, v10.16b, v10.16b\n"
                        "eor  v11.16b, v11.16b, v11.16b\n"
                        "eor  v12.16b, v12.16b, v12.16b\n"
                        "0:\n"
                        "ldr q3, [x3, 16]!\n"
                        "ldr q29, [x0, 16]\n"
                        "fmla  v5.4s, v0.4s, v1.s[0]\n"
                        "fmla  v7.4s, v0.4s, v1.s[1]\n"
                        "fmla  v9.4s, v0.4s, v1.s[2]\n"
                        "fmla v11.4s, v0.4s, v1.s[3]\n"

                        "fmla  v6.4s, v29.4s, v1.s[0]\n"
                        "fmla  v8.4s, v29.4s, v1.s[1]\n"
                        "ldr q0, [x0, 32]!\n"
                        "subs x2, x2, #1\n"
                        "fmla v10.4s, v29.4s, v1.s[2]\n"
                        "fmla v12.4s, v29.4s, v1.s[3]\n"

                        "mov	v1.16b, v3.16b\n"
                        "bne 0b\n"
                        "str   q5, [%[out_0]]\n"
                        "str   q6, [%[out_0], #16]\n"
                        "str   q7, [%[out_0], #32]\n"
                        "str   q8, [%[out_0], #48]\n"
                        "str   q9, [%[out_0], #64]\n"
                        "str   q10, [%[out_0], #80]\n"
                        "str   q11, [%[out_0], #96]\n"
                        "str   q12, [%[out_0], #112]\n"
                        : [out_0] "+r"(out_o0hw0), [in_0] "+r"(itm_0), [f_0] "+r"(f_o0c0)
                        : [ic] "r"((I64)ic * 8)
                        : "memory", "cc", "v0", "v1", "v3", "v5", "v6", "v7", "v8", "v9", "v10",
                        "v11", "v12", "v27", "v28", "v29", "x0", "x1", "x2", "x3");
                }
                // out trans
                // O*(6*6)*hw4*o8 => NOHWo8
                for (U32 hw4 = 0; hw4 < 4; hw4++) {
                    U32 h = (hw + hw4) / tile_w;
                    U32 w = (hw + hw4) % tile_w;
                    F32 *out_0 = outArray + n * oc * oh * ow * 8 + o * oh * ow * 8 +
                        h * 4 * ow * 8 + w * 4 * 8;

                    F32 *Ow_0[36];
                    F32 *Ow_1[36];
                    F32 *O_0[16];
                    F32 *O_1[16];
                    for (U32 idx = 0; idx < 36; idx++) {
                        Ow_0[idx] = otmArray + idx * 4 * 8 + hw4 * 8;
                        Ow_1[idx] = Ow_0[idx] + 4;
                    }
                    for (U32 i = 0; i < 4; ++i) {
                        for (U32 j = 0; j < 4; ++j) {
                            O_0[i * 4 + j] = out_0 + i * ow * 8 + j * 8;
                            O_1[i * 4 + j] = O_0[i * 4 + j] + 4;
                        }
                    }
                    CHECK_STATUS(trans_O_4x4_3x3(Ow_0, O_0, b_0, h, w, pad_h_mod_4, pad_w_mod_4,
                        tile_h - 1, tile_w - 1, activationDesc));
                    CHECK_STATUS(trans_O_4x4_3x3(Ow_1, O_1, b_1, h, w, pad_h_mod_4, pad_w_mod_4,
                        tile_h - 1, tile_w - 1, activationDesc));
                }
            }
            tiles_s += 4;
            tiles_tail -= 4;
        }

        for (I32 hw = tiles_s; hw < tiles; hw++) {
            // in trans
            // NCHWc8 => (6*6)*C*c8*hw1
            for (U32 c = 0; c < ic; c++) {
                F32 *inArray_pad_mov = inArray_pad + c * ih_pad * iw_pad * 8;
                F32 *itmArray_mov = itmArray + c * 8;
                F32 *Iw_ptr0[36];
                F32 *Iw_ptr1[36];
                F32 Iw[36][8];
                F32 *I0[36];
                F32 *I1[36];
                U32 h = (hw / tile_w) * 4;
                U32 w = (hw % tile_w) * 4;
                for (U32 i = 0; i < 6; i++) {
                    for (U32 j = 0; j < 6; j++) {
                        I0[i * 6 + j] = inArray_pad_mov + (h + i) * iw_pad * 8 + (w + j) * 8;
                        I1[i * 6 + j] = inArray_pad_mov + (h + i) * iw_pad * 8 + (w + j) * 8 + 4;
                    }
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr0[i] = Iw[i];
                    Iw_ptr1[i] = Iw_ptr0[i] + 4;
                }
                trans_I_4x4_3x3(Iw_ptr0, I0);
                trans_I_4x4_3x3(Iw_ptr1, I1);
                for (U32 i = 0; i < 36; i++) {
                    F32 *itm = itmArray_mov + i * ic * 8;
                    memcpy(itm, Iw[i], 8 * bytesOf(idt));
                }
            }
            for (I32 o = 0; o < I32(oc); o++) {
                const F32 *b_0 = biasArray + o * 8;
                const F32 *b_1 = b_0 + 4;
                // dot prod
                // (6*6)*C*c8*hw1 times O*(6*6)*C*c8*o8 = O*(6*6)*hw1*o8
                for (U32 idx = 0; idx < 36; idx++) {
                    F32 *itm_0 = itmArray + idx * ic * 8;
                    const F32 *f_o0c0 = filterArray + o * 8 * 36 * ic * 8 + idx * 8 * ic * 8;
                    F32 *out_o0hw0 = otmArray + idx * 8;
                    __asm__ __volatile__(
                        "ldr  s1, [%[in_0]]\n"     // in_hw0
                        "ldp q0, q29, [%[f_0]]\n"  // f_o0c0
                        // give in address to x3
                        "mov x3, %[in_0]\n"

                        // give f address to x0
                        "mov x0, %[f_0]\n"

                        "mov  x2, %[ic]\n"

                        "eor  v5.16b, v5.16b, v5.16b\n"
                        "eor  v6.16b, v6.16b, v6.16b\n"
                        "0:\n"
                        "ldp q30, q28, [x0, #32]\n"
                        "ldr s3, [x3, #4]\n"
                        "fmla  v5.4s, v0.4s, v1.s[0]\n"
                        "fmla  v6.4s, v29.4s, v1.s[0]\n"

                        "ldr q0, [x0, #64]!\n"
                        "subs x2, x2, #2\n"
                        "ldr q29, [x0, #16]\n"
                        "ldr s1, [x3, #8]!\n"
                        "fmla  v5.4s, v30.4s, v3.s[0]\n"
                        "fmla  v6.4s, v28.4s, v3.s[0]\n"
                        "bne 0b\n"
                        "str   q5, [%[out_0]]\n"
                        "str   q6, [%[out_0], #16]\n"
                        : [out_0] "+r"(out_o0hw0), [in_0] "+r"(itm_0), [f_0] "+r"(f_o0c0)
                        : [ic] "r"((I64)ic * 8)
                        : "memory", "cc", "v0", "v1", "v3", "v5", "v6", "v28", "v29", "v30", "x0",
                        "x1", "x2", "x3");
                }
                // out trans
                // O*(6*6)*hw1*o8 => NOHWo8
                U32 h = hw / tile_w;
                U32 w = hw % tile_w;
                F32 *out_0 =
                    outArray + n * oc * oh * ow * 8 + o * oh * ow * 8 + h * 4 * ow * 8 + w * 4 * 8;

                F32 *Ow_0[36];
                F32 *Ow_1[36];
                F32 *O_0[16];
                F32 *O_1[16];
                for (U32 idx = 0; idx < 36; idx++) {
                    Ow_0[idx] = otmArray + idx * 8;
                    Ow_1[idx] = Ow_0[idx] + 4;
                }
                for (U32 i = 0; i < 4; ++i) {
                    for (U32 j = 0; j < 4; ++j) {
                        O_0[i * 4 + j] = out_0 + i * ow * 8 + j * 8;
                        O_1[i * 4 + j] = O_0[i * 4 + j] + 4;
                    }
                }
                CHECK_STATUS(trans_O_4x4_3x3(Ow_0, O_0, b_0, h, w, pad_h_mod_4, pad_w_mod_4,
                    tile_h - 1, tile_w - 1, activationDesc));
                CHECK_STATUS(trans_O_4x4_3x3(Ow_1, O_1, b_1, h, w, pad_h_mod_4, pad_w_mod_4,
                    tile_h - 1, tile_w - 1, activationDesc));
            }
        }
    }
    return ret;
}
