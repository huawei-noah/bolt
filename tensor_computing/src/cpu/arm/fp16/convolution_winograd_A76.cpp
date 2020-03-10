// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "cpu/arm/fp16/convolution_winograd_transform.h"
#include "cpu/arm/fp16/convolution_winograd.h"

EE convolution_winograd_A76(TensorDesc inputDesc, F16* inArray,
    TensorDesc filterDesc, const F16* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const F16* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F16* outArray,
    ActivationMode activationMode)
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
    U32 paddingT = convDesc.padding_top;
    U32 paddingB = convDesc.padding_bottom;
    U32 paddingL = convDesc.padding_left;
    U32 paddingR = convDesc.padding_right;

    if (fdf != DF_HWNCN16)
        CHECK_STATUS(NOT_MATCH);
    if (!(fh == 6 && fw == 6))
        CHECK_STATUS(NOT_SUPPORTED);

    oc /= 8;
    ic /= 8;

    U32 tile_h = (oh + 3) / 4;
    U32 tile_w = (ow + 3) / 4;
    // num of 6x6 tiles
    I32 tiles = tile_h * tile_w;
    U32 pad_left = paddingL;
    U32 pad_right = paddingR + (tile_w*4 - ow);
    U32 pad_w_mod_4 = tile_w*4 - ow;
    U32 pad_top = paddingT;
    U32 pad_bottom = paddingB + (tile_h*4 - oh);
    U32 pad_h_mod_4 = tile_h*4 - oh;
    U32 ih_pad = ih + pad_top + pad_bottom;
    U32 iw_pad = iw + pad_left + pad_right;
    // tmp = in_pad + itm + otm
    // in_pad: ic*ih_pad*iw_pad*8
    // itm: 6*6*ic*8*8
    // otm: oc*6*6*8*8
    F16* inArray_pad = (F16*)tmp;
    F16* itmArray = inArray_pad + ic*ih_pad*iw_pad*8;
    F16* otmArray = itmArray + 6*6*ic*8*8;

    EE ret = SUCCESS;
    // copy input into a input with padding
    for (U32 n = 0; n < in; n++) {
        F16 *inArray_pad_mov = inArray_pad;
        F16 *inArray_mov = inArray + n*ic*ih*iw*8;
        for (U32 c = 0; c < ic; c++) {
            memset(inArray_pad_mov, 0, pad_top*iw_pad*8*bytesOf(idt));
            inArray_pad_mov += pad_top*iw_pad*8;
            for (U32 h = pad_top; h < ih_pad - pad_bottom; h++) {
                memset(inArray_pad_mov, 0, pad_left*8*bytesOf(idt));
                inArray_pad_mov += pad_left*8;
                memcpy(inArray_pad_mov, inArray_mov, iw*8*bytesOf(idt));
                inArray_pad_mov += iw*8;
                inArray_mov += iw*8;
                memset(inArray_pad_mov, 0, pad_right*8*bytesOf(idt));
                inArray_pad_mov += pad_right*8;
            }
            memset(inArray_pad_mov, 0, pad_bottom*iw_pad*8*bytesOf(idt));
            inArray_pad_mov += pad_bottom*iw_pad*8;
        }

        // tiles / 8
        for (I32 hw = 0; hw < tiles-7; hw+=8) {
            const F16 *ftm_0 = filterArray;
            F16 *otm_0 = otmArray;
            // in trans
            // NCHWc8 => (6*6)*C*c8*hw8
            for (U32 c = 0; c < ic; c++) {
                F16 *inArray_pad_mov = inArray_pad + c*ih_pad*iw_pad*8;
                F16 *itmArray_mov = itmArray + c*8*8;
                F16 *Iw_ptr[36];
                F16 Iw[8][36][8];
                F16 *I[8][36];
                U32 h[8];
                U32 w[8];
                for (U32 index = 0; index < 8; index++) {
                   h[index] = ((hw + index) / tile_w) * 4;
                   w[index] = ((hw + index) % tile_w) * 4;
                }
                for (U32 i = 0; i < 6; i++) {
                    for (U32 j = 0; j < 6; j++) {
                        for (U32 index = 0; index < 8; index++) {
                            I[index][i*6 + j] = inArray_pad_mov + (h[index] + i) * iw_pad * 8 + (w[index] + j) * 8;
                        }
                    }
                }
                for (U32 index = 0; index < 8; index++) {
                    for (U32 i = 0; i < 36; i++) {
                        Iw_ptr[i] = Iw[index][i];
                    }
                    trans_I_4x4_3x3(Iw_ptr, I[index]);
                }
                for (U32 i = 0; i < 36; i++) {
                    F16* itm = itmArray_mov + i*ic*8*8;
                    float16x8_t v0 = vld1q_f16(Iw[0][i]);
                    float16x8_t v1 = vld1q_f16(Iw[1][i]);
                    float16x8_t v2 = vld1q_f16(Iw[2][i]);
                    float16x8_t v3 = vld1q_f16(Iw[3][i]);
                    float16x8_t v4 = vld1q_f16(Iw[4][i]);
                    float16x8_t v5 = vld1q_f16(Iw[5][i]);
                    float16x8_t v6 = vld1q_f16(Iw[6][i]);
                    float16x8_t v7 = vld1q_f16(Iw[7][i]);
                    vst1q_f16(itm,
                        vzip1q_f16(
                            vzip1q_f16(vzip1q_f16(v0, v4), vzip1q_f16(v2, v6)),
                            vzip1q_f16(vzip1q_f16(v1, v5), vzip1q_f16(v3, v7))));
                    vst1q_f16(itm + 8,
                        vzip2q_f16(
                            vzip1q_f16(vzip1q_f16(v0, v4), vzip1q_f16(v2, v6)),
                            vzip1q_f16(vzip1q_f16(v1, v5), vzip1q_f16(v3, v7))));
                    vst1q_f16(itm + 8*2,
                        vzip1q_f16(
                            vzip2q_f16(vzip1q_f16(v0, v4), vzip1q_f16(v2, v6)),
                            vzip2q_f16(vzip1q_f16(v1, v5), vzip1q_f16(v3, v7))));
                    vst1q_f16(itm + 8*3,
                        vzip2q_f16(
                            vzip2q_f16(vzip1q_f16(v0, v4), vzip1q_f16(v2, v6)),
                            vzip2q_f16(vzip1q_f16(v1, v5), vzip1q_f16(v3, v7))));
                    vst1q_f16(itm + 8*4,
                        vzip1q_f16(
                            vzip1q_f16(vzip2q_f16(v0, v4), vzip2q_f16(v2, v6)),
                            vzip1q_f16(vzip2q_f16(v1, v5), vzip2q_f16(v3, v7))));
                    vst1q_f16(itm + 8*5,
                        vzip2q_f16(
                            vzip1q_f16(vzip2q_f16(v0, v4), vzip2q_f16(v2, v6)),
                            vzip1q_f16(vzip2q_f16(v1, v5), vzip2q_f16(v3, v7))));
                    vst1q_f16(itm + 8*6,
                        vzip1q_f16(
                            vzip2q_f16(vzip2q_f16(v0, v4), vzip2q_f16(v2, v6)),
                            vzip2q_f16(vzip2q_f16(v1, v5), vzip2q_f16(v3, v7))));
                    vst1q_f16(itm + 8*7,
                        vzip2q_f16(
                            vzip2q_f16(vzip2q_f16(v0, v4), vzip2q_f16(v2, v6)),
                            vzip2q_f16(vzip2q_f16(v1, v5), vzip2q_f16(v3, v7))));
                }
            }
            for (I32 o = 0; o < I32(oc-1); o+=2) {
                const F16 *b_0 = biasArray + o*8;
                const F16 *b_1 = b_0 + 8;
                F16 *itm_0 = itmArray;
                // dot prod
                // (6*6)*C*c8*hw8 times O*(6*6)*C*c8*o16 = O*(6*6)*hw8*o16
                for (U32 idx = 0; idx < 36; idx++) {
                    __asm__ __volatile__(
                        "mov  x0, %[ic]\n"             //ic_blk
                        "eor  v2.16b,  v2.16b,  v2.16b\n"      //out_o0hw0
                        "ldr  q0, [%[in]]\n"           //in_hw0
                        "eor  v4.16b,  v4.16b,  v4.16b\n"      //out_o0hw1
                        "eor  v6.16b,  v6.16b,  v6.16b\n"      //out_o0hw2
                        "eor  v8.16b,  v8.16b,  v8.16b\n"      //out_o0hw3
                        "ldr q18, [%[f]]\n"            //f_o0c0
                        "eor v10.16b, v10.16b, v10.16b\n"      //out_o0hw4
                        "eor v12.16b, v12.16b, v12.16b\n"      //out_o0hw5
                        "eor v14.16b, v14.16b, v14.16b\n"      //out_o0hw6
                        "ldr q19, [%[f], #16]\n"            //f_o1c0
                        "eor v16.16b, v16.16b, v16.16b\n"      //out_o0hw7
                        "eor  v3.16b,  v3.16b,  v3.16b\n"      //out_o1hw0
                        "eor  v5.16b,  v5.16b,  v5.16b\n"      //out_o1hw1
                        "eor  v7.16b,  v7.16b,  v7.16b\n"      //out_o1hw2
                        "eor  v9.16b,  v9.16b,  v9.16b\n"      //out_o1hw3
                        "eor v11.16b, v11.16b, v11.16b\n"      //out_o1hw4
                        "eor v13.16b, v13.16b, v13.16b\n"      //out_o1hw5
                        "eor v15.16b, v15.16b, v15.16b\n"      //out_o1hw6
                        "eor v17.16b, v17.16b, v17.16b\n"      //out_o1hw7
                        "0:\n"
                        "ldr  q1, [%[in], #16]\n"           //in_hw0
                        "fmla  v2.8h, v18.8h, v0.h[0]\n"
                        "fmla  v4.8h, v18.8h, v0.h[1]\n"
                        "ldr q20, [%[f], #32]\n"            //f_o0c0
                        "fmla  v6.8h, v18.8h, v0.h[2]\n"
                        "fmla  v8.8h, v18.8h, v0.h[3]\n"
                        "ldr q21, [%[f], #48]\n"            //f_o1c0
                        "fmla v10.8h, v18.8h, v0.h[4]\n"
                        "fmla v12.8h, v18.8h, v0.h[5]\n"
                        "fmla v14.8h, v18.8h, v0.h[6]\n"
                        "fmla v16.8h, v18.8h, v0.h[7]\n"
                        "fmla  v3.8h, v19.8h, v0.h[0]\n"
                        "fmla  v5.8h, v19.8h, v0.h[1]\n"
                        "fmla  v7.8h, v19.8h, v0.h[2]\n"
                        "fmla v9.8h, v19.8h, v0.h[3]\n"
                        "fmla v11.8h, v19.8h, v0.h[4]\n"
                        "fmla v13.8h, v19.8h, v0.h[5]\n"
                        "fmla v15.8h, v19.8h, v0.h[6]\n"
                        "fmla v17.8h, v19.8h, v0.h[7]\n"

                        "ldr  q0, [%[in], #32]\n"           //in_hw0
                        "fmla  v2.8h, v20.8h, v1.h[0]\n"
                        "fmla  v4.8h, v20.8h, v1.h[1]\n"
                        "ldr q18, [%[f], #64]\n"            //f_o0c0
                        "fmla  v6.8h, v20.8h, v1.h[2]\n"
                        "fmla  v8.8h, v20.8h, v1.h[3]\n"
                        "ldr q19, [%[f], #80]\n"            //f_o1c0
                        "fmla v10.8h, v20.8h, v1.h[4]\n"
                        "fmla v12.8h, v20.8h, v1.h[5]\n"
                        "fmla v14.8h, v20.8h, v1.h[6]\n"
                        "fmla v16.8h, v20.8h, v1.h[7]\n"
                        "fmla  v3.8h, v21.8h, v1.h[0]\n"
                        "add %[in], %[in], #32\n"
                        "fmla  v5.8h, v21.8h, v1.h[1]\n"
                        "add %[f], %[f], #64\n"
                        "fmla  v7.8h, v21.8h, v1.h[2]\n"
                        "subs x0, x0, #2\n"
                        "fmla  v9.8h, v21.8h, v1.h[3]\n"
                        "fmla v11.8h, v21.8h, v1.h[4]\n"
                        "fmla v13.8h, v21.8h, v1.h[5]\n"
                        "fmla v15.8h, v21.8h, v1.h[6]\n"
                        "fmla v17.8h, v21.8h, v1.h[7]\n"
                        "bne 0b\n"
                        "st1 { v2.8h,  v3.8h,  v4.8h,  v5.8h}, [%[out]], #64\n"
                        "st1 { v6.8h,  v7.8h,  v8.8h,  v9.8h}, [%[out]], #64\n"
                        "st1 {v10.8h, v11.8h, v12.8h, v13.8h}, [%[out]], #64\n"
                        "st1 {v14.8h, v15.8h, v16.8h, v17.8h}, [%[out]], #64\n"
                        :[out]"+r"(otm_0),
                         [in]"+r"(itm_0),
                         [f]"+r"(ftm_0)
                        :[ic]"r"((I64)ic*8)
                        :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                            "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "x0"
                    );
                }
                // out trans
                // O*(6*6)*hw8*o16 => NOHWo8
                for (U32 hw8 = 0; hw8 < 8; hw8++) {
                    U32 h = (hw+hw8) / tile_w;
                    U32 w = (hw+hw8) % tile_w;
                    F16 *out_0 = outArray + n*oc*oh*ow*8 + o*oh*ow*8 + h*4*ow*8 + w*4*8;
                    F16 *out_1 = out_0 + oh*ow*8;
                    U32 otm_off_0 = o*8*36*8 + hw8*16;
                    U32 otm_off_1 = otm_off_0 + 8;

                    F16 *Ow_0[36];
                    F16 *Ow_1[36];
                    F16 *O_0[16];
                    F16 *O_1[16];
                    for (U32 idx = 0; idx < 36; idx++) {
                        Ow_0[idx] = otmArray + otm_off_0 + idx*8*16;
                        Ow_1[idx] = otmArray + otm_off_1 + idx*8*16;
                    }
                    for (U32 i = 0; i < 4; ++i) {
                        for (U32 j = 0; j < 4; ++j) {
                            O_0[i*4 + j] = out_0 + i*ow*8 + j*8;
                            O_1[i*4 + j] = out_1 + i*ow*8 + j*8;
                        }
                    }
                    CHECK_STATUS(trans_O_4x4_3x3(Ow_0, O_0, b_0, h, w, pad_h_mod_4, pad_w_mod_4, tile_h-1, tile_w-1, activationMode));
                    CHECK_STATUS(trans_O_4x4_3x3(Ow_1, O_1, b_1, h, w, pad_h_mod_4, pad_w_mod_4, tile_h-1, tile_w-1, activationMode));
                }
            }
            if (oc & 1) {
                F16 *itm_0 = itmArray;
                const F16 *ftm_0 = filterArray + (oc-1)*36*ic*8*8;
                F16 *otm_0 = otmArray + (oc-1)*36*8*8;
                const F16 *b_0 = biasArray + (oc-1)*8;
                // dot prod
                // (6*6)*C*c8*hw8 times O*(6*6)*C*c8*o8 = O*(6*6)*hw8*o8
                for (U32 idx = 0; idx < 36; idx++) {
                    __asm__ __volatile__(
                        "mov x0, %[ic]\n"             //ic_blk
                        "eor v2.16b, v2.16b, v2.16b\n"      //out_o0hw0
                        "ldr q0, [%[in]]\n"           //in_hw0
                        "eor v3.16b, v3.16b, v3.16b\n"      //out_o0hw1
                        "eor v4.16b, v4.16b, v4.16b\n"      //out_o0hw2
                        "eor v5.16b, v5.16b, v5.16b\n"      //out_o0hw3
                        "ldr q18, [%[f]]\n"            //f_o0c0
                        "eor v6.16b, v6.16b, v6.16b\n"      //out_o0hw4
                        "eor v7.16b, v7.16b, v7.16b\n"      //out_o0hw5
                        "eor v8.16b, v8.16b, v8.16b\n"      //out_o0hw6
                        "eor v9.16b, v9.16b, v9.16b\n"      //out_o0hw7
                        "0:\n"
                        "ldr  q1, [%[in], #16]\n"           //in_hw0
                        "fmla  v2.8h, v18.8h, v0.h[0]\n"
                        "fmla  v3.8h, v18.8h, v0.h[1]\n"
                        "fmla  v4.8h, v18.8h, v0.h[2]\n"
                        "ldr q20, [%[f], #16]\n"            //f_o0c0
                        "fmla  v5.8h, v18.8h, v0.h[3]\n"
                        "fmla  v6.8h, v18.8h, v0.h[4]\n"
                        "fmla  v7.8h, v18.8h, v0.h[5]\n"
                        "fmla  v8.8h, v18.8h, v0.h[6]\n"
                        "subs x0, x0, #2\n"
                        "fmla  v9.8h, v18.8h, v0.h[7]\n"

                        "ldr  q0, [%[in], #32]\n"           //in_hw0
                        "fmla  v2.8h, v20.8h, v1.h[0]\n"
                        "fmla  v3.8h, v20.8h, v1.h[1]\n"
                        "fmla  v4.8h, v20.8h, v1.h[2]\n"
                        "ldr q18, [%[f], #32]\n"            //f_o0c0
                        "fmla  v5.8h, v20.8h, v1.h[3]\n"
                        "fmla  v6.8h, v20.8h, v1.h[4]\n"
                        "fmla  v7.8h, v20.8h, v1.h[5]\n"
                        "add %[in], %[in], #32\n"
                        "fmla  v8.8h, v20.8h, v1.h[6]\n"
                        "add %[f], %[f], #32\n"
                        "fmla  v9.8h, v20.8h, v1.h[7]\n"
                        "bne 0b\n"
                        "st1 {v2.8h, v3.8h, v4.8h, v5.8h}, [%[out]], #64\n"
                        "st1 {v6.8h, v7.8h, v8.8h, v9.8h}, [%[out]], #64\n"
                        :[out]"+r"(otm_0),
                         [in]"+r"(itm_0),
                         [f]"+r"(ftm_0)
                        :[ic]"r"((I64)ic*8)
                        :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v18", "v20", "x0");
                }
                // out trans
                // O*(6*6)*hw8*o8 => NOWHo8
                for (U32 hw8 = 0; hw8 < 8; hw8++) {
                    U32 h = (hw+hw8) / tile_w;
                    U32 w = (hw+hw8) % tile_w;
                    F16 *out_0 = outArray + n*oc*oh*ow*8 + (oc-1)*oh*ow*8 + h*4*ow*8 + w*4*8;
                    U32 otm_off_0 = (oc-1)*8*36*8 + hw8*8;

                    F16 *Ow_0[36];
                    F16 *O_0[16];
                    for (U32 idx = 0; idx < 36; idx++) {
                        Ow_0[idx] = otmArray + otm_off_0 + idx*8*8;
                    }
                    for (U32 i = 0; i < 4; ++i) {
                        for (U32 j = 0; j < 4; ++j) {
                            O_0[i*4 + j] = out_0 + i*ow*8 + j*8;
                        }
                    }
                    CHECK_STATUS(trans_O_4x4_3x3(Ow_0, O_0, b_0, h, w, pad_h_mod_4, pad_w_mod_4, tile_h-1, tile_w-1, activationMode));
                }
            }
        }

        // tiles_reminder % 8 / 4
        I32 tiles_s = (tiles / 8) * 8;
        for (I32 hw = tiles_s; hw < tiles-3; hw+=4) {
            const F16 *ftm_0 = filterArray;
            F16 *otm_0 = otmArray;
            // in trans
            // NCHWc8 => (6*6)*C*c8*hw4
            for (U32 c = 0; c < ic; c++) {
                F16 *inArray_pad_mov = inArray_pad + c*ih_pad*iw_pad*8;
                F16 *Iw_ptr[36];
                F16 Iw0[36][8];
                F16 *I0[36];
                F16 Iw1[36][8];
                F16 *I1[36];
                F16 Iw2[36][8];
                F16 *I2[36];
                F16 Iw3[36][8];
                F16 *I3[36];
                F16 *itmArray_mov = itmArray + c*8*4;
                U32 h0 = (hw/tile_w)*4;
                U32 w0 = (hw%tile_w)*4;
                U32 h1 = ((hw+1)/tile_w)*4;
                U32 w1 = ((hw+1)%tile_w)*4;
                U32 h2 = ((hw+2)/tile_w)*4;
                U32 w2 = ((hw+2)%tile_w)*4;
                U32 h3 = ((hw+3)/tile_w)*4;
                U32 w3 = ((hw+3)%tile_w)*4;
                for (U32 i = 0; i < 6; i++) {
                    for (U32 j = 0; j < 6; j++) {
                        I0[i*6 + j] = inArray_pad_mov + (h0+i)*iw_pad*8 + (w0+j)*8;
                        I1[i*6 + j] = inArray_pad_mov + (h1+i)*iw_pad*8 + (w1+j)*8;
                        I2[i*6 + j] = inArray_pad_mov + (h2+i)*iw_pad*8 + (w2+j)*8;
                        I3[i*6 + j] = inArray_pad_mov + (h3+i)*iw_pad*8 + (w3+j)*8;
                    }
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw0[i];
                }
                trans_I_4x4_3x3(Iw_ptr, I0);
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw1[i];
                }
                trans_I_4x4_3x3(Iw_ptr, I1);
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw2[i];
                }
                trans_I_4x4_3x3(Iw_ptr, I2);
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw3[i];
                }
                trans_I_4x4_3x3(Iw_ptr, I3);
                for (U32 i = 0; i < 36; i++) {
                    F16* itm = itmArray_mov + i*ic*8*4;

                    // for (U32 c8 = 0; c8 < 8; c8++) {
                    //     itm[c8*4] = Iw0[i][c8];
                    //     itm[c8*4 + 1] = Iw1[i][c8];
                    //     itm[c8*4 + 2] = Iw2[i][c8];
                    //     itm[c8*4 + 3] = Iw3[i][c8];
                    // }

                    __asm__ __volatile__(
                        "ldr q0, [%[in_0]]\n"
                        "ldr q1, [%[in_1]]\n"
                        "ldr q2, [%[in_2]]\n"
                        "ldr q3, [%[in_3]]\n"
                        "st4 {v0.8h, v1.8h, v2.8h, v3.8h}, [%[itm]]\n"
                        :[itm]"+r"(itm)
                        :[in_0]"r"(Iw0[i]),
                         [in_1]"r"(Iw1[i]),
                         [in_2]"r"(Iw2[i]),
                         [in_3]"r"(Iw3[i])
                        :"memory", "cc", "v0", "v1", "v2", "v3"
                    );
                }
            }
            for (I32 o = 0; o < I32(oc-1); o+=2) {
                const F16 *b_0 = biasArray + o*8;
                const F16 *b_1 = b_0 + 8;
                F16 *itm_0 = itmArray;
                // dot prod
                // (6*6)*C*c8*hw4 times O*(6*6)*C*c8*o16 = O*(6*6)*hw4*o16
                for (U32 idx = 0; idx < 36; idx++) {
                    __asm__ __volatile__(
                        "mov  x0, %[ic]\n"                  //ic_blk
                        "eor  v2.16b,  v2.16b,  v2.16b\n"   //out_o0hw0
                        "ldr  d0, [%[in]]\n"                //in_hw0
                        "eor  v4.16b,  v4.16b,  v4.16b\n"   //out_o0hw1
                        "ldr q18, [%[f]]\n"                 //f_o0c0
                        "eor  v6.16b,  v6.16b,  v6.16b\n"   //out_o0hw2
                        "eor  v8.16b,  v8.16b,  v8.16b\n"   //out_o0hw3
                        "ldr q19, [%[f], #16]\n"            //f_o1c0
                        "eor  v3.16b,  v3.16b,  v3.16b\n"   //out_o1hw0
                        "eor  v5.16b,  v5.16b,  v5.16b\n"   //out_o1hw1
                        "eor  v7.16b,  v7.16b,  v7.16b\n"   //out_o1hw2
                        "eor  v9.16b,  v9.16b,  v9.16b\n"   //out_o1hw3
                        "0:\n"
                        "ldr  d1, [%[in], #8]\n"            //in_hw0
                        "fmla  v2.8h, v18.8h, v0.h[0]\n"
                        "ldr q20, [%[f], #32]\n"            //f_o0c0
                        "fmla  v4.8h, v18.8h, v0.h[1]\n"
                        "fmla  v6.8h, v18.8h, v0.h[2]\n"
                        "fmla  v8.8h, v18.8h, v0.h[3]\n"
                        "ldr q21, [%[f], #48]\n"            //f_o1c0
                        "fmla  v3.8h, v19.8h, v0.h[0]\n"
                        "fmla  v5.8h, v19.8h, v0.h[1]\n"
                        "fmla  v7.8h, v19.8h, v0.h[2]\n"
                        "subs x0, x0, #2\n"
                        "fmla v9.8h, v19.8h, v0.h[3]\n"

                        "ldr  d0, [%[in], #16]\n"           //in_hw0
                        "fmla  v2.8h, v20.8h, v1.h[0]\n"
                        "ldr q18, [%[f], #64]\n"            //f_o0c0
                        "fmla  v4.8h, v20.8h, v1.h[1]\n"
                        "fmla  v6.8h, v20.8h, v1.h[2]\n"
                        "fmla  v8.8h, v20.8h, v1.h[3]\n"
                        "ldr q19, [%[f], #80]\n"            //f_o1c0
                        "fmla  v3.8h, v21.8h, v1.h[0]\n"
                        "fmla  v5.8h, v21.8h, v1.h[1]\n"
                        "fmla  v7.8h, v21.8h, v1.h[2]\n"
                        "add %[in], %[in], #16\n"
                        "fmla  v9.8h, v21.8h, v1.h[3]\n"
                        "add %[f], %[f], #64\n"
                        "bne 0b\n"
                        "st1 { v2.8h,  v3.8h,  v4.8h,  v5.8h}, [%[out]], #64\n"
                        "st1 { v6.8h,  v7.8h,  v8.8h,  v9.8h}, [%[out]], #64\n"
                        :[out]"+r"(otm_0),
                         [in]"+r"(itm_0),
                         [f]"+r"(ftm_0)
                        :[ic]"r"((I64)ic*8)
                        :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v18", "v19", "v20", "v21", "x0"
                    );
                }
                // out trans
                // O*(6*6)*hw4*o16 => NOWHo8
                for (U32 hw4 = 0; hw4 < 4; hw4++) {
                    U32 h = (hw+hw4) / tile_w;
                    U32 w = (hw+hw4) % tile_w;
                    F16 *out_0 = outArray + n*oc*oh*ow*8 + o*oh*ow*8 + h*4*ow*8 + w*4*8;
                    F16 *out_1 = out_0 + oh*ow*8;
                    U32 otm_off_0 = o*8*36*4 + hw4*16;
                    U32 otm_off_1 = otm_off_0 + 8;

                    F16 *Ow_0[36];
                    F16 *Ow_1[36];
                    F16 *O_0[16];
                    F16 *O_1[16];
                    for (U32 idx = 0; idx < 36; idx++) {
                        Ow_0[idx] = otmArray + otm_off_0 + idx*4*16;
                        Ow_1[idx] = otmArray + otm_off_1 + idx*4*16;
                    }
                    for (U32 i = 0; i < 4; ++i) {
                        for (U32 j = 0; j < 4; ++j) {
                            O_0[i*4 + j] = out_0 + i*ow*8 + j*8;
                            O_1[i*4 + j] = out_1 + i*ow*8 + j*8;
                        }
                    }
                    CHECK_STATUS(trans_O_4x4_3x3(Ow_0, O_0, b_0, h, w, pad_h_mod_4, pad_w_mod_4, tile_h-1, tile_w-1, activationMode));
                    CHECK_STATUS(trans_O_4x4_3x3(Ow_1, O_1, b_1, h, w, pad_h_mod_4, pad_w_mod_4, tile_h-1, tile_w-1, activationMode));
                }
            }
            if (oc & 1) {
                F16 *itm_0 = itmArray;
                const F16 *ftm_0 = filterArray + (oc-1)*8*36*ic*8;
                F16 *otm_0 = otmArray + (oc-1)*8*36*4;
                const F16 *b_0 = biasArray + (oc-1)*8;
                // dot prod
                // (6*6)*C*c8*hw4 times O*(6*6)*C*c8*o8 = O*(6*6)*hw4*o8
                for (U32 idx = 0; idx < 36; idx++) {
                    __asm__ __volatile__(
                        "mov x0, %[ic]\n"             //ic_blk
                        "eor v2.16b, v2.16b, v2.16b\n"      //out_o0hw0
                        "ldr d0, [%[in]]\n"           //in_hw0
                        "eor v3.16b, v3.16b, v3.16b\n"      //out_o0hw1
                        "ldr q18, [%[f]]\n"            //f_o0c0
                        "eor v4.16b, v4.16b, v4.16b\n"      //out_o0hw2
                        "eor v5.16b, v5.16b, v5.16b\n"      //out_o0hw3
                        "0:\n"
                        "ldr  d1, [%[in], #8]\n"           //in_hw0
                        "fmla  v2.8h, v18.8h, v0.h[0]\n"
                        "ldr q20, [%[f], #16]\n"            //f_o0c0
                        "fmla  v3.8h, v18.8h, v0.h[1]\n"
                        "fmla  v4.8h, v18.8h, v0.h[2]\n"
                        "fmla  v5.8h, v18.8h, v0.h[3]\n"
                        "subs x0, x0, #2\n"

                        "ldr  d0, [%[in], #16]\n"           //in_hw0
                        "fmla  v2.8h, v20.8h, v1.h[0]\n"
                        "ldr q18, [%[f], #32]\n"            //f_o0c0
                        "fmla  v3.8h, v20.8h, v1.h[1]\n"
                        "fmla  v4.8h, v20.8h, v1.h[2]\n"
                        "fmla  v5.8h, v20.8h, v1.h[3]\n"
                        "add %[in], %[in], #16\n"
                        "add %[f], %[f], #32\n"
                        "bne 0b\n"
                        "st1 {v2.8h, v3.8h, v4.8h, v5.8h}, [%[out]], #64\n"
                        :[out]"+r"(otm_0),
                         [in]"+r"(itm_0),
                         [f]"+r"(ftm_0)
                        :[ic]"r"((I64)ic*8)
                        :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v18", "v20", "x0"
                    );
                }
                // out trans
                // O*(6*6)*hw4*o8 => NOWHo8
                for (U32 hw4 = 0; hw4 < 4; hw4++) {
                    U32 h = (hw+hw4) / tile_w;
                    U32 w = (hw+hw4) % tile_w;
                    F16 *out_0 = outArray + n*oc*oh*ow*8 + (oc-1)*oh*ow*8 + h*4*ow*8 + w*4*8;
                    U32 otm_off_0 = (oc-1)*8*36*4 + hw4*8;

                    F16 *Ow_0[36];
                    F16 *O_0[16];
                    for (U32 idx = 0; idx < 36; idx++) {
                        Ow_0[idx] = otmArray + otm_off_0 + idx*4*8;
                    }
                    for (U32 i = 0; i < 4; ++i) {
                        for (U32 j = 0; j < 4; ++j) {
                            O_0[i*4 + j] = out_0 + i*ow*8 + j*8;
                        }
                    }
                    CHECK_STATUS(trans_O_4x4_3x3(Ow_0, O_0, b_0, h, w, pad_h_mod_4, pad_w_mod_4, tile_h-1, tile_w-1, activationMode));
                }
            }
        }

        // tiles_reminder % 4
        tiles_s = (tiles / 4) * 4;
        for (I32 hw = tiles_s; hw < tiles; hw++) {
            const F16 *ftm_0 = filterArray;
            F16 *otm_0 = otmArray;
            // in trans
            // NCHWc8 => (6*6)*C*c8*hw1
            for (U32 c = 0; c < ic; c++) {
                F16 *inArray_pad_mov = inArray_pad + c*ih_pad*iw_pad*8;
                F16 *Iw_ptr[36];
                F16 Iw0[36][8];
                F16 *I0[36];
                F16 *itmArray_mov = itmArray + c*8;
                U32 h0 = (hw/tile_w)*4;
                U32 w0 = (hw%tile_w)*4;
                for (U32 i = 0; i < 6; i++) {
                    for (U32 j = 0; j < 6; j++) {
                        I0[i*6 + j] = inArray_pad_mov + (h0+i)*iw_pad*8 + (w0+j)*8;
                    }
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw0[i];
                }
                trans_I_4x4_3x3(Iw_ptr, I0);
                for (U32 i = 0; i < 36; i++) {
                    F16* itm = itmArray_mov + i*ic*8;

                    // for (U32 c8 = 0; c8 < 8; c8++) {
                    //     itm[c8] = Iw0[i][c8];
                    // }
                    memcpy(itm, Iw0[i], 8*bytesOf(idt));
                }
            }
            for (I32 o = 0; o < I32(oc-1); o+=2) {
                const F16 *b_0 = biasArray + o*8;
                const F16 *b_1 = b_0 + 8;
                F16 *itm_0 = itmArray;
                // dot prod
                // (6*6)*C*c8*hw1 times O*(6*6)*C*c8*o16 = O*(6*6)*hw1*o16
                for (U32 idx = 0; idx < 36; idx++) {
                    __asm__ __volatile__(
                        "mov  x0, %[ic]\n"                  //ic_blk
                        "eor  v2.16b,  v2.16b,  v2.16b\n"   //out_o0hw0
                        "ldr  h0, [%[in]]\n"                //in_hw0
                        "eor  v3.16b,  v3.16b,  v3.16b\n"   //out_o1hw0
                        "ldr q18, [%[f]]\n"                 //f_o0c0
                        "ldr q19, [%[f], #16]\n"            //f_o1c0
                        "0:\n"
                        "ldr  h1, [%[in], #2]\n"            //in_hw0
                        "fmla  v2.8h, v18.8h, v0.h[0]\n"
                        "ldr q20, [%[f], #32]\n"            //f_o0c0
                        "fmla  v3.8h, v19.8h, v0.h[0]\n"
                        "ldr q21, [%[f], #48]\n"            //f_o1c0
                        "subs x0, x0, #2\n"

                        "ldr  h0, [%[in], #4]\n"           //in_hw0
                        "fmla  v2.8h, v20.8h, v1.h[0]\n"
                        "ldr q18, [%[f], #64]\n"            //f_o0c0
                        "fmla  v3.8h, v21.8h, v1.h[0]\n"
                        "ldr q19, [%[f], #80]\n"            //f_o1c0
                        "add %[in], %[in], #4\n"
                        "add %[f], %[f], #64\n"
                        "bne 0b\n"
                        "st1 {v2.8h,  v3.8h}, [%[out]], #32\n"
                        :[out]"+r"(otm_0),
                         [in]"+r"(itm_0),
                         [f]"+r"(ftm_0)
                        :[ic]"r"((I64)ic*8)
                        :"memory", "cc", "v0", "v1", "v2", "v3", "v18", "v19", "v20", "v21", "x0"
                    );
                }
                // out trans
                // O*(6*6)*hw1*o16 => NOWHo8
                U32 h = hw / tile_w;
                U32 w = hw % tile_w;
                F16 *out_0 = outArray + n*oc*oh*ow*8 + o*oh*ow*8 + h*4*ow*8 + w*4*8;
                F16 *out_1 = out_0 + oh*ow*8;
                U32 otm_off_0 = o*8*36;
                U32 otm_off_1 = otm_off_0 + 8;

                F16 *Ow_0[36];
                F16 *Ow_1[36];
                F16 *O_0[16];
                F16 *O_1[16];
                for (U32 idx = 0; idx < 36; idx++) {
                    Ow_0[idx] = otmArray + otm_off_0 + idx*16;
                    Ow_1[idx] = otmArray + otm_off_1 + idx*16;
                }
                for (U32 i = 0; i < 4; ++i) {
                    for (U32 j = 0; j < 4; ++j) {
                        O_0[i*4 + j] = out_0 + i*ow*8 + j*8;
                        O_1[i*4 + j] = out_1 + i*ow*8 + j*8;
                    }
                }
                CHECK_STATUS(trans_O_4x4_3x3(Ow_0, O_0, b_0, h, w, pad_h_mod_4, pad_w_mod_4, tile_h-1, tile_w-1, activationMode));
                CHECK_STATUS(trans_O_4x4_3x3(Ow_1, O_1, b_1, h, w, pad_h_mod_4, pad_w_mod_4, tile_h-1, tile_w-1, activationMode));
            }
            if (oc & 1) {
                F16 *itm_0 = itmArray;
                const F16 *ftm_0 = filterArray + (oc-1)*8*36*ic*8;
                F16 *otm_0 = otmArray + (oc-1)*8*36;
                const F16 *b_0 = biasArray + (oc-1)*8;
                // dot prod
                // (6*6)*C*c8*hw1 times O*(6*6)*C*c8*o8 = O*(6*6)*hw1*o8
                for (U32 idx = 0; idx < 36; idx++) {
                    __asm__ __volatile__(
                        "mov x0, %[ic]\n"             //ic_blk
                        "eor v2.16b, v2.16b, v2.16b\n"      //out_o0hw0
                        "ldr s0, [%[in]]\n"           //in_hw0
                        "ldr q18, [%[f]]\n"            //f_o0c0
                        "0:\n"
                        "ldr  h1, [%[in], #2]\n"           //in_hw0
                        "fmla  v2.8h, v18.8h, v0.h[0]\n"
                        "ldr q20, [%[f], #16]\n"            //f_o0c0
                        "subs x0, x0, #2\n"

                        "ldr  h0, [%[in], #4]\n"           //in_hw0
                        "fmla  v2.8h, v20.8h, v1.h[0]\n"
                        "ldr q18, [%[f], #32]\n"            //f_o0c0
                        "add %[in], %[in], #4\n"
                        "add %[f], %[f], #32\n"
                        "bne 0b\n"
                        "st1 {v2.8h}, [%[out]], #16\n"
                        :[out]"+r"(otm_0),
                         [in]"+r"(itm_0),
                         [f]"+r"(ftm_0)
                        :[ic]"r"((I64)ic*8)
                        :"memory", "cc", "v0", "v1", "v2", "v18", "v20", "x0"
                    );
                }
                // out trans
                // O*(6*6)*hw1*o8 => NOWHo8
                U32 h = hw / tile_w;
                U32 w = hw % tile_w;
                F16 *out_0 = outArray + n*oc*oh*ow*8 + (oc-1)*oh*ow*8 + h*4*ow*8 + w*4*8;
                U32 otm_off_0 = (oc-1)*8*36;

                F16 *Ow_0[36];
                F16 *O_0[16];
                for (U32 idx = 0; idx < 36; idx++) {
                    Ow_0[idx] = otmArray + otm_off_0 + idx*8;
                }
                for (U32 i = 0; i < 4; ++i) {
                    for (U32 j = 0; j < 4; ++j) {
                        O_0[i*4 + j] = out_0 + i*ow*8 + j*8;
                    }
                }
                CHECK_STATUS(trans_O_4x4_3x3(Ow_0, O_0, b_0, h, w, pad_h_mod_4, pad_w_mod_4, tile_h-1, tile_w-1, activationMode));
            }
        }
    }
    return ret;
}
