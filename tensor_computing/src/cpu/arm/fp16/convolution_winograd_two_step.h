// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_CONVOLUTION_WINOGRAD_TWO_STEP
#define _H_CONVOLUTION_WINOGRAD_TWO_STEP

#include "type.h"
#include "error.h"
#include "tensor_desc.h"
#include "cpu/arm/fp16/convolution_winograd_transform.h"

inline EE convolution_winograd_two_step(TensorDesc inputDesc, F16* inArray,
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
    CHECK_STATUS_WITH_RETURN(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 padding = convDesc.padding;

    if (fdf != DF_HWNCN16)
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    if (!(ic == fc && oc == fn))
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    if (!(fh == 6 && fw == 6))
        CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);

    oc /= 8;
    ic /= 8;

    U32 ih_pad = ih + 2*padding;
    U32 iw_pad = iw + 2*padding;
    U32 tile_h = oh / 4;
    U32 tile_w = ow / 4;
    U32 tiles = tile_h * tile_w;  // num of 6x6 tile, assume 8|tiles

    // tmp = otm + itm + in_pad
    F16* otmArray = (F16*)tmp;
    F16* itmArray = otmArray + oc*tiles*6*6*8;
    F16* inArray_pad = itmArray + ic*tiles*6*6*8;

    EE ret = SUCCESS;
    // copy input into a input with padding
    for (U32 n = 0; n < in; n++) {
        F16 *inArray_pad_mov = inArray_pad;
        F16 *inArray_mov = inArray + n*ic*ih*iw*8;
        for (U32 c = 0; c < ic; c++) {
            for (U32 h = 0; h < padding; h++) {
                memset(inArray_pad_mov, 0, iw_pad*8*bytesOf(idt));
                inArray_pad_mov += iw_pad*8;
            }
            for (U32 h = padding; h < ih_pad - padding; h++) {
                memset(inArray_pad_mov, 0, padding*8*bytesOf(idt));
                inArray_pad_mov += padding*8;
                memcpy(inArray_pad_mov, inArray_mov, iw*8*bytesOf(idt));
                inArray_pad_mov += iw*8;
                inArray_mov += iw*8;
                memset(inArray_pad_mov, 0, padding*8*bytesOf(idt));
                inArray_pad_mov += padding*8;
            }
            for (U32 h = ih_pad - padding; h < ih_pad; h++) {
                memset(inArray_pad_mov, 0, iw_pad*8*bytesOf(idt));
                inArray_pad_mov += iw_pad*8;
            }
        }

        // input transform, assume ih_pad/iw_pad = 4k + 2
        // NCHWc8 => N*tile*(6*6)*C*c8*t8
        // assume tile_h > 1, tile_w > 1
        for (U32 c = 0; c < ic; c++) {
            F16 *inArray_pad_mov = inArray_pad + c*ih_pad*iw_pad*8;
            for (U32 hw = 0; hw < tiles; hw+=8) {
                F16 *Iw_ptr[36];
                F16 Iw0[36][8];
                F16 *I0[36];
                F16 Iw1[36][8];
                F16 *I1[36];
                F16 Iw2[36][8];
                F16 *I2[36];
                F16 Iw3[36][8];
                F16 *I3[36];
                F16 Iw4[36][8];
                F16 *I4[36];
                F16 Iw5[36][8];
                F16 *I5[36];
                F16 Iw6[36][8];
                F16 *I6[36];
                F16 Iw7[36][8];
                F16 *I7[36];
                F16 *itmArray_mov = itmArray + hw*36*ic*8 + c*8*8;
                U32 h0 = (hw/tile_w)*4;
                U32 w0 = (hw%tile_w)*4;
                U32 h1 = ((hw+1)/tile_w)*4;
                U32 w1 = ((hw+1)%tile_w)*4;
                U32 h2 = ((hw+2)/tile_w)*4;
                U32 w2 = ((hw+2)%tile_w)*4;
                U32 h3 = ((hw+3)/tile_w)*4;
                U32 w3 = ((hw+3)%tile_w)*4;
                U32 h4 = ((hw+4)/tile_w)*4;
                U32 w4 = ((hw+4)%tile_w)*4;
                U32 h5 = ((hw+5)/tile_w)*4;
                U32 w5 = ((hw+5)%tile_w)*4;
                U32 h6 = ((hw+6)/tile_w)*4;
                U32 w6 = ((hw+6)%tile_w)*4;
                U32 h7 = ((hw+7)/tile_w)*4;
                U32 w7 = ((hw+7)%tile_w)*4;
                for (U32 i = 0; i < 6; i++) {
                    for (U32 j = 0; j < 6; j++) {
                        I0[i*6 + j] = inArray_pad_mov + (h0+i)*iw_pad*8 + (w0+j)*8;
                        I1[i*6 + j] = inArray_pad_mov + (h1+i)*iw_pad*8 + (w1+j)*8;
                        I2[i*6 + j] = inArray_pad_mov + (h2+i)*iw_pad*8 + (w2+j)*8;
                        I3[i*6 + j] = inArray_pad_mov + (h3+i)*iw_pad*8 + (w3+j)*8;
                        I4[i*6 + j] = inArray_pad_mov + (h4+i)*iw_pad*8 + (w4+j)*8;
                        I5[i*6 + j] = inArray_pad_mov + (h5+i)*iw_pad*8 + (w5+j)*8;
                        I6[i*6 + j] = inArray_pad_mov + (h6+i)*iw_pad*8 + (w6+j)*8;
                        I7[i*6 + j] = inArray_pad_mov + (h7+i)*iw_pad*8 + (w7+j)*8;
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
                    Iw_ptr[i] = Iw4[i];
                }
                trans_I_4x4_3x3(Iw_ptr, I4);
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw5[i];
                }
                trans_I_4x4_3x3(Iw_ptr, I5);
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw6[i];
                }
                trans_I_4x4_3x3(Iw_ptr, I6);
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw7[i];
                }
                trans_I_4x4_3x3(Iw_ptr, I7);
                for (U32 i = 0; i < 36; i++) {
                    F16* itm = itmArray_mov + i*ic*8*8;
                    /*
                     * for (U32 c8 = 0; c8 < 8; c8++) {
                     *     itm[c8*8] = Iw0[i][c8];
                     *     itm[c8*8 + 1] = Iw1[i][c8];
                     *     itm[c8*8 + 2] = Iw2[i][c8];
                     *     itm[c8*8 + 3] = Iw3[i][c8];
                     *     itm[c8*8 + 4] = Iw4[i][c8];
                     *     itm[c8*8 + 5] = Iw5[i][c8];
                     *     itm[c8*8 + 6] = Iw6[i][c8];
                     *     itm[c8*8 + 7] = Iw7[i][c8];
                     * }
                     */
                    float16x8_t v0 = vld1q_f16(Iw0[i]);
                    float16x8_t v1 = vld1q_f16(Iw1[i]);
                    float16x8_t v2 = vld1q_f16(Iw2[i]);
                    float16x8_t v3 = vld1q_f16(Iw3[i]);
                    float16x8_t v4 = vld1q_f16(Iw4[i]);
                    float16x8_t v5 = vld1q_f16(Iw5[i]);
                    float16x8_t v6 = vld1q_f16(Iw6[i]);
                    float16x8_t v7 = vld1q_f16(Iw7[i]);
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
        }

        // dot product
        F16 *otm = otmArray;
        for (U32 blk = 0; blk < tiles; blk+=8) {
            F16 *itm0 = itmArray + blk*36*ic*8;
            const F16 *ftm_0 = filterArray;
            for (U32 o = 0; o < oc; o+=2) {
                F16 *itm_0 = itm0;
                for (U32 idx = 0; idx < 36; idx++) {
                    __asm__ __volatile__(
                        "mov  x0, %[ic]\n"             //ic_blk
                        "eor  v2.16b,  v2.16b,  v2.16b\n"      //out_o0hw0
                        "ldr  d0, [%[in]]\n"           //in_hw0
                        "eor  v4.16b,  v4.16b,  v4.16b\n"      //out_o0hw1
                        "ldr  x1, [%[in], #8]\n"
                        "eor  v6.16b,  v6.16b,  v6.16b\n"      //out_o0hw2
                        "ins  v0.d[1], x1\n"
                        "eor  v8.16b,  v8.16b,  v8.16b\n"      //out_o0hw3
                        "ldr d18, [%[f]]\n"            //f_o0c0
                        "eor v10.16b, v10.16b, v10.16b\n"      //out_o0hw4
                        "ldr  x2, [%[f], #8]\n"
                        "eor v12.16b, v12.16b, v12.16b\n"      //out_o0hw5
                        "ins v18.d[1], x2\n"
                        "eor v14.16b, v14.16b, v14.16b\n"      //out_o0hw6
                        "ldr d19, [%[f], #16]\n"            //f_o1c0
                        "eor v16.16b, v16.16b, v16.16b\n"      //out_o0hw7
                        "ldr  x3, [%[f], #24]\n"
                        "eor  v3.16b,  v3.16b,  v3.16b\n"      //out_o1hw0
                        "ins v19.d[1], x3\n"
                        "eor  v5.16b,  v5.16b,  v5.16b\n"      //out_o1hw1
                        "eor  v7.16b,  v7.16b,  v7.16b\n"      //out_o1hw2
                        "eor  v9.16b,  v9.16b,  v9.16b\n"      //out_o1hw3
                        "eor v11.16b, v11.16b, v11.16b\n"      //out_o1hw4
                        "eor v13.16b, v13.16b, v13.16b\n"      //out_o1hw5
                        "eor v15.16b, v15.16b, v15.16b\n"      //out_o1hw6
                        "eor v17.16b, v17.16b, v17.16b\n"      //out_o1hw7
                        "0:\n"
                        "ldr  d1, [%[in], #16]\n"           //in_hw0
                        "fmla  v2.8h, v18.8h, v0.h[0]\n"
                        "ldr  x1, [%[in], #24]\n"
                        "fmla  v4.8h, v18.8h, v0.h[1]\n"
                        "ins  v1.d[1], x1\n"
                        "fmla  v6.8h, v18.8h, v0.h[2]\n"
                        "ldr d20, [%[f], #32]\n"            //f_o0c0
                        "fmla  v8.8h, v18.8h, v0.h[3]\n"
                        "ldr  x2, [%[f], #40]\n"
                        "fmla v10.8h, v18.8h, v0.h[4]\n"
                        "ins v20.d[1], x2\n"
                        "fmla v12.8h, v18.8h, v0.h[5]\n"
                        "ldr d21, [%[f], #48]\n"            //f_o1c0
                        "fmla v14.8h, v18.8h, v0.h[6]\n"
                        "ldr  x3, [%[f], #56]\n"
                        "fmla v16.8h, v18.8h, v0.h[7]\n"
                        "ins v21.d[1], x3\n"
                        "fmla  v3.8h, v19.8h, v0.h[0]\n"
                        "fmla  v5.8h, v19.8h, v0.h[1]\n"
                        "fmla  v7.8h, v19.8h, v0.h[2]\n"
                        "fmla v9.8h, v19.8h, v0.h[3]\n"
                        "fmla v11.8h, v19.8h, v0.h[4]\n"
                        "fmla v13.8h, v19.8h, v0.h[5]\n"
                        "fmla v15.8h, v19.8h, v0.h[6]\n"
                        "fmla v17.8h, v19.8h, v0.h[7]\n"

                        "ldr  d0, [%[in], #32]\n"           //in_hw0
                        "fmla  v2.8h, v20.8h, v1.h[0]\n"
                        "ldr  x1, [%[in], #40]\n"
                        "fmla  v4.8h, v20.8h, v1.h[1]\n"
                        "ins  v0.d[1], x1\n"
                        "fmla  v6.8h, v20.8h, v1.h[2]\n"
                        "ldr d18, [%[f], #64]\n"            //f_o0c0
                        "fmla  v8.8h, v20.8h, v1.h[3]\n"
                        "ldr  x2, [%[f], #72]\n"
                        "fmla v10.8h, v20.8h, v1.h[4]\n"
                        "ins v18.d[1], x2\n"
                        "fmla v12.8h, v20.8h, v1.h[5]\n"
                        "ldr d19, [%[f], #80]\n"            //f_o1c0
                        "fmla v14.8h, v20.8h, v1.h[6]\n"
                        "ldr  x3, [%[f], #88]\n"
                        "fmla v16.8h, v20.8h, v1.h[7]\n"
                        "ins v19.d[1], x3\n"
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
                        :[out]"+r"(otm),
                         [in]"+r"(itm_0),
                         [f]"+r"(ftm_0)
                        :[ic]"r"(ic*8)
                        :"memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q18", "q19", "q20", "q21", "x0", "x1", "x2", "x3"
                    );
                }
            }
        }

        // output transform
        // N*tile*O*(6*6)*t8*o16 => NOHWo8
        for (U32 o = 0; o < oc; o++) {
            const F16 *b = biasArray + o*8;
            for (U32 h = 0; h < tile_h; h++) {
                for (U32 w = 0; w < tile_w; w++) {
                    F16 *out = outArray + n*oc*oh*ow*8 + o*oh*ow*8 + h*4*ow*8 + w*4*8;
                    U32 tile = (h*tile_w + w) / 8;
                    U32 t8 = (h*tile_w + w) % 8;
                    U32 otm_off = tile*oc*36*64 + (o/2)*36*8*16 + t8*16 + (o%2)*8;

                    F16 *Ow[36];
                    F16 *O[16];
                    for (U32 i = 0; i < 6; ++i) {
                        for (U32 j = 0; j < 6; ++j) {
                            Ow[i*6 + j] = otmArray + otm_off + i*6*8*16 + j*8*16;
                        }
                    }
                    for (U32 i = 0; i < 4; ++i) {
                        for (U32 j = 0; j < 4; ++j) {
                            O[i*4 + j] = out + i*ow*8 + j*8;
                        }
                    }
                    CHECK_STATUS_WITH_RETURN(trans_O_4x4_3x3(Ow, O, b, 0, 0, 0, 0, 1, 1, activationMode));
                }
            }
        }
    }
    return ret;
}
#endif
