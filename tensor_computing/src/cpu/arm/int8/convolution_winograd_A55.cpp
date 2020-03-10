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
#include "cpu/arm/int8/convolution_winograd_transform.h"
#include "cpu/arm/int8/convolution_winograd.h"

template<typename OT>
EE convolution_winograd_A55(TensorDesc inputDesc, const void* input, F16* input_scale, TensorDesc filterDesc, const void* filter, F16* filterScale,
    ConvolutionDesc convDesc, TensorDesc biasDesc, const void* bias, U32 tmpBytes, void* tmp, TensorDesc outputDesc,
    void* output, F16* outputScale, ActivationMode am)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    
    // not truely one_step. Compute hw12*(6*6)*ic at one time.
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

    if (fdf != DF_HWNCN8C4) {
        return NOT_MATCH;
    }
    if (!(fh == 6 && fw == 6)) {
        return NOT_MATCH;
    }

    // Assume IT is the same as OT
    OT* inArray = (OT*)input;
    INT8* filterArray = (INT8*)filter;
    F16* outArray = (F16*)output;
    F16* biasArray = (F16*)bias;

    // both input and output are stored with C8 
    oc /= 8;
    ic /= 8;

    U32 tile_h = (oh + 3) / 4;
    U32 tile_w = (ow + 3) / 4;
    I32 tiles = tile_h * tile_w;  // num of 6x6 tiles
    U32 pad_left = paddingL;
    U32 pad_right = paddingR + (tile_w*4 - ow);
    U32 pad_w_mod_4 = tile_w*4 - ow;
    U32 pad_top = paddingT;
    U32 pad_bottom = paddingB + (tile_h*4 - oh);
    U32 pad_h_mod_4 = tile_h*4 - oh;
    U32 ih_pad = ih + pad_top + pad_bottom;
    U32 iw_pad = iw + pad_left + pad_right;

    U32 ohow = oh*ow;
    U32 ihiw = ih_pad*iw_pad;

    // tmp = in_pad + itm + otm + inQ + ...
    // in_pad: ic*ih_pad*iw_pad*8
    // itm: 6*6*ic*12*8 (int16 or fp16)
    // otm: 6*6*12*8 (F16)
    // inQ: 6*6*ic*12*8 (int8)
    OT* inArray_pad = (OT*)tmp;
    short* itmArray = (short*)(inArray_pad + ic*ihiw*8);  // will be cast to fp16 for fp16 inputs
    F16* otmArray = (F16*)(itmArray + 6*6*ic*12*8);
    INT8* inQ = (INT8*)(otmArray + 6*6*12*8);
    if (DT_I8 == odt) {
        outArray = (F16*)(inQ + 6*6*ic*12*8);  // After otmArray and pack
    }

    // To track the range of the final outputs and prepare for quantization
    F16 max[8] = {0};
    F16 min[8] = {0};

    for (U32 n = 0; n < in; n++) {  // for each batch
        OT *inArray_pad_mov = inArray_pad;
        OT *inArray_mov = inArray + n*ic*ih*iw*8;
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

        // tiles / 12
        for (I32 hw = 0; hw < tiles-11; hw+=12) {
            // in trans
            // NCHWc8 => (6*6)*(C/4)*hw12*c4
            // transform hw1c8 at a time, so we need 12 times to cover hw12c8
            // pack into hw12c4 after quantizing (reuse the space of itmArray)
            for (U32 c = 0; c < ic; c++) {
                OT *inArray_pad_mov = inArray_pad + c*ihiw*8;
                short *Iw_ptr[36];
                short *Iw0[36];
                OT *I0[36];
                short *Iw1[36];
                OT *I1[36];
                short *Iw2[36];
                OT *I2[36];
                short *Iw3[36];
                OT *I3[36];
                short *Iw4[36];
                OT *I4[36];
                short *Iw5[36];
                OT *I5[36];
                short *Iw6[36];
                OT *I6[36];
                short *Iw7[36];
                OT *I7[36];
                short *Iw8[36];
                OT *I8[36];
                short *Iw9[36];
                OT *I9[36];
                short *Iw10[36];
                OT *I10[36];
                short *Iw11[36];
                OT *I11[36];

                // Store transformed hw12c8 to itmArray
                for (U32 i = 0; i < 36; i++) {
                    Iw0[i] = itmArray + i*12*ic*8 + c*8*12;
                    Iw1[i] = itmArray + i*12*ic*8 + c*8*12 + 1*8;
                    Iw2[i] = itmArray + i*12*ic*8 + c*8*12 + 2*8;
                    Iw3[i] = itmArray + i*12*ic*8 + c*8*12 + 3*8;
                    Iw4[i] = itmArray + i*12*ic*8 + c*8*12 + 4*8;
                    Iw5[i] = itmArray + i*12*ic*8 + c*8*12 + 5*8;
                    Iw6[i] = itmArray + i*12*ic*8 + c*8*12 + 6*8;
                    Iw7[i] = itmArray + i*12*ic*8 + c*8*12 + 7*8;
                    Iw8[i] = itmArray + i*12*ic*8 + c*8*12 + 8*8;
                    Iw9[i] = itmArray + i*12*ic*8 + c*8*12 + 9*8;
                    Iw10[i] = itmArray + i*12*ic*8 + c*8*12 + 10*8;
                    Iw11[i] = itmArray + i*12*ic*8 + c*8*12 + 11*8;
                }

                U32 h0 = (hw/tile_w)*4;  // stride is 4
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
                U32 h8 = ((hw+8)/tile_w)*4;
                U32 w8 = ((hw+8)%tile_w)*4;
                U32 h9 = ((hw+9)/tile_w)*4;
                U32 w9 = ((hw+9)%tile_w)*4;
                U32 h10 = ((hw+10)/tile_w)*4;
                U32 w10 = ((hw+10)%tile_w)*4;
                U32 h11 = ((hw+11)/tile_w)*4;
                U32 w11 = ((hw+11)%tile_w)*4;

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
                        I8[i*6 + j] = inArray_pad_mov + (h8+i)*iw_pad*8 + (w8+j)*8;
                        I9[i*6 + j] = inArray_pad_mov + (h9+i)*iw_pad*8 + (w9+j)*8;
                        I10[i*6 + j] = inArray_pad_mov + (h10+i)*iw_pad*8 + (w10+j)*8;
                        I11[i*6 + j] = inArray_pad_mov + (h11+i)*iw_pad*8 + (w11+j)*8;
                    }
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw0[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I0);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I0);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw1[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I1);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I1);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw2[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I2);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I2);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw3[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I3);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I3);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw4[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I4);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I4);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw5[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I5);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I5);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw6[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I6);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I6);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw7[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I7);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I7);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw8[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I8);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I8);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw9[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I9);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I9);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw10[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I10);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I10);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw11[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I11);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I11);
                }
            }

            F32 inputScale[36];

            if (DT_I8 == idt) {
                quantize_wino_input_s16(itmArray, 12*ic*8, inQ, inputScale, *input_scale);
            } else {
                quantize_wino_input((F16*)itmArray, 12*ic*8, inQ, inputScale);
            }

            F32 factor_v[36][4];
            for (U32 i = 0; i < 36; i++) {
                if (inputScale[i] == 0) {
                    factor_v[i][0] = 0;
                    continue;
                } else {
                    factor_v[i][0] = 1.0 / inputScale[i] / (F32)filterScale[i];
                }

                factor_v[i][1] = factor_v[i][0];
                factor_v[i][2] = factor_v[i][0];
                factor_v[i][3] = factor_v[i][0];
            }

            INT8 *in_pack = (INT8*)itmArray;  // Reuse the space
            
            for (U32 idx=0; idx<36; idx++) {
                if (factor_v[idx][0] == 0) {  // input pixels are all 0
                    continue;
                }
                for (U32 c = 0; c < ic; c++) {  // for each 8 channels
                    INT8 *in_hw12c8 = inQ + idx*12*ic*8 + c*12*8;

                    INT8 *in_0 = in_hw12c8;
                    INT8 *in_1 = in_hw12c8 + 1*8;
                    INT8 *in_2 = in_hw12c8 + 2*8;
                    INT8 *in_3 = in_hw12c8 + 3*8;
                    INT8 *in_4 = in_hw12c8 + 4*8;
                    INT8 *in_5 = in_hw12c8 + 5*8;
                    INT8 *in_6 = in_hw12c8 + 6*8;
                    INT8 *in_7 = in_hw12c8 + 7*8;
                    INT8 *in_8 = in_hw12c8 + 8*8;
                    INT8 *in_9 = in_hw12c8 + 9*8;
                    INT8 *in_10 = in_hw12c8 + 10*8;
                    INT8 *in_11 = in_hw12c8 + 11*8;
                            
                    // NHWChw12c4
                    INT8 *in_pack_0 = in_pack + idx*12*ic*8 + c*12*8;
                    INT8 *in_pack_1 = in_pack_0 + 12*4;

                    __asm__ __volatile__(
                        "ldr d0, [%[in_0]]\n"
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
                        "ldr d8, [%[in_8]]\n"
                        "ldr x10, [%[in_10]]\n"
                        "trn1 v24.4s, v4.4s, v5.4s\n"
                        "trn2 v25.4s, v4.4s, v5.4s\n"
                        "ldr d9, [%[in_9]]\n"
                        "ldr x11, [%[in_11]]\n"
                        "ins v8.d[1], x10\n"
                        "ins v9.d[1], x11\n"
                            
                        "str   q20, [%[pack_0]]\n"
                        "trn1 v28.4s, v8.4s, v9.4s\n"
                        "trn2 v29.4s, v8.4s, v9.4s\n"
                        "str   q24, [%[pack_0], #16]\n"
                        "str   q28, [%[pack_0], #32]\n"
                        "str   q21, [%[pack_1]]\n"
                        "str   q25, [%[pack_1], #16]\n"
                        "str   q29, [%[pack_1], #32]\n"
                        :
                        :[pack_0]"r"(in_pack_0),
                        [pack_1]"r"(in_pack_1),
                        [in_0]"r"(in_0),
                        [in_1]"r"(in_1),
                        [in_2]"r"(in_2),
                        [in_3]"r"(in_3),
                        [in_4]"r"(in_4),
                        [in_5]"r"(in_5),
                        [in_6]"r"(in_6),
                        [in_7]"r"(in_7),
                        [in_8]"r"(in_8),
                        [in_9]"r"(in_9),
                        [in_10]"r"(in_10),
                        [in_11]"r"(in_11)
                        :"memory", "cc", "v0", "v1", "v4", "v5", "v8", "v9", "v20", "v21", "v24", "v25", "v28", "v29", "x2", "x3", "x6", "x7", "x10", "x11"
                    );
                }
            }

            // compute
            for (U32 o = 0; o < oc; o++) {  // 8 output channels at a time
                // bias
                F16 *b_0 = biasArray + o*8;
                for (U32 idx = 0; idx < 36; idx++) {
                    INT8 *in_hw0 = in_pack + idx*12*ic*8;
                    INT8 *f_o0c0 = filterArray + o*8*36*ic*8 + idx*8*ic*8;
                    F16 *out_o0hw0 = otmArray + idx*12*8;
                    if (factor_v[idx][0] == 0) {  // input pixels are all 0
                        memset(out_o0hw0, 0, 12*8*sizeof(OT));
                        continue;
                    }
                    F32 *fac = factor_v[idx];
                    __asm__ __volatile__(
                        "eor v5.16b, v5.16b, v5.16b\n"
                        "ldr  d1, [%[in_0]]\n"           //in_0
                        "eor v6.16b, v6.16b, v6.16b\n"
                        "ldr  x1, [%[in_0], #8]\n"
                        "eor v7.16b, v7.16b, v7.16b\n"
                        "ins  v1.d[1], x1\n"
                        "eor v8.16b, v8.16b, v8.16b\n"
                        "ldr d0, [%[f_0]]\n"            //f_0
                        "eor v9.16b, v9.16b, v9.16b\n"
                        "ldr  x2, [%[f_0], #8]\n"
                        "eor v10.16b, v10.16b, v10.16b\n"
                        "ins v0.d[1], x2\n"
                        "eor v11.16b, v11.16b, v11.16b\n"
                        "ldr  d3, [%[in_0], #16]\n"     //in_1
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

                        //give in address to x3
                        "mov x3, %[in_0]\n"

                        //give f address to x0
                        "mov x0, %[f_0]\n"

                        "mov  x2, %[ic]\n"             //ic_blk
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
                        "scvtf v5.4s, v5.4s\n"
                        "scvtf v6.4s, v6.4s\n"
                        "ldr d1, [%[factor]]\n"
                        "ldr x1, [%[factor], #8]\n"
                        "scvtf v7.4s, v7.4s\n"
                        "scvtf v8.4s, v8.4s\n"
                        "ins v1.d[1], x1\n"
                        "scvtf v9.4s, v9.4s\n"
                        "scvtf v10.4s, v10.4s\n"
                        "scvtf v11.4s, v11.4s\n"
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
                        :
                        :[out_0]"r"(out_o0hw0),
                        [in_0]"r"(in_hw0),
                        [f_0]"r"(f_o0c0),
                        [ic]"r"((I64)ic*8),
                        [factor]"r"(fac)
                        :"memory", "cc", "v0", "v1", "v2", "v3", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "x0", "x1", "x2", "x3","x17","x16"
                    );
                }
                // out trans
                // (6*6)*hw12*o8 => NOHWo8
                for (U32 hw12 = 0; hw12 < 12; hw12++) {
                    U32 h = (hw+hw12) / tile_w;
                    U32 w = (hw+hw12) % tile_w;
                    F16 *out_0 = outArray + n*oc*ohow*8 + o*ohow*8 + h*4*ow*8 + w*4*8;

                    F16 *Ow_0[36];
                    F16 *O_0[16];

                    for (U32 idx = 0; idx < 36; idx++) {
                        Ow_0[idx] = otmArray + idx*12*8 + hw12*8;
                    }
                    for (U32 i = 0; i < 4; ++i) {
                        for (U32 j = 0; j < 4; ++j) {
                            O_0[i*4 + j] = out_0 + i*ow*8 + j*8;
                        }
                    }
                    trans_O(Ow_0, O_0, b_0, h, w, pad_h_mod_4, pad_w_mod_4, tile_h-1, tile_w-1, max, min, am);
                }
            }
        }

        // tiles_reminder % 12 / 8
        I32 tiles_s = (tiles / 12) * 12;
        I32 tiles_tail = tiles - tiles_s;
        
        if (tiles_tail >= 8) {
            I32 hw = tiles_s;
            // in trans
            // NCHWc8 => (6*6)*(C/4)*hw8*c4
            // transform hw1c8 at a time, so we need 8 times to cover hw8c8
            // pack into hw8c4 after quantizing (reuse the space of itmArray)
            for (U32 c = 0; c < ic; c++) {
                OT *inArray_pad_mov = inArray_pad + c*ihiw*8;
                short *Iw_ptr[36];
                short *Iw0[36];
                OT *I0[36];
                short *Iw1[36];
                OT *I1[36];
                short *Iw2[36];
                OT *I2[36];
                short *Iw3[36];
                OT *I3[36];
                short *Iw4[36];
                OT *I4[36];
                short *Iw5[36];
                OT *I5[36];
                short *Iw6[36];
                OT *I6[36];
                short *Iw7[36];
                OT *I7[36];

                // Store transformed hw12c8 to itmArray
                for (U32 i = 0; i < 36; i++) {
                    Iw0[i] = itmArray + i*8*ic*8 + c*8*8;
                    Iw1[i] = itmArray + i*8*ic*8 + c*8*8 + 1*8;
                    Iw2[i] = itmArray + i*8*ic*8 + c*8*8 + 2*8;
                    Iw3[i] = itmArray + i*8*ic*8 + c*8*8 + 3*8;
                    Iw4[i] = itmArray + i*8*ic*8 + c*8*8 + 4*8;
                    Iw5[i] = itmArray + i*8*ic*8 + c*8*8 + 5*8;
                    Iw6[i] = itmArray + i*8*ic*8 + c*8*8 + 6*8;
                    Iw7[i] = itmArray + i*8*ic*8 + c*8*8 + 7*8;
                }

                U32 h0 = (hw/tile_w)*4;  // stride is 4
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
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I0);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I0);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw1[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I1);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I1);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw2[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I2);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I2);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw3[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I3);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I3);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw4[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I4);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I4);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw5[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I5);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I5);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw6[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I6);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I6);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw7[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I7);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I7);
                }
            }

            F32 inputScale[36];

            if (idt == DT_I8) {
                quantize_wino_input_s16(itmArray, 8*ic*8, inQ, inputScale, *input_scale);
            } else {
                quantize_wino_input((F16*)itmArray, 8*ic*8, inQ, inputScale);
            }

            F32 factor_v[36][4];
            for (U32 i = 0; i < 36; i++) {
                if (inputScale[i] == 0) {
                    factor_v[i][0] = 0;
                    continue;
                } else {
                    factor_v[i][0] = 1.0 / inputScale[i] / (F32)filterScale[i];
                }
                factor_v[i][1] = factor_v[i][0];
                factor_v[i][2] = factor_v[i][0];
                factor_v[i][3] = factor_v[i][0];
            }

            INT8 *in_pack = (INT8*)itmArray;  // Reuse the space
            
            for (U32 idx=0; idx<36; idx++) {
                if (factor_v[idx][0] == 0) {  // input pixels are all 0
                    continue;
                }
                for (U32 c = 0; c < ic; c++) {  // for each 8 channels
                    INT8 *in_hw8c8 = inQ + idx*8*ic*8 + c*8*8;

                    INT8 *in_0 = in_hw8c8;
                    INT8 *in_1 = in_hw8c8 + 1*8;
                    INT8 *in_2 = in_hw8c8 + 2*8;
                    INT8 *in_3 = in_hw8c8 + 3*8;
                    INT8 *in_4 = in_hw8c8 + 4*8;
                    INT8 *in_5 = in_hw8c8 + 5*8;
                    INT8 *in_6 = in_hw8c8 + 6*8;
                    INT8 *in_7 = in_hw8c8 + 7*8;
                            
                    // NHWChw8c4
                    INT8 *in_pack_0 = in_pack + idx*8*ic*8 + c*8*8;
                    INT8 *in_pack_1 = in_pack_0 + 8*4;

                    __asm__ __volatile__(
                        "ldr d0, [%[in_0]]\n"
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
                        :[pack_0]"r"(in_pack_0),
                        [pack_1]"r"(in_pack_1),
                        [in_0]"r"(in_0),
                        [in_1]"r"(in_1),
                        [in_2]"r"(in_2),
                        [in_3]"r"(in_3),
                        [in_4]"r"(in_4),
                        [in_5]"r"(in_5),
                        [in_6]"r"(in_6),
                        [in_7]"r"(in_7)
                        :"memory", "cc", "v0", "v1", "v4", "v5", "v20", "v21", "v24", "v25", "x2", "x3", "x6", "x7"
                    );
                }
            }

            // compute
            for (U32 o = 0; o < oc; o++) {  // 8 output channels at a time
                // bias
                F16 *b_0 = biasArray + o*8;
                for (U32 idx = 0; idx < 36; idx++) {
                    INT8 *in_hw0 = in_pack + idx*8*ic*8;
                    INT8 *f_o0c0 = filterArray + o*8*36*ic*8 + idx*8*ic*8;
                    F16 *out_o0hw0 = otmArray + idx*8*8;
                    if (factor_v[idx][0] == 0) {  // input pixels are all 0
                        memset(out_o0hw0, 0, 8*8*sizeof(OT));
                        continue;
                    }
                    F32 *fac = factor_v[idx];
                    __asm__ __volatile__(
                        // Bias should be applied after transform
                        "eor v5.16b, v5.16b, v5.16b\n"
                        "ldr  d1, [%[in_0]]\n"           //in_0
                        "eor v6.16b, v6.16b, v6.16b\n"
                        "ldr  x1, [%[in_0], #8]\n"
                        "eor v7.16b, v7.16b, v7.16b\n"
                        "ins  v1.d[1], x1\n"
                        "eor v8.16b, v8.16b, v8.16b\n"
                        "ldr d0, [%[f_0]]\n"            //f_0
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

                        //give in address to x3
                        "mov x3, %[in_0]\n"

                        //give f address to x0
                        "mov x0, %[f_0]\n"

                        "mov  x2, %[ic]\n"             //ic_blk
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
                        "scvtf v5.4s, v5.4s\n"
                        "scvtf v6.4s, v6.4s\n"
                        "ldr d1, [%[factor]]\n"
                        "ldr x1, [%[factor], #8]\n"
                        "scvtf v7.4s, v7.4s\n"
                        "scvtf v8.4s, v8.4s\n"
                        "ins v1.d[1], x1\n"
                        "scvtf v9.4s, v9.4s\n"
                        "scvtf v10.4s, v10.4s\n"
                        "scvtf v11.4s, v11.4s\n"
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

                        "str   q5, [%[out_0]]\n"
                        "str   q7, [%[out_0], #16]\n"
                        "str   q9, [%[out_0], #32]\n"
                        "str   q11, [%[out_0], #48]\n"
                        "str   q13, [%[out_0], #64]\n"
                        "str   q15, [%[out_0], #80]\n"
                        "str   q17, [%[out_0], #96]\n"
                        "str   q19, [%[out_0], #112]\n"
                        :
                        :[out_0]"r"(out_o0hw0),
                        [in_0]"r"(in_hw0),
                        [f_0]"r"(f_o0c0),
                        [ic]"r"((I64)ic*8),
                        [factor]"r"(fac)
                        :"memory", "cc", "v0", "v1", "v3", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v29", "v30", "x0", "x1", "x2", "x3","x17","x16"
                    );
                }
                // out trans
                // (6*6)*hw8*o8 => NOHWo8
                for (U32 hw8 = 0; hw8 < 8; hw8++) {
                    U32 h = (hw+hw8) / tile_w;
                    U32 w = (hw+hw8) % tile_w;
                    F16 *out_0 = outArray + n*oc*ohow*8 + o*ohow*8 + h*4*ow*8 + w*4*8;

                    F16 *Ow_0[36];
                    F16 *O_0[16];

                    for (U32 idx = 0; idx < 36; idx++) {
                        Ow_0[idx] = otmArray + idx*8*8 + hw8*8;
                    }
                    for (U32 i = 0; i < 4; ++i) {
                        for (U32 j = 0; j < 4; ++j) {
                            O_0[i*4 + j] = out_0 + i*ow*8 + j*8;
                        }
                    }
                    trans_O(Ow_0, O_0, b_0, h, w, pad_h_mod_4, pad_w_mod_4, tile_h-1, tile_w-1, max, min, am);
                }
            }
            tiles_s += 8;
            tiles_tail -= 8;
        }

        if (tiles_tail >= 4) {
            I32 hw = tiles_s;
            // in trans
            // NCHWc8 => (6*6)*(C/4)*hw4*c4
            // transform hw4c8 at a time, so we need 4 times to cover hw4c8
            // pack into hw4c4 after quantizing (reuse the space of itmArray)
            for (U32 c = 0; c < ic; c++) {
                OT *inArray_pad_mov = inArray_pad + c*ihiw*8;
                short *Iw_ptr[36];
                short *Iw0[36];
                OT *I0[36];
                short *Iw1[36];
                OT *I1[36];
                short *Iw2[36];
                OT *I2[36];
                short *Iw3[36];
                OT *I3[36];

                // Store transformed hw4c8 to itmArray
                for (U32 i = 0; i < 36; i++) {
                    Iw0[i] = itmArray + i*4*ic*8 + c*4*8;
                    Iw1[i] = itmArray + i*4*ic*8 + c*4*8 + 1*8;
                    Iw2[i] = itmArray + i*4*ic*8 + c*4*8 + 2*8;
                    Iw3[i] = itmArray + i*4*ic*8 + c*4*8 + 3*8;
                }

                U32 h0 = (hw/tile_w)*4;  // stride is 4
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
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I0);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I0);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw1[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I1);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I1);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw2[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I2);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I2);
                }
                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw3[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I3);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I3);
                }
            }

            F32 inputScale[36];

            if (idt == DT_I8) {
                quantize_wino_input_s16(itmArray, 4*ic*8, inQ, inputScale, *input_scale);
            } else {
                quantize_wino_input((F16*)itmArray, 4*ic*8, inQ, inputScale);
            }
            
            F32 factor_v[36][4];
            for (U32 i = 0; i < 36; i++) {
                if (inputScale[i] == 0) {
                    factor_v[i][0] = 0;
                    continue;
                } else {
                    factor_v[i][0] = 1.0 / inputScale[i] / (F32)filterScale[i];
                }
                factor_v[i][1] = factor_v[i][0];
                factor_v[i][2] = factor_v[i][0];
                factor_v[i][3] = factor_v[i][0];
            }

            F16 *b0 = biasArray;
            INT8 *in_pack = (INT8*)itmArray;  // Reuse the space
            
            for (U32 idx=0; idx<36; idx++) {
                if (factor_v[idx][0] == 0) {  // input pixels are all 0
                    continue;
                }
                for (U32 c = 0; c < ic; c++) {  // for each 8 channels
                    INT8 *in_hw4c8 = inQ + idx*4*ic*8 + c*4*8;

                    INT8 *in_0 = in_hw4c8;
                    INT8 *in_1 = in_hw4c8 + 1*8;
                    INT8 *in_2 = in_hw4c8 + 2*8;
                    INT8 *in_3 = in_hw4c8 + 3*8;
                            
                    // NHWChw8c4
                    INT8 *in_pack_0 = in_pack + idx*4*ic*8 + c*4*8;
                    INT8 *in_pack_1 = in_pack_0 + 4*4;

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
                        :[pack_0]"r"(in_pack_0),
                        [pack_1]"r"(in_pack_1),
                        [in_0]"r"(in_0),
                        [in_1]"r"(in_1),
                        [in_2]"r"(in_2),
                        [in_3]"r"(in_3)
                        :"memory", "cc", "v0", "v1", "v20", "v21", "x2", "x3"
                    );
                }
            }

            // compute
            for (U32 o = 0; o < oc; o++) {  // 8 output channels at a time
                // bias
                F16 *b_0 = b0 + o*8;
                for (U32 idx = 0; idx < 36; idx++) {
                    INT8 *in_hw0 = in_pack + idx*4*ic*8;
                    INT8 *f_o0c0 = filterArray + o*8*36*ic*8 + idx*8*ic*8;
                    F16 *out_o0hw0 = otmArray + idx*4*8;
                    if (factor_v[idx][0] == 0) {
                        memset(out_o0hw0, 0, 4*8*sizeof(OT));
                        continue;
                    }
                    F32 *fac = factor_v[idx];
                    __asm__ __volatile__(
                        "eor v5.16b, v5.16b, v5.16b\n"
                        "ldr  d1, [%[in_0]]\n"           //in_0
                        "eor v6.16b, v6.16b, v6.16b\n"
                        "ldr  x1, [%[in_0], #8]\n"
                        "eor v7.16b, v7.16b, v7.16b\n"
                        "ins  v1.d[1], x1\n"
                        "eor v8.16b, v8.16b, v8.16b\n"
                        "ldr d0, [%[f_0]]\n"            //f_0
                        
                        "eor v9.16b, v9.16b, v9.16b\n"
                        "ldr  x2, [%[f_0], #8]\n"
                        "eor v10.16b, v10.16b, v10.16b\n"
                        "ins v0.d[1], x2\n"
                        "eor v11.16b, v11.16b, v11.16b\n"
                        "eor v12.16b, v12.16b, v12.16b\n"

                        //give in address to x3
                        "mov x3, %[in_0]\n"

                        //give f address to x0
                        "mov x0, %[f_0]\n"

                        "mov  x2, %[ic]\n"             //ic_blk
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

                        "scvtf v5.4s, v5.4s\n"
                        "scvtf v6.4s, v6.4s\n"
                        "ldr d1, [%[factor]]\n"
                        "ldr x1, [%[factor], #8]\n"
                        "scvtf v7.4s, v7.4s\n"
                        "scvtf v8.4s, v8.4s\n"
                        "ins v1.d[1], x1\n"
                        "scvtf v9.4s, v9.4s\n"
                        "scvtf v10.4s, v10.4s\n"
                        "scvtf v11.4s, v11.4s\n"
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

                        "str   q5, [%[out_0]]\n"
                        "str   q7, [%[out_0], #16]\n"
                        "str   q9, [%[out_0], #32]\n"
                        "str   q11, [%[out_0], #48]\n"
                        :
                        :[out_0]"r"(out_o0hw0),
                        [in_0]"r"(in_hw0),
                        [f_0]"r"(f_o0c0),
                        [ic]"r"((I64)ic*8),
                        [factor]"r"(fac)
                        :"memory", "cc", "v0", "v1", "v2", "v3", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v29", "x0", "x1", "x2", "x3","x17","x16"
                    );
                }
                // out trans
                // (6*6)*hw4*o8 => NOHWo8
                for (U32 hw4 = 0; hw4 < 4; hw4++) {
                    U32 h = (hw+hw4) / tile_w;
                    U32 w = (hw+hw4) % tile_w;
                    F16 *out_0 = outArray + n*oc*ohow*8 + o*ohow*8 + h*4*ow*8 + w*4*8;

                    F16 *Ow_0[36];
                    F16 *O_0[16];

                    for (U32 idx = 0; idx < 36; idx++) {
                        Ow_0[idx] = otmArray + idx*4*8 + hw4*8;
                    }
                    for (U32 i = 0; i < 4; ++i) {
                        for (U32 j = 0; j < 4; ++j) {
                            O_0[i*4 + j] = out_0 + i*ow*8 + j*8;
                        }
                    }
                    trans_O(Ow_0, O_0, b_0, h, w, pad_h_mod_4, pad_w_mod_4, tile_h-1, tile_w-1, max, min, am);
                }
            }
            tiles_s += 4;
        }

        for (I32 hw = tiles_s; hw < tiles; hw++) {
            // in trans
            // NCHWc8 => (6*6)*(C/4)*hw1*c4
            // transform hw1c8
            // pack into hw1c4 after quantizing (reuse the space of itmArray)
            for (U32 c = 0; c < ic; c++) {
                OT *inArray_pad_mov = inArray_pad + c*ihiw*8;
                short *Iw_ptr[36];
                short *Iw0[36];
                OT *I0[36];

                // Store transformed hw12c8 to itmArray
                for (U32 i = 0; i < 36; i++) {
                    Iw0[i] = itmArray + i*ic*8 + c*8;
                }

                U32 h0 = (hw/tile_w)*4;  // stride is 4
                U32 w0 = (hw%tile_w)*4;

                for (U32 i = 0; i < 6; i++) {
                    for (U32 j = 0; j < 6; j++) {
                        I0[i*6 + j] = inArray_pad_mov + (h0+i)*iw_pad*8 + (w0+j)*8;
                    }
                }

                for (U32 i = 0; i < 36; i++) {
                    Iw_ptr[i] = Iw0[i];
                }
                if (idt == DT_I8) {
                    trans_I_int8(Iw_ptr, (INT8* const*)I0);
                } else {
                    trans_I_4x4_3x3((F16**)Iw_ptr, (F16* const*)I0);
                }
            }

            F32 inputScale[36];

            if (idt == DT_I8) {
                quantize_wino_input_s16(itmArray, ic*8, inQ, inputScale, *input_scale);
            } else {
                quantize_wino_input((F16*)itmArray, ic*8, inQ, inputScale);
            }

            F32 factor_v[36][4];
            for (U32 i = 0; i < 36; i++) {
                if (inputScale[i] == 0) {
                    factor_v[i][0] = 0;
                    continue;
                } else {
                    factor_v[i][0] = 1.0 / inputScale[i] / (F32)filterScale[i];
                }
                factor_v[i][1] = factor_v[i][0];
                factor_v[i][2] = factor_v[i][0];
                factor_v[i][3] = factor_v[i][0];
            }

            F16 *b0 = biasArray;
            INT8 *in_pack = (INT8*)itmArray;  // Reuse the space
            
            for (U32 idx=0; idx<36; idx++) {
                if (factor_v[idx][0] == 0) {
                    continue;
                }
                for (U32 c = 0; c < ic; c++) {  // for each 8 channels
                    INT8 *in_0 = inQ + idx*ic*8 + c*8;
                            
                    // NHWChw8c4
                    INT8 *in_pack_0 = in_pack + idx*ic*8 + c*8;
                    INT8 *in_pack_1 = in_pack_0 + 4;

                    memcpy(in_pack_0, in_0, 4*bytesOf(DT_I8));
                    memcpy(in_pack_1, in_0+4, 4*bytesOf(DT_I8));
                }
            }

            // compute
            for (U32 o = 0; o < oc; o++) {  // 8 output channels at a time
                // bias
                F16 *b_0 = b0 + o*8;
                for (U32 idx = 0; idx < 36; idx++) {
                    INT8 *in_hw = in_pack + idx*ic*8;
                    INT8 *f_o = filterArray + o*8*36*ic*8 + idx*8*ic*8;
                    F16 *out_o0hw0 = otmArray + idx*8;
                    if (factor_v[idx][0] == 0) {
                        memset(out_o0hw0, 0, 8*sizeof(OT));
                        continue;
                    }
                    int32x4_t res[2] = {0};

                    for(U32 c = 0; c < ic; c++) {
                        int8x8_t in_2 = vld1_s8(in_hw);
                        in_hw += 8;
                        int8x16_t f_8o[4];
                        f_8o[0] = vld1q_s8(f_o);
                        f_8o[1] = vld1q_s8(f_o+16);
                        res[0] = vdotq_lane_s32(res[0], f_8o[0], in_2, 0);
                        res[1] = vdotq_lane_s32(res[1], f_8o[1], in_2, 0);

                        f_8o[2] = vld1q_s8(f_o+32);
                        f_8o[3] = vld1q_s8(f_o+48);
                        f_o += 64;
                        res[0] = vdotq_lane_s32(res[0], f_8o[2], in_2, 1);
                        res[1] = vdotq_lane_s32(res[1], f_8o[3], in_2, 1);
                    }
                    float32x4_t fac = vld1q_f32(factor_v[idx]);
                    float32x4_t resf0 = vcvtq_f32_s32(res[0]);
                    float32x4_t resf1 = vcvtq_f32_s32(res[1]);
                    resf0 = vmulq_f32(resf0, fac);
                    resf1 = vmulq_f32(resf1, fac);

                    float16x4_t resh0 = vcvt_f16_f32(resf0);
                    float16x4_t resh1 = vcvt_f16_f32(resf1);

                    vst1_f16(out_o0hw0, resh0);
                    vst1_f16(out_o0hw0+4, resh1);
                }
                // out trans
                // (6*6)*hw1*o8 => NOHWo8
                U32 h = hw / tile_w;
                U32 w = hw % tile_w;
                F16 *out_0 = outArray + n*oc*ohow*8 + o*ohow*8 + h*4*ow*8 + w*4*8;

                F16 *Ow_0[36];
                F16 *O_0[16];

                for (U32 idx = 0; idx < 36; idx++) {
                    Ow_0[idx] = otmArray + idx*8;
                }
                for (U32 i = 0; i < 4; ++i) {
                    for (U32 j = 0; j < 4; ++j) {
                        O_0[i*4 + j] = out_0 + i*ow*8 + j*8;
                    }
                }
                trans_O(Ow_0, O_0, b_0, h, w, pad_h_mod_4, pad_w_mod_4, tile_h-1, tile_w-1, max, min, am);
            }
        }
    }

    if (DT_I8 == odt) {
        F16 max_s = max[0];
        F16 min_s = min[0];
        for (U32 i = 1; i < 8; i++) {
            if (max_s < max[i]) {
                max_s = max[i];
            }
            if (min_s > min[i]) {
                min_s = min[i];
            }
        }

        if (max_s == 0 && min_s == 0) {
            return NOT_SUPPORTED;
        }

        F16 scale_o;
        if (max_s > 0 && min_s < 0) {
            F16 scale_max = 127.0 / max_s;
            F16 scale_min = -128.0 / min_s;
            scale_o = (scale_max < scale_min) ? scale_max : scale_min;
        } else if (max_s > 0) {
            scale_o = 127.0 / max_s;
        } else {
            scale_o = -128.0 / min_s;
        }
        *outputScale = scale_o;

        apply_scale_f16(on*oc*ohow*8, outArray, scale_o, (INT8*)output);
    }
    return SUCCESS;
}

template EE convolution_winograd_A55<INT8>(TensorDesc inputDesc, const void* input, F16* input_scale, TensorDesc filterDesc, const void* filter, F16* filterScale,
    ConvolutionDesc convDesc, TensorDesc biasDesc, const void* bias, U32 tmpBytes, void* tmp, TensorDesc outputDesc,
    void* output, F16* outputScale, ActivationMode am);

template EE convolution_winograd_A55<F16>(TensorDesc inputDesc, const void* input, F16* input_scale, TensorDesc filterDesc, const void* filter, F16* filterScale,
    ConvolutionDesc convDesc, TensorDesc biasDesc, const void* bias, U32 tmpBytes, void* tmp, TensorDesc outputDesc,
    void* output, F16* outputScale, ActivationMode am);
#endif
