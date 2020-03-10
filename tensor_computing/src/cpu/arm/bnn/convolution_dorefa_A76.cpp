// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifdef _USE_FP16
#include "sys.h"
#include "type.h"
#include "tensor_desc.h"
#include "error.h"

#include "cpu/arm/bnn/convolution_dorefa.h"

EE convolution_dorefa_A76(TensorDesc inputDesc, const F16* input,
    TensorDesc filterDesc, const BIN8* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc scaleDesc, const F16* scaleArray,
    TensorDesc biasDesc, const F16* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F16* outArray,
    ActivationMode activationMode)
{
    UNUSED(scaleDesc);
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    UNUSED(activationMode);

    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 strideH = convDesc.stride_h;
    U32 strideW = convDesc.stride_w;
    U32 paddingT = convDesc.padding_top;
    U32 paddingB = convDesc.padding_bottom;
    U32 paddingL = convDesc.padding_left;
    U32 paddingR = convDesc.padding_right;

    if (fdf != DF_NCHWN16C8)
        CHECK_STATUS(NOT_MATCH);
    if (!(ic == fc && oc == fn))
        CHECK_STATUS(NOT_MATCH);

    oc /= 8;
    ic /= 8;

    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;
    U32 ohow = oh*ow;
    U32 ihiw = ih_pad*iw_pad;

    BIN8* inArray = ((BIN8*)tmp) + ic*ihiw + 8*fh*fw*ic;  // ic has been divided by 8
    BIN8 *inArray_pad;
    for (U32 n = 0; n < in; n++) {
        const F16 *in = input + n*ic*ih*iw*8;
        for (U32 i = 0; i < ic*ih*iw; i++) {
            BIN8 temp = 0;
            for (U32 j = 0; j < 8; j++) {
                if (in[i*8+j] >= 0.5) {
                    temp |= (1 << (7-j));  // set
                } 
            }
            inArray[i] = temp;
        }

        if (paddingT == 0 && paddingB == 0 && paddingL == 0 && paddingR == 0) {
            inArray_pad = inArray + n*ic*ih*iw;  // ic has been divided by 8
        } else {
            // copy input into a input with padding
            inArray_pad = (BIN8*)tmp;
            BIN8 *inArray_pad_mov = inArray_pad;
            BIN8 *inArray_mov = inArray + n*ic*ih*iw;
            for (U32 c = 0; c < ic; c++) {  // All divide by 8
                for (U32 h = 0; h < paddingT; h++) {
                    memset(inArray_pad_mov, 0, iw_pad*bytesOf(DT_BIN01));
                    inArray_pad_mov += iw_pad;
                }
                for (U32 h = paddingT; h < ih_pad - paddingB; h++) {
                    memset(inArray_pad_mov, 0, paddingL*bytesOf(DT_BIN01));
                    inArray_pad_mov += paddingL;
                    memcpy(inArray_pad_mov, inArray_mov, iw*bytesOf(DT_BIN01));
                    inArray_pad_mov += iw;
                    inArray_mov += iw;
                    memset(inArray_pad_mov, 0, paddingR*bytesOf(DT_BIN01));
                    inArray_pad_mov += paddingR;
                }
                for (U32 h = ih_pad - paddingB; h < ih_pad; h++) {
                    memset(inArray_pad_mov, 0, iw_pad*bytesOf(DT_BIN01));
                    inArray_pad_mov += iw_pad;
                }
            }
        }
        for (U32 hw = 0; hw < ohow-7; hw+=8) {
            const F16 *s0 = scaleArray;
            const F16 *s1 = scaleArray + 8;
            const F16 *b0 = biasArray;
            const F16 *b1 = biasArray + 8;
            BIN8 *in_order = ((BIN8*)tmp) + ic*ihiw;  // ic has been divided by 8
            // reorder input
            // NCHWc8 => NHWChw8c8 + im2col
            U32 in_h[8];
            U32 in_w[8];
            for (U32 i = 0; i < 8; i++) {
                in_h[i] = ((hw+i)/ow)*strideH;
                in_w[i] = ((hw+i)%ow)*strideW;
            }
            for (U32 c = 0; c < ic; c++) {
                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        BIN8 *in_hw8c8 = inArray_pad + c*ihiw + fh_idx*iw_pad + fw_idx;
                        // NHWChw8c8
                        BIN8 *in_order_hw8c8 = in_order + c*fh*fw*8 + fh_idx*fw*8 + fw_idx*8;  // This 8 comes from hw8
                        for (U32 i = 0; i < 8; i++) {
                            in_order_hw8c8[i] = *(in_hw8c8 + in_h[i]*iw_pad + in_w[i]);
                        }
                    }
                }
            }

            // compute
            for (U32 o = 0; o < oc; o+=2) {  // oc should be multiple of 32. It will at least be multiple of 16 in the future.
                BIN8 *in_hw0 = in_order;
                const BIN8 *f_o0c0 = filterArray + o*8*fh*fw*ic;  // ic has been divided by 8
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                F16 *out_o1hw0 = out_o0hw0 + ohow*8;
                // scale and bias
                const F16 *s_o0 = s0;
                const F16 *s_o1 = s1;
                const F16 *b_o0 = b0;
                const F16 *b_o1 = b1;
                __asm__ __volatile__(
                    "ldr d29, [%[in_0]]\n"           //in_0
                    "ldr q0, [%[f_0]]\n"            //f_0
            /* Layout
            5   6
            7   8
            9   10
            11  12

            13  14
            15  16
            17  18
            19  20
            */
                    "eor v5.16b, v5.16b, v5.16b\n"
                    "eor v6.16b, v6.16b, v6.16b\n"
                    "eor v7.16b, v7.16b, v7.16b\n"
                    "eor v8.16b, v8.16b, v8.16b\n"
                    "eor v9.16b, v9.16b, v9.16b\n"
                    "dup v1.16b, v29.b[0]\n"        //duplicate a full register
                    "eor v10.16b, v10.16b, v10.16b\n"
                    "dup v2.16b, v29.b[1]\n"
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
                        "eor v21.16b, v21.16b, v21.16b\n"
                        "mov  x9, %[fhfw]\n"
                        "eor v22.16b, v22.16b, v22.16b\n"
                        "eor v23.16b, v23.16b, v23.16b\n"
                        "eor v24.16b, v24.16b, v24.16b\n"
                        "eor v25.16b, v25.16b, v25.16b\n"
                        "eor v26.16b, v26.16b, v26.16b\n"
                        "eor v27.16b, v27.16b, v27.16b\n"
                        "eor v28.16b, v28.16b, v28.16b\n"

                        "mov x4, #4\n"

                        "1:\n"
                            "and v3.16b, v1.16b, v0.16b\n"
                            "and v4.16b, v2.16b, v0.16b\n"

                            "cnt v3.16b, v3.16b\n"
                            "subs x4, x4, #1\n"

                            "cnt v4.16b, v4.16b\n"
                            "dup v1.16b, v29.b[2]\n"
                            
                            "add v21.16b, v21.16b, v3.16b\n" // Use add because the latency is shorter
                            "dup v2.16b, v29.b[3]\n"

                            "add v22.16b, v22.16b, v4.16b\n"

                            "and v3.16b, v1.16b, v0.16b\n"
                            "and v4.16b, v2.16b, v0.16b\n"
                            "cnt v3.16b, v3.16b\n"
                            "cnt v4.16b, v4.16b\n"
                            "dup v1.16b, v29.b[4]\n"
                            "add v23.16b, v23.16b, v3.16b\n"
                            "dup v2.16b, v29.b[5]\n"
                            "add v24.16b, v24.16b, v4.16b\n"

                            "and v3.16b, v1.16b, v0.16b\n"
                            "and v4.16b, v2.16b, v0.16b\n"
                            "cnt v3.16b, v3.16b\n"
                            "cnt v4.16b, v4.16b\n"
                            "dup v1.16b, v29.b[6]\n"
                            "add v25.16b, v25.16b, v3.16b\n"
                            "dup v2.16b, v29.b[7]\n"
                            "add v26.16b, v26.16b, v4.16b\n"

                            "and v3.16b, v1.16b, v0.16b\n"
                            "ldr d29, [x3, 8]!\n"
                            "and v4.16b, v2.16b, v0.16b\n"
                            "cnt v3.16b, v3.16b\n"
                            "ldr q0, [x0, 16]!\n" // next filter
                            "cnt v4.16b, v4.16b\n"
                            "dup v1.16b, v29.b[0]\n"
                            "add v27.16b, v27.16b, v3.16b\n"
                            "dup v2.16b, v29.b[1]\n"
                            "add v28.16b, v28.16b, v4.16b\n"
                        "bne 1b\n"

                        "movi v3.16b, #1\n"
                        "umlal v5.8h, v21.8b, v3.8b\n"
                        "umlal v7.8h, v22.8b, v3.8b\n"
                        "umlal v9.8h, v23.8b, v3.8b\n"
                        "umlal v11.8h, v24.8b, v3.8b\n"
                        "umlal v13.8h, v25.8b, v3.8b\n"
                        "umlal v15.8h, v26.8b, v3.8b\n"
                        "umlal v17.8h, v27.8b, v3.8b\n"
                        "umlal v19.8h, v28.8b, v3.8b\n"
                        
                        "umlal2 v6.8h, v21.16b, v3.16b\n"
                        "umlal2 v8.8h, v22.16b, v3.16b\n"
                        "umlal2 v10.8h, v23.16b, v3.16b\n"
                        "umlal2 v12.8h, v24.16b, v3.16b\n"
                        "umlal2 v14.8h, v25.16b, v3.16b\n"
                        "umlal2 v16.8h, v26.16b, v3.16b\n"
                        "umlal2 v18.8h, v27.16b, v3.16b\n"
                        "umlal2 v20.8h, v28.16b, v3.16b\n"

                        "subs x9, x9, #1\n"
                        "beq 4f\n" // 1x1, continue with the next 32 input channels

                        "2:\n"
                        "eor v21.16b, v21.16b, v21.16b\n"
                        "eor v22.16b, v22.16b, v22.16b\n"
                        "eor v23.16b, v23.16b, v23.16b\n"
                        "eor v24.16b, v24.16b, v24.16b\n"
                        "eor v25.16b, v25.16b, v25.16b\n"
                        "eor v26.16b, v26.16b, v26.16b\n"
                        "eor v27.16b, v27.16b, v27.16b\n"
                        "eor v28.16b, v28.16b, v28.16b\n"

                        "mov x4, #32\n" // Assume 256 will not happen
                        "3:\n"
                            "and v3.16b, v1.16b, v0.16b\n"
                            "and v4.16b, v2.16b, v0.16b\n"

                            "cnt v3.16b, v3.16b\n"
                            "subs x4, x4, #1\n"

                            "cnt v4.16b, v4.16b\n"
                            "dup v1.16b, v29.b[2]\n"
                            
                            "uqadd v21.16b, v21.16b, v3.16b\n"
                            "dup v2.16b, v29.b[3]\n"

                            "uqadd v22.16b, v22.16b, v4.16b\n"

                            "and v3.16b, v1.16b, v0.16b\n"
                            "and v4.16b, v2.16b, v0.16b\n"
                            "cnt v3.16b, v3.16b\n"
                            "cnt v4.16b, v4.16b\n"
                            "dup v1.16b, v29.b[4]\n"
                            "uqadd v23.16b, v23.16b, v3.16b\n"
                            "dup v2.16b, v29.b[5]\n"
                            "uqadd v24.16b, v24.16b, v4.16b\n"

                            "and v3.16b, v1.16b, v0.16b\n"
                            "and v4.16b, v2.16b, v0.16b\n"
                            "cnt v3.16b, v3.16b\n"
                            "cnt v4.16b, v4.16b\n"
                            "dup v1.16b, v29.b[6]\n"
                            "uqadd v25.16b, v25.16b, v3.16b\n"
                            "dup v2.16b, v29.b[7]\n"
                            "uqadd v26.16b, v26.16b, v4.16b\n"

                            "and v3.16b, v1.16b, v0.16b\n"
                            "ldr d29, [x3, 8]!\n"
                            "and v4.16b, v2.16b, v0.16b\n"
                            "cnt v3.16b, v3.16b\n"
                            "ldr q0, [x0, 16]!\n" // next filter
                            "cnt v4.16b, v4.16b\n"
                            "dup v1.16b, v29.b[0]\n"
                            "uqadd v27.16b, v27.16b, v3.16b\n"
                            "dup v2.16b, v29.b[1]\n"
                            "uqadd v28.16b, v28.16b, v4.16b\n"
                        "bne 3b\n"

                        "movi v3.16b, #1\n"
                        "umlal v5.8h, v21.8b, v3.8b\n"
                        "umlal v7.8h, v22.8b, v3.8b\n"
                        "umlal v9.8h, v23.8b, v3.8b\n"
                        "umlal v11.8h, v24.8b, v3.8b\n"
                        "umlal v13.8h, v25.8b, v3.8b\n"
                        "umlal v15.8h, v26.8b, v3.8b\n"
                        "umlal v17.8h, v27.8b, v3.8b\n"
                        "umlal v19.8h, v28.8b, v3.8b\n"
                        
                        "umlal2 v6.8h, v21.16b, v3.16b\n"
                        "umlal2 v8.8h, v22.16b, v3.16b\n"
                        "umlal2 v10.8h, v23.16b, v3.16b\n"
                        "umlal2 v12.8h, v24.16b, v3.16b\n"
                        "umlal2 v14.8h, v25.16b, v3.16b\n"
                        "umlal2 v16.8h, v26.16b, v3.16b\n"
                        "umlal2 v18.8h, v27.16b, v3.16b\n"
                        "umlal2 v20.8h, v28.16b, v3.16b\n"

                        "subs x9, x9, #8\n"
                        "bne 2b\n"

                        "4:\n" // Wrap up computation for 32 input channels
                        "subs x2, x2, #32\n"
                    "bne 0b\n"

                    // pipelined
                    "ucvtf v5.8h, v5.8h\n"
                    "ucvtf v6.8h, v6.8h\n"
                    "ldr q21, [%[b_0]]\n"
                    "ucvtf v7.8h, v7.8h\n"
                    "ldr q22, [%[b_1]]\n"
                    "ucvtf v8.8h, v8.8h\n"
                    "ldr q23, [%[s_0]]\n"
                    "ucvtf v9.8h, v9.8h\n"
                    "ldr q24, [%[s_1]]\n"
                    "ucvtf v10.8h, v10.8h\n"
                    "ucvtf v11.8h, v11.8h\n"
                    "mov v1.16b, v21.16b\n"
                    "ucvtf v12.8h, v12.8h\n"
                    "mov v2.16b, v22.16b\n"
                    "ucvtf v13.8h, v13.8h\n"
                    "fmla v1.8h, v5.8h, v23.8h\n"
                    "ucvtf v14.8h, v14.8h\n"
                    "fmla v2.8h, v6.8h, v24.8h\n"
                    "ucvtf v15.8h, v15.8h\n"
                    "mov v3.16b, v21.16b\n"
                    "ucvtf v16.8h, v16.8h\n"
                    "mov v4.16b, v22.16b\n"
                    "ucvtf v17.8h, v17.8h\n"
                    "fmla v3.8h, v7.8h, v23.8h\n"
                    "ucvtf v18.8h, v18.8h\n"
                    "fmla v4.8h, v8.8h, v24.8h\n"
                    "ucvtf v19.8h, v19.8h\n"
                    "mov v5.16b, v21.16b\n"
                    "ucvtf v20.8h, v20.8h\n"
                    "mov v6.16b, v22.16b\n"

                    "fmla v5.8h, v9.8h, v23.8h\n"
                    "mov v7.16b, v21.16b\n"
                    "fmla v6.8h, v10.8h, v24.8h\n"
                    "mov v8.16b, v22.16b\n"
                    "fmla v7.8h, v11.8h, v23.8h\n"
                    "mov v9.16b, v21.16b\n"
                    "fmla v8.8h, v12.8h, v24.8h\n"
                    "mov v10.16b, v22.16b\n"
                    "fmla v9.8h, v13.8h, v23.8h\n"
                    "mov v11.16b, v21.16b\n"
                    "fmla v10.8h, v14.8h, v24.8h\n"
                    "mov v12.16b, v22.16b\n"
                    "fmla v11.8h, v15.8h, v23.8h\n"
                    "mov v13.16b, v21.16b\n"
                    "fmla v12.8h, v16.8h, v24.8h\n"
                    "mov v14.16b, v22.16b\n"
                    "fmla v13.8h, v17.8h, v23.8h\n"
                    "mov v15.16b, v21.16b\n"
                    "fmla v14.8h, v18.8h, v24.8h\n"
                    "mov v16.16b, v22.16b\n"
                    "fmla v15.8h, v19.8h, v23.8h\n"
                    "fmla v16.8h, v20.8h, v24.8h\n"
                    
                    "str   q1, [%[out_0]]\n"           //out_o0hw0
                    "str   q3, [%[out_0], #16]\n"      //out_o0hw1
                    "str   q5, [%[out_0], #32]\n"      //out_o0hw2
                    "str   q7, [%[out_0], #48]\n"      //out_o0hw3
                    "str   q9, [%[out_0], #64]\n"      //out_o0hw4
                    "str   q11, [%[out_0], #80]\n"      //out_o0hw5
                    "str   q13, [%[out_0], #96]\n"      //out_o0hw6
                    "str   q15, [%[out_0], #112]\n"     //out_o0hw7

                    "str  q2, [%[out_1]]\n"           //out_o1hw0
                    "str  q4, [%[out_1], #16]\n"      //out_o1hw1
                    "str  q6, [%[out_1], #32]\n"      //out_o1hw2
                    "str  q8, [%[out_1], #48]\n"      //out_o1hw3
                    "str  q10, [%[out_1], #64]\n"      //out_o1hw4
                    "str  q12, [%[out_1], #80]\n"      //out_o1hw5
                    "str  q14, [%[out_1], #96]\n"      //out_o1hw6
                    "str  q16, [%[out_1], #112]\n"     //out_o1hw7
                    :[out_0]"+r"(out_o0hw0),
                     [out_1]"+r"(out_o1hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*8),
                     [fhfw]"r"((I64)fh*fw),
                     [s_0]"r"(s_o0),
                     [s_1]"r"(s_o1),
                     [b_0]"r"(b_o0),
                     [b_1]"r"(b_o1)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "x0", "x1", "x2", "x3", "x4", "x9"
                );
                s0 += 16;
                s1 += 16;
                b0 += 16;
                b1 += 16;
            }
        }
        // ohow_remainder % 8 / 4
        U32 ohow_s = (ohow / 8) * 8;

        for (U32 hw = ohow_s; hw < ohow-3; hw+=4) {
            const F16 *s0 = scaleArray;
            const F16 *s1 = scaleArray + 8;
            const F16 *b0 = biasArray;
            const F16 *b1 = biasArray + 8;
            BIN8 *in_order = ((BIN8*)tmp) + ic*ihiw;  // ic has been divided by 8
            // reorder input
            // NCHWc8 => NHWChw4c8 + im2col
            U32 in_h[4];
            U32 in_w[4];
            for (U32 i = 0; i < 4; i++) {
                in_h[i] = ((hw+i)/ow)*strideH;
                in_w[i] = ((hw+i)%ow)*strideW;
            }
            for (U32 c = 0; c < ic; c++) {
                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        BIN8 *in_hw4c8 = inArray_pad + c*ihiw + fh_idx*iw_pad + fw_idx;
                        // NHWChw4c8
                        BIN8 *in_order_hw4c8 = in_order + c*fh*fw*4 + fh_idx*fw*4 + fw_idx*4;
                        for (U32 i = 0; i < 4; i++) {
                            in_order_hw4c8[i] = *(in_hw4c8 + in_h[i]*iw_pad + in_w[i]);
                        }
                    }
                }
            }

            // compute
            for (U32 o = 0; o < oc; o+=2) {  // oc should be multiple of 32. It will at least be multiple of 16 in the future.
                BIN8 *in_hw0 = in_order;
                const BIN8 *f_o0c0 = filterArray + o*8*fh*fw*ic;  // ic has been divided by 8
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                F16 *out_o1hw0 = out_o0hw0 + ohow*8;
                // scale and bias
                const F16 *s_o0 = s0;
                const F16 *s_o1 = s1;
                const F16 *b_o0 = b0;
                const F16 *b_o1 = b1;
                __asm__ __volatile__(
                    "ldr q0, [%[f_0]]\n"            //f_0
                    "ldr s29, [%[in_0]]\n"           //in_0
            /* Layout
            5   6
            7   8
            9   10
            11  12
            */
                    "eor v5.16b, v5.16b, v5.16b\n"
                    "eor v6.16b, v6.16b, v6.16b\n"
                    "eor v7.16b, v7.16b, v7.16b\n"
                    "eor v8.16b, v8.16b, v8.16b\n"
                    "eor v9.16b, v9.16b, v9.16b\n"
                    "dup v1.16b, v29.b[0]\n"        //duplicate a full register
                    "eor v10.16b, v10.16b, v10.16b\n"
                    "dup v2.16b, v29.b[1]\n"
                    "eor v11.16b, v11.16b, v11.16b\n"
                    "eor v12.16b, v12.16b, v12.16b\n"

                    //give in address to x3
                    "mov x3, %[in_0]\n"

                    //give f address to x0
                    "mov x0, %[f_0]\n"

                    "mov  x2, %[ic]\n"             //ic_blk

                    "0:\n"
                        "eor v21.16b, v21.16b, v21.16b\n"
                        "mov  x9, %[fhfw]\n"
                        "eor v22.16b, v22.16b, v22.16b\n"
                        "eor v23.16b, v23.16b, v23.16b\n"
                        "eor v24.16b, v24.16b, v24.16b\n"

                        "mov x4, #4\n"

                        "1:\n"
                            "and v3.16b, v1.16b, v0.16b\n"
                            "and v4.16b, v2.16b, v0.16b\n"

                            "cnt v3.16b, v3.16b\n"
                            "subs x4, x4, #1\n"

                            "cnt v4.16b, v4.16b\n"
                            "dup v1.16b, v29.b[2]\n"
                            
                            "add v21.16b, v21.16b, v3.16b\n" // Use add because the latency is shorter
                            "dup v2.16b, v29.b[3]\n"

                            "add v22.16b, v22.16b, v4.16b\n"

                            "and v3.16b, v1.16b, v0.16b\n"
                            "ldr s29, [x3, 4]!\n"
                            "and v4.16b, v2.16b, v0.16b\n"
                            "cnt v3.16b, v3.16b\n"
                            "ldr q0, [x0, 16]!\n" // next filter
                            "cnt v4.16b, v4.16b\n"
                            "dup v1.16b, v29.b[0]\n"
                            "add v23.16b, v23.16b, v3.16b\n"
                            "dup v2.16b, v29.b[1]\n"
                            "add v24.16b, v24.16b, v4.16b\n"
                        "bne 1b\n"

                        "movi v3.16b, #1\n"
                        "umlal v5.8h, v21.8b, v3.8b\n"
                        "umlal v7.8h, v22.8b, v3.8b\n"
                        "umlal v9.8h, v23.8b, v3.8b\n"
                        "umlal v11.8h, v24.8b, v3.8b\n"
                        
                        "umlal2 v6.8h, v21.16b, v3.16b\n"
                        "umlal2 v8.8h, v22.16b, v3.16b\n"
                        "umlal2 v10.8h, v23.16b, v3.16b\n"
                        "umlal2 v12.8h, v24.16b, v3.16b\n"

                        "subs x9, x9, #1\n"
                        "beq 4f\n" // 1x1, continue with the next 32 input channels

                        "2:\n"
                        "eor v21.16b, v21.16b, v21.16b\n"
                        "eor v22.16b, v22.16b, v22.16b\n"
                        "eor v23.16b, v23.16b, v23.16b\n"
                        "eor v24.16b, v24.16b, v24.16b\n"

                        "mov x4, #32\n" // Assume 256 will not happen
                        "3:\n"
                            "and v3.16b, v1.16b, v0.16b\n"
                            "and v4.16b, v2.16b, v0.16b\n"

                            "cnt v3.16b, v3.16b\n"
                            "subs x4, x4, #1\n"

                            "cnt v4.16b, v4.16b\n"
                            "dup v1.16b, v29.b[2]\n"
                            
                            "uqadd v21.16b, v21.16b, v3.16b\n"
                            "dup v2.16b, v29.b[3]\n"

                            "uqadd v22.16b, v22.16b, v4.16b\n"

                            "and v3.16b, v1.16b, v0.16b\n"
                            "ldr s29, [x3, 4]!\n"
                            "and v4.16b, v2.16b, v0.16b\n"
                            "cnt v3.16b, v3.16b\n"
                            "ldr q0, [x0, 16]!\n" // next filter
                            "cnt v4.16b, v4.16b\n"
                            "dup v1.16b, v29.b[0]\n"
                            "uqadd v23.16b, v23.16b, v3.16b\n"
                            "dup v2.16b, v29.b[1]\n"
                            "uqadd v24.16b, v24.16b, v4.16b\n"
                        "bne 3b\n"

                        "movi v3.16b, #1\n"
                        "umlal v5.8h, v21.8b, v3.8b\n"
                        "umlal v7.8h, v22.8b, v3.8b\n"
                        "umlal v9.8h, v23.8b, v3.8b\n"
                        "umlal v11.8h, v24.8b, v3.8b\n"
                        
                        "umlal2 v6.8h, v21.16b, v3.16b\n"
                        "umlal2 v8.8h, v22.16b, v3.16b\n"
                        "umlal2 v10.8h, v23.16b, v3.16b\n"
                        "umlal2 v12.8h, v24.16b, v3.16b\n"

                        "subs x9, x9, #8\n"
                        "bne 2b\n"

                        "4:\n" // Wrap up computation for 32 input channels
                        "subs x2, x2, #32\n"
                    "bne 0b\n"

                    // pipelined
                    "ucvtf v5.8h, v5.8h\n"
                    "ucvtf v6.8h, v6.8h\n"
                    "ldr q21, [%[b_0]]\n"
                    "ucvtf v7.8h, v7.8h\n"
                    "ldr q22, [%[b_1]]\n"
                    "ucvtf v8.8h, v8.8h\n"
                    "ldr q23, [%[s_0]]\n"
                    "ucvtf v9.8h, v9.8h\n"
                    "ldr q24, [%[s_1]]\n"
                    "ucvtf v10.8h, v10.8h\n"
                    "ucvtf v11.8h, v11.8h\n"
                    "mov v1.16b, v21.16b\n"
                    "ucvtf v12.8h, v12.8h\n"
                    "mov v2.16b, v22.16b\n"
                    "fmla v1.8h, v5.8h, v23.8h\n"
                    "fmla v2.8h, v6.8h, v24.8h\n"
                    "mov v3.16b, v21.16b\n"
                    "mov v4.16b, v22.16b\n"
                    "fmla v3.8h, v7.8h, v23.8h\n"
                    "fmla v4.8h, v8.8h, v24.8h\n"
                    "mov v5.16b, v21.16b\n"
                    "mov v6.16b, v22.16b\n"

                    "fmla v5.8h, v9.8h, v23.8h\n"
                    "mov v7.16b, v21.16b\n"
                    "fmla v6.8h, v10.8h, v24.8h\n"
                    "mov v8.16b, v22.16b\n"
                    "fmla v7.8h, v11.8h, v23.8h\n"
                    "fmla v8.8h, v12.8h, v24.8h\n"
                    
                    "str   q1, [%[out_0]]\n"           //out_o0hw0
                    "str   q3, [%[out_0], #16]\n"      //out_o0hw1
                    "str   q5, [%[out_0], #32]\n"      //out_o0hw2
                    "str   q7, [%[out_0], #48]\n"      //out_o0hw3

                    "str  q2, [%[out_1]]\n"           //out_o1hw0
                    "str  q4, [%[out_1], #16]\n"      //out_o1hw1
                    "str  q6, [%[out_1], #32]\n"      //out_o1hw2
                    "str  q8, [%[out_1], #48]\n"      //out_o1hw3
                    :[out_0]"+r"(out_o0hw0),
                     [out_1]"+r"(out_o1hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*8),
                     [fhfw]"r"((I64)fh*fw),
                     [s_0]"r"(s_o0),
                     [s_1]"r"(s_o1),
                     [b_0]"r"(b_o0),
                     [b_1]"r"(b_o1)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v21", "v22", "v23", "v24", "v29", "v30", "x0", "x1", "x2", "x3", "x4", "x9"
                );
                s0 += 16;
                s1 += 16;
                b0 += 16;
                b1 += 16;
            }
        }
        // ohow_reminder % 4
        ohow_s = (ohow / 4) * 4;
        for (U32 hw = ohow_s; hw < ohow; hw++) {
            const F16 *s0 = scaleArray;
            const F16 *s1 = scaleArray + 8;
            const F16 *b0 = biasArray;
            const F16 *b1 = biasArray + 8;
            BIN8 *in_order = ((BIN8*)tmp) + ic*ih_pad*iw_pad;  // ic has been divided by 8
            // reorder input
            // NCHWc8 => NHWChw1c8 + im2col
            U32 in_h_0 = (hw/ow)*strideH;
            U32 in_w_0 = (hw%ow)*strideW;
            for (U32 c = 0; c < ic; c++) {
                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        BIN8 *in_hw1c8 = inArray_pad + c*ihiw + fh_idx*iw_pad + fw_idx;
                        BIN8 *in_0 = in_hw1c8 + in_h_0*iw_pad + in_w_0;
                        BIN8 *in_order_hw1c8 = in_order + c*fh*fw + fh_idx*fw + fw_idx;
                        *in_order_hw1c8 = (*in_0);
                    }
                }
            }
            // compute
            for (U32 o = 0; o < oc; o+=2) {
                BIN8 *in_hw0 = in_order;
                const BIN8 *f_o = filterArray + o*8*fh*fw*ic;
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                F16 *out_o1hw0 = out_o0hw0 + ohow*8;
                
                uint16x8_t sum[2] = {0};
                uint8x8_t v1 = vdup_n_u8(1);
                for (U32 i = 0; i < ic*8; i += 32) {
                    uint8x8_t sub0[2] = {0};

                    for (U32 j = 0; j < 4; j++) {
                        uint8x8_t f_0 = vld1_u8(f_o);
                        uint8x8_t f_1 = vld1_u8(f_o+8);
                        f_o += 16;
                        uint8x8_t in_1 = vdup_n_u8(*in_hw0);
                        in_hw0++;
                        f_0 = vand_u8(in_1, f_0);
                        f_1 = vand_u8(in_1, f_1);
                        f_0 = vcnt_u8(f_0);
                        f_1 = vcnt_u8(f_1);
                        sub0[0] = vadd_u8(sub0[0], f_0);
                        sub0[1] = vadd_u8(sub0[1], f_1);
                    }
                    sum[0] = vmlal_u8(sum[0], sub0[0], v1);
                    sum[1] = vmlal_u8(sum[1], sub0[1], v1);

                    for (U32 j = 1; j < fh*fw; j += 8) {
                        uint8x8_t sub1[2] = {0};
                        for (U32 k = 0; k < 32; k++) {
                            uint8x8_t f_0 = vld1_u8(f_o);
                            uint8x8_t f_1 = vld1_u8(f_o+8);
                            f_o += 16;
                            uint8x8_t in_1 = vdup_n_u8(*in_hw0);
                            in_hw0++;
                            f_0 = vand_u8(in_1, f_0);
                            f_1 = vand_u8(in_1, f_1);
                            f_0 = vcnt_u8(f_0);
                            f_1 = vcnt_u8(f_1);
                            sub1[0] = vadd_u8(sub1[0], f_0);
                            sub1[1] = vadd_u8(sub1[1], f_1);
                        }
                        sum[0] = vmlal_u8(sum[0], sub1[0], v1);
                        sum[1] = vmlal_u8(sum[1], sub1[1], v1);
                    }
                }

                float16x8_t res_o0 = vcvtq_f16_u16(sum[0]);
                float16x8_t res_o1 = vcvtq_f16_u16(sum[1]);
                float16x8_t scale_o0 = vld1q_f16(s0);
                s0 += 16;
                float16x8_t scale_o1 = vld1q_f16(s1);
                s1 += 16;
                float16x8_t bias_o0 = vld1q_f16(b0);
                b0 += 16;
                float16x8_t bias_o1 = vld1q_f16(b1);
                b1 += 16;
                res_o0 = vmulq_f16(res_o0, scale_o0);
                res_o1 = vmulq_f16(res_o1, scale_o1);
                res_o0 = vaddq_f16(res_o0, bias_o0);
                res_o1 = vaddq_f16(res_o1, bias_o1);
                vst1q_f16(out_o0hw0, res_o0);
                vst1q_f16(out_o1hw0, res_o1);
            }
        }
    }
    return SUCCESS;
}
#endif
