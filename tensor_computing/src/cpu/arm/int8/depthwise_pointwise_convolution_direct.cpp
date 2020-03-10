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
#include "cpu/arm/int8/depthwise_convolution.h"

EE depthwise_pointwise_convolution_direct(TensorDesc inputDesc, INT8* inArray,
    TensorDesc filterDesc, const INT8* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const I32* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, I32* outArray,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode,
    Arch arch)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    UNUSED(arch);

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
    U32 dilateH = convDesc.dilatedRate_h;
    U32 dilateW = convDesc.dilatedRate_w;

    if (fdf != DF_CHWC8_NCN8C4)
        CHECK_STATUS(NOT_MATCH);

    oc /= 8;
    ic /= 8;

    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;
    U32 ihiw = ih*iw;
    I32 ohow = oh*ow;
    INT8 *pwArray = (INT8*)tmp + ic*ih_pad*iw_pad*8;
    I32 *dw_out = (I32 *)(pwArray + ic*ohow*8);

    for (U32 n = 0; n < in; n++) {
        // copy input into a input with padding
        INT8 *inArray_pad = (INT8*)tmp;
        INT8 *inArray_pad_mov = inArray_pad;
        INT8 *inArray_mov = inArray + n*ic*ihiw*8;
        for (U32 c = 0; c < ic; c++) {
            for (U32 h = 0; h < paddingT; h++) {
                memset(inArray_pad_mov, 0, iw_pad*8*bytesOf(idt));
                inArray_pad_mov += iw_pad*8;
            }
            for (U32 h = paddingT; h < ih_pad - paddingB; h++) {
                memset(inArray_pad_mov, 0, paddingL*8*bytesOf(idt));
                inArray_pad_mov += paddingL*8;
                memcpy(inArray_pad_mov, inArray_mov, iw*8*bytesOf(idt));
                inArray_pad_mov += iw*8;
                inArray_mov += iw*8;
                memset(inArray_pad_mov, 0, paddingR*8*bytesOf(idt));
                inArray_pad_mov += paddingR*8;
            }
            for (U32 h = ih_pad - paddingB; h < ih_pad; h++) {
                memset(inArray_pad_mov, 0, iw_pad*8*bytesOf(idt));
                inArray_pad_mov += iw_pad*8;
            }
        }

        // dw_conv
        for (U32 c = 0; c < ic ; c++) {
            const I32 *b = biasArray + c*8;
            INT8 *in_pad = inArray_pad + c*ih_pad*iw_pad*8;
            const INT8 *f = filterArray + c*fh*fw*8;

            // ohow / 12
            for (I32 hw = 0; hw < ohow-11; hw+=12) {
                U32 in_h_0 = hw/ow*strideH;
                U32 in_w_0 = hw%ow*strideW;
                U32 in_h_1 = (hw+1)/ow*strideH;
                U32 in_w_1 = (hw+1)%ow*strideW;
                U32 in_h_2 = (hw+2)/ow*strideH;
                U32 in_w_2 = (hw+2)%ow*strideW;
                U32 in_h_3 = (hw+3)/ow*strideH;
                U32 in_w_3 = (hw+3)%ow*strideW;
                U32 in_h_4 = (hw+4)/ow*strideH;
                U32 in_w_4 = (hw+4)%ow*strideW;
                U32 in_h_5 = (hw+5)/ow*strideH;
                U32 in_w_5 = (hw+5)%ow*strideW;
                U32 in_h_6 = (hw+6)/ow*strideH;
                U32 in_w_6 = (hw+6)%ow*strideW;
                U32 in_h_7 = (hw+7)/ow*strideH;
                U32 in_w_7 = (hw+7)%ow*strideW;
                U32 in_h_8 = (hw+8)/ow*strideH;
                U32 in_w_8 = (hw+8)%ow*strideW;
                U32 in_h_9 = (hw+9)/ow*strideH;
                U32 in_w_9 = (hw+9)%ow*strideW;
                U32 in_h_10 = (hw+10)/ow*strideH;
                U32 in_w_10 = (hw+10)%ow*strideW;
                U32 in_h_11 = (hw+11)/ow*strideH;
                U32 in_w_11 = (hw+11)%ow*strideW;

                I32 *pw_pack_0 = dw_out + hw*ic*8 + c*12*8;
                I32 *pw_pack_1 = pw_pack_0 + 48; // Second half
                //TODO handle asm combined with c. No guarantee that compiler will not use vec reg in c.
                __asm__ __volatile__(
                    "ldr d29, [%[b]]\n"       //b_0
                    "ldr  x1, [%[b], #8]\n"
                    "ins v29.d[1], x1\n"
                    "ldr d30, [%[b], #16]\n"       //b_1
                    "ldr  x2, [%[b], #24]\n"
                    "ins v30.d[1], x2\n"
                    "mov  v5.16b, v29.16b\n"
                    "mov  v7.16b, v29.16b\n"
                    "mov  v9.16b, v29.16b\n"
                    "mov  v11.16b, v29.16b\n"
                    "mov  v13.16b, v29.16b\n"
                    "mov  v15.16b, v29.16b\n"
                    "mov  v17.16b, v29.16b\n"
                    "mov  v19.16b, v29.16b\n"
                    "mov  v21.16b, v29.16b\n"
                    "mov  v23.16b, v29.16b\n"
                    "mov  v25.16b, v29.16b\n"
                    "mov  v27.16b, v29.16b\n"
                    
                    "mov  v6.16b, v30.16b\n"
                    "mov  v8.16b, v30.16b\n"
                    "mov  v10.16b, v30.16b\n"
                    "mov  v12.16b, v30.16b\n"
                    "mov  v14.16b, v30.16b\n"
                    "mov  v16.16b, v30.16b\n"
                    "mov  v18.16b, v30.16b\n"
                    "mov  v20.16b, v30.16b\n"
                    "mov  v22.16b, v30.16b\n"
                    "mov  v24.16b, v30.16b\n"
                    "mov  v26.16b, v30.16b\n"
                    "mov  v28.16b, v30.16b\n"
                    :
                    :[b]"r"(b)
                    :"memory", "cc", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "x1", "x2"
                );

                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        const INT8 *f_0 = f + fh_idx*fw*8 + fw_idx*8;
                        INT8 *in_idx = in_pad + fh_idx*dilateH*iw_pad*8 + fw_idx*dilateW*8;
                        INT8 *in_0 = in_idx + in_h_0*iw_pad*8 + in_w_0*8;
                        INT8 *in_1 = in_idx + in_h_1*iw_pad*8 + in_w_1*8;
                        INT8 *in_2 = in_idx + in_h_2*iw_pad*8 + in_w_2*8;
                        INT8 *in_3 = in_idx + in_h_3*iw_pad*8 + in_w_3*8;
                        INT8 *in_4 = in_idx + in_h_4*iw_pad*8 + in_w_4*8;
                        INT8 *in_5 = in_idx + in_h_5*iw_pad*8 + in_w_5*8;
                        INT8 *in_6 = in_idx + in_h_6*iw_pad*8 + in_w_6*8;
                        INT8 *in_7 = in_idx + in_h_7*iw_pad*8 + in_w_7*8;
                        INT8 *in_8 = in_idx + in_h_8*iw_pad*8 + in_w_8*8;
                        INT8 *in_9 = in_idx + in_h_9*iw_pad*8 + in_w_9*8;
                        INT8 *in_10 = in_idx + in_h_10*iw_pad*8 + in_w_10*8;
                        INT8 *in_11 = in_idx + in_h_11*iw_pad*8 + in_w_11*8;
                        __asm__ __volatile__(
                            "ldr d29, [%[f0]]\n"
                            "ldr d0, [%[in0]]\n"
                            "ldr d1, [%[in1]]\n"
                            "ldr d2, [%[in2]]\n"
                            "sshll v29.8h, v29.8b, #0\n"
                            "ldr d30, [%[in3]]\n"
                            "sshll v0.8h, v0.8b, #0\n"
                            "sshll v1.8h, v1.8b, #0\n"

                            "smlal v5.4s, v29.4h, v0.4h\n"
                            "sshll v2.8h, v2.8b, #0\n"
                            "smlal2 v6.4s, v29.8h, v0.8h\n"
                            "sshll v30.8h, v30.8b, #0\n"
                            "smlal v7.4s, v29.4h, v1.4h\n"
                            "ldr d0, [%[in4]]\n"
                            "smlal2 v8.4s, v29.8h, v1.8h\n"
                            "smlal v9.4s, v29.4h, v2.4h\n"
                            "ldr d1, [%[in5]]\n"
                            "smlal2 v10.4s, v29.8h, v2.8h\n"
                            "sshll v0.8h, v0.8b, #0\n"
                            "smlal v11.4s, v29.4h, v30.4h\n"
                            "ldr d2, [%[in6]]\n"
                            "smlal2 v12.4s, v29.8h, v30.8h\n"
                            "sshll v1.8h, v1.8b, #0\n"

                            "smlal v13.4s, v29.4h, v0.4h\n"
                            "ldr d30, [%[in7]]\n"
                            "smlal2 v14.4s, v29.8h, v0.8h\n"
                            "sshll v2.8h, v2.8b, #0\n"
                            "smlal v15.4s, v29.4h, v1.4h\n"
                            "ldr d0, [%[in8]]\n"
                            "smlal2 v16.4s, v29.8h, v1.8h\n"
                            "sshll v30.8h, v30.8b, #0\n"
                            "smlal v17.4s, v29.4h, v2.4h\n"
                            "ldr d1, [%[in9]]\n"
                            "smlal2 v18.4s, v29.8h, v2.8h\n"
                            "sshll v0.8h, v0.8b, #0\n"
                            "smlal v19.4s, v29.4h, v30.4h\n"
                            "ldr d2, [%[in10]]\n"
                            "smlal2 v20.4s, v29.8h, v30.8h\n"
                            "sshll v1.8h, v1.8b, #0\n"

                            "smlal v21.4s, v29.4h, v0.4h\n"
                            "ldr d30, [%[in11]]\n"
                            "smlal2 v22.4s, v29.8h, v0.8h\n"
                            "sshll v2.8h, v2.8b, #0\n"
                            "smlal v23.4s, v29.4h, v1.4h\n"
                            "sshll v30.8h, v30.8b, #0\n"
                            "smlal2 v24.4s, v29.8h, v1.8h\n"
                            "smlal v25.4s, v29.4h, v2.4h\n"
                            "smlal2 v26.4s, v29.8h, v2.8h\n"
                            "smlal v27.4s, v29.4h, v30.4h\n"
                            "smlal2 v28.4s, v29.8h, v30.8h\n"
                            :
                            :[in0]"r"(in_0),
                             [in1]"r"(in_1),
                             [in2]"r"(in_2),
                             [in3]"r"(in_3),
                             [in4]"r"(in_4),
                             [in5]"r"(in_5),
                             [in6]"r"(in_6),
                             [in7]"r"(in_7),
                             [in8]"r"(in_8),
                             [in9]"r"(in_9),
                             [in10]"r"(in_10),
                             [in11]"r"(in_11),
                             [f0]"r"(f_0)
                            :"memory", "cc", "v0", "v1", "v2", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30"
                        );
                    }
                }

                // activation
                switch (depthwiseActivationMode){
                    case ACTIVATION_NULL: {
                        break;
                    }
                    case ACTIVATION_RELU:{
                        __asm__ __volatile__(
                            "eor v0.16b, v0.16b, v0.16b\n" // zero
                            
                            "smax v5.4s, v0.4s, v5.4s\n"
                            "smax v6.4s, v0.4s, v6.4s\n"
                            "smax v7.4s, v0.4s, v7.4s\n"
                            "smax v8.4s, v0.4s, v8.4s\n"
                            "smax v9.4s, v0.4s, v9.4s\n"
                            "smax v10.4s, v0.4s, v10.4s\n"
                            "smax v11.4s, v0.4s, v11.4s\n"
                            "smax v12.4s, v0.4s, v12.4s\n"
                            "smax v13.4s, v0.4s, v13.4s\n"
                            "smax v14.4s, v0.4s, v14.4s\n"
                            "smax v15.4s, v0.4s, v15.4s\n"
                            "smax v16.4s, v0.4s, v16.4s\n"
                            "smax v17.4s, v0.4s, v17.4s\n"
                            "smax v18.4s, v0.4s, v18.4s\n"
                            "smax v19.4s, v0.4s, v19.4s\n"
                            "smax v20.4s, v0.4s, v20.4s\n"
                            "smax v21.4s, v0.4s, v21.4s\n"
                            "smax v22.4s, v0.4s, v22.4s\n"
                            "smax v23.4s, v0.4s, v23.4s\n"
                            "smax v24.4s, v0.4s, v24.4s\n"
                            "smax v25.4s, v0.4s, v25.4s\n"
                            "smax v26.4s, v0.4s, v26.4s\n"
                            "smax v27.4s, v0.4s, v27.4s\n"
                            "smax v28.4s, v0.4s, v28.4s\n"
                            :
                            :
                            :"memory", "cc", "v0", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28"
                        );
                        break;
                    }
                    case ACTIVATION_RELU6:{
                        INT8* pw_in0 = pwArray + hw*ic*8 + c*12*8;
                        INT8* pw_in1 = pw_in0 + 48;
                        __asm__ __volatile__(
                            "eor v0.16b, v0.16b, v0.16b\n" // zero
                            "movi v30.4s, #6\n"  // six

                            "smax v5.4s, v0.4s, v5.4s\n"
                            "smax v6.4s, v0.4s, v6.4s\n"
                            "smax v7.4s, v0.4s, v7.4s\n"
                            "smax v8.4s, v0.4s, v8.4s\n"
                            "smax v9.4s, v0.4s, v9.4s\n"
                            "smax v10.4s, v0.4s, v10.4s\n"
                            "smax v11.4s, v0.4s, v11.4s\n"
                            "smax v12.4s, v0.4s, v12.4s\n"
                            "smax v13.4s, v0.4s, v13.4s\n"
                            "smax v14.4s, v0.4s, v14.4s\n"
                            "smax v15.4s, v0.4s, v15.4s\n"
                            "smax v16.4s, v0.4s, v16.4s\n"
                            "smax v17.4s, v0.4s, v17.4s\n"
                            "smax v18.4s, v0.4s, v18.4s\n"
                            "smax v19.4s, v0.4s, v19.4s\n"
                            "smax v20.4s, v0.4s, v20.4s\n"
                            "smax v21.4s, v0.4s, v21.4s\n"
                            "smax v22.4s, v0.4s, v22.4s\n"
                            "smax v23.4s, v0.4s, v23.4s\n"
                            "smax v24.4s, v0.4s, v24.4s\n"
                            "smax v25.4s, v0.4s, v25.4s\n"
                            "smax v26.4s, v0.4s, v26.4s\n"
                            "smax v27.4s, v0.4s, v27.4s\n"
                            "smax v28.4s, v0.4s, v28.4s\n"

                            "smin v5.4s, v30.4s, v5.4s\n"
                            "smin v6.4s, v30.4s, v6.4s\n"
                            "smin v7.4s, v30.4s, v7.4s\n"
                            "smin v8.4s, v30.4s, v8.4s\n"
                            "smin v9.4s, v30.4s, v9.4s\n"
                            "smin v10.4s, v30.4s, v10.4s\n"
                            "smin v11.4s, v30.4s, v11.4s\n"
                            "smin v12.4s, v30.4s, v12.4s\n"
                            "smin v13.4s, v30.4s, v13.4s\n"
                            "smin v14.4s, v30.4s, v14.4s\n"
                            "smin v15.4s, v30.4s, v15.4s\n"
                            "smin v16.4s, v30.4s, v16.4s\n"
                            "smin v17.4s, v30.4s, v17.4s\n"
                            "smin v18.4s, v30.4s, v18.4s\n"
                            "smin v19.4s, v30.4s, v19.4s\n"
                            "smin v20.4s, v30.4s, v20.4s\n"
                            "smin v21.4s, v30.4s, v21.4s\n"
                            "smin v22.4s, v30.4s, v22.4s\n"
                            "smin v23.4s, v30.4s, v23.4s\n"
                            "smin v24.4s, v30.4s, v24.4s\n"
                            "smin v25.4s, v30.4s, v25.4s\n"
                            "smin v26.4s, v30.4s, v26.4s\n"
                            "smin v27.4s, v30.4s, v27.4s\n"
                            "smin v28.4s, v30.4s, v28.4s\n"

                            // No need to quantize for ReLU6
                            "sqshl v5.4s, v5.4s, #2\n"
                            "sqshl v6.4s, v6.4s, #2\n"
                            "sqshl v7.4s, v7.4s, #2\n"
                            "sqshl v8.4s, v8.4s, #2\n"
                            "sqshl v9.4s, v9.4s, #2\n"
                            "sqshl v10.4s, v10.4s, #2\n"
                            "sqshl v11.4s, v11.4s, #2\n"
                            "sqshl v12.4s, v12.4s, #2\n"
                            "sqshl v13.4s, v13.4s, #2\n"
                            "sqshl v14.4s, v14.4s, #2\n"
                            "sqshl v15.4s, v15.4s, #2\n"
                            "sqshl v16.4s, v16.4s, #2\n"
                            "sqshl v17.4s, v17.4s, #2\n"
                            "sqshl v18.4s, v18.4s, #2\n"
                            "sqshl v19.4s, v19.4s, #2\n"
                            "sqshl v20.4s, v20.4s, #2\n"
                            "sqshl v21.4s, v21.4s, #2\n"
                            "sqshl v22.4s, v22.4s, #2\n"
                            "sqshl v23.4s, v23.4s, #2\n"
                            "sqshl v24.4s, v24.4s, #2\n"
                            "sqshl v25.4s, v25.4s, #2\n"
                            "sqshl v26.4s, v26.4s, #2\n"
                            "sqshl v27.4s, v27.4s, #2\n"
                            "sqshl v28.4s, v28.4s, #2\n"

                            "sqshrn v5.4h, v5.4s, #1\n"
                            "sqshrn v9.4h, v9.4s, #1\n"
                            "sqshrn2 v5.8h, v7.4s, #1\n"
                            "sqshrn2 v9.8h, v11.4s, #1\n"
                            "sqshrn v13.4h, v13.4s, #1\n"
                            "sqshrn v17.4h, v17.4s, #1\n"
                            "sqshrn2 v13.8h, v15.4s, #1\n"
                            "sqshrn2 v17.8h, v19.4s, #1\n"

                            "sqshrn v21.4h, v21.4s, #1\n"
                            "sqshrn v25.4h, v25.4s, #1\n"
                            "sqshrn2 v21.8h, v23.4s, #1\n"
                            "sqshrn2 v25.8h, v27.4s, #1\n"

                            "sqshrn v5.8b, v5.8h, #1\n"
                            "sqshrn v13.8b, v13.8h, #1\n"
                            "sqshrn v21.8b, v21.8h, #1\n"

                            "sqshrn2 v5.16b, v9.8h, #1\n"
                            "sqshrn2 v13.16b, v17.8h, #1\n"
                            "sqshrn2 v21.16b, v25.8h, #1\n"
                            "str q5, [%[in0]]\n"
                            "str q13, [%[in0], #16]\n"
                            "str q21, [%[in0], #32]\n"

                            "sqshrn v6.4h, v6.4s, #1\n"
                            "sqshrn v10.4h, v10.4s, #1\n"
                            "sqshrn2 v6.8h, v8.4s, #1\n"
                            "sqshrn2 v10.8h, v12.4s, #1\n"
                            
                            "sqshrn v14.4h, v14.4s, #1\n"
                            "sqshrn v18.4h, v18.4s, #1\n"
                            "sqshrn2 v14.8h, v16.4s, #1\n"
                            "sqshrn2 v18.8h, v20.4s, #1\n"

                            "sqshrn v22.4h, v22.4s, #1\n"
                            "sqshrn v26.4h, v26.4s, #1\n"
                            "sqshrn2 v22.8h, v24.4s, #1\n"
                            "sqshrn2 v26.8h, v28.4s, #1\n"

                            "sqshrn v6.8b, v6.8h, #1\n"
                            "sqshrn v14.8b, v14.8h, #1\n"
                            "sqshrn v22.8b, v22.8h, #1\n"

                            "sqshrn2 v6.16b, v10.8h, #1\n"
                            "sqshrn2 v14.16b, v18.8h, #1\n"
                            "sqshrn2 v22.16b, v26.8h, #1\n"
                            "str q6, [%[in1]]\n"
                            "str q14, [%[in1], #16]\n"
                            "str q22, [%[in1], #32]\n"
                            :
                            :[in0]"r"(pw_in0),
                             [in1]"r"(pw_in1)
                            :"memory", "cc", "v0", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v30"
                        );
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }

                if (depthwiseActivationMode != ACTIVATION_RELU6) {
                    __asm__ __volatile__(
                        "str q5, [%[pw0]]\n"
                        "str q7, [%[pw0], #16]\n"
                        "str q9, [%[pw0], #32]\n"
                        "str q11, [%[pw0], #48]\n"
                        "str q13, [%[pw0], #64]\n"
                        "str q15, [%[pw0], #80]\n"
                        "str q17, [%[pw0], #96]\n"
                        "str q19, [%[pw0], #112]\n"
                        "str q21, [%[pw0], #128]\n"
                        "str q23, [%[pw0], #144]\n"
                        "str q25, [%[pw0], #160]\n"
                        "str q27, [%[pw0], #176]\n"

                        "str q6, [%[pw1]]\n"
                        "str q8, [%[pw1], #16]\n"
                        "str q10, [%[pw1], #32]\n"
                        "str q12, [%[pw1], #48]\n"
                        "str q14, [%[pw1], #64]\n"
                        "str q16, [%[pw1], #80]\n"
                        "str q18, [%[pw1], #96]\n"
                        "str q20, [%[pw1], #112]\n"
                        "str q22, [%[pw1], #128]\n"
                        "str q24, [%[pw1], #144]\n"
                        "str q26, [%[pw1], #160]\n"
                        "str q28, [%[pw1], #176]\n"
                        :
                        :[pw0]"r"(pw_pack_0),
                         [pw1]"r"(pw_pack_1)
                        :"memory", "cc", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28"
                    );
                }
            }

            // ohow_reminder % 12 / 8
            U32 ohow_s = (ohow / 12) * 12;
            U32 ohow_tail = ohow - ohow_s;

            if (ohow_tail >= 8) {
                U32 hw = ohow_s;
                U32 in_h_0 = hw/ow*strideH;
                U32 in_w_0 = hw%ow*strideW;
                U32 in_h_1 = (hw+1)/ow*strideH;
                U32 in_w_1 = (hw+1)%ow*strideW;
                U32 in_h_2 = (hw+2)/ow*strideH;
                U32 in_w_2 = (hw+2)%ow*strideW;
                U32 in_h_3 = (hw+3)/ow*strideH;
                U32 in_w_3 = (hw+3)%ow*strideW;
                U32 in_h_4 = ((hw+4)/ow)*strideH;
                U32 in_w_4 = ((hw+4)%ow)*strideW;
                U32 in_h_5 = ((hw+5)/ow)*strideH;
                U32 in_w_5 = ((hw+5)%ow)*strideW;
                U32 in_h_6 = ((hw+6)/ow)*strideH;
                U32 in_w_6 = ((hw+6)%ow)*strideW;
                U32 in_h_7 = ((hw+7)/ow)*strideH;
                U32 in_w_7 = ((hw+7)%ow)*strideW;
                I32 *pw_pack_0 = dw_out + hw*ic*8 + c*8*8;
                I32 *pw_pack_1 = pw_pack_0 + 32;
                //TODO handle asm combined with c. No guarantee that compile will not use vec reg in c.
                __asm__ __volatile__(
                    "ldr d29, [%[b]]\n"       //b_0
                    "ldr  x1, [%[b], #8]\n"
                    "ins v29.d[1], x1\n"
                    "ldr d30, [%[b], #16]\n"       //b_1
                    "ldr  x2, [%[b], #24]\n"
                    "ins v30.d[1], x2\n"
                    "mov  v5.16b, v29.16b\n"
                    "mov  v7.16b, v29.16b\n"
                    "mov  v9.16b, v29.16b\n"
                    "mov  v11.16b, v29.16b\n"
                    "mov  v13.16b, v29.16b\n"
                    "mov  v15.16b, v29.16b\n"
                    "mov  v17.16b, v29.16b\n"
                    "mov  v19.16b, v29.16b\n"
                    
                    "mov  v6.16b, v30.16b\n"
                    "mov  v8.16b, v30.16b\n"
                    "mov  v10.16b, v30.16b\n"
                    "mov  v12.16b, v30.16b\n"
                    "mov  v14.16b, v30.16b\n"
                    "mov  v16.16b, v30.16b\n"
                    "mov  v18.16b, v30.16b\n"
                    "mov  v20.16b, v30.16b\n"
                    :
                    :[b]"r"(b)
                    :"memory", "cc", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v29", "v30", "x1", "x2"
                );

                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        const INT8 *f_0 = f + fh_idx*fw*8 + fw_idx*8;
                        INT8 *in_idx = in_pad + fh_idx*dilateH*iw_pad*8 + fw_idx*dilateW*8;
                        INT8 *in_0 = in_idx + in_h_0*iw_pad*8 + in_w_0*8;
                        INT8 *in_1 = in_idx + in_h_1*iw_pad*8 + in_w_1*8;
                        INT8 *in_2 = in_idx + in_h_2*iw_pad*8 + in_w_2*8;
                        INT8 *in_3 = in_idx + in_h_3*iw_pad*8 + in_w_3*8;
                        INT8 *in_4 = in_idx + in_h_4*iw_pad*8 + in_w_4*8;
                        INT8 *in_5 = in_idx + in_h_5*iw_pad*8 + in_w_5*8;
                        INT8 *in_6 = in_idx + in_h_6*iw_pad*8 + in_w_6*8;
                        INT8 *in_7 = in_idx + in_h_7*iw_pad*8 + in_w_7*8;
                        __asm__ __volatile__(
                            "ldr d29, [%[f0]]\n"
                            "ldr d0, [%[in0]]\n"
                            "ldr d1, [%[in1]]\n"
                            "ldr d2, [%[in2]]\n"
                            "sshll v29.8h, v29.8b, #0\n"
                            "ldr d30, [%[in3]]\n"
                            "sshll v0.8h, v0.8b, #0\n"
                            "sshll v1.8h, v1.8b, #0\n"

                            "smlal v5.4s, v29.4h, v0.4h\n"
                            "sshll v2.8h, v2.8b, #0\n"
                            "smlal2 v6.4s, v29.8h, v0.8h\n"
                            "sshll v30.8h, v30.8b, #0\n"
                            "smlal v7.4s, v29.4h, v1.4h\n"
                            "ldr d0, [%[in4]]\n"
                            "smlal2 v8.4s, v29.8h, v1.8h\n"
                            "smlal v9.4s, v29.4h, v2.4h\n"
                            "ldr d1, [%[in5]]\n"
                            "smlal2 v10.4s, v29.8h, v2.8h\n"
                            "sshll v0.8h, v0.8b, #0\n"
                            "smlal v11.4s, v29.4h, v30.4h\n"
                            "ldr d2, [%[in6]]\n"
                            "smlal2 v12.4s, v29.8h, v30.8h\n"
                            "sshll v1.8h, v1.8b, #0\n"

                            "smlal v13.4s, v29.4h, v0.4h\n"
                            "ldr d30, [%[in7]]\n"
                            "smlal2 v14.4s, v29.8h, v0.8h\n"
                            "sshll v2.8h, v2.8b, #0\n"
                            "smlal v15.4s, v29.4h, v1.4h\n"
                            "smlal2 v16.4s, v29.8h, v1.8h\n"
                            "sshll v30.8h, v30.8b, #0\n"
                            "smlal v17.4s, v29.4h, v2.4h\n"
                            "smlal2 v18.4s, v29.8h, v2.8h\n"
                            "smlal v19.4s, v29.4h, v30.4h\n"
                            "smlal2 v20.4s, v29.8h, v30.8h\n"
                            :
                            :[in0]"r"(in_0),
                             [in1]"r"(in_1),
                             [in2]"r"(in_2),
                             [in3]"r"(in_3),
                             [in4]"r"(in_4),
                             [in5]"r"(in_5),
                             [in6]"r"(in_6),
                             [in7]"r"(in_7),
                             [f0]"r"(f_0)
                            :"memory", "cc", "v0", "v1", "v2", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v29", "v30"
                        );
                    }
                }

                // activation
                switch (depthwiseActivationMode){
                    case ACTIVATION_NULL: {
                        break;
                    }
                    case ACTIVATION_RELU:{
                        __asm__ __volatile__(
                            "eor v0.16b, v0.16b, v0.16b\n" // zero
                            
                            "smax v5.4s, v0.4s, v5.4s\n"
                            "smax v6.4s, v0.4s, v6.4s\n"
                            "smax v7.4s, v0.4s, v7.4s\n"
                            "smax v8.4s, v0.4s, v8.4s\n"
                            "smax v9.4s, v0.4s, v9.4s\n"
                            "smax v10.4s, v0.4s, v10.4s\n"
                            "smax v11.4s, v0.4s, v11.4s\n"
                            "smax v12.4s, v0.4s, v12.4s\n"
                            "smax v13.4s, v0.4s, v13.4s\n"
                            "smax v14.4s, v0.4s, v14.4s\n"
                            "smax v15.4s, v0.4s, v15.4s\n"
                            "smax v16.4s, v0.4s, v16.4s\n"
                            "smax v17.4s, v0.4s, v17.4s\n"
                            "smax v18.4s, v0.4s, v18.4s\n"
                            "smax v19.4s, v0.4s, v19.4s\n"
                            "smax v20.4s, v0.4s, v20.4s\n"
                            :
                            :
                            :"memory", "cc", "v0", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20"
                        );
                        break;
                    }
                    case ACTIVATION_RELU6:{
                        INT8* pw_in0 = pwArray + hw*ic*8 + c*8*8;
                        INT8* pw_in1 = pw_in0 + 32;
                        __asm__ __volatile__(
                            "eor v0.16b, v0.16b, v0.16b\n" // zero
                            "movi v30.4s, #6\n"  // six

                            "smax v5.4s, v0.4s, v5.4s\n"
                            "smax v6.4s, v0.4s, v6.4s\n"
                            "smax v7.4s, v0.4s, v7.4s\n"
                            "smax v8.4s, v0.4s, v8.4s\n"
                            "smax v9.4s, v0.4s, v9.4s\n"
                            "smax v10.4s, v0.4s, v10.4s\n"
                            "smax v11.4s, v0.4s, v11.4s\n"
                            "smax v12.4s, v0.4s, v12.4s\n"
                            "smax v13.4s, v0.4s, v13.4s\n"
                            "smax v14.4s, v0.4s, v14.4s\n"
                            "smax v15.4s, v0.4s, v15.4s\n"
                            "smax v16.4s, v0.4s, v16.4s\n"
                            "smax v17.4s, v0.4s, v17.4s\n"
                            "smax v18.4s, v0.4s, v18.4s\n"
                            "smax v19.4s, v0.4s, v19.4s\n"
                            "smax v20.4s, v0.4s, v20.4s\n"

                            "smin v5.4s, v30.4s, v5.4s\n"
                            "smin v6.4s, v30.4s, v6.4s\n"
                            "smin v7.4s, v30.4s, v7.4s\n"
                            "smin v8.4s, v30.4s, v8.4s\n"
                            "smin v9.4s, v30.4s, v9.4s\n"
                            "smin v10.4s, v30.4s, v10.4s\n"
                            "smin v11.4s, v30.4s, v11.4s\n"
                            "smin v12.4s, v30.4s, v12.4s\n"
                            "smin v13.4s, v30.4s, v13.4s\n"
                            "smin v14.4s, v30.4s, v14.4s\n"
                            "smin v15.4s, v30.4s, v15.4s\n"
                            "smin v16.4s, v30.4s, v16.4s\n"
                            "smin v17.4s, v30.4s, v17.4s\n"
                            "smin v18.4s, v30.4s, v18.4s\n"
                            "smin v19.4s, v30.4s, v19.4s\n"
                            "smin v20.4s, v30.4s, v20.4s\n"

                            // No need to quantize for ReLU6
                            "sqshl v5.4s, v5.4s, #2\n"
                            "sqshl v6.4s, v6.4s, #2\n"
                            "sqshl v7.4s, v7.4s, #2\n"
                            "sqshl v8.4s, v8.4s, #2\n"
                            "sqshl v9.4s, v9.4s, #2\n"
                            "sqshl v10.4s, v10.4s, #2\n"
                            "sqshl v11.4s, v11.4s, #2\n"
                            "sqshl v12.4s, v12.4s, #2\n"
                            "sqshl v13.4s, v13.4s, #2\n"
                            "sqshl v14.4s, v14.4s, #2\n"
                            "sqshl v15.4s, v15.4s, #2\n"
                            "sqshl v16.4s, v16.4s, #2\n"
                            "sqshl v17.4s, v17.4s, #2\n"
                            "sqshl v18.4s, v18.4s, #2\n"
                            "sqshl v19.4s, v19.4s, #2\n"
                            "sqshl v20.4s, v20.4s, #2\n"

                            "sqshrn v5.4h, v5.4s, #1\n"
                            "sqshrn v9.4h, v9.4s, #1\n"
                            "sqshrn2 v5.8h, v7.4s, #1\n"
                            "sqshrn2 v9.8h, v11.4s, #1\n"
                            
                            "sqshrn v13.4h, v13.4s, #1\n"
                            "sqshrn v17.4h, v17.4s, #1\n"
                            "sqshrn2 v13.8h, v15.4s, #1\n"
                            "sqshrn2 v17.8h, v19.4s, #1\n"

                            "sqshrn v5.8b, v5.8h, #1\n"
                            "sqshrn v13.8b, v13.8h, #1\n"

                            "sqshrn2 v5.16b, v9.8h, #1\n"
                            "sqshrn2 v13.16b, v17.8h, #1\n"
                            "str q5, [%[in0]]\n"
                            "str q13, [%[in0], #16]\n"

                            "sqshrn v6.4h, v6.4s, #1\n"
                            "sqshrn v10.4h, v10.4s, #1\n"
                            "sqshrn2 v6.8h, v8.4s, #1\n"
                            "sqshrn2 v10.8h, v12.4s, #1\n"
                            
                            "sqshrn v14.4h, v14.4s, #1\n"
                            "sqshrn v18.4h, v18.4s, #1\n"
                            "sqshrn2 v14.8h, v16.4s, #1\n"
                            "sqshrn2 v18.8h, v20.4s, #1\n"

                            "sqshrn v6.8b, v6.8h, #1\n"
                            "sqshrn v14.8b, v14.8h, #1\n"

                            "sqshrn2 v6.16b, v10.8h, #1\n"
                            "sqshrn2 v14.16b, v18.8h, #1\n"
                            "str q6, [%[in1]]\n"
                            "str q14, [%[in1], #16]\n"
                            :
                            :[in0]"r"(pw_in0),
                             [in1]"r"(pw_in1)
                            :"memory", "cc", "v0", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v30"
                        );
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }

                if (depthwiseActivationMode != ACTIVATION_RELU6) {
                    __asm__ __volatile__(
                        "str q5, [%[pw0]]\n"
                        "str q7, [%[pw0], #16]\n"
                        "str q9, [%[pw0], #32]\n"
                        "str q11, [%[pw0], #48]\n"
                        "str q13, [%[pw0], #64]\n"
                        "str q15, [%[pw0], #80]\n"
                        "str q17, [%[pw0], #96]\n"
                        "str q19, [%[pw0], #112]\n"

                        "str q6, [%[pw1]]\n"
                        "str q8, [%[pw1], #16]\n"
                        "str q10, [%[pw1], #32]\n"
                        "str q12, [%[pw1], #48]\n"
                        "str q14, [%[pw1], #64]\n"
                        "str q16, [%[pw1], #80]\n"
                        "str q18, [%[pw1], #96]\n"
                        "str q20, [%[pw1], #112]\n"
                        :
                        :[pw0]"r"(pw_pack_0),
                         [pw1]"r"(pw_pack_1)
                        :"memory", "cc", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20"
                    );
                }
                ohow_s += 8;
                ohow_tail -= 8;
            }

            if (ohow_tail >= 4) {
                U32 hw = ohow_s;
                U32 in_h_0 = hw/ow*strideH;
                U32 in_w_0 = hw%ow*strideW;
                U32 in_h_1 = (hw+1)/ow*strideH;
                U32 in_w_1 = (hw+1)%ow*strideW;
                U32 in_h_2 = (hw+2)/ow*strideH;
                U32 in_w_2 = (hw+2)%ow*strideW;
                U32 in_h_3 = (hw+3)/ow*strideH;
                U32 in_w_3 = (hw+3)%ow*strideW;
                I32 *pw_pack_0 = dw_out + hw*ic*8 + c*4*8;
                I32 *pw_pack_1 = pw_pack_0 + 16;
                //TODO handle asm combined with c. No guarantee that compile will not use vec reg in c.
                __asm__ __volatile__(
                    "ldr d29, [%[b]]\n"       //b_0
                    "ldr  x1, [%[b], #8]\n"
                    "ins v29.d[1], x1\n"
                    "ldr d30, [%[b], #16]\n"       //b_1
                    "ldr  x2, [%[b], #24]\n"
                    "ins v30.d[1], x2\n"
                    "mov  v5.16b, v29.16b\n"
                    "mov  v7.16b, v29.16b\n"
                    "mov  v9.16b, v29.16b\n"
                    "mov  v11.16b, v29.16b\n"
                    
                    "mov  v6.16b, v30.16b\n"
                    "mov  v8.16b, v30.16b\n"
                    "mov  v10.16b, v30.16b\n"
                    "mov  v12.16b, v30.16b\n"
                    :
                    :[b]"r"(b)
                    :"memory", "cc", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v29", "v30", "x1", "x2"
                );

                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        const INT8 *f_0 = f + fh_idx*fw*8 + fw_idx*8;
                        INT8 *in_idx = in_pad + fh_idx*dilateH*iw_pad*8 + fw_idx*dilateW*8;
                        INT8 *in_0 = in_idx + in_h_0*iw_pad*8 + in_w_0*8;
                        INT8 *in_1 = in_idx + in_h_1*iw_pad*8 + in_w_1*8;
                        INT8 *in_2 = in_idx + in_h_2*iw_pad*8 + in_w_2*8;
                        INT8 *in_3 = in_idx + in_h_3*iw_pad*8 + in_w_3*8;
                        __asm__ __volatile__(
                            "ldr d29, [%[f0]]\n"
                            "ldr d0, [%[in0]]\n"
                            "ldr d1, [%[in1]]\n"
                            "ldr d2, [%[in2]]\n"
                            "sshll v29.8h, v29.8b, #0\n"
                            "ldr d30, [%[in3]]\n"
                            "sshll v0.8h, v0.8b, #0\n"
                            "sshll v1.8h, v1.8b, #0\n"

                            "smlal v5.4s, v29.4h, v0.4h\n"
                            "sshll v2.8h, v2.8b, #0\n"
                            "smlal2 v6.4s, v29.8h, v0.8h\n"
                            "sshll v30.8h, v30.8b, #0\n"
                            "smlal v7.4s, v29.4h, v1.4h\n"
                            "smlal2 v8.4s, v29.8h, v1.8h\n"
                            "smlal v9.4s, v29.4h, v2.4h\n"
                            "smlal2 v10.4s, v29.8h, v2.8h\n"
                            "smlal v11.4s, v29.4h, v30.4h\n"
                            "smlal2 v12.4s, v29.8h, v30.8h\n"
                            :
                            :[in0]"r"(in_0),
                             [in1]"r"(in_1),
                             [in2]"r"(in_2),
                             [in3]"r"(in_3),
                             [f0]"r"(f_0)
                            :"memory", "cc", "v0", "v1", "v2", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v29", "v30"
                        );
                    }
                }

                // activation
                switch (depthwiseActivationMode){
                    case ACTIVATION_NULL: {
                        break;
                    }
                    case ACTIVATION_RELU:{
                        __asm__ __volatile__(
                            "eor v0.16b, v0.16b, v0.16b\n" // zero
                            
                            "smax v5.4s, v0.4s, v5.4s\n"
                            "smax v6.4s, v0.4s, v6.4s\n"
                            "smax v7.4s, v0.4s, v7.4s\n"
                            "smax v8.4s, v0.4s, v8.4s\n"
                            "smax v9.4s, v0.4s, v9.4s\n"
                            "smax v10.4s, v0.4s, v10.4s\n"
                            "smax v11.4s, v0.4s, v11.4s\n"
                            "smax v12.4s, v0.4s, v12.4s\n"
                            :
                            :
                            :"memory", "cc", "v0", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"
                        );
                        break;
                    }
                    case ACTIVATION_RELU6:{
                        INT8* pw_in0 = pwArray + hw*ic*8 + c*4*8;
                        INT8* pw_in1 = pw_in0 + 16;
                        __asm__ __volatile__(
                            "eor v0.16b, v0.16b, v0.16b\n" // zero
                            "movi v30.4s, #6\n"  // six

                            "smax v5.4s, v0.4s, v5.4s\n"
                            "smax v6.4s, v0.4s, v6.4s\n"
                            "smax v7.4s, v0.4s, v7.4s\n"
                            "smax v8.4s, v0.4s, v8.4s\n"
                            "smax v9.4s, v0.4s, v9.4s\n"
                            "smax v10.4s, v0.4s, v10.4s\n"
                            "smax v11.4s, v0.4s, v11.4s\n"
                            "smax v12.4s, v0.4s, v12.4s\n"

                            "smin v5.4s, v30.4s, v5.4s\n"
                            "smin v6.4s, v30.4s, v6.4s\n"
                            "smin v7.4s, v30.4s, v7.4s\n"
                            "smin v8.4s, v30.4s, v8.4s\n"
                            "smin v9.4s, v30.4s, v9.4s\n"
                            "smin v10.4s, v30.4s, v10.4s\n"
                            "smin v11.4s, v30.4s, v11.4s\n"
                            "smin v12.4s, v30.4s, v12.4s\n"

                            // No need to quantize for ReLU6
                            "sqshl v5.4s, v5.4s, #2\n"
                            "sqshl v6.4s, v6.4s, #2\n"
                            "sqshl v7.4s, v7.4s, #2\n"
                            "sqshl v8.4s, v8.4s, #2\n"
                            "sqshl v9.4s, v9.4s, #2\n"
                            "sqshl v10.4s, v10.4s, #2\n"
                            "sqshl v11.4s, v11.4s, #2\n"
                            "sqshl v12.4s, v12.4s, #2\n"

                            "sqshrn v5.4h, v5.4s, #1\n"
                            "sqshrn v9.4h, v9.4s, #1\n"
                            "sqshrn2 v5.8h, v7.4s, #1\n"
                            "sqshrn2 v9.8h, v11.4s, #1\n"

                            "sqshrn v5.8b, v5.8h, #1\n"
                            "sqshrn2 v5.16b, v9.8h, #1\n"
                            "str q5, [%[in0]]\n"

                            "sqshrn v6.4h, v6.4s, #1\n"
                            "sqshrn v10.4h, v10.4s, #1\n"
                            "sqshrn2 v6.8h, v8.4s, #1\n"
                            "sqshrn2 v10.8h, v12.4s, #1\n"

                            "sqshrn v6.8b, v6.8h, #1\n"

                            "sqshrn2 v6.16b, v10.8h, #1\n"
                            "str q6, [%[in1]]\n"
                            :
                            :[in0]"r"(pw_in0),
                             [in1]"r"(pw_in1)
                            :"memory", "cc", "v0", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v30"
                        );
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }

                if (depthwiseActivationMode != ACTIVATION_RELU6) {
                    __asm__ __volatile__(
                        "str q5, [%[pw0]]\n"
                        "str q7, [%[pw0], #16]\n"
                        "str q9, [%[pw0], #32]\n"
                        "str q11, [%[pw0], #48]\n"

                        "str q6, [%[pw1]]\n"
                        "str q8, [%[pw1], #16]\n"
                        "str q10, [%[pw1], #32]\n"
                        "str q12, [%[pw1], #48]\n"
                        :
                        :[pw0]"r"(pw_pack_0),
                         [pw1]"r"(pw_pack_1)
                        :"memory", "cc", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"
                    );
                }
                ohow_s += 4;
                ohow_tail -= 4;
            }

            // ohow_reminder % 4
            for (I32 hw = ohow_s; hw < ohow; hw++) {
                U32 in_h_0 = hw/ow*strideH;
                U32 in_w_0 = hw%ow*strideW;
                I32 *pw_pack_0 = dw_out + hw*ic*8 + c*8;
                I32 *pw_pack_1 = pw_pack_0 + 4;
                //TODO handle asm combined with c. No guarantee that compile will not use vec reg in c.
                __asm__ __volatile__(
                    "ldr d5, [%[b]]\n"       //b_0
                    "ldr  x1, [%[b], #8]\n"
                    "ins v5.d[1], x1\n"
                    "ldr d6, [%[b], #16]\n"       //b_1
                    "ldr  x2, [%[b], #24]\n"
                    "ins v6.d[1], x2\n"
                    :
                    :[b]"r"(b)
                    :"memory", "cc", "v5", "v6", "x1", "x2"
                );

                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        const INT8 *f_0 = f + fh_idx*fw*8 + fw_idx*8;
                        INT8 *in_idx = in_pad + fh_idx*dilateH*iw_pad*8 + fw_idx*dilateW*8;
                        INT8 *in_0 = in_idx + in_h_0*iw_pad*8 + in_w_0*8;
                        __asm__ __volatile__(
                            "ldr d29, [%[f0]]\n"
                            "ldr d0, [%[in0]]\n"
                            "sshll v29.8h, v29.8b, #0\n"
                            "sshll v0.8h, v0.8b, #0\n"
                            "smlal v5.4s, v29.4h, v0.4h\n"
                            "smlal2 v6.4s, v29.8h, v0.8h\n"
                            :
                            :[in0]"r"(in_0),
                             [f0]"r"(f_0)
                            :"memory", "cc", "v0", "v5", "v6", "v29"
                        );
                    }
                }

                // activation
                switch (depthwiseActivationMode){
                    case ACTIVATION_NULL: {
                        break;
                    }
                    case ACTIVATION_RELU:{
                        __asm__ __volatile__(
                            "eor v0.16b, v0.16b, v0.16b\n" // zero
                            
                            "smax v5.4s, v0.4s, v5.4s\n"
                            "smax v6.4s, v0.4s, v6.4s\n"
                            :
                            :
                            :"memory", "cc", "v0", "v5", "v6"
                        );
                        break;
                    }
                    case ACTIVATION_RELU6:{
                        INT8* pw_in0 = pwArray + hw*ic*8 + c*8;
                        __asm__ __volatile__(
                            "eor v0.16b, v0.16b, v0.16b\n" // zero
                            "movi v30.4s, #6\n"  // six

                            "smax v5.4s, v0.4s, v5.4s\n"
                            "smax v6.4s, v0.4s, v6.4s\n"

                            "smin v5.4s, v30.4s, v5.4s\n"
                            "smin v6.4s, v30.4s, v6.4s\n"

                            // No need to quantize for ReLU6
                            "sqshl v5.4s, v5.4s, #2\n"
                            "sqshl v6.4s, v6.4s, #2\n"

                            "sqshrn v5.4h, v5.4s, #1\n"
                            "sqshrn2 v5.8h, v6.4s, #1\n"

                            "sqshrn v5.8b, v5.8h, #1\n"
                            "str d5, [%[in0]]\n"
                            :
                            :[in0]"r"(pw_in0)
                            :"memory", "cc", "v0", "v5", "v6", "v30"
                        );
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }

                if (depthwiseActivationMode != ACTIVATION_RELU6) {
                    __asm__ __volatile__(
                        "str q5, [%[pw0]]\n"
                        "str q6, [%[pw1]]\n"
                        :
                        :[pw0]"r"(pw_pack_0),
                         [pw1]"r"(pw_pack_1)
                        :"memory", "cc", "v5", "v6"
                    );
                }
            }
        }
        
        I32 scale = 1;
        if (depthwiseActivationMode != ACTIVATION_RELU6) {
            // quantization
            I32 factor = 16777216; // 24 bits
            switch (depthwiseActivationMode) {
                case ACTIVATION_NULL: {
                    I32 max_s = dw_out[0];
                    I32 min_s = dw_out[0];
                    for (U32 i=1; i<ohow*ic*8; i++) {
                        I32 cur = dw_out[i];
                        if (cur > max_s) {
                            max_s = cur;
                        } 
                        if (cur < min_s) {
                            min_s = cur;
                        }
                    }

                    if (max_s <= 127 && min_s >= -128) { // No need to scale
                        break;
                    }

                    if (max_s == 0 && min_s == 0) {
                        break;
                    }
                    
                    if (max_s>0 && min_s<0) {
                        I32 factor_p = (factor * 127) / max_s;
                        I32 factor_n = (factor * -128) / min_s;
                        factor = (factor_p < factor_n) ? factor_p : factor_n;
                    } else if (max_s < 0) {
                        factor = (factor * -128) / min_s;
                    } else { // min_s > 0
                        factor = (factor * 127) / max_s;
                    }
                    scale = 16777216 / factor;
                    break;
                }
                case ACTIVATION_RELU: {
                    I32 max_s = dw_out[0];
                    for (U32 i=1; i<ohow*ic*8; i++) {
                        I32 cur = dw_out[i];
                        if (cur > max_s) {
                            max_s = cur;
                        } 
                    }
                    if (max_s <= 127) { // No need to scale
                        break;
                    }

                    if (max_s == 0) {
                        break;
                    }

                    factor = (factor * 127) / max_s;
                    scale = 16777216 / factor;
                    break;
                }
                default:
                    return NOT_SUPPORTED;
            }
            I32 factor_v[4];
            for (U32 i=0; i<4; i++) {
                factor_v[i] = factor;
            }
            __asm__ __volatile__(
                "ldr q0, [%[factor]]\n"
                "mov x0, %[dw_out]\n"
                "mov x1, %[pw_in]\n"
                "mov x2, %[num]\n"
                "0:\n"
                "ldr q1, [x0], #16\n"
                "ldr q2, [x0], #16\n"
                "mul v1.4s, v0.4s, v1.4s\n"
                "mul v2.4s, v0.4s, v2.4s\n"

                "shrn v1.4h, v1.4s, #16\n"
                "shrn2 v1.8h, v2.4s, #16\n"

                "shrn v1.8b, v1.8h, #8\n"
                "subs x2, x2, #8\n"

                "str d1, [x1], #8\n"
                "bne 0b\n"
                :
                :[factor]"r"(factor_v),
                 [dw_out]"r"(dw_out),
                 [pw_in]"r"(pwArray),
                 [num]"r"((I64)ohow*ic*8)
                :"memory", "cc", "v0", "v1", "v2", "x0", "x1", "x2"
            );
        }

        I32 scale_v[4];
        for (U32 i=0; i<4; i++) {
            scale_v[i] = scale;
        }

        // pw_conv
        const INT8 *f_base = filterArray + ic*fh*fw*8;

        // ohow / 12
        for (I32 hw = 0; hw < ohow-11; hw+=12) {
            const I32 *b0 = biasArray + ic*8;
            const I32 *b1 = b0 + 4;
            INT8 *in_pack = pwArray + hw*ic*8;
            for (U32 o = 0; o < oc; o++) {
                INT8 *in_hw0 = in_pack;
                const INT8 *f_o0c0 = f_base + o*8*ic*8;
                I32 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                // bias
                const I32 *b_0 = b0;
                const I32 *b_1 = b1;
                __asm__ __volatile__(
                    // Bias should be applied after scaling
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

                    "cmp %[pointwiseActivationMode], %[am_relu6]\n"       //No need to scale for relu6
                    "ldr q3, [%[b_0]]\n"
                    "ldr q4, [%[b_1]]\n"
                    "beq 11f\n"

                    "ldr q0, [%[scale]]\n"
                    "mul v5.4s, v0.4s, v5.4s\n"
                    "mul v6.4s, v0.4s, v6.4s\n"
                    "mul v7.4s, v0.4s, v7.4s\n"
                    "mul v8.4s, v0.4s, v8.4s\n"
                    "mul v9.4s, v0.4s, v9.4s\n"
                    "mul v10.4s, v0.4s, v10.4s\n"
                    "mul v11.4s, v0.4s, v11.4s\n"
                    "mul v12.4s, v0.4s, v12.4s\n"
                    "mul v13.4s, v0.4s, v13.4s\n"
                    "mul v14.4s, v0.4s, v14.4s\n"
                    "mul v15.4s, v0.4s, v15.4s\n"
                    "mul v16.4s, v0.4s, v16.4s\n"
                    "mul v17.4s, v0.4s, v17.4s\n"
                    "mul v18.4s, v0.4s, v18.4s\n"
                    "mul v19.4s, v0.4s, v19.4s\n"
                    "mul v20.4s, v0.4s, v20.4s\n"
                    "mul v21.4s, v0.4s, v21.4s\n"
                    "mul v22.4s, v0.4s, v22.4s\n"
                    "mul v23.4s, v0.4s, v23.4s\n"
                    "mul v24.4s, v0.4s, v24.4s\n"
                    "mul v25.4s, v0.4s, v25.4s\n"
                    "mul v26.4s, v0.4s, v26.4s\n"
                    "mul v27.4s, v0.4s, v27.4s\n"
                    "mul v28.4s, v0.4s, v28.4s\n"

                    "add v5.4s, v3.4s, v5.4s\n"
                    "add v6.4s, v4.4s, v6.4s\n"
                    "add v7.4s, v3.4s, v7.4s\n"
                    "add v8.4s, v4.4s, v8.4s\n"
                    "add v9.4s, v3.4s, v9.4s\n"
                    "add v10.4s, v4.4s, v10.4s\n"
                    "add v11.4s, v3.4s, v11.4s\n"
                    "add v12.4s, v4.4s, v12.4s\n"
                    "add v13.4s, v3.4s, v13.4s\n"
                    "add v14.4s, v4.4s, v14.4s\n"
                    "add v15.4s, v3.4s, v15.4s\n"
                    "add v16.4s, v4.4s, v16.4s\n"
                    "add v17.4s, v3.4s, v17.4s\n"
                    "add v18.4s, v4.4s, v18.4s\n"
                    "add v19.4s, v3.4s, v19.4s\n"
                    "add v20.4s, v4.4s, v20.4s\n"
                    "add v21.4s, v3.4s, v21.4s\n"
                    "add v22.4s, v4.4s, v22.4s\n"
                    "add v23.4s, v3.4s, v23.4s\n"
                    "add v24.4s, v4.4s, v24.4s\n"
                    "add v25.4s, v3.4s, v25.4s\n"
                    "add v26.4s, v4.4s, v26.4s\n"
                    "add v27.4s, v3.4s, v27.4s\n"
                    "add v28.4s, v4.4s, v28.4s\n"

                    "cmp %[pointwiseActivationMode], %[am_relu]\n"
                    "bne 13f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"     //zero
                    "smax  v5.4s,  v5.4s, v1.4s\n"
                    "smax  v6.4s,  v6.4s, v1.4s\n"
                    "smax  v7.4s,  v7.4s, v1.4s\n"
                    "smax  v8.4s,  v8.4s, v1.4s\n"
                    "smax  v9.4s,  v9.4s, v1.4s\n"
                    "smax  v10.4s,  v10.4s, v1.4s\n"
                    "smax  v11.4s,  v11.4s, v1.4s\n"
                    "smax  v12.4s,  v12.4s, v1.4s\n"
                    "smax  v13.4s,  v13.4s, v1.4s\n"
                    "smax  v14.4s,  v14.4s, v1.4s\n"
                    "smax  v15.4s,  v15.4s, v1.4s\n"
                    "smax  v16.4s,  v16.4s, v1.4s\n"
                    "smax  v17.4s,  v17.4s, v1.4s\n"
                    "smax  v18.4s,  v18.4s, v1.4s\n"
                    "smax  v19.4s,  v19.4s, v1.4s\n"
                    "smax  v20.4s,  v20.4s, v1.4s\n"
                    "smax  v21.4s,  v21.4s, v1.4s\n"
                    "smax  v22.4s,  v22.4s, v1.4s\n"
                    "smax  v23.4s,  v23.4s, v1.4s\n"
                    "smax  v24.4s,  v24.4s, v1.4s\n"
                    "smax  v25.4s,  v25.4s, v1.4s\n"
                    "smax  v26.4s,  v26.4s, v1.4s\n"
                    "smax  v27.4s,  v27.4s, v1.4s\n"
                    "smax  v28.4s,  v28.4s, v1.4s\n"

                    "11:\n"
                    "cmp %[pointwiseActivationMode], %[am_relu6]\n"
                    "bne 13f\n"
                    // Apply bias
                    "add v5.4s, v3.4s, v5.4s\n"
                    "add v6.4s, v4.4s, v6.4s\n"
                    "add v7.4s, v3.4s, v7.4s\n"
                    "add v8.4s, v4.4s, v8.4s\n"
                    "add v9.4s, v3.4s, v9.4s\n"
                    "add v10.4s, v4.4s, v10.4s\n"
                    "add v11.4s, v3.4s, v11.4s\n"
                    "add v12.4s, v4.4s, v12.4s\n"
                    "add v13.4s, v3.4s, v13.4s\n"
                    "add v14.4s, v4.4s, v14.4s\n"
                    "add v15.4s, v3.4s, v15.4s\n"
                    "add v16.4s, v4.4s, v16.4s\n"
                    "add v17.4s, v3.4s, v17.4s\n"
                    "add v18.4s, v4.4s, v18.4s\n"
                    "add v19.4s, v3.4s, v19.4s\n"
                    "add v20.4s, v4.4s, v20.4s\n"
                    "add v21.4s, v3.4s, v21.4s\n"
                    "add v22.4s, v4.4s, v22.4s\n"
                    "add v23.4s, v3.4s, v23.4s\n"
                    "add v24.4s, v4.4s, v24.4s\n"
                    "add v25.4s, v3.4s, v25.4s\n"
                    "add v26.4s, v4.4s, v26.4s\n"
                    "add v27.4s, v3.4s, v27.4s\n"
                    "add v28.4s, v4.4s, v28.4s\n"

                    "eor v1.16b, v0.16b, v0.16b\n"     //zero
                    "movi v2.4s, #6\n"                 //six
                    "smax  v5.4s,  v5.4s, v1.4s\n"
                    "smax  v6.4s,  v6.4s, v1.4s\n"
                    "smax  v7.4s,  v7.4s, v1.4s\n"
                    "smax  v8.4s,  v8.4s, v1.4s\n"
                    "smax  v9.4s,  v9.4s, v1.4s\n"
                    "smax  v10.4s,  v10.4s, v1.4s\n"
                    "smax  v11.4s,  v11.4s, v1.4s\n"
                    "smax  v12.4s,  v12.4s, v1.4s\n"
                    "smax  v13.4s,  v13.4s, v1.4s\n"
                    "smax  v14.4s,  v14.4s, v1.4s\n"
                    "smax  v15.4s,  v15.4s, v1.4s\n"
                    "smax  v16.4s,  v16.4s, v1.4s\n"
                    "smax  v17.4s,  v17.4s, v1.4s\n"
                    "smax  v18.4s,  v18.4s, v1.4s\n"
                    "smax  v19.4s,  v19.4s, v1.4s\n"
                    "smax  v20.4s,  v20.4s, v1.4s\n"
                    "smax  v21.4s,  v21.4s, v1.4s\n"
                    "smax  v22.4s,  v22.4s, v1.4s\n"
                    "smax  v23.4s,  v23.4s, v1.4s\n"
                    "smax  v24.4s,  v24.4s, v1.4s\n"
                    "smax  v25.4s,  v25.4s, v1.4s\n"
                    "smax  v26.4s,  v26.4s, v1.4s\n"
                    "smax  v27.4s,  v27.4s, v1.4s\n"
                    "smax  v28.4s,  v28.4s, v1.4s\n"

                    "smin  v5.4s,  v5.4s, v2.4s\n"
                    "smin  v6.4s,  v6.4s, v2.4s\n"
                    "smin  v7.4s,  v7.4s, v2.4s\n"
                    "smin  v8.4s,  v8.4s, v2.4s\n"
                    "smin  v9.4s,  v9.4s, v2.4s\n"
                    "smin  v10.4s,  v10.4s, v2.4s\n"
                    "smin  v11.4s,  v11.4s, v2.4s\n"
                    "smin  v12.4s,  v12.4s, v2.4s\n"
                    "smin  v13.4s,  v13.4s, v2.4s\n"
                    "smin  v14.4s,  v14.4s, v2.4s\n"
                    "smin  v15.4s,  v15.4s, v2.4s\n"
                    "smin  v16.4s,  v16.4s, v2.4s\n"
                    "smin  v17.4s,  v17.4s, v2.4s\n"
                    "smin  v18.4s,  v18.4s, v2.4s\n"
                    "smin  v19.4s,  v19.4s, v2.4s\n"
                    "smin  v20.4s,  v20.4s, v2.4s\n"
                    "smin  v21.4s,  v21.4s, v2.4s\n"
                    "smin  v22.4s,  v22.4s, v2.4s\n"
                    "smin  v23.4s,  v23.4s, v2.4s\n"
                    "smin  v24.4s,  v24.4s, v2.4s\n"
                    "smin  v25.4s,  v25.4s, v2.4s\n"
                    "smin  v26.4s,  v26.4s, v2.4s\n"
                    "smin  v27.4s,  v27.4s, v2.4s\n"
                    "smin  v28.4s,  v28.4s, v2.4s\n"

                    "13:\n"
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
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*8),
                     [b_0]"r"(b_0),
                     [b_1]"r"(b_1),
                     [pointwiseActivationMode]"r"((I64)pointwiseActivationMode),
                     [am_relu]"r"((I64)ACTIVATION_RELU),
                     [am_relu6]"r"((I64)ACTIVATION_RELU6),
                     [scale]"r"(scale_v)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "x0", "x1", "x2", "x3","x17","x16"
                );
                b0 += 8;
                b1 += 8;
            }
        }

        // ohow_reminder % 12 / 8
        U32 ohow_s = (ohow / 12) * 12;
        U32 ohow_tail = ohow - ohow_s;
        
        if (ohow_tail >= 8) {
            U32 hw = ohow_s;
            const I32 *b0 = biasArray + ic*8;
            const I32 *b1 = b0 + 4;
            INT8 *in_pack = pwArray + hw*ic*8;
            for (U32 o = 0; o < oc; o++) {
                INT8 *in_hw0 = in_pack;
                const INT8 *f_o0c0 = f_base + o*8*ic*8;
                I32 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                // bias
                const I32 *b_0 = b0;
                const I32 *b_1 = b1;
                __asm__ __volatile__(
                    // Bias should be applied after scaling
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

                    "cmp %[pointwiseActivationMode], %[am_relu6]\n"       //No need to scale for relu6
                    "ldr q3, [%[b_0]]\n"
                    "ldr q4, [%[b_1]]\n"
                    "beq 11f\n"

                    "ldr q0, [%[scale]]\n"
                    "mul v5.4s, v0.4s, v5.4s\n"
                    "mul v6.4s, v0.4s, v6.4s\n"
                    "mul v7.4s, v0.4s, v7.4s\n"
                    "mul v8.4s, v0.4s, v8.4s\n"
                    "mul v9.4s, v0.4s, v9.4s\n"
                    "mul v10.4s, v0.4s, v10.4s\n"
                    "mul v11.4s, v0.4s, v11.4s\n"
                    "mul v12.4s, v0.4s, v12.4s\n"
                    "mul v13.4s, v0.4s, v13.4s\n"
                    "mul v14.4s, v0.4s, v14.4s\n"
                    "mul v15.4s, v0.4s, v15.4s\n"
                    "mul v16.4s, v0.4s, v16.4s\n"
                    "mul v17.4s, v0.4s, v17.4s\n"
                    "mul v18.4s, v0.4s, v18.4s\n"
                    "mul v19.4s, v0.4s, v19.4s\n"
                    "mul v20.4s, v0.4s, v20.4s\n"

                    "add v5.4s, v3.4s, v5.4s\n"
                    "add v6.4s, v4.4s, v6.4s\n"
                    "add v7.4s, v3.4s, v7.4s\n"
                    "add v8.4s, v4.4s, v8.4s\n"
                    "add v9.4s, v3.4s, v9.4s\n"
                    "add v10.4s, v4.4s, v10.4s\n"
                    "add v11.4s, v3.4s, v11.4s\n"
                    "add v12.4s, v4.4s, v12.4s\n"
                    "add v13.4s, v3.4s, v13.4s\n"
                    "add v14.4s, v4.4s, v14.4s\n"
                    "add v15.4s, v3.4s, v15.4s\n"
                    "add v16.4s, v4.4s, v16.4s\n"
                    "add v17.4s, v3.4s, v17.4s\n"
                    "add v18.4s, v4.4s, v18.4s\n"
                    "add v19.4s, v3.4s, v19.4s\n"
                    "add v20.4s, v4.4s, v20.4s\n"

                    "cmp %[pointwiseActivationMode], %[am_relu]\n"
                    "bne 13f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"     //zero
                    "smax  v5.4s,  v5.4s, v1.4s\n"
                    "smax  v6.4s,  v6.4s, v1.4s\n"
                    "smax  v7.4s,  v7.4s, v1.4s\n"
                    "smax  v8.4s,  v8.4s, v1.4s\n"
                    "smax  v9.4s,  v9.4s, v1.4s\n"
                    "smax  v10.4s,  v10.4s, v1.4s\n"
                    "smax  v11.4s,  v11.4s, v1.4s\n"
                    "smax  v12.4s,  v12.4s, v1.4s\n"
                    "smax  v13.4s,  v13.4s, v1.4s\n"
                    "smax  v14.4s,  v14.4s, v1.4s\n"
                    "smax  v15.4s,  v15.4s, v1.4s\n"
                    "smax  v16.4s,  v16.4s, v1.4s\n"
                    "smax  v17.4s,  v17.4s, v1.4s\n"
                    "smax  v18.4s,  v18.4s, v1.4s\n"
                    "smax  v19.4s,  v19.4s, v1.4s\n"
                    "smax  v20.4s,  v20.4s, v1.4s\n"

                    "11:\n"
                    "cmp %[pointwiseActivationMode], %[am_relu6]\n"
                    "bne 13f\n"
                    // Apply bias
                    "add v5.4s, v3.4s, v5.4s\n"
                    "add v6.4s, v4.4s, v6.4s\n"
                    "add v7.4s, v3.4s, v7.4s\n"
                    "add v8.4s, v4.4s, v8.4s\n"
                    "add v9.4s, v3.4s, v9.4s\n"
                    "add v10.4s, v4.4s, v10.4s\n"
                    "add v11.4s, v3.4s, v11.4s\n"
                    "add v12.4s, v4.4s, v12.4s\n"
                    "add v13.4s, v3.4s, v13.4s\n"
                    "add v14.4s, v4.4s, v14.4s\n"
                    "add v15.4s, v3.4s, v15.4s\n"
                    "add v16.4s, v4.4s, v16.4s\n"
                    "add v17.4s, v3.4s, v17.4s\n"
                    "add v18.4s, v4.4s, v18.4s\n"
                    "add v19.4s, v3.4s, v19.4s\n"
                    "add v20.4s, v4.4s, v20.4s\n"

                    "eor v1.16b, v0.16b, v0.16b\n"     //zero
                    "movi v2.4s, #6\n"                   //six
                    "smax  v5.4s,  v5.4s, v1.4s\n"
                    "smax  v6.4s,  v6.4s, v1.4s\n"
                    "smax  v7.4s,  v7.4s, v1.4s\n"
                    "smax  v8.4s,  v8.4s, v1.4s\n"
                    "smax  v9.4s,  v9.4s, v1.4s\n"
                    "smax  v10.4s,  v10.4s, v1.4s\n"
                    "smax  v11.4s,  v11.4s, v1.4s\n"
                    "smax  v12.4s,  v12.4s, v1.4s\n"
                    "smax  v13.4s,  v13.4s, v1.4s\n"
                    "smax  v14.4s,  v14.4s, v1.4s\n"
                    "smax  v15.4s,  v15.4s, v1.4s\n"
                    "smax  v16.4s,  v16.4s, v1.4s\n"
                    "smax  v17.4s,  v17.4s, v1.4s\n"
                    "smax  v18.4s,  v18.4s, v1.4s\n"
                    "smax  v19.4s,  v19.4s, v1.4s\n"
                    "smax  v20.4s,  v20.4s, v1.4s\n"

                    "smin  v5.4s,  v5.4s, v2.4s\n"
                    "smin  v6.4s,  v6.4s, v2.4s\n"
                    "smin  v7.4s,  v7.4s, v2.4s\n"
                    "smin  v8.4s,  v8.4s, v2.4s\n"
                    "smin  v9.4s,  v9.4s, v2.4s\n"
                    "smin  v10.4s,  v10.4s, v2.4s\n"
                    "smin  v11.4s,  v11.4s, v2.4s\n"
                    "smin  v12.4s,  v12.4s, v2.4s\n"
                    "smin  v13.4s,  v13.4s, v2.4s\n"
                    "smin  v14.4s,  v14.4s, v2.4s\n"
                    "smin  v15.4s,  v15.4s, v2.4s\n"
                    "smin  v16.4s,  v16.4s, v2.4s\n"
                    "smin  v17.4s,  v17.4s, v2.4s\n"
                    "smin  v18.4s,  v18.4s, v2.4s\n"
                    "smin  v19.4s,  v19.4s, v2.4s\n"
                    "smin  v20.4s,  v20.4s, v2.4s\n"

                    "13:\n"
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
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*8),
                     [b_0]"r"(b_0),
                     [b_1]"r"(b_1),
                     [pointwiseActivationMode]"r"((I64)pointwiseActivationMode),
                     [am_relu]"r"((I64)ACTIVATION_RELU),
                     [am_relu6]"r"((I64)ACTIVATION_RELU6),
                     [scale]"r"(scale_v)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v29", "v30", "x0", "x1", "x2", "x3","x17","x16"
                );
                b0 += 8;
                b1 += 8;
            }
            ohow_s += 8;
            ohow_tail -= 8;
        }

        if (ohow_tail >= 4) {
            U32 hw = ohow_s;
            const I32 *b0 = biasArray + ic*8;
            const I32 *b1 = b0 + 4;
            INT8 *in_pack = pwArray + hw*ic*8;
            for (U32 o = 0; o < oc; o++) {
                INT8 *in_hw0 = in_pack;
                const INT8 *f_o0c0 = f_base + o*8*ic*8;
                I32 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                // bias
                const I32 *b_0 = b0;
                const I32 *b_1 = b1;
                __asm__ __volatile__(
                    // Bias should be applied after scaling
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

                    "cmp %[pointwiseActivationMode], %[am_relu6]\n"       //No need to scale for relu6
                    "ldr q3, [%[b_0]]\n"
                    "ldr q4, [%[b_1]]\n"
                    "beq 11f\n"

                    "ldr q0, [%[scale]]\n"
                    "mul v5.4s, v0.4s, v5.4s\n"
                    "mul v6.4s, v0.4s, v6.4s\n"
                    "mul v7.4s, v0.4s, v7.4s\n"
                    "mul v8.4s, v0.4s, v8.4s\n"
                    "mul v9.4s, v0.4s, v9.4s\n"
                    "mul v10.4s, v0.4s, v10.4s\n"
                    "mul v11.4s, v0.4s, v11.4s\n"
                    "mul v12.4s, v0.4s, v12.4s\n"

                    "add v5.4s, v3.4s, v5.4s\n"
                    "add v6.4s, v4.4s, v6.4s\n"
                    "add v7.4s, v3.4s, v7.4s\n"
                    "add v8.4s, v4.4s, v8.4s\n"
                    "add v9.4s, v3.4s, v9.4s\n"
                    "add v10.4s, v4.4s, v10.4s\n"
                    "add v11.4s, v3.4s, v11.4s\n"
                    "add v12.4s, v4.4s, v12.4s\n"

                    "cmp %[pointwiseActivationMode], %[am_relu]\n"
                    "bne 13f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"     //zero
                    "smax  v5.4s,  v5.4s, v1.4s\n"
                    "smax  v6.4s,  v6.4s, v1.4s\n"
                    "smax  v7.4s,  v7.4s, v1.4s\n"
                    "smax  v8.4s,  v8.4s, v1.4s\n"
                    "smax  v9.4s,  v9.4s, v1.4s\n"
                    "smax  v10.4s,  v10.4s, v1.4s\n"
                    "smax  v11.4s,  v11.4s, v1.4s\n"
                    "smax  v12.4s,  v12.4s, v1.4s\n"

                    "11:\n"
                    "cmp %[pointwiseActivationMode], %[am_relu6]\n"
                    "bne 13f\n"
                    // Apply bias
                    "add v5.4s, v3.4s, v5.4s\n"
                    "add v6.4s, v4.4s, v6.4s\n"
                    "add v7.4s, v3.4s, v7.4s\n"
                    "add v8.4s, v4.4s, v8.4s\n"
                    "add v9.4s, v3.4s, v9.4s\n"
                    "add v10.4s, v4.4s, v10.4s\n"
                    "add v11.4s, v3.4s, v11.4s\n"
                    "add v12.4s, v4.4s, v12.4s\n"

                    "eor v1.16b, v0.16b, v0.16b\n"     //zero
                    "movi v2.4s, #0x06\n"               //six
                    "smax  v5.4s,  v5.4s, v1.4s\n"
                    "smax  v6.4s,  v6.4s, v1.4s\n"
                    "smax  v7.4s,  v7.4s, v1.4s\n"
                    "smax  v8.4s,  v8.4s, v1.4s\n"
                    "smax  v9.4s,  v9.4s, v1.4s\n"
                    "smax  v10.4s,  v10.4s, v1.4s\n"
                    "smax  v11.4s,  v11.4s, v1.4s\n"
                    "smax  v12.4s,  v12.4s, v1.4s\n"

                    "smin  v5.4s,  v5.4s, v2.4s\n"
                    "smin  v6.4s,  v6.4s, v2.4s\n"
                    "smin  v7.4s,  v7.4s, v2.4s\n"
                    "smin  v8.4s,  v8.4s, v2.4s\n"
                    "smin  v9.4s,  v9.4s, v2.4s\n"
                    "smin  v10.4s,  v10.4s, v2.4s\n"
                    "smin  v11.4s,  v11.4s, v2.4s\n"
                    "smin  v12.4s,  v12.4s, v2.4s\n"

                    "13:\n"
                    "str   q5, [%[out_0]]\n"
                    "str   q6, [%[out_0], #16]\n"
                    "str   q7, [%[out_0], #32]\n"
                    "str   q8, [%[out_0], #48]\n"
                    "str   q9, [%[out_0], #64]\n"
                    "str   q10, [%[out_0], #80]\n"
                    "str   q11, [%[out_0], #96]\n"
                    "str   q12, [%[out_0], #112]\n"
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*8),
                     [b_0]"r"(b_0),
                     [b_1]"r"(b_1),
                     [pointwiseActivationMode]"r"((I64)pointwiseActivationMode),
                     [am_relu]"r"((I64)ACTIVATION_RELU),
                     [am_relu6]"r"((I64)ACTIVATION_RELU6),
                     [scale]"r"(scale_v)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v29", "v30", "x0", "x1", "x2", "x3","x17","x16"
                );
                b0 += 8;
                b1 += 8;
            }
            ohow_s += 4;
            ohow_tail -= 4;
        }

        for (I32 hw = ohow_s; hw < ohow; hw++) {
            const I32 *b0 = biasArray + ic*8;
            INT8 *in_pack = pwArray + hw*ic*8;

            // compute
            for (U32 o = 0; o < oc; o++) {
                INT8 *in_hw = in_pack;
                const INT8 *f_o = f_base + o*8*ic*8;
                I32 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                
                int32x4_t res[2] = {0};

                for(U32 c=0; c<ic; c++) {
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

                if (pointwiseActivationMode!=ACTIVATION_RELU6 && scale!=1) { // Scale
                    int32x4_t sc = vld1q_s32(scale_v);
                    res[0] = vmulq_s32(res[0], sc);
                    res[1] = vmulq_s32(res[1], sc);
                }

                int32x4_t bias[2];
                bias[0] = vld1q_s32(b0);
                bias[1] = vld1q_s32(b0+4);

                res[0] = vaddq_s32(res[0], bias[0]);
                res[1] = vaddq_s32(res[1], bias[1]);

                switch (pointwiseActivationMode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        int32x4_t z = vdupq_n_s32(0);
                        res[0] = vmaxq_s32(res[0], z);
                        res[1] = vmaxq_s32(res[1], z);
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        int32x4_t z = vdupq_n_s32(0);
                        int32x4_t s = vdupq_n_s32(6);
                        res[0] = vmaxq_s32(res[0], z);
                        res[1] = vmaxq_s32(res[1], z);
                        res[0] = vminq_s32(res[0], s);
                        res[1] = vminq_s32(res[1], s);
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }
                vst1q_s32(out_o0hw0, res[0]);
                vst1q_s32(out_o0hw0+4, res[1]);
                b0 += 8;
            }
        }
    }
    return SUCCESS;
}
#endif
