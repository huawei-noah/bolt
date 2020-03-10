// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "cpu/arm/fp16/depthwise_pointwise_convolution_3x3s1p1.h"

EE depthwise_pointwise_convolution_3x3s1p1_A76(TensorDesc inputDesc, F16* inArray,
    TensorDesc filterDesc, const F16* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const F16* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F16* outArray,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    UNUSED(convDesc);

    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (fdf != DF_CHWC8_NCN16)
        CHECK_STATUS(NOT_MATCH);

    oc /= 8;
    ic /= 8;

    I32 ohow = oh * ow;
    F16 *pwArray = (F16*)tmp;

    for (U32 n = 0; n < in; n++) {
        // dw_conv + padding
        for (U32 c = 0; c < ic; c++) {
            const F16 *b = biasArray + c*8;
            F16 *in_c = inArray + c*ih*iw*8;
            const F16 *f = filterArray + c*fh*fw*8;
            F16 *out = pwArray + c*ohow*8;
            F16 *in0 = in_c;
            F16 *in1 = in0 + iw*8;
            F16 *in2 = in1 + iw*8;
            __asm__ __volatile__(
                "mov x0, %[w]\n"
                "ldr q28, [%[b]]\n"
                "ldr q3, [%[f], #48]\n"
                "ldr q4, [%[f], #64]\n"
                "ldr q5, [%[f], #80]\n"
                "ldr q6, [%[f], #96]\n"
                "ldr q7, [%[f], #112]\n"
                "ldr q8, [%[f], #128]\n"
                "ldr q13, [%[in_0]]\n"
                "ldr q14, [%[in_0], #16]\n"
                "ldr q15, [%[in_0], #32]\n"
                "ldr q16, [%[in_0], #48]\n"
                "ldr q18, [%[in_1]]\n"
                "ldr q19, [%[in_1], #16]\n"
                "ldr q20, [%[in_1], #32]\n"
                "ldr q21, [%[in_1], #48]\n"
                "mov  v9.16b, v28.16b\n"    //out_0
                "mov v10.16b, v28.16b\n"    //out_1
                "mov v11.16b, v28.16b\n"    //out_2
                "mov v12.16b, v28.16b\n"    //out_3

                "ldr q17, [%[in_0], #64]\n"
                "fmla v10.8h, v3.8h, v13.8h\n"
                "fmla v11.8h, v3.8h, v14.8h\n"
                "fmla v12.8h, v3.8h, v15.8h\n"
                "ldr q22, [%[in_1], #64]\n"
                "fmla v10.8h, v6.8h, v18.8h\n"
                "fmla v11.8h, v6.8h, v19.8h\n"
                "fmla v12.8h, v6.8h, v20.8h\n"

                "fmla  v9.8h, v4.8h, v13.8h\n"
                "fmla v10.8h, v4.8h, v14.8h\n"
                "fmla v11.8h, v4.8h, v15.8h\n"
                "fmla v12.8h, v4.8h, v16.8h\n"
                "fmla  v9.8h, v7.8h, v18.8h\n"
                "fmla v10.8h, v7.8h, v19.8h\n"
                "fmla v11.8h, v7.8h, v20.8h\n"
                "fmla v12.8h, v7.8h, v21.8h\n"

                "ldr q13, [%[in_0], #80]\n"
                "fmla  v9.8h, v5.8h, v14.8h\n"
                "fmla v10.8h, v5.8h, v15.8h\n"
                "fmla v11.8h, v5.8h, v16.8h\n"
                "fmla v12.8h, v5.8h, v17.8h\n"
                "ldr q18, [%[in_1], #80]\n"
                "fmla  v9.8h, v8.8h, v19.8h\n"
                "fmla v10.8h, v8.8h, v20.8h\n"
                "fmla v11.8h, v8.8h, v21.8h\n"
                "fmla v12.8h, v8.8h, v22.8h\n"

                "mov v14.16b, v17.16b\n"
                "mov v19.16b, v22.16b\n"
                "mov v15.16b, v13.16b\n"
                "mov v20.16b, v18.16b\n"
                "mov v13.16b, v16.16b\n"
                "mov v18.16b, v21.16b\n"
                "ldr q16, [%[in_0], #96]\n"
                "ldr q21, [%[in_1], #96]\n"
                "add %[in_0], %[in_0], #48\n"
                "add %[in_1], %[in_1], #48\n"

                "cmp %[depthwiseActivationMode], %[am_relu]\n"        // v17, v22, v27, 29, 30, 31 will be reuse
                "bne 111f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                "fmax v10.8h, v10.8h, v17.8h\n"
                "fmax v11.8h, v11.8h, v17.8h\n"
                "fmax v12.8h, v12.8h, v17.8h\n"

                "111:\n"
                "cmp %[depthwiseActivationMode], %[am_relu6]\n"
                "bne 112f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "movi v22.8h, #0x46, lsl #8\n"      //six
                "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                "fmax v10.8h, v10.8h, v17.8h\n"
                "fmax v11.8h, v11.8h, v17.8h\n"
                "fmax v12.8h, v12.8h, v17.8h\n"
                "fmin  v9.8h,  v9.8h, v22.8h\n"     //min(v9, 6)
                "fmin v10.8h, v10.8h, v22.8h\n"
                "fmin v11.8h, v11.8h, v22.8h\n"
                "fmin v12.8h, v12.8h, v22.8h\n"

                "112:\n"
                "cmp %[depthwiseActivationMode], %[am_h_swish]\n"
                "bne 113f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "movi v22.8h, #0x42, lsl #8\n"      //three
                "fadd v27.8h,  v9.8h, v22.8h\n"
                "fadd v29.8h, v10.8h, v22.8h\n"
                "fadd v30.8h, v11.8h, v22.8h\n"
                "fadd v31.8h, v12.8h, v22.8h\n"
                "movi v22.8h, #0x46, lsl #8\n"      //six
                "fmax v27.8h, v27.8h, v17.8h\n"
                "fmax v29.8h, v29.8h, v17.8h\n"
                "fmax v30.8h, v30.8h, v17.8h\n"
                "fmax v31.8h, v31.8h, v17.8h\n"
                "fmin v27.8h, v27.8h, v22.8h\n"
                "fmin v29.8h, v29.8h, v22.8h\n"
                "fmin v30.8h, v30.8h, v22.8h\n"
                "fmin v31.8h, v31.8h, v22.8h\n"
                "fdiv v27.8h, v27.8h, v22.8h\n"
                "fdiv v29.8h, v29.8h, v22.8h\n"
                "fdiv v30.8h, v30.8h, v22.8h\n"
                "fdiv v31.8h, v31.8h, v22.8h\n"
                "fmul  v9.8h, v27.8h,  v9.8h\n"
                "fmul v10.8h, v29.8h, v10.8h\n"
                "fmul v11.8h, v30.8h, v11.8h\n"
                "fmul v12.8h, v31.8h, v12.8h\n"

                "113:\n"
                "st1 {v9.8h, v10.8h, v11.8h, v12.8h}, [%[out]], #64\n"

                "0:\n"
                "mov  v9.16b, v28.16b\n"    //out_0
                "mov v10.16b, v28.16b\n"    //out_1
                "mov v11.16b, v28.16b\n"    //out_2
                "mov v12.16b, v28.16b\n"    //out_3

                "ldr q17, [%[in_0], #64]\n"
                "fmla  v9.8h, v3.8h, v13.8h\n"
                "fmla v10.8h, v3.8h, v14.8h\n"
                "fmla v11.8h, v3.8h, v15.8h\n"
                "ldr q22, [%[in_1], #64]\n"
                "fmla v12.8h, v3.8h, v16.8h\n"
                "fmla  v9.8h, v6.8h, v18.8h\n"
                "fmla v10.8h, v6.8h, v19.8h\n"
                "fmla v11.8h, v6.8h, v20.8h\n"
                "fmla v12.8h, v6.8h, v21.8h\n"

                "ldr q13, [%[in_0], #80]\n"
                "fmla  v9.8h, v4.8h, v14.8h\n"
                "fmla v10.8h, v4.8h, v15.8h\n"
                "fmla v11.8h, v4.8h, v16.8h\n"
                "ldr q18, [%[in_1], #80]\n"
                "fmla v12.8h, v4.8h, v17.8h\n"
                "fmla  v9.8h, v7.8h, v19.8h\n"
                "fmla v10.8h, v7.8h, v20.8h\n"
                "fmla v11.8h, v7.8h, v21.8h\n"
                "fmla v12.8h, v7.8h, v22.8h\n"

                "ldr q14, [%[in_0], #96]\n"
                "fmla  v9.8h, v5.8h, v15.8h\n"
                "fmla v10.8h, v5.8h, v16.8h\n"
                "fmla v11.8h, v5.8h, v17.8h\n"
                "ldr q19, [%[in_1], #96]\n"
                "fmla v12.8h, v5.8h, v13.8h\n"
                "fmla  v9.8h, v8.8h, v20.8h\n"
                "fmla v10.8h, v8.8h, v21.8h\n"
                "fmla v11.8h, v8.8h, v22.8h\n"
                "fmla v12.8h, v8.8h, v18.8h\n"

                "ldr q16, [%[in_0], #112]\n"
                "mov v15.16b, v14.16b\n"
                "mov v20.16b, v19.16b\n"
                "mov v14.16b, v13.16b\n"
                "ldr q21, [%[in_1], #112]\n"
                "mov v19.16b, v18.16b\n"
                "mov v13.16b, v17.16b\n"
                "mov v18.16b, v22.16b\n"

                "add %[in_0], %[in_0], #64\n"
                "add %[in_1], %[in_1], #64\n"

                "cmp %[depthwiseActivationMode], %[am_relu]\n"        // v17, v22, v27, 29, 30, 31 will be reuse
                "bne 211f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                "fmax v10.8h, v10.8h, v17.8h\n"
                "fmax v11.8h, v11.8h, v17.8h\n"
                "fmax v12.8h, v12.8h, v17.8h\n"

                "211:\n"
                "cmp %[depthwiseActivationMode], %[am_relu6]\n"
                "bne 212f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "movi v22.8h, #0x46, lsl #8\n"      //six
                "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                "fmax v10.8h, v10.8h, v17.8h\n"
                "fmax v11.8h, v11.8h, v17.8h\n"
                "fmax v12.8h, v12.8h, v17.8h\n"
                "fmin  v9.8h,  v9.8h, v22.8h\n"     //min(v9, 6)
                "fmin v10.8h, v10.8h, v22.8h\n"
                "fmin v11.8h, v11.8h, v22.8h\n"
                "fmin v12.8h, v12.8h, v22.8h\n"

                "212:\n"
                "cmp %[depthwiseActivationMode], %[am_h_swish]\n"
                "bne 213f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "movi v22.8h, #0x42, lsl #8\n"      //three
                "fadd v27.8h,  v9.8h, v22.8h\n"
                "fadd v29.8h, v10.8h, v22.8h\n"
                "fadd v30.8h, v11.8h, v22.8h\n"
                "fadd v31.8h, v12.8h, v22.8h\n"
                "movi v22.8h, #0x46, lsl #8\n"      //six
                "fmax v27.8h, v27.8h, v17.8h\n"
                "fmax v29.8h, v29.8h, v17.8h\n"
                "fmax v30.8h, v30.8h, v17.8h\n"
                "fmax v31.8h, v31.8h, v17.8h\n"
                "fmin v27.8h, v27.8h, v22.8h\n"
                "fmin v29.8h, v29.8h, v22.8h\n"
                "fmin v30.8h, v30.8h, v22.8h\n"
                "fmin v31.8h, v31.8h, v22.8h\n"
                "fdiv v27.8h, v27.8h, v22.8h\n"
                "fdiv v29.8h, v29.8h, v22.8h\n"
                "fdiv v30.8h, v30.8h, v22.8h\n"
                "fdiv v31.8h, v31.8h, v22.8h\n"
                "fmul  v9.8h, v27.8h,  v9.8h\n"
                "fmul v10.8h, v29.8h, v10.8h\n"
                "fmul v11.8h, v30.8h, v11.8h\n"
                "fmul v12.8h, v31.8h, v12.8h\n"

                "213:\n"
                "subs x0, x0, #4\n"
                "st1 {v9.8h, v10.8h, v11.8h, v12.8h}, [%[out]], #64\n"
                "bne 0b\n"

                "mov  v9.16b, v28.16b\n"    //out_0
                "mov v10.16b, v28.16b\n"    //out_1
                "mov v11.16b, v28.16b\n"    //out_2
                "mov v12.16b, v28.16b\n"    //out_3

                "ldr q17, [%[in_0], #64]\n"
                "fmla  v9.8h, v3.8h, v13.8h\n"
                "fmla v10.8h, v3.8h, v14.8h\n"
                "fmla v11.8h, v3.8h, v15.8h\n"
                "fmla v12.8h, v3.8h, v16.8h\n"
                "ldr q22, [%[in_1], #64]\n"
                "fmla  v9.8h, v6.8h, v18.8h\n"
                "fmla v10.8h, v6.8h, v19.8h\n"
                "fmla v11.8h, v6.8h, v20.8h\n"
                "fmla v12.8h, v6.8h, v21.8h\n"

                "fmla  v9.8h, v4.8h, v14.8h\n"
                "fmla v10.8h, v4.8h, v15.8h\n"
                "fmla v11.8h, v4.8h, v16.8h\n"
                "fmla v12.8h, v4.8h, v17.8h\n"
                "fmla  v9.8h, v7.8h, v19.8h\n"
                "fmla v10.8h, v7.8h, v20.8h\n"
                "fmla v11.8h, v7.8h, v21.8h\n"
                "fmla v12.8h, v7.8h, v22.8h\n"

                "fmla  v9.8h, v5.8h, v15.8h\n"
                "fmla v10.8h, v5.8h, v16.8h\n"
                "fmla v11.8h, v5.8h, v17.8h\n"
                "fmla  v9.8h, v8.8h, v20.8h\n"
                "fmla v10.8h, v8.8h, v21.8h\n"
                "fmla v11.8h, v8.8h, v22.8h\n"

                "cmp %[depthwiseActivationMode], %[am_relu]\n"        // v17, v22, v27, 29, 30, 31 will be reuse
                "bne 311f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                "fmax v10.8h, v10.8h, v17.8h\n"
                "fmax v11.8h, v11.8h, v17.8h\n"
                "fmax v12.8h, v12.8h, v17.8h\n"

                "311:\n"
                "cmp %[depthwiseActivationMode], %[am_relu6]\n"
                "bne 312f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "movi v22.8h, #0x46, lsl #8\n"      //six
                "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                "fmax v10.8h, v10.8h, v17.8h\n"
                "fmax v11.8h, v11.8h, v17.8h\n"
                "fmax v12.8h, v12.8h, v17.8h\n"
                "fmin  v9.8h,  v9.8h, v22.8h\n"     //min(v9, 6)
                "fmin v10.8h, v10.8h, v22.8h\n"
                "fmin v11.8h, v11.8h, v22.8h\n"
                "fmin v12.8h, v12.8h, v22.8h\n"

                "312:\n"
                "cmp %[depthwiseActivationMode], %[am_h_swish]\n"
                "bne 313f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "movi v22.8h, #0x42, lsl #8\n"      //three
                "fadd v27.8h,  v9.8h, v22.8h\n"
                "fadd v29.8h, v10.8h, v22.8h\n"
                "fadd v30.8h, v11.8h, v22.8h\n"
                "fadd v31.8h, v12.8h, v22.8h\n"
                "movi v22.8h, #0x46, lsl #8\n"      //six
                "fmax v27.8h, v27.8h, v17.8h\n"
                "fmax v29.8h, v29.8h, v17.8h\n"
                "fmax v30.8h, v30.8h, v17.8h\n"
                "fmax v31.8h, v31.8h, v17.8h\n"
                "fmin v27.8h, v27.8h, v22.8h\n"
                "fmin v29.8h, v29.8h, v22.8h\n"
                "fmin v30.8h, v30.8h, v22.8h\n"
                "fmin v31.8h, v31.8h, v22.8h\n"
                "fdiv v27.8h, v27.8h, v22.8h\n"
                "fdiv v29.8h, v29.8h, v22.8h\n"
                "fdiv v30.8h, v30.8h, v22.8h\n"
                "fdiv v31.8h, v31.8h, v22.8h\n"
                "fmul  v9.8h, v27.8h,  v9.8h\n"
                "fmul v10.8h, v29.8h, v10.8h\n"
                "fmul v11.8h, v30.8h, v11.8h\n"
                "fmul v12.8h, v31.8h, v12.8h\n"

                "313:\n"
                "st1 {v9.8h, v10.8h, v11.8h, v12.8h}, [%[out]], #64\n"
                :[out]"+r"(out),
                 [in_0]"+r"(in0),
                 [in_1]"+r"(in1)
                :[f]"r"(f),
                 [b]"r"(b),
                 [w]"r"((I64)ow-8),
                 [depthwiseActivationMode]"r"((I64)depthwiseActivationMode),
                 [am_relu]"r"((I64)ACTIVATION_RELU),
                 [am_relu6]"r"((I64)ACTIVATION_RELU6),
                 [am_h_swish]"r"((I64)ACTIVATION_H_SWISH)
                :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                                 "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                                 "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x0", "x1", "x2", "x3"
            );

            for (U32 h = 0; h < oh-2; h++) {
                in0 = in_c + h*iw*8;
                in1 = in0 + iw*8;
                in2 = in1 + iw*8;
                __asm__ __volatile__(
                    "mov x0, %[w]\n"
                    "ldr q28, [%[b]]\n"
                    "ldr q0, [%[f]]\n"
                    "ldr q1, [%[f], #16]\n"
                    "ldr q2, [%[f], #32]\n"
                    "ldr q3, [%[f], #48]\n"
                    "ldr q4, [%[f], #64]\n"
                    "ldr q5, [%[f], #80]\n"
                    "ldr q6, [%[f], #96]\n"
                    "ldr q7, [%[f], #112]\n"
                    "ldr q8, [%[f], #128]\n"
                    "ldr q13, [%[in_0]]\n"
                    "ldr q14, [%[in_0], #16]\n"
                    "ldr q15, [%[in_0], #32]\n"
                    "ldr q16, [%[in_0], #48]\n"
                    "ldr q18, [%[in_1]]\n"
                    "ldr q19, [%[in_1], #16]\n"
                    "ldr q20, [%[in_1], #32]\n"
                    "ldr q21, [%[in_1], #48]\n"
                    "ldr q23, [%[in_2]]\n"
                    "ldr q24, [%[in_2], #16]\n"
                    "ldr q25, [%[in_2], #32]\n"
                    "ldr q26, [%[in_2], #48]\n"

                    "mov  v9.16b, v28.16b\n"    //out_0
                    "mov v10.16b, v28.16b\n"    //out_1
                    "mov v11.16b, v28.16b\n"    //out_2
                    "mov v12.16b, v28.16b\n"    //out_3

                    "ldr q17, [%[in_0], #64]\n"
                    "fmla v10.8h, v0.8h, v13.8h\n"
                    "fmla v11.8h, v0.8h, v14.8h\n"
                    "fmla v12.8h, v0.8h, v15.8h\n"
                    "ldr q22, [%[in_1], #64]\n"
                    "fmla v10.8h, v3.8h, v18.8h\n"
                    "fmla v11.8h, v3.8h, v19.8h\n"
                    "fmla v12.8h, v3.8h, v20.8h\n"
                    "ldr q27, [%[in_2], #64]\n"
                    "fmla v10.8h, v6.8h, v23.8h\n"
                    "fmla v11.8h, v6.8h, v24.8h\n"
                    "fmla v12.8h, v6.8h, v25.8h\n"

                    "fmla  v9.8h, v1.8h, v13.8h\n"
                    "fmla v10.8h, v1.8h, v14.8h\n"
                    "fmla v11.8h, v1.8h, v15.8h\n"
                    "fmla v12.8h, v1.8h, v16.8h\n"
                    "fmla  v9.8h, v4.8h, v18.8h\n"
                    "fmla v10.8h, v4.8h, v19.8h\n"
                    "fmla v11.8h, v4.8h, v20.8h\n"
                    "fmla v12.8h, v4.8h, v21.8h\n"
                    "fmla  v9.8h, v7.8h, v23.8h\n"
                    "fmla v10.8h, v7.8h, v24.8h\n"
                    "fmla v11.8h, v7.8h, v25.8h\n"
                    "fmla v12.8h, v7.8h, v26.8h\n"

                    "ldr q13, [%[in_0], #80]\n"
                    "fmla  v9.8h, v2.8h, v14.8h\n"
                    "fmla v10.8h, v2.8h, v15.8h\n"
                    "fmla v11.8h, v2.8h, v16.8h\n"
                    "fmla v12.8h, v2.8h, v17.8h\n"
                    "ldr q18, [%[in_1], #80]\n"
                    "fmla  v9.8h, v5.8h, v19.8h\n"
                    "fmla v10.8h, v5.8h, v20.8h\n"
                    "fmla v11.8h, v5.8h, v21.8h\n"
                    "fmla v12.8h, v5.8h, v22.8h\n"
                    "ldr q23, [%[in_2], #80]\n"
                    "fmla  v9.8h, v8.8h, v24.8h\n"
                    "fmla v10.8h, v8.8h, v25.8h\n"
                    "fmla v11.8h, v8.8h, v26.8h\n"
                    "fmla v12.8h, v8.8h, v27.8h\n"

                    "mov v14.16b, v17.16b\n"
                    "mov v19.16b, v22.16b\n"
                    "mov v24.16b, v27.16b\n"
                    "mov v15.16b, v13.16b\n"
                    "mov v20.16b, v18.16b\n"
                    "mov v25.16b, v23.16b\n"
                    "mov v13.16b, v16.16b\n"
                    "mov v18.16b, v21.16b\n"
                    "mov v23.16b, v26.16b\n"
                    "ldr q16, [%[in_0], #96]\n"
                    "ldr q21, [%[in_1], #96]\n"
                    "ldr q26, [%[in_2], #96]\n"
                    "add %[in_0], %[in_0], #48\n"
                    "add %[in_1], %[in_1], #48\n"
                    "add %[in_2], %[in_2], #48\n"

                    "cmp %[depthwiseActivationMode], %[am_relu]\n"        // v17, v22, v27, 29, 30, 31 will be reuse
                    "bne 111f\n"
                    "eor v17.16b, v17.16b, v17.16b\n"   //zero
                    "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                    "fmax v10.8h, v10.8h, v17.8h\n"
                    "fmax v11.8h, v11.8h, v17.8h\n"
                    "fmax v12.8h, v12.8h, v17.8h\n"

                    "111:\n"
                    "cmp %[depthwiseActivationMode], %[am_relu6]\n"
                    "bne 112f\n"
                    "eor v17.16b, v17.16b, v17.16b\n"   //zero
                    "movi v22.8h, #0x46, lsl #8\n"      //six
                    "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                    "fmax v10.8h, v10.8h, v17.8h\n"
                    "fmax v11.8h, v11.8h, v17.8h\n"
                    "fmax v12.8h, v12.8h, v17.8h\n"
                    "fmin  v9.8h,  v9.8h, v22.8h\n"     //min(v9, 6)
                    "fmin v10.8h, v10.8h, v22.8h\n"
                    "fmin v11.8h, v11.8h, v22.8h\n"
                    "fmin v12.8h, v12.8h, v22.8h\n"

                    "112:\n"
                    "cmp %[depthwiseActivationMode], %[am_h_swish]\n"
                    "bne 113f\n"
                    "eor v17.16b, v17.16b, v17.16b\n"   //zero
                    "movi v22.8h, #0x42, lsl #8\n"      //three
                    "fadd v27.8h,  v9.8h, v22.8h\n"
                    "fadd v29.8h, v10.8h, v22.8h\n"
                    "fadd v30.8h, v11.8h, v22.8h\n"
                    "fadd v31.8h, v12.8h, v22.8h\n"
                    "movi v22.8h, #0x46, lsl #8\n"      //six
                    "fmax v27.8h, v27.8h, v17.8h\n"
                    "fmax v29.8h, v29.8h, v17.8h\n"
                    "fmax v30.8h, v30.8h, v17.8h\n"
                    "fmax v31.8h, v31.8h, v17.8h\n"
                    "fmin v27.8h, v27.8h, v22.8h\n"
                    "fmin v29.8h, v29.8h, v22.8h\n"
                    "fmin v30.8h, v30.8h, v22.8h\n"
                    "fmin v31.8h, v31.8h, v22.8h\n"
                    "fdiv v27.8h, v27.8h, v22.8h\n"
                    "fdiv v29.8h, v29.8h, v22.8h\n"
                    "fdiv v30.8h, v30.8h, v22.8h\n"
                    "fdiv v31.8h, v31.8h, v22.8h\n"
                    "fmul  v9.8h, v27.8h,  v9.8h\n"
                    "fmul v10.8h, v29.8h, v10.8h\n"
                    "fmul v11.8h, v30.8h, v11.8h\n"
                    "fmul v12.8h, v31.8h, v12.8h\n"

                    "113:\n"
                    "st1 {v9.8h, v10.8h, v11.8h, v12.8h}, [%[out]], #64\n"

                    "0:\n"
                    "mov  v9.16b, v28.16b\n"    //out_0
                    "mov v10.16b, v28.16b\n"    //out_1
                    "mov v11.16b, v28.16b\n"    //out_2
                    "mov v12.16b, v28.16b\n"    //out_3

                    "ldr q17, [%[in_0], #64]\n"
                    "fmla  v9.8h, v0.8h, v13.8h\n"
                    "fmla v10.8h, v0.8h, v14.8h\n"
                    "fmla v11.8h, v0.8h, v15.8h\n"
                    "ldr q22, [%[in_1], #64]\n"
                    "fmla v12.8h, v0.8h, v16.8h\n"
                    "fmla  v9.8h, v3.8h, v18.8h\n"
                    "fmla v10.8h, v3.8h, v19.8h\n"
                    "ldr q27, [%[in_2], #64]\n"
                    "fmla v11.8h, v3.8h, v20.8h\n"
                    "fmla v12.8h, v3.8h, v21.8h\n"
                    "fmla  v9.8h, v6.8h, v23.8h\n"
                    "fmla v10.8h, v6.8h, v24.8h\n"
                    "fmla v11.8h, v6.8h, v25.8h\n"
                    "fmla v12.8h, v6.8h, v26.8h\n"

                    "ldr q13, [%[in_0], #80]\n"
                    "fmla  v9.8h, v1.8h, v14.8h\n"
                    "fmla v10.8h, v1.8h, v15.8h\n"
                    "fmla v11.8h, v1.8h, v16.8h\n"
                    "ldr q18, [%[in_1], #80]\n"
                    "fmla v12.8h, v1.8h, v17.8h\n"
                    "fmla  v9.8h, v4.8h, v19.8h\n"
                    "fmla v10.8h, v4.8h, v20.8h\n"
                    "ldr q23, [%[in_2], #80]\n"
                    "fmla v11.8h, v4.8h, v21.8h\n"
                    "fmla v12.8h, v4.8h, v22.8h\n"
                    "fmla  v9.8h, v7.8h, v24.8h\n"
                    "fmla v10.8h, v7.8h, v25.8h\n"
                    "fmla v11.8h, v7.8h, v26.8h\n"
                    "fmla v12.8h, v7.8h, v27.8h\n"

                    "ldr q14, [%[in_0], #96]\n"
                    "fmla  v9.8h, v2.8h, v15.8h\n"
                    "fmla v10.8h, v2.8h, v16.8h\n"
                    "fmla v11.8h, v2.8h, v17.8h\n"
                    "ldr q19, [%[in_1], #96]\n"
                    "fmla v12.8h, v2.8h, v13.8h\n"
                    "fmla  v9.8h, v5.8h, v20.8h\n"
                    "fmla v10.8h, v5.8h, v21.8h\n"
                    "ldr q24, [%[in_2], #96]\n"
                    "fmla v11.8h, v5.8h, v22.8h\n"
                    "fmla v12.8h, v5.8h, v18.8h\n"
                    "fmla  v9.8h, v8.8h, v25.8h\n"
                    "fmla v10.8h, v8.8h, v26.8h\n"
                    "fmla v11.8h, v8.8h, v27.8h\n"
                    "fmla v12.8h, v8.8h, v23.8h\n"


                    "ldr q16, [%[in_0], #112]\n"
                    "mov v15.16b, v14.16b\n"
                    "mov v20.16b, v19.16b\n"
                    "mov v25.16b, v24.16b\n"
                    "ldr q21, [%[in_1], #112]\n"
                    "mov v14.16b, v13.16b\n"
                    "mov v19.16b, v18.16b\n"
                    "mov v24.16b, v23.16b\n"
                    "ldr q26, [%[in_2], #112]\n"
                    "mov v13.16b, v17.16b\n"
                    "mov v18.16b, v22.16b\n"
                    "mov v23.16b, v27.16b\n"

                    "add %[in_0], %[in_0], #64\n"
                    "add %[in_1], %[in_1], #64\n"
                    "add %[in_2], %[in_2], #64\n"

                    "cmp %[depthwiseActivationMode], %[am_relu]\n"        // v17, v22, v27, 29, 30, 31 will be reuse
                    "bne 211f\n"
                    "eor v17.16b, v17.16b, v17.16b\n"   //zero
                    "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                    "fmax v10.8h, v10.8h, v17.8h\n"
                    "fmax v11.8h, v11.8h, v17.8h\n"
                    "fmax v12.8h, v12.8h, v17.8h\n"

                    "211:\n"
                    "cmp %[depthwiseActivationMode], %[am_relu6]\n"
                    "bne 212f\n"
                    "eor v17.16b, v17.16b, v17.16b\n"   //zero
                    "movi v22.8h, #0x46, lsl #8\n"      //six
                    "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                    "fmax v10.8h, v10.8h, v17.8h\n"
                    "fmax v11.8h, v11.8h, v17.8h\n"
                    "fmax v12.8h, v12.8h, v17.8h\n"
                    "fmin  v9.8h,  v9.8h, v22.8h\n"     //min(v9, 6)
                    "fmin v10.8h, v10.8h, v22.8h\n"
                    "fmin v11.8h, v11.8h, v22.8h\n"
                    "fmin v12.8h, v12.8h, v22.8h\n"

                    "212:\n"
                    "cmp %[depthwiseActivationMode], %[am_h_swish]\n"
                    "bne 213f\n"
                    "eor v17.16b, v17.16b, v17.16b\n"   //zero
                    "movi v22.8h, #0x42, lsl #8\n"      //three
                    "fadd v27.8h,  v9.8h, v22.8h\n"
                    "fadd v29.8h, v10.8h, v22.8h\n"
                    "fadd v30.8h, v11.8h, v22.8h\n"
                    "fadd v31.8h, v12.8h, v22.8h\n"
                    "movi v22.8h, #0x46, lsl #8\n"      //six
                    "fmax v27.8h, v27.8h, v17.8h\n"
                    "fmax v29.8h, v29.8h, v17.8h\n"
                    "fmax v30.8h, v30.8h, v17.8h\n"
                    "fmax v31.8h, v31.8h, v17.8h\n"
                    "fmin v27.8h, v27.8h, v22.8h\n"
                    "fmin v29.8h, v29.8h, v22.8h\n"
                    "fmin v30.8h, v30.8h, v22.8h\n"
                    "fmin v31.8h, v31.8h, v22.8h\n"
                    "fdiv v27.8h, v27.8h, v22.8h\n"
                    "fdiv v29.8h, v29.8h, v22.8h\n"
                    "fdiv v30.8h, v30.8h, v22.8h\n"
                    "fdiv v31.8h, v31.8h, v22.8h\n"
                    "fmul  v9.8h, v27.8h,  v9.8h\n"
                    "fmul v10.8h, v29.8h, v10.8h\n"
                    "fmul v11.8h, v30.8h, v11.8h\n"
                    "fmul v12.8h, v31.8h, v12.8h\n"

                    "213:\n"
                    "subs x0, x0, #4\n"
                    "st1 {v9.8h, v10.8h, v11.8h, v12.8h}, [%[out]], #64\n"
                    "bne 0b\n"

                    "mov  v9.16b, v28.16b\n"    //out_0
                    "mov v10.16b, v28.16b\n"    //out_1
                    "mov v11.16b, v28.16b\n"    //out_2
                    "mov v12.16b, v28.16b\n"    //out_3

                    "ldr q17, [%[in_0], #64]\n"
                    "fmla  v9.8h, v0.8h, v13.8h\n"
                    "fmla v10.8h, v0.8h, v14.8h\n"
                    "fmla v11.8h, v0.8h, v15.8h\n"
                    "fmla v12.8h, v0.8h, v16.8h\n"
                    "ldr q22, [%[in_1], #64]\n"
                    "fmla  v9.8h, v3.8h, v18.8h\n"
                    "fmla v10.8h, v3.8h, v19.8h\n"
                    "fmla v11.8h, v3.8h, v20.8h\n"
                    "fmla v12.8h, v3.8h, v21.8h\n"
                    "ldr q27, [%[in_2], #64]\n"
                    "fmla  v9.8h, v6.8h, v23.8h\n"
                    "fmla v10.8h, v6.8h, v24.8h\n"
                    "fmla v11.8h, v6.8h, v25.8h\n"
                    "fmla v12.8h, v6.8h, v26.8h\n"

                    "fmla  v9.8h, v1.8h, v14.8h\n"
                    "fmla v10.8h, v1.8h, v15.8h\n"
                    "fmla v11.8h, v1.8h, v16.8h\n"
                    "fmla v12.8h, v1.8h, v17.8h\n"
                    "fmla  v9.8h, v4.8h, v19.8h\n"
                    "fmla v10.8h, v4.8h, v20.8h\n"
                    "fmla v11.8h, v4.8h, v21.8h\n"
                    "fmla v12.8h, v4.8h, v22.8h\n"
                    "fmla  v9.8h, v7.8h, v24.8h\n"
                    "fmla v10.8h, v7.8h, v25.8h\n"
                    "fmla v11.8h, v7.8h, v26.8h\n"
                    "fmla v12.8h, v7.8h, v27.8h\n"

                    "fmla  v9.8h, v2.8h, v15.8h\n"
                    "fmla v10.8h, v2.8h, v16.8h\n"
                    "fmla v11.8h, v2.8h, v17.8h\n"
                    "fmla  v9.8h, v5.8h, v20.8h\n"
                    "fmla v10.8h, v5.8h, v21.8h\n"
                    "fmla v11.8h, v5.8h, v22.8h\n"
                    "fmla  v9.8h, v8.8h, v25.8h\n"
                    "fmla v10.8h, v8.8h, v26.8h\n"
                    "fmla v11.8h, v8.8h, v27.8h\n"

                    "cmp %[depthwiseActivationMode], %[am_relu]\n"        // v17, v22, v27, 29, 30, 31 will be reuse
                    "bne 311f\n"
                    "eor v17.16b, v17.16b, v17.16b\n"   //zero
                    "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                    "fmax v10.8h, v10.8h, v17.8h\n"
                    "fmax v11.8h, v11.8h, v17.8h\n"
                    "fmax v12.8h, v12.8h, v17.8h\n"

                    "311:\n"
                    "cmp %[depthwiseActivationMode], %[am_relu6]\n"
                    "bne 312f\n"
                    "eor v17.16b, v17.16b, v17.16b\n"   //zero
                    "movi v22.8h, #0x46, lsl #8\n"      //six
                    "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                    "fmax v10.8h, v10.8h, v17.8h\n"
                    "fmax v11.8h, v11.8h, v17.8h\n"
                    "fmax v12.8h, v12.8h, v17.8h\n"
                    "fmin  v9.8h,  v9.8h, v22.8h\n"     //min(v9, 6)
                    "fmin v10.8h, v10.8h, v22.8h\n"
                    "fmin v11.8h, v11.8h, v22.8h\n"
                    "fmin v12.8h, v12.8h, v22.8h\n"

                    "312:\n"
                    "cmp %[depthwiseActivationMode], %[am_h_swish]\n"
                    "bne 313f\n"
                    "eor v17.16b, v17.16b, v17.16b\n"   //zero
                    "movi v22.8h, #0x42, lsl #8\n"      //three
                    "fadd v27.8h,  v9.8h, v22.8h\n"
                    "fadd v29.8h, v10.8h, v22.8h\n"
                    "fadd v30.8h, v11.8h, v22.8h\n"
                    "fadd v31.8h, v12.8h, v22.8h\n"
                    "movi v22.8h, #0x46, lsl #8\n"      //six
                    "fmax v27.8h, v27.8h, v17.8h\n"
                    "fmax v29.8h, v29.8h, v17.8h\n"
                    "fmax v30.8h, v30.8h, v17.8h\n"
                    "fmax v31.8h, v31.8h, v17.8h\n"
                    "fmin v27.8h, v27.8h, v22.8h\n"
                    "fmin v29.8h, v29.8h, v22.8h\n"
                    "fmin v30.8h, v30.8h, v22.8h\n"
                    "fmin v31.8h, v31.8h, v22.8h\n"
                    "fdiv v27.8h, v27.8h, v22.8h\n"
                    "fdiv v29.8h, v29.8h, v22.8h\n"
                    "fdiv v30.8h, v30.8h, v22.8h\n"
                    "fdiv v31.8h, v31.8h, v22.8h\n"
                    "fmul  v9.8h, v27.8h,  v9.8h\n"
                    "fmul v10.8h, v29.8h, v10.8h\n"
                    "fmul v11.8h, v30.8h, v11.8h\n"
                    "fmul v12.8h, v31.8h, v12.8h\n"

                    "313:\n"
                    "st1 {v9.8h, v10.8h, v11.8h, v12.8h}, [%[out]], #64\n"
                    :[out]"+r"(out),
                     [in_0]"+r"(in0),
                     [in_1]"+r"(in1),
                     [in_2]"+r"(in2)
                    :[f]"r"(f),
                     [b]"r"(b),
                     [w]"r"((I64)ow-8),
                     [depthwiseActivationMode]"r"((I64)depthwiseActivationMode),
                     [am_relu]"r"((I64)ACTIVATION_RELU),
                     [am_relu6]"r"((I64)ACTIVATION_RELU6),
                     [am_h_swish]"r"((I64)ACTIVATION_H_SWISH)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                                     "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                                     "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x0", "x1", "x2", "x3"
                );
            }
            in0 = in_c + (ih-2)*iw*8;
            in1 = in0 + iw*8;
            in2 = in1 + iw*8;
            __asm__ __volatile__(
                "mov x0, %[w]\n"
                "ldr q28, [%[b]]\n"
                "ldr q0, [%[f]]\n"
                "ldr q1, [%[f], #16]\n"
                "ldr q2, [%[f], #32]\n"
                "ldr q3, [%[f], #48]\n"
                "ldr q4, [%[f], #64]\n"
                "ldr q5, [%[f], #80]\n"
                "ldr q13, [%[in_0]]\n"
                "ldr q14, [%[in_0], #16]\n"
                "ldr q15, [%[in_0], #32]\n"
                "ldr q16, [%[in_0], #48]\n"
                "ldr q18, [%[in_1]]\n"
                "ldr q19, [%[in_1], #16]\n"
                "ldr q20, [%[in_1], #32]\n"
                "ldr q21, [%[in_1], #48]\n"

                "mov  v9.16b, v28.16b\n"    //out_0
                "mov v10.16b, v28.16b\n"    //out_1
                "mov v11.16b, v28.16b\n"    //out_2
                "mov v12.16b, v28.16b\n"    //out_3

                "ldr q17, [%[in_0], #64]\n"
                "fmla v10.8h, v0.8h, v13.8h\n"
                "fmla v11.8h, v0.8h, v14.8h\n"
                "fmla v12.8h, v0.8h, v15.8h\n"
                "ldr q22, [%[in_1], #64]\n"
                "fmla v10.8h, v3.8h, v18.8h\n"
                "fmla v11.8h, v3.8h, v19.8h\n"
                "fmla v12.8h, v3.8h, v20.8h\n"

                "fmla  v9.8h, v1.8h, v13.8h\n"
                "fmla v10.8h, v1.8h, v14.8h\n"
                "fmla v11.8h, v1.8h, v15.8h\n"
                "fmla v12.8h, v1.8h, v16.8h\n"
                "fmla  v9.8h, v4.8h, v18.8h\n"
                "fmla v10.8h, v4.8h, v19.8h\n"
                "fmla v11.8h, v4.8h, v20.8h\n"
                "fmla v12.8h, v4.8h, v21.8h\n"

                "ldr q13, [%[in_0], #80]\n"
                "fmla  v9.8h, v2.8h, v14.8h\n"
                "fmla v10.8h, v2.8h, v15.8h\n"
                "fmla v11.8h, v2.8h, v16.8h\n"
                "fmla v12.8h, v2.8h, v17.8h\n"
                "ldr q18, [%[in_1], #80]\n"
                "fmla  v9.8h, v5.8h, v19.8h\n"
                "fmla v10.8h, v5.8h, v20.8h\n"
                "fmla v11.8h, v5.8h, v21.8h\n"
                "fmla v12.8h, v5.8h, v22.8h\n"

                "mov v14.16b, v17.16b\n"
                "mov v19.16b, v22.16b\n"
                "mov v15.16b, v13.16b\n"
                "mov v20.16b, v18.16b\n"
                "mov v13.16b, v16.16b\n"
                "mov v18.16b, v21.16b\n"
                "ldr q16, [%[in_0], #96]\n"
                "ldr q21, [%[in_1], #96]\n"
                "add %[in_0], %[in_0], #48\n"
                "add %[in_1], %[in_1], #48\n"

                "cmp %[depthwiseActivationMode], %[am_relu]\n"        // v17, v22, v27, 29, 30, 31 will be reuse
                "bne 111f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                "fmax v10.8h, v10.8h, v17.8h\n"
                "fmax v11.8h, v11.8h, v17.8h\n"
                "fmax v12.8h, v12.8h, v17.8h\n"

                "111:\n"
                "cmp %[depthwiseActivationMode], %[am_relu6]\n"
                "bne 112f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "movi v22.8h, #0x46, lsl #8\n"      //six
                "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                "fmax v10.8h, v10.8h, v17.8h\n"
                "fmax v11.8h, v11.8h, v17.8h\n"
                "fmax v12.8h, v12.8h, v17.8h\n"
                "fmin  v9.8h,  v9.8h, v22.8h\n"     //min(v9, 6)
                "fmin v10.8h, v10.8h, v22.8h\n"
                "fmin v11.8h, v11.8h, v22.8h\n"
                "fmin v12.8h, v12.8h, v22.8h\n"

                "112:\n"
                "cmp %[depthwiseActivationMode], %[am_h_swish]\n"
                "bne 113f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "movi v22.8h, #0x42, lsl #8\n"      //three
                "fadd v27.8h,  v9.8h, v22.8h\n"
                "fadd v29.8h, v10.8h, v22.8h\n"
                "fadd v30.8h, v11.8h, v22.8h\n"
                "fadd v31.8h, v12.8h, v22.8h\n"
                "movi v22.8h, #0x46, lsl #8\n"      //six
                "fmax v27.8h, v27.8h, v17.8h\n"
                "fmax v29.8h, v29.8h, v17.8h\n"
                "fmax v30.8h, v30.8h, v17.8h\n"
                "fmax v31.8h, v31.8h, v17.8h\n"
                "fmin v27.8h, v27.8h, v22.8h\n"
                "fmin v29.8h, v29.8h, v22.8h\n"
                "fmin v30.8h, v30.8h, v22.8h\n"
                "fmin v31.8h, v31.8h, v22.8h\n"
                "fdiv v27.8h, v27.8h, v22.8h\n"
                "fdiv v29.8h, v29.8h, v22.8h\n"
                "fdiv v30.8h, v30.8h, v22.8h\n"
                "fdiv v31.8h, v31.8h, v22.8h\n"
                "fmul  v9.8h, v27.8h,  v9.8h\n"
                "fmul v10.8h, v29.8h, v10.8h\n"
                "fmul v11.8h, v30.8h, v11.8h\n"
                "fmul v12.8h, v31.8h, v12.8h\n"

                "113:\n"
                "st1 {v9.8h, v10.8h, v11.8h, v12.8h}, [%[out]], #64\n"

                "0:\n"
                "mov  v9.16b, v28.16b\n"    //out_0
                "mov v10.16b, v28.16b\n"    //out_1
                "mov v11.16b, v28.16b\n"    //out_2
                "mov v12.16b, v28.16b\n"    //out_3

                "ldr q17, [%[in_0], #64]\n"
                "fmla  v9.8h, v0.8h, v13.8h\n"
                "fmla v10.8h, v0.8h, v14.8h\n"
                "fmla v11.8h, v0.8h, v15.8h\n"
                "ldr q22, [%[in_1], #64]\n"
                "fmla v12.8h, v0.8h, v16.8h\n"
                "fmla  v9.8h, v3.8h, v18.8h\n"
                "fmla v10.8h, v3.8h, v19.8h\n"
                "fmla v11.8h, v3.8h, v20.8h\n"
                "fmla v12.8h, v3.8h, v21.8h\n"

                "ldr q13, [%[in_0], #80]\n"
                "fmla  v9.8h, v1.8h, v14.8h\n"
                "fmla v10.8h, v1.8h, v15.8h\n"
                "fmla v11.8h, v1.8h, v16.8h\n"
                "ldr q18, [%[in_1], #80]\n"
                "fmla v12.8h, v1.8h, v17.8h\n"
                "fmla  v9.8h, v4.8h, v19.8h\n"
                "fmla v10.8h, v4.8h, v20.8h\n"
                "fmla v11.8h, v4.8h, v21.8h\n"
                "fmla v12.8h, v4.8h, v22.8h\n"

                "ldr q14, [%[in_0], #96]\n"
                "fmla  v9.8h, v2.8h, v15.8h\n"
                "fmla v10.8h, v2.8h, v16.8h\n"
                "fmla v11.8h, v2.8h, v17.8h\n"
                "ldr q19, [%[in_1], #96]\n"
                "fmla v12.8h, v2.8h, v13.8h\n"
                "fmla  v9.8h, v5.8h, v20.8h\n"
                "fmla v10.8h, v5.8h, v21.8h\n"
                "fmla v11.8h, v5.8h, v22.8h\n"
                "fmla v12.8h, v5.8h, v18.8h\n"


                "ldr q16, [%[in_0], #112]\n"
                "mov v15.16b, v14.16b\n"
                "mov v20.16b, v19.16b\n"
                "ldr q21, [%[in_1], #112]\n"
                "mov v14.16b, v13.16b\n"
                "mov v19.16b, v18.16b\n"
                "mov v13.16b, v17.16b\n"
                "mov v18.16b, v22.16b\n"

                "add %[in_0], %[in_0], #64\n"
                "add %[in_1], %[in_1], #64\n"

                "cmp %[depthwiseActivationMode], %[am_relu]\n"        // v17, v22, v27, 29, 30, 31 will be reuse
                "bne 211f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                "fmax v10.8h, v10.8h, v17.8h\n"
                "fmax v11.8h, v11.8h, v17.8h\n"
                "fmax v12.8h, v12.8h, v17.8h\n"

                "211:\n"
                "cmp %[depthwiseActivationMode], %[am_relu6]\n"
                "bne 212f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "movi v22.8h, #0x46, lsl #8\n"      //six
                "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                "fmax v10.8h, v10.8h, v17.8h\n"
                "fmax v11.8h, v11.8h, v17.8h\n"
                "fmax v12.8h, v12.8h, v17.8h\n"
                "fmin  v9.8h,  v9.8h, v22.8h\n"     //min(v9, 6)
                "fmin v10.8h, v10.8h, v22.8h\n"
                "fmin v11.8h, v11.8h, v22.8h\n"
                "fmin v12.8h, v12.8h, v22.8h\n"

                "212:\n"
                "cmp %[depthwiseActivationMode], %[am_h_swish]\n"
                "bne 213f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "movi v22.8h, #0x42, lsl #8\n"      //three
                "fadd v27.8h,  v9.8h, v22.8h\n"
                "fadd v29.8h, v10.8h, v22.8h\n"
                "fadd v30.8h, v11.8h, v22.8h\n"
                "fadd v31.8h, v12.8h, v22.8h\n"
                "movi v22.8h, #0x46, lsl #8\n"      //six
                "fmax v27.8h, v27.8h, v17.8h\n"
                "fmax v29.8h, v29.8h, v17.8h\n"
                "fmax v30.8h, v30.8h, v17.8h\n"
                "fmax v31.8h, v31.8h, v17.8h\n"
                "fmin v27.8h, v27.8h, v22.8h\n"
                "fmin v29.8h, v29.8h, v22.8h\n"
                "fmin v30.8h, v30.8h, v22.8h\n"
                "fmin v31.8h, v31.8h, v22.8h\n"
                "fdiv v27.8h, v27.8h, v22.8h\n"
                "fdiv v29.8h, v29.8h, v22.8h\n"
                "fdiv v30.8h, v30.8h, v22.8h\n"
                "fdiv v31.8h, v31.8h, v22.8h\n"
                "fmul  v9.8h, v27.8h,  v9.8h\n"
                "fmul v10.8h, v29.8h, v10.8h\n"
                "fmul v11.8h, v30.8h, v11.8h\n"
                "fmul v12.8h, v31.8h, v12.8h\n"

                "213:\n"
                "subs x0, x0, #4\n"
                "st1 {v9.8h, v10.8h, v11.8h, v12.8h}, [%[out]], #64\n"
                "bne 0b\n"

                "mov  v9.16b, v28.16b\n"    //out_0
                "mov v10.16b, v28.16b\n"    //out_1
                "mov v11.16b, v28.16b\n"    //out_2
                "mov v12.16b, v28.16b\n"    //out_3

                "ldr q17, [%[in_0], #64]\n"
                "fmla  v9.8h, v0.8h, v13.8h\n"
                "fmla v10.8h, v0.8h, v14.8h\n"
                "fmla v11.8h, v0.8h, v15.8h\n"
                "fmla v12.8h, v0.8h, v16.8h\n"
                "ldr q22, [%[in_1], #64]\n"
                "fmla  v9.8h, v3.8h, v18.8h\n"
                "fmla v10.8h, v3.8h, v19.8h\n"
                "fmla v11.8h, v3.8h, v20.8h\n"
                "fmla v12.8h, v3.8h, v21.8h\n"

                "fmla  v9.8h, v1.8h, v14.8h\n"
                "fmla v10.8h, v1.8h, v15.8h\n"
                "fmla v11.8h, v1.8h, v16.8h\n"
                "fmla v12.8h, v1.8h, v17.8h\n"
                "fmla  v9.8h, v4.8h, v19.8h\n"
                "fmla v10.8h, v4.8h, v20.8h\n"
                "fmla v11.8h, v4.8h, v21.8h\n"
                "fmla v12.8h, v4.8h, v22.8h\n"

                "fmla  v9.8h, v2.8h, v15.8h\n"
                "fmla v10.8h, v2.8h, v16.8h\n"
                "fmla v11.8h, v2.8h, v17.8h\n"
                "fmla  v9.8h, v5.8h, v20.8h\n"
                "fmla v10.8h, v5.8h, v21.8h\n"
                "fmla v11.8h, v5.8h, v22.8h\n"

                "cmp %[depthwiseActivationMode], %[am_relu]\n"        // v17, v22, v27, 29, 30, 31 will be reuse
                "bne 311f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                "fmax v10.8h, v10.8h, v17.8h\n"
                "fmax v11.8h, v11.8h, v17.8h\n"
                "fmax v12.8h, v12.8h, v17.8h\n"

                "311:\n"
                "cmp %[depthwiseActivationMode], %[am_relu6]\n"
                "bne 312f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "movi v22.8h, #0x46, lsl #8\n"      //six
                "fmax  v9.8h,  v9.8h, v17.8h\n"     //max(v9, 0)
                "fmax v10.8h, v10.8h, v17.8h\n"
                "fmax v11.8h, v11.8h, v17.8h\n"
                "fmax v12.8h, v12.8h, v17.8h\n"
                "fmin  v9.8h,  v9.8h, v22.8h\n"     //min(v9, 6)
                "fmin v10.8h, v10.8h, v22.8h\n"
                "fmin v11.8h, v11.8h, v22.8h\n"
                "fmin v12.8h, v12.8h, v22.8h\n"

                "312:\n"
                "cmp %[depthwiseActivationMode], %[am_h_swish]\n"
                "bne 313f\n"
                "eor v17.16b, v17.16b, v17.16b\n"   //zero
                "movi v22.8h, #0x42, lsl #8\n"      //three
                "fadd v27.8h,  v9.8h, v22.8h\n"
                "fadd v29.8h, v10.8h, v22.8h\n"
                "fadd v30.8h, v11.8h, v22.8h\n"
                "fadd v31.8h, v12.8h, v22.8h\n"
                "movi v22.8h, #0x46, lsl #8\n"      //six
                "fmax v27.8h, v27.8h, v17.8h\n"
                "fmax v29.8h, v29.8h, v17.8h\n"
                "fmax v30.8h, v30.8h, v17.8h\n"
                "fmax v31.8h, v31.8h, v17.8h\n"
                "fmin v27.8h, v27.8h, v22.8h\n"
                "fmin v29.8h, v29.8h, v22.8h\n"
                "fmin v30.8h, v30.8h, v22.8h\n"
                "fmin v31.8h, v31.8h, v22.8h\n"
                "fdiv v27.8h, v27.8h, v22.8h\n"
                "fdiv v29.8h, v29.8h, v22.8h\n"
                "fdiv v30.8h, v30.8h, v22.8h\n"
                "fdiv v31.8h, v31.8h, v22.8h\n"
                "fmul  v9.8h, v27.8h,  v9.8h\n"
                "fmul v10.8h, v29.8h, v10.8h\n"
                "fmul v11.8h, v30.8h, v11.8h\n"
                "fmul v12.8h, v31.8h, v12.8h\n"

                "313:\n"
                "st1 {v9.8h, v10.8h, v11.8h, v12.8h}, [%[out]], #64\n"
                :[out]"+r"(out),
                 [in_0]"+r"(in0),
                 [in_1]"+r"(in1)
                :[f]"r"(f),
                 [b]"r"(b),
                 [w]"r"((I64)ow-8),
                 [depthwiseActivationMode]"r"((I64)depthwiseActivationMode),
                 [am_relu]"r"((I64)ACTIVATION_RELU),
                 [am_relu6]"r"((I64)ACTIVATION_RELU6),
                 [am_h_swish]"r"((I64)ACTIVATION_H_SWISH)
                :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                                 "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                                 "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x0", "x1", "x2", "x3"
            );
        }

        // pw_conv
        for (I32 hw = 0; hw < ohow-7; hw+=8) {
            const F16 *b0 = biasArray + ic*8;
            const F16 *b1 = b0 + 8;
            const F16 *f_o0c0 = filterArray + ic*fh*fw*8;
            F16 *in_pack = pwArray + ohow*ic*8;
            // pack input
            // NCHWc8 => NHWChw8
            for (U32 c = 0; c < ic; c++) {
                F16 *in_pack_c8hw8 = in_pack + c*8*8;
                // it is 2% faster than in_hw8c8 = ... + hw*8; Amazing!
                F16 *in_hw8c8 = pwArray + c*ohow*8;
                //
                // for (U32 c8 = 0; c8 < 8; c8++) {
                //     for (U32 hw8 = 0; hw8 < 8; hw8++) {
                //         in_pack_c8hw8[c8*8 + hw8] = in_hw8c8[hw8*8 + c8];
                //     }
                // }
                //
                float16x8_t v0 = vld1q_f16(in_hw8c8 + hw*8);
                float16x8_t v1 = vld1q_f16(in_hw8c8 + hw*8 + 8);
                float16x8_t v2 = vld1q_f16(in_hw8c8 + hw*8 + 8*2);
                float16x8_t v3 = vld1q_f16(in_hw8c8 + hw*8 + 8*3);
                float16x8_t v4 = vld1q_f16(in_hw8c8 + hw*8 + 8*4);
                float16x8_t v5 = vld1q_f16(in_hw8c8 + hw*8 + 8*5);
                float16x8_t v6 = vld1q_f16(in_hw8c8 + hw*8 + 8*6);
                float16x8_t v7 = vld1q_f16(in_hw8c8 + hw*8 + 8*7);
                vst1q_f16(in_pack_c8hw8,
                    vzip1q_f16(
                        vzip1q_f16(vzip1q_f16(v0, v4), vzip1q_f16(v2, v6)),
                        vzip1q_f16(vzip1q_f16(v1, v5), vzip1q_f16(v3, v7))));
                vst1q_f16(in_pack_c8hw8 + 8,
                    vzip2q_f16(
                        vzip1q_f16(vzip1q_f16(v0, v4), vzip1q_f16(v2, v6)),
                        vzip1q_f16(vzip1q_f16(v1, v5), vzip1q_f16(v3, v7))));
                vst1q_f16(in_pack_c8hw8 + 8*2,
                    vzip1q_f16(
                        vzip2q_f16(vzip1q_f16(v0, v4), vzip1q_f16(v2, v6)),
                        vzip2q_f16(vzip1q_f16(v1, v5), vzip1q_f16(v3, v7))));
                vst1q_f16(in_pack_c8hw8 + 8*3,
                    vzip2q_f16(
                        vzip2q_f16(vzip1q_f16(v0, v4), vzip1q_f16(v2, v6)),
                        vzip2q_f16(vzip1q_f16(v1, v5), vzip1q_f16(v3, v7))));
                vst1q_f16(in_pack_c8hw8 + 8*4,
                    vzip1q_f16(
                        vzip1q_f16(vzip2q_f16(v0, v4), vzip2q_f16(v2, v6)),
                        vzip1q_f16(vzip2q_f16(v1, v5), vzip2q_f16(v3, v7))));
                vst1q_f16(in_pack_c8hw8 + 8*5,
                    vzip2q_f16(
                        vzip1q_f16(vzip2q_f16(v0, v4), vzip2q_f16(v2, v6)),
                        vzip1q_f16(vzip2q_f16(v1, v5), vzip2q_f16(v3, v7))));
                vst1q_f16(in_pack_c8hw8 + 8*6,
                    vzip1q_f16(
                        vzip2q_f16(vzip2q_f16(v0, v4), vzip2q_f16(v2, v6)),
                        vzip2q_f16(vzip2q_f16(v1, v5), vzip2q_f16(v3, v7))));
                vst1q_f16(in_pack_c8hw8 + 8*7,
                    vzip2q_f16(
                        vzip2q_f16(vzip2q_f16(v0, v4), vzip2q_f16(v2, v6)),
                        vzip2q_f16(vzip2q_f16(v1, v5), vzip2q_f16(v3, v7))));
            }
            // compute
            for (I32 o = 0; o < I32(oc-1); o+=2) {
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                F16 *out_o1hw0 = out_o0hw0 + ohow*8;
                // bias
                const F16 *b_o0 = b0;
                const F16 *b_o1 = b1;
                __asm__ __volatile__(
                    "ldr q22, [%[b_0]]\n"       //b_o0
                    "ldr q23, [%[b_1]]\n"       //b_o1
                    "mov  x0, %[ic]\n"             //ic_blk
                    "mov  v2.16b, v22.16b\n"      //out_o0hw0
                    "ldr  q0, [%[in_0]]\n"           //in_hw0
                    "mov  v3.16b, v22.16b\n"      //out_o0hw1
                    "mov  v4.16b, v22.16b\n"      //out_o0hw2
                    "mov  v5.16b, v22.16b\n"      //out_o0hw3
                    "ldr q18, [%[f_0]]\n"            //f_o0c0
                    "mov  v6.16b, v22.16b\n"      //out_o0hw4
                    "mov  v7.16b, v22.16b\n"      //out_o0hw5
                    "mov  v8.16b, v22.16b\n"      //out_o0hw6
                    "ldr q19, [%[f_0], #16]\n"            //f_o1c0
                    "mov  v9.16b, v22.16b\n"      //out_o0hw7
                    "mov v10.16b, v23.16b\n"      //out_o1hw0
                    "mov v11.16b, v23.16b\n"      //out_o1hw1
                    "mov v12.16b, v23.16b\n"      //out_o1hw2
                    "mov v13.16b, v23.16b\n"      //out_o1hw3
                    "mov v14.16b, v23.16b\n"      //out_o1hw4
                    "mov v15.16b, v23.16b\n"      //out_o1hw5
                    "mov v16.16b, v23.16b\n"      //out_o1hw6
                    "mov v17.16b, v23.16b\n"      //out_o1hw7
                    "0:\n"
                    "ldr  q1, [%[in_0], #16]\n"           //in_hw0
                    "fmla  v2.8h, v18.8h, v0.h[0]\n"
                    "fmla  v3.8h, v18.8h, v0.h[1]\n"
                    "fmla  v4.8h, v18.8h, v0.h[2]\n"
                    "ldr q20, [%[f_0], #32]\n"            //f_o0c0
                    "fmla  v5.8h, v18.8h, v0.h[3]\n"
                    "fmla  v6.8h, v18.8h, v0.h[4]\n"
                    "fmla  v7.8h, v18.8h, v0.h[5]\n"
                    "ldr q21, [%[f_0], #48]\n"            //f_o1c0
                    "fmla  v8.8h, v18.8h, v0.h[6]\n"
                    "fmla  v9.8h, v18.8h, v0.h[7]\n"
                    "fmla v10.8h, v19.8h, v0.h[0]\n"
                    "fmla v11.8h, v19.8h, v0.h[1]\n"
                    "fmla v12.8h, v19.8h, v0.h[2]\n"
                    "fmla v13.8h, v19.8h, v0.h[3]\n"
                    "fmla v14.8h, v19.8h, v0.h[4]\n"
                    "fmla v15.8h, v19.8h, v0.h[5]\n"
                    "fmla v16.8h, v19.8h, v0.h[6]\n"
                    "fmla v17.8h, v19.8h, v0.h[7]\n"

                    "ldr  q0, [%[in_0], #32]\n"           //in_hw0
                    "fmla  v2.8h, v20.8h, v1.h[0]\n"
                    "fmla  v3.8h, v20.8h, v1.h[1]\n"
                    "fmla  v4.8h, v20.8h, v1.h[2]\n"
                    "ldr q18, [%[f_0], #64]\n"            //f_o0c0
                    "fmla  v5.8h, v20.8h, v1.h[3]\n"
                    "fmla  v6.8h, v20.8h, v1.h[4]\n"
                    "fmla  v7.8h, v20.8h, v1.h[5]\n"
                    "ldr q19, [%[f_0], #80]\n"            //f_o1c0
                    "fmla  v8.8h, v20.8h, v1.h[6]\n"
                    "fmla  v9.8h, v20.8h, v1.h[7]\n"
                    "fmla v10.8h, v21.8h, v1.h[0]\n"
                    "add %[in_0], %[in_0], #32\n"
                    "fmla v11.8h, v21.8h, v1.h[1]\n"
                    "add %[f_0], %[f_0], #64\n"
                    "fmla v12.8h, v21.8h, v1.h[2]\n"
                    "subs x0, x0, #2\n"
                    "fmla v13.8h, v21.8h, v1.h[3]\n"
                    "fmla v14.8h, v21.8h, v1.h[4]\n"
                    "fmla v15.8h, v21.8h, v1.h[5]\n"
                    "fmla v16.8h, v21.8h, v1.h[6]\n"
                    "fmla v17.8h, v21.8h, v1.h[7]\n"
                    "bne 0b\n"

                    "cmp %[pointwiseActivationMode], %[am_relu]\n"
                    "bne 11f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "fmax  v2.8h,  v2.8h, v0.8h\n"
                    "fmax  v3.8h,  v3.8h, v0.8h\n"
                    "fmax  v4.8h,  v4.8h, v0.8h\n"
                    "fmax  v5.8h,  v5.8h, v0.8h\n"
                    "fmax  v6.8h,  v6.8h, v0.8h\n"
                    "fmax  v7.8h,  v7.8h, v0.8h\n"
                    "fmax  v8.8h,  v8.8h, v0.8h\n"
                    "fmax  v9.8h,  v9.8h, v0.8h\n"
                    "fmax v10.8h, v10.8h, v0.8h\n"
                    "fmax v11.8h, v11.8h, v0.8h\n"
                    "fmax v12.8h, v12.8h, v0.8h\n"
                    "fmax v13.8h, v13.8h, v0.8h\n"
                    "fmax v14.8h, v14.8h, v0.8h\n"
                    "fmax v15.8h, v15.8h, v0.8h\n"
                    "fmax v16.8h, v16.8h, v0.8h\n"
                    "fmax v17.8h, v17.8h, v0.8h\n"

                    "11:\n"
                    "cmp %[pointwiseActivationMode], %[am_relu6]\n"
                    "bne 12f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "movi v1.8h, #0x46, lsl #8\n"     //six
                    "fmax  v2.8h,  v2.8h, v0.8h\n"
                    "fmax  v3.8h,  v3.8h, v0.8h\n"
                    "fmax  v4.8h,  v4.8h, v0.8h\n"
                    "fmax  v5.8h,  v5.8h, v0.8h\n"
                    "fmax  v6.8h,  v6.8h, v0.8h\n"
                    "fmax  v7.8h,  v7.8h, v0.8h\n"
                    "fmax  v8.8h,  v8.8h, v0.8h\n"
                    "fmax  v9.8h,  v9.8h, v0.8h\n"
                    "fmax v10.8h, v10.8h, v0.8h\n"
                    "fmax v11.8h, v11.8h, v0.8h\n"
                    "fmax v12.8h, v12.8h, v0.8h\n"
                    "fmax v13.8h, v13.8h, v0.8h\n"
                    "fmax v14.8h, v14.8h, v0.8h\n"
                    "fmax v15.8h, v15.8h, v0.8h\n"
                    "fmax v16.8h, v16.8h, v0.8h\n"
                    "fmax v17.8h, v17.8h, v0.8h\n"
                    "fmin  v2.8h,  v2.8h, v1.8h\n"
                    "fmin  v3.8h,  v3.8h, v1.8h\n"
                    "fmin  v4.8h,  v4.8h, v1.8h\n"
                    "fmin  v5.8h,  v5.8h, v1.8h\n"
                    "fmin  v6.8h,  v6.8h, v1.8h\n"
                    "fmin  v7.8h,  v7.8h, v1.8h\n"
                    "fmin  v8.8h,  v8.8h, v1.8h\n"
                    "fmin  v9.8h,  v9.8h, v1.8h\n"
                    "fmin v10.8h, v10.8h, v1.8h\n"
                    "fmin v11.8h, v11.8h, v1.8h\n"
                    "fmin v12.8h, v12.8h, v1.8h\n"
                    "fmin v13.8h, v13.8h, v1.8h\n"
                    "fmin v14.8h, v14.8h, v1.8h\n"
                    "fmin v15.8h, v15.8h, v1.8h\n"
                    "fmin v16.8h, v16.8h, v1.8h\n"
                    "fmin v17.8h, v17.8h, v1.8h\n"

                    "12:\n"
                    "cmp %[pointwiseActivationMode], %[am_h_swish]\n"
                    "bne 13f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "movi v1.8h, #0x46, lsl #8\n"      //six
                    "movi v18.8h, #0x42, lsl #8\n"     //three
                    "fadd v19.8h,  v2.8h, v18.8h\n"
                    "fadd v20.8h,  v3.8h, v18.8h\n"
                    "fadd v21.8h,  v4.8h, v18.8h\n"
                    "fadd v22.8h,  v5.8h, v18.8h\n"
                    "fadd v23.8h,  v6.8h, v18.8h\n"
                    "fadd v24.8h,  v7.8h, v18.8h\n"
                    "fadd v25.8h,  v8.8h, v18.8h\n"
                    "fadd v26.8h,  v9.8h, v18.8h\n"
                    "fmax v19.8h, v19.8h,  v0.8h\n"
                    "fmax v20.8h, v20.8h,  v0.8h\n"
                    "fmax v21.8h, v21.8h,  v0.8h\n"
                    "fmax v22.8h, v22.8h,  v0.8h\n"
                    "fmax v23.8h, v23.8h,  v0.8h\n"
                    "fmax v24.8h, v24.8h,  v0.8h\n"
                    "fmax v25.8h, v25.8h,  v0.8h\n"
                    "fmax v26.8h, v26.8h,  v0.8h\n"
                    "fmin v19.8h, v19.8h,  v1.8h\n"
                    "fmin v20.8h, v20.8h,  v1.8h\n"
                    "fmin v21.8h, v21.8h,  v1.8h\n"
                    "fmin v22.8h, v22.8h,  v1.8h\n"
                    "fmin v23.8h, v23.8h,  v1.8h\n"
                    "fmin v24.8h, v24.8h,  v1.8h\n"
                    "fmin v25.8h, v25.8h,  v1.8h\n"
                    "fmin v26.8h, v26.8h,  v1.8h\n"
                    "fdiv v19.8h, v19.8h,  v1.8h\n"
                    "fdiv v20.8h, v20.8h,  v1.8h\n"
                    "fdiv v21.8h, v21.8h,  v1.8h\n"
                    "fdiv v22.8h, v22.8h,  v1.8h\n"
                    "fdiv v23.8h, v23.8h,  v1.8h\n"
                    "fdiv v24.8h, v24.8h,  v1.8h\n"
                    "fdiv v25.8h, v25.8h,  v1.8h\n"
                    "fdiv v26.8h, v26.8h,  v1.8h\n"
                    "fmul  v2.8h, v19.8h,  v2.8h\n"
                    "fmul  v3.8h, v20.8h,  v3.8h\n"
                    "fmul  v4.8h, v21.8h,  v4.8h\n"
                    "fmul  v5.8h, v22.8h,  v5.8h\n"
                    "fmul  v6.8h, v23.8h,  v6.8h\n"
                    "fmul  v7.8h, v24.8h,  v7.8h\n"
                    "fmul  v8.8h, v25.8h,  v8.8h\n"
                    "fmul  v9.8h, v26.8h,  v9.8h\n"

                    "fadd v19.8h, v10.8h, v18.8h\n"
                    "fadd v20.8h, v11.8h, v18.8h\n"
                    "fadd v21.8h, v12.8h, v18.8h\n"
                    "fadd v22.8h, v13.8h, v18.8h\n"
                    "fadd v23.8h, v14.8h, v18.8h\n"
                    "fadd v24.8h, v15.8h, v18.8h\n"
                    "fadd v25.8h, v16.8h, v18.8h\n"
                    "fadd v26.8h, v17.8h, v18.8h\n"
                    "fmax v19.8h, v19.8h,  v0.8h\n"
                    "fmax v20.8h, v20.8h,  v0.8h\n"
                    "fmax v21.8h, v21.8h,  v0.8h\n"
                    "fmax v22.8h, v22.8h,  v0.8h\n"
                    "fmax v23.8h, v23.8h,  v0.8h\n"
                    "fmax v24.8h, v24.8h,  v0.8h\n"
                    "fmax v25.8h, v25.8h,  v0.8h\n"
                    "fmax v26.8h, v26.8h,  v0.8h\n"
                    "fmin v19.8h, v19.8h,  v1.8h\n"
                    "fmin v20.8h, v20.8h,  v1.8h\n"
                    "fmin v21.8h, v21.8h,  v1.8h\n"
                    "fmin v22.8h, v22.8h,  v1.8h\n"
                    "fmin v23.8h, v23.8h,  v1.8h\n"
                    "fmin v24.8h, v24.8h,  v1.8h\n"
                    "fmin v25.8h, v25.8h,  v1.8h\n"
                    "fmin v26.8h, v26.8h,  v1.8h\n"
                    "fdiv v19.8h, v19.8h,  v1.8h\n"
                    "fdiv v20.8h, v20.8h,  v1.8h\n"
                    "fdiv v21.8h, v21.8h,  v1.8h\n"
                    "fdiv v22.8h, v22.8h,  v1.8h\n"
                    "fdiv v23.8h, v23.8h,  v1.8h\n"
                    "fdiv v24.8h, v24.8h,  v1.8h\n"
                    "fdiv v25.8h, v25.8h,  v1.8h\n"
                    "fdiv v26.8h, v26.8h,  v1.8h\n"
                    "fmul v10.8h, v19.8h, v10.8h\n"
                    "fmul v11.8h, v20.8h, v11.8h\n"
                    "fmul v12.8h, v21.8h, v12.8h\n"
                    "fmul v13.8h, v22.8h, v13.8h\n"
                    "fmul v14.8h, v23.8h, v14.8h\n"
                    "fmul v15.8h, v24.8h, v15.8h\n"
                    "fmul v16.8h, v25.8h, v16.8h\n"
                    "fmul v17.8h, v26.8h, v17.8h\n"

                    "13:\n"
                    "str   q2, [%[out_0]]\n"           //out_o0hw0
                    "str   q3, [%[out_0], #16]\n"      //out_o0hw1
                    "str   q4, [%[out_0], #32]\n"      //out_o0hw2
                    "str   q5, [%[out_0], #48]\n"      //out_o0hw3
                    "str   q6, [%[out_0], #64]\n"      //out_o0hw4
                    "str   q7, [%[out_0], #80]\n"      //out_o0hw5
                    "str   q8, [%[out_0], #96]\n"      //out_o0hw6
                    "str   q9, [%[out_0], #112]\n"     //out_o0hw7
                    "str  q10, [%[out_1]]\n"           //out_o1hw0
                    "str  q11, [%[out_1], #16]\n"      //out_o1hw1
                    "str  q12, [%[out_1], #32]\n"      //out_o1hw2
                    "str  q13, [%[out_1], #48]\n"      //out_o1hw3
                    "str  q14, [%[out_1], #64]\n"      //out_o1hw4
                    "str  q15, [%[out_1], #80]\n"      //out_o1hw5
                    "str  q16, [%[out_1], #96]\n"      //out_o1hw6
                    "str  q17, [%[out_1], #112]\n"     //out_o1hw7
                    :[out_0]"+r"(out_o0hw0),
                     [out_1]"+r"(out_o1hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*8),
                     [b_0]"r"(b_o0),
                     [b_1]"r"(b_o1),
                     [pointwiseActivationMode]"r"((I64)pointwiseActivationMode),
                     [am_relu]"r"((I64)ACTIVATION_RELU),
                     [am_relu6]"r"((I64)ACTIVATION_RELU6),
                     [am_h_swish]"r"((I64)ACTIVATION_H_SWISH)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                                     "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                                     "v21", "v22", "v23", "v24", "v25", "v26", "x0", "x1", "x2", "x3"
                );
                b0 += 16;
                b1 += 16;
            }
            if (oc & 1) {
                // oc%2 != 0
                const F16 *f_r = filterArray + ic*fh*fw*8 + (oc-1)*8*ic*8;
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + (oc-1)*ohow*8 + hw*8;
                // bias
                const F16 *b_o0 = biasArray + ic*8 + (oc-1)*8;
                __asm__ __volatile__(
                    "ldr q12, [%[b_0]]\n"       //b_o0
                    "mov x0, %[ic]\n" // ic_blk
                    "ldr q0, [%[in_0]]\n"   //in_hw0
                    "mov v2.16b, v12.16b\n" //out_o0hw0
                    "mov v3.16b, v12.16b\n" //out_o0hw1
                    "mov v4.16b, v12.16b\n" //out_o0hw2
                    "ldr q10, [%[f_0]]\n"   //f_o0c0
                    "mov v5.16b, v12.16b\n" //out_o0hw3
                    "mov v6.16b, v12.16b\n" //out_o0hw4
                    "mov v7.16b, v12.16b\n" //out_o0hw5
                    "mov v8.16b, v12.16b\n" //out_o0hw6
                    "mov v9.16b, v12.16b\n" //out_o0hw7
                    "0:\n"
                    "ldr q1, [%[in_0], #16]\n" //in_hw0
                    "fmla v2.8h, v10.8h, v0.h[0]\n"
                    "fmla v3.8h, v10.8h, v0.h[1]\n"
                    "fmla v4.8h, v10.8h, v0.h[2]\n"
                    "ldr q11, [%[f_0], #16]\n" //f_o0c0
                    "fmla v5.8h, v10.8h, v0.h[3]\n"
                    "fmla v6.8h, v10.8h, v0.h[4]\n"
                    "fmla v7.8h, v10.8h, v0.h[5]\n"
                    "subs x0, x0, #2\n"
                    "fmla v8.8h, v10.8h, v0.h[6]\n"
                    "fmla v9.8h, v10.8h, v0.h[7]\n"

                    "ldr q0, [%[in_0], #32]\n" //in_hw0
                    "fmla v2.8h, v11.8h, v1.h[0]\n"
                    "fmla v3.8h, v11.8h, v1.h[1]\n"
                    "fmla v4.8h, v11.8h, v1.h[2]\n"
                    "ldr q10, [%[f_0], #32]\n" //f_o0c0
                    "fmla v5.8h, v11.8h, v1.h[3]\n"
                    "fmla v6.8h, v11.8h, v1.h[4]\n"
                    "fmla v7.8h, v11.8h, v1.h[5]\n"
                    "add %[in_0], %[in_0], #32\n"
                    "fmla v8.8h, v11.8h, v1.h[6]\n"
                    "add %[f_0], %[f_0], #32\n"
                    "fmla v9.8h, v11.8h, v1.h[7]\n"
                    "bne 0b\n"

                    "cmp %[pointwiseActivationMode], %[am_relu]\n"
                    "bne 11f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "fmax  v2.8h,  v2.8h, v0.8h\n"
                    "fmax  v3.8h,  v3.8h, v0.8h\n"
                    "fmax  v4.8h,  v4.8h, v0.8h\n"
                    "fmax  v5.8h,  v5.8h, v0.8h\n"
                    "fmax  v6.8h,  v6.8h, v0.8h\n"
                    "fmax  v7.8h,  v7.8h, v0.8h\n"
                    "fmax  v8.8h,  v8.8h, v0.8h\n"
                    "fmax  v9.8h,  v9.8h, v0.8h\n"

                    "11:\n"
                    "cmp %[pointwiseActivationMode], %[am_relu6]\n"
                    "bne 12f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "movi v1.8h, #0x46, lsl #8\n"     //six
                    "fmax  v2.8h,  v2.8h, v0.8h\n"
                    "fmax  v3.8h,  v3.8h, v0.8h\n"
                    "fmax  v4.8h,  v4.8h, v0.8h\n"
                    "fmax  v5.8h,  v5.8h, v0.8h\n"
                    "fmax  v6.8h,  v6.8h, v0.8h\n"
                    "fmax  v7.8h,  v7.8h, v0.8h\n"
                    "fmax  v8.8h,  v8.8h, v0.8h\n"
                    "fmax  v9.8h,  v9.8h, v0.8h\n"
                    "fmin  v2.8h,  v2.8h, v1.8h\n"
                    "fmin  v3.8h,  v3.8h, v1.8h\n"
                    "fmin  v4.8h,  v4.8h, v1.8h\n"
                    "fmin  v5.8h,  v5.8h, v1.8h\n"
                    "fmin  v6.8h,  v6.8h, v1.8h\n"
                    "fmin  v7.8h,  v7.8h, v1.8h\n"
                    "fmin  v8.8h,  v8.8h, v1.8h\n"
                    "fmin  v9.8h,  v9.8h, v1.8h\n"

                   "12:\n"
                    "cmp %[pointwiseActivationMode], %[am_h_swish]\n"
                    "bne 13f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "movi v1.8h, #0x46, lsl #8\n"      //six
                    "movi v10.8h, #0x42, lsl #8\n"     //three
                    "fadd v11.8h,  v2.8h, v10.8h\n"
                    "fadd v12.8h,  v3.8h, v10.8h\n"
                    "fadd v13.8h,  v4.8h, v10.8h\n"
                    "fadd v14.8h,  v5.8h, v10.8h\n"
                    "fadd v15.8h,  v6.8h, v10.8h\n"
                    "fadd v16.8h,  v7.8h, v10.8h\n"
                    "fadd v17.8h,  v8.8h, v10.8h\n"
                    "fadd v18.8h,  v9.8h, v10.8h\n"
                    "fmax v11.8h, v11.8h,  v0.8h\n"
                    "fmax v12.8h, v12.8h,  v0.8h\n"
                    "fmax v13.8h, v13.8h,  v0.8h\n"
                    "fmax v14.8h, v14.8h,  v0.8h\n"
                    "fmax v15.8h, v15.8h,  v0.8h\n"
                    "fmax v16.8h, v16.8h,  v0.8h\n"
                    "fmax v17.8h, v17.8h,  v0.8h\n"
                    "fmax v18.8h, v18.8h,  v0.8h\n"
                    "fmin v11.8h, v11.8h,  v1.8h\n"
                    "fmin v12.8h, v12.8h,  v1.8h\n"
                    "fmin v13.8h, v13.8h,  v1.8h\n"
                    "fmin v14.8h, v14.8h,  v1.8h\n"
                    "fmin v15.8h, v15.8h,  v1.8h\n"
                    "fmin v16.8h, v16.8h,  v1.8h\n"
                    "fmin v17.8h, v17.8h,  v1.8h\n"
                    "fmin v18.8h, v18.8h,  v1.8h\n"
                    "fdiv v11.8h, v11.8h,  v1.8h\n"
                    "fdiv v12.8h, v12.8h,  v1.8h\n"
                    "fdiv v13.8h, v13.8h,  v1.8h\n"
                    "fdiv v14.8h, v14.8h,  v1.8h\n"
                    "fdiv v15.8h, v15.8h,  v1.8h\n"
                    "fdiv v16.8h, v16.8h,  v1.8h\n"
                    "fdiv v17.8h, v17.8h,  v1.8h\n"
                    "fdiv v18.8h, v18.8h,  v1.8h\n"
                    "fmul  v2.8h, v11.8h,  v2.8h\n"
                    "fmul  v3.8h, v12.8h,  v3.8h\n"
                    "fmul  v4.8h, v13.8h,  v4.8h\n"
                    "fmul  v5.8h, v14.8h,  v5.8h\n"
                    "fmul  v6.8h, v15.8h,  v6.8h\n"
                    "fmul  v7.8h, v16.8h,  v7.8h\n"
                    "fmul  v8.8h, v17.8h,  v8.8h\n"
                    "fmul  v9.8h, v18.8h,  v9.8h\n"

                    "13:\n"
                    "str q2, [%[out_0]]\n" //out_o0hw0
                    "str q3, [%[out_0], #16]\n" //out_o0hw0
                    "str q4, [%[out_0], #32]\n" //out_o0hw0
                    "str q5, [%[out_0], #48]\n" //out_o0hw0
                    "str q6, [%[out_0], #64]\n" //out_o0hw0
                    "str q7, [%[out_0], #80]\n" //out_o0hw0
                    "str q8, [%[out_0], #96]\n" //out_o0hw0
                    "str q9, [%[out_0], #112]\n" //out_o0hw0
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_r)
                    :[ic]"r"((I64)ic*8),
                     [b_0]"r"(b_o0),
                     [pointwiseActivationMode]"r"((I64)pointwiseActivationMode),
                     [am_relu]"r"((I64)ACTIVATION_RELU),
                     [am_relu6]"r"((I64)ACTIVATION_RELU6),
                     [am_h_swish]"r"((I64)ACTIVATION_H_SWISH)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                                     "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "x0", "x1", "x2"
                );
            }
        }
    }
    return SUCCESS;
}
