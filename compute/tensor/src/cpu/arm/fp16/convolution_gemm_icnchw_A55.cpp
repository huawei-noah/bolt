// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/fp16/convolution_gemm_icnchw.h"
#include "cpu/arm/transform_functions.h"
#include "thread_affinity.h"
#ifdef _USE_OPENMP
#include <omp.h>
#endif

EE convolution_gemm_icnchw_A55(TensorDesc inputDesc,
    F16 *inArray,
    TensorDesc filterDesc,
    const F16 *filterArray,
    ConvolutionParamSpec p,
    TensorDesc biasDesc,
    const F16 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F16 *outArray,
    ActivationParamSpec activationDesc)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);

    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, it, ih, iw;
    U32 fn, fc, ft, fh, fw;
    U32 on, oc, ot, oh, ow;
    if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
        it = ft = ot = 1;
        p.dilatedRate_t = p.stride_t = 1;
        p.padding_before = p.padding_after = 0;
    } else if (tensorIs5d(inputDesc)) {
        CHECK_STATUS(tensor5dGet(inputDesc, &idt, &idf, &in, &ic, &it, &ih, &iw));
        CHECK_STATUS(tensor5dGet(filterDesc, &fdt, &fdf, &fn, &fc, &ft, &fh, &fw));
        CHECK_STATUS(tensor5dGet(outputDesc, &odt, &odf, &on, &oc, &ot, &oh, &ow));
    } else {
        return NOT_SUPPORTED;
    }

    if (fdf != DF_NHWCN16) {
        CHECK_STATUS(NOT_MATCH);
    }

    oc /= 8;
    U32 it_pad = it + p.padding_before + p.padding_after;
    U32 ih_pad = ih + p.padding_top + p.padding_bottom;
    U32 iw_pad = iw + p.padding_left + p.padding_right;
    I64 K = ic * ft * fh * fw;
    I32 ohow = ot * oh * ow;
    int oc_1 = oc - 1;
    F16 *in_pack = ((F16 *)tmp) + ic * it_pad * ih_pad * iw_pad;
    EE ret = SUCCESS;
    for (U32 n = 0; n < in; n++) {
        F16 *inArray_pad = convolution_input_padding_per_channel<F16, 1>(
            n, ic, it, ih, iw, p, inArray, (F16 *)tmp);

        // ohow / 8
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (I32 hw = 0; hw < ohow - 7; hw += 8) {
#ifdef _USE_OPENMP
            F16 *thread_in_pack = in_pack + 8 * K * omp_get_thread_num();
#else
            F16 *thread_in_pack = in_pack;
#endif
            const F16 *f_o0c0 = filterArray;
            // pack input
            // NCHW => NHWChw8 + im2col
            convolution_nchw_input_pack<F16, 8>(ic, it_pad, ih_pad, iw_pad, p, ft, fh, fw, ot, oh,
                ow, inArray_pad, hw, thread_in_pack);

            // compute
            for (I32 o = 0; o < oc_1; o += 2) {
                F16 *in_hw0 = thread_in_pack;
                F16 *out_o0hw0 = outArray + ((n * oc + o) * ohow + hw) * 8;
                F16 *out_o1hw0 = out_o0hw0 + ohow * 8;
                // bias
                const F16 *b_o0 = biasArray + o * 8;
                const F16 *b_o1 = b_o0 + 8;
                __asm__ __volatile__("ldr d22, [%[b_0]]\n"  // b_o0
                                     "ldr  x1, [%[b_0], #8]\n"
                                     "ins v22.d[1], x1\n"
                                     "ldr d23, [%[b_1]]\n"  // b_o1
                                     "ldr  x2, [%[b_1], #8]\n"
                                     "ins v23.d[1], x2\n"
                                     "mov  x0, %[ic]\n"        // ic_blk
                                     "mov  v2.16b, v22.16b\n"  // out_o0hw0
                                     "ldr  d0, [%[in_0]]\n"    // in_hw0
                                     "mov  v3.16b, v22.16b\n"  // out_o0hw1
                                     "ldr  x1, [%[in_0], #8]\n"
                                     "mov  v4.16b, v22.16b\n"  // out_o0hw2
                                     "ins  v0.d[1], x1\n"
                                     "mov  v5.16b, v22.16b\n"  // out_o0hw3
                                     "ldr d18, [%[f_0]]\n"     // f_o0c0
                                     "mov  v6.16b, v22.16b\n"  // out_o0hw4
                                     "ldr  x2, [%[f_0], #8]\n"
                                     "mov  v7.16b, v22.16b\n"  // out_o0hw5
                                     "ins v18.d[1], x2\n"
                                     "mov  v8.16b, v22.16b\n"    // out_o0hw6
                                     "ldr d19, [%[f_0], #16]\n"  // f_o1c0
                                     "mov  v9.16b, v22.16b\n"    // out_o0hw7
                                     "ldr  x3, [%[f_0], #24]\n"
                                     "mov v10.16b, v23.16b\n"  // out_o1hw0
                                     "ins v19.d[1], x3\n"
                                     "mov v11.16b, v23.16b\n"  // out_o1hw1
                                     "mov v12.16b, v23.16b\n"  // out_o1hw2
                                     "mov v13.16b, v23.16b\n"  // out_o1hw3
                                     "mov v14.16b, v23.16b\n"  // out_o1hw4
                                     "mov v15.16b, v23.16b\n"  // out_o1hw5
                                     "mov v16.16b, v23.16b\n"  // out_o1hw6
                                     "mov v17.16b, v23.16b\n"  // out_o1hw7

                                     "0:\n"
                                     "cmp x0, #1\n"
                                     "ble 1f\n"
                                     "ldr  d1, [%[in_0], #16]\n"  // in_hw0
                                     "fmla  v2.8h, v18.8h, v0.h[0]\n"
                                     "ldr  x1, [%[in_0], #24]\n"
                                     "fmla  v3.8h, v18.8h, v0.h[1]\n"
                                     "ins  v1.d[1], x1\n"
                                     "fmla  v4.8h, v18.8h, v0.h[2]\n"
                                     "ldr d20, [%[f_0], #32]\n"  // f_o0c0
                                     "fmla  v5.8h, v18.8h, v0.h[3]\n"
                                     "ldr  x2, [%[f_0], #40]\n"
                                     "fmla  v6.8h, v18.8h, v0.h[4]\n"
                                     "ins v20.d[1], x2\n"
                                     "fmla  v7.8h, v18.8h, v0.h[5]\n"
                                     "ldr d21, [%[f_0], #48]\n"  // f_o1c0
                                     "fmla  v8.8h, v18.8h, v0.h[6]\n"
                                     "ldr  x3, [%[f_0], #56]\n"
                                     "fmla  v9.8h, v18.8h, v0.h[7]\n"
                                     "ins v21.d[1], x3\n"
                                     "fmla v10.8h, v19.8h, v0.h[0]\n"
                                     "fmla v11.8h, v19.8h, v0.h[1]\n"
                                     "fmla v12.8h, v19.8h, v0.h[2]\n"
                                     "fmla v13.8h, v19.8h, v0.h[3]\n"
                                     "fmla v14.8h, v19.8h, v0.h[4]\n"
                                     "fmla v15.8h, v19.8h, v0.h[5]\n"
                                     "fmla v16.8h, v19.8h, v0.h[6]\n"
                                     "fmla v17.8h, v19.8h, v0.h[7]\n"

                                     "ldr  d0, [%[in_0], #32]\n"  // in_hw0
                                     "fmla  v2.8h, v20.8h, v1.h[0]\n"
                                     "ldr  x1, [%[in_0], #40]\n"
                                     "fmla  v3.8h, v20.8h, v1.h[1]\n"
                                     "ins  v0.d[1], x1\n"
                                     "fmla  v4.8h, v20.8h, v1.h[2]\n"
                                     "ldr d18, [%[f_0], #64]\n"  // f_o0c0
                                     "fmla  v5.8h, v20.8h, v1.h[3]\n"
                                     "ldr  x2, [%[f_0], #72]\n"
                                     "fmla  v6.8h, v20.8h, v1.h[4]\n"
                                     "ins v18.d[1], x2\n"
                                     "fmla  v7.8h, v20.8h, v1.h[5]\n"
                                     "ldr d19, [%[f_0], #80]\n"  // f_o1c0
                                     "fmla  v8.8h, v20.8h, v1.h[6]\n"
                                     "ldr  x3, [%[f_0], #88]\n"
                                     "fmla  v9.8h, v20.8h, v1.h[7]\n"
                                     "ins v19.d[1], x3\n"
                                     "fmla v10.8h, v21.8h, v1.h[0]\n"
                                     "add %[in_0], %[in_0], #32\n"
                                     "fmla v11.8h, v21.8h, v1.h[1]\n"
                                     "add %[f_0], %[f_0], #64\n"
                                     "fmla v12.8h, v21.8h, v1.h[2]\n"
                                     "sub x0, x0, #2\n"
                                     "fmla v13.8h, v21.8h, v1.h[3]\n"
                                     "fmla v14.8h, v21.8h, v1.h[4]\n"
                                     "fmla v15.8h, v21.8h, v1.h[5]\n"
                                     "fmla v16.8h, v21.8h, v1.h[6]\n"
                                     "fmla v17.8h, v21.8h, v1.h[7]\n"
                                     "b 0b\n"

                                     "1:\n"
                                     "blt 2f\n"
                                     "fmla  v2.8h, v18.8h, v0.h[0]\n"
                                     "fmla  v3.8h, v18.8h, v0.h[1]\n"
                                     "fmla  v4.8h, v18.8h, v0.h[2]\n"
                                     "fmla  v5.8h, v18.8h, v0.h[3]\n"
                                     "fmla  v6.8h, v18.8h, v0.h[4]\n"
                                     "fmla  v7.8h, v18.8h, v0.h[5]\n"
                                     "fmla  v8.8h, v18.8h, v0.h[6]\n"
                                     "fmla  v9.8h, v18.8h, v0.h[7]\n"
                                     "add %[f_0], %[f_0], #32\n"
                                     "fmla v10.8h, v19.8h, v0.h[0]\n"
                                     "fmla v11.8h, v19.8h, v0.h[1]\n"
                                     "fmla v12.8h, v19.8h, v0.h[2]\n"
                                     "fmla v13.8h, v19.8h, v0.h[3]\n"
                                     "fmla v14.8h, v19.8h, v0.h[4]\n"
                                     "fmla v15.8h, v19.8h, v0.h[5]\n"
                                     "fmla v16.8h, v19.8h, v0.h[6]\n"
                                     "fmla v17.8h, v19.8h, v0.h[7]\n"
                                     "2:\n"
                                     : [in_0] "+r"(in_hw0), [f_0] "+r"(f_o0c0)
                                     : [ic] "r"(K), [b_0] "r"(b_o0), [b_1] "r"(b_o1)
                                     : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                                     "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                                     "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "x0",
                                     "x1", "x2", "x3");
                switch (activationDesc.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("eor v1.16b, v1.16b, v1.16b\n"  // zero
                                             "fmax  v2.8h,  v2.8h, v1.8h\n"  // max(v2, 0)
                                             "fmax  v3.8h,  v3.8h, v1.8h\n"
                                             "fmax  v4.8h,  v4.8h, v1.8h\n"
                                             "fmax  v5.8h,  v5.8h, v1.8h\n"
                                             "fmax  v6.8h,  v6.8h, v1.8h\n"
                                             "fmax  v7.8h,  v7.8h, v1.8h\n"
                                             "fmax  v8.8h,  v8.8h, v1.8h\n"
                                             "fmax  v9.8h,  v9.8h, v1.8h\n"
                                             "fmax v10.8h, v10.8h, v1.8h\n"
                                             "fmax v11.8h, v11.8h, v1.8h\n"
                                             "fmax v12.8h, v12.8h, v1.8h\n"
                                             "fmax v13.8h, v13.8h, v1.8h\n"
                                             "fmax v14.8h, v14.8h, v1.8h\n"
                                             "fmax v15.8h, v15.8h, v1.8h\n"
                                             "fmax v16.8h, v16.8h, v1.8h\n"
                                             "fmax v17.8h, v17.8h, v1.8h\n"
                                             :
                                             :
                                             : "memory", "cc", "v1", "v2", "v3", "v4", "v5", "v6",
                                             "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
                                             "v15", "v16", "v17");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("eor v31.16b, v31.16b, v31.16b\n"  // zero
                                             "movi v30.8h, #0x46, lsl #8\n"     // six
                                             "fmax v2.8h, v2.8h, v31.8h\n"
                                             "fmax v3.8h, v3.8h, v31.8h\n"
                                             "fmax v4.8h, v4.8h, v31.8h\n"
                                             "fmax v5.8h, v5.8h, v31.8h\n"
                                             "fmax v6.8h, v6.8h, v31.8h\n"
                                             "fmax v7.8h, v7.8h, v31.8h\n"
                                             "fmax  v8.8h,  v8.8h, v31.8h\n"
                                             "fmax  v9.8h,  v9.8h, v31.8h\n"
                                             "fmax v10.8h, v10.8h, v31.8h\n"
                                             "fmax v11.8h, v11.8h, v31.8h\n"
                                             "fmax v12.8h, v12.8h, v31.8h\n"
                                             "fmax v13.8h, v13.8h, v31.8h\n"
                                             "fmax v14.8h, v14.8h, v31.8h\n"
                                             "fmax v15.8h, v15.8h, v31.8h\n"
                                             "fmax v16.8h, v16.8h, v31.8h\n"
                                             "fmax v17.8h, v17.8h, v31.8h\n"

                                             "fmin v2.8h, v2.8h, v30.8h\n"
                                             "fmin v3.8h, v3.8h, v30.8h\n"
                                             "fmin v4.8h, v4.8h, v30.8h\n"
                                             "fmin v5.8h, v5.8h, v30.8h\n"
                                             "fmin v6.8h, v6.8h, v30.8h\n"
                                             "fmin v7.8h, v7.8h, v30.8h\n"
                                             "fmin  v8.8h,  v8.8h, v30.8h\n"
                                             "fmin  v9.8h,  v9.8h, v30.8h\n"
                                             "fmin v10.8h, v10.8h, v30.8h\n"
                                             "fmin v11.8h, v11.8h, v30.8h\n"
                                             "fmin v12.8h, v12.8h, v30.8h\n"
                                             "fmin v13.8h, v13.8h, v30.8h\n"
                                             "fmin v14.8h, v14.8h, v30.8h\n"
                                             "fmin v15.8h, v15.8h, v30.8h\n"
                                             "fmin v16.8h, v16.8h, v30.8h\n"
                                             "fmin v17.8h, v17.8h, v30.8h\n"
                                             :
                                             :
                                             : "memory", "cc", "v2", "v3", "v4", "v5", "v6", "v7",
                                             "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                                             "v16", "v17", "v30", "v31");
                        break;
                    }
                    default: {
                        ret = NOT_SUPPORTED;
                        break;
                    }
                }

                __asm__ __volatile__("str   q2, [%[out_0]]\n"        // out_o0hw0
                                     "str   q3, [%[out_0], #16]\n"   // out_o0hw1
                                     "str   q4, [%[out_0], #32]\n"   // out_o0hw2
                                     "str   q5, [%[out_0], #48]\n"   // out_o0hw3
                                     "str   q6, [%[out_0], #64]\n"   // out_o0hw4
                                     "str   q7, [%[out_0], #80]\n"   // out_o0hw5
                                     "str   q8, [%[out_0], #96]\n"   // out_o0hw6
                                     "str   q9, [%[out_0], #112]\n"  // out_o0hw7
                                     "str  q10, [%[out_1]]\n"        // out_o1hw0
                                     "str  q11, [%[out_1], #16]\n"   // out_o1hw1
                                     "str  q12, [%[out_1], #32]\n"   // out_o1hw2
                                     "str  q13, [%[out_1], #48]\n"   // out_o1hw3
                                     "str  q14, [%[out_1], #64]\n"   // out_o1hw4
                                     "str  q15, [%[out_1], #80]\n"   // out_o1hw5
                                     "str  q16, [%[out_1], #96]\n"   // out_o1hw6
                                     "str  q17, [%[out_1], #112]\n"  // out_o1hw7
                                     : [out_0] "+r"(out_o0hw0), [out_1] "+r"(out_o1hw0)
                                     :
                                     : "memory", "cc", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                                     "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17");
            }
            if (oc & 1) {
                // oc%2 != 0
                const F16 *f_r = filterArray + oc_1 * 8 * K;
                F16 *in_hw0 = thread_in_pack;
                F16 *out_o0hw0 = outArray + ((n * oc + oc_1) * ohow + hw) * 8;
                // bias
                const F16 *b_o0 = biasArray + oc_1 * 8;
                __asm__ __volatile__("ldr q12, [%[b_0]]\n"    // b_o0
                                     "mov x0, %[ic]\n"        // ic_blk
                                     "ldr d0, [%[in_0]]\n"    // in_hw0
                                     "mov v2.16b, v12.16b\n"  // out_o0hw0
                                     "ldr x1, [%[in_0], #8]\n"
                                     "mov v3.16b, v12.16b\n"  // out_o0hw1
                                     "ins v0.d[1], x1\n"
                                     "mov v4.16b, v12.16b\n"  // out_o0hw2
                                     "ldr d10, [%[f_0]]\n"    // f_o0c0
                                     "mov v5.16b, v12.16b\n"  // out_o0hw3
                                     "ldr x2, [%[f_0], #8]\n"
                                     "mov v6.16b, v12.16b\n"  // out_o0hw4
                                     "ins v10.d[1], x2\n"
                                     "mov v7.16b, v12.16b\n"  // out_o0hw5
                                     "mov v8.16b, v12.16b\n"  // out_o0hw6
                                     "mov v9.16b, v12.16b\n"  // out_o0hw7

                                     "0:\n"
                                     "cmp x0, #1\n"
                                     "ble 1f\n"
                                     "ldr d1, [%[in_0], #16]\n"  // in_hw0
                                     "fmla v2.8h, v10.8h, v0.h[0]\n"
                                     "ldr x1, [%[in_0], #24]\n"
                                     "fmla v3.8h, v10.8h, v0.h[1]\n"
                                     "ins v1.d[1], x1\n"
                                     "fmla v4.8h, v10.8h, v0.h[2]\n"
                                     "ldr d11, [%[f_0], #16]\n"  // f_o0c0
                                     "fmla v5.8h, v10.8h, v0.h[3]\n"
                                     "ldr x2, [%[f_0], #24]\n"
                                     "fmla v6.8h, v10.8h, v0.h[4]\n"
                                     "ins v11.d[1], x2\n"
                                     "fmla v7.8h, v10.8h, v0.h[5]\n"
                                     "sub x0, x0, #2\n"
                                     "fmla v8.8h, v10.8h, v0.h[6]\n"
                                     "fmla v9.8h, v10.8h, v0.h[7]\n"

                                     "ldr d0, [%[in_0], #32]\n"  // in_hw0
                                     "fmla v2.8h, v11.8h, v1.h[0]\n"
                                     "ldr x1, [%[in_0], #40]\n"
                                     "fmla v3.8h, v11.8h, v1.h[1]\n"
                                     "ins v0.d[1], x1\n"
                                     "fmla v4.8h, v11.8h, v1.h[2]\n"
                                     "ldr d10, [%[f_0], #32]\n"  // f_o0c0
                                     "fmla v5.8h, v11.8h, v1.h[3]\n"
                                     "ldr x2, [%[f_0], #40]\n"
                                     "fmla v6.8h, v11.8h, v1.h[4]\n"
                                     "ins v10.d[1], x2\n"
                                     "fmla v7.8h, v11.8h, v1.h[5]\n"
                                     "add %[in_0], %[in_0], #32\n"
                                     "fmla v8.8h, v11.8h, v1.h[6]\n"
                                     "add %[f_0], %[f_0], #32\n"
                                     "fmla v9.8h, v11.8h, v1.h[7]\n"
                                     "b 0b\n"

                                     "1:\n"
                                     "blt 2f\n"
                                     "fmla  v2.8h, v10.8h, v0.h[0]\n"
                                     "fmla  v3.8h, v10.8h, v0.h[1]\n"
                                     "fmla  v4.8h, v10.8h, v0.h[2]\n"
                                     "fmla  v5.8h, v10.8h, v0.h[3]\n"
                                     "add %[f_0], %[f_0], #16\n"
                                     "fmla  v6.8h, v10.8h, v0.h[4]\n"
                                     "fmla  v7.8h, v10.8h, v0.h[5]\n"
                                     "fmla  v8.8h, v10.8h, v0.h[6]\n"
                                     "fmla  v9.8h, v10.8h, v0.h[7]\n"
                                     "2:\n"
                                     : [in_0] "+r"(in_hw0), [f_0] "+r"(f_r)
                                     : [ic] "r"(K), [b_0] "r"(b_o0)
                                     : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                                     "v7", "v8", "v9", "v10", "v11", "v12", "x0", "x1", "x2");
                switch (activationDesc.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__(
                            "eor v1.16b, v1.16b, v1.16b\n"  // zero
                            "fmax  v2.8h,  v2.8h, v1.8h\n"  // max(v2, 0)
                            "fmax  v3.8h,  v3.8h, v1.8h\n"
                            "fmax  v4.8h,  v4.8h, v1.8h\n"
                            "fmax  v5.8h,  v5.8h, v1.8h\n"
                            "fmax  v6.8h,  v6.8h, v1.8h\n"
                            "fmax  v7.8h,  v7.8h, v1.8h\n"
                            "fmax  v8.8h,  v8.8h, v1.8h\n"
                            "fmax  v9.8h,  v9.8h, v1.8h\n"
                            :
                            :
                            : "memory", "cc", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("eor v31.16b, v31.16b, v31.16b\n"  // zero
                                             "movi v30.8h, #0x46, lsl #8\n"     // six
                                             "fmax v2.8h, v2.8h, v31.8h\n"
                                             "fmax v3.8h, v3.8h, v31.8h\n"
                                             "fmax v4.8h, v4.8h, v31.8h\n"
                                             "fmax v5.8h, v5.8h, v31.8h\n"
                                             "fmax v6.8h, v6.8h, v31.8h\n"
                                             "fmax v7.8h, v7.8h, v31.8h\n"
                                             "fmax  v8.8h,  v8.8h, v31.8h\n"
                                             "fmax  v9.8h,  v9.8h, v31.8h\n"

                                             "fmin v2.8h, v2.8h, v30.8h\n"
                                             "fmin v3.8h, v3.8h, v30.8h\n"
                                             "fmin v4.8h, v4.8h, v30.8h\n"
                                             "fmin v5.8h, v5.8h, v30.8h\n"
                                             "fmin v6.8h, v6.8h, v30.8h\n"
                                             "fmin v7.8h, v7.8h, v30.8h\n"
                                             "fmin  v8.8h,  v8.8h, v30.8h\n"
                                             "fmin  v9.8h,  v9.8h, v30.8h\n"
                                             :
                                             :
                                             : "memory", "cc", "v2", "v3", "v4", "v5", "v6", "v7",
                                             "v8", "v9", "v30", "v31");
                        break;
                    }
                    default: {
                        ret = NOT_SUPPORTED;
                        break;
                    }
                }

                __asm__ __volatile__(
                    "str q2, [%[out_0]]\n"        // out_o0hw0
                    "str q3, [%[out_0], #16]\n"   // out_o0hw0
                    "str q4, [%[out_0], #32]\n"   // out_o0hw0
                    "str q5, [%[out_0], #48]\n"   // out_o0hw0
                    "str q6, [%[out_0], #64]\n"   // out_o0hw0
                    "str q7, [%[out_0], #80]\n"   // out_o0hw0
                    "str q8, [%[out_0], #96]\n"   // out_o0hw0
                    "str q9, [%[out_0], #112]\n"  // out_o0hw0
                    : [out_0] "+r"(out_o0hw0)
                    :
                    : "memory", "cc", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9");
            }
        }
        // ohow_reminder % 8 / 4
        U32 ohow_s = (ohow / 8) * 8;
        // U32 ohow_s = (ohow/8)*8;
        for (I32 hw = ohow_s; hw < ohow - 3; hw += 4) {
            const F16 *f_o0c0 = filterArray;
            // pack input
            // NCHWc8 => NHWChw4 + im2col
            convolution_nchw_input_pack<F16, 4>(
                ic, it_pad, ih_pad, iw_pad, p, ft, fh, fw, ot, oh, ow, inArray_pad, hw, in_pack);

            // compute
            for (I32 o = 0; o < oc_1; o += 2) {
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + ((n * oc + o) * ohow + hw) * 8;
                F16 *out_o1hw0 = out_o0hw0 + ohow * 8;
                // bias
                const F16 *b_o0 = biasArray + o * 8;
                const F16 *b_o1 = b_o0 + 8;
                __asm__ __volatile__("ldr d22, [%[b_0]]\n"  // b_o0
                                     "ldr  x1, [%[b_0], #8]\n"
                                     "ins v22.d[1], x1\n"
                                     "ldr d23, [%[b_1]]\n"  // b_o1
                                     "ldr  x2, [%[b_1], #8]\n"
                                     "ins v23.d[1], x2\n"
                                     "mov  x0, %[ic]\n"        // ic_blk
                                     "mov  v2.16b, v22.16b\n"  // out_o0hw0
                                     "ldr  d0, [%[in_0]]\n"    // in_hw0
                                     "mov  v3.16b, v22.16b\n"  // out_o0hw1
                                     "ldr d18, [%[f_0]]\n"     // f_o0c0
                                     "mov  v4.16b, v22.16b\n"  // out_o0hw2
                                     "ldr  x2, [%[f_0], #8]\n"
                                     "mov  v5.16b, v22.16b\n"  // out_o0hw3
                                     "ins v18.d[1], x2\n"
                                     "mov v10.16b, v23.16b\n"    // out_o1hw0
                                     "ldr d19, [%[f_0], #16]\n"  // f_o1c0
                                     "mov v11.16b, v23.16b\n"    // out_o1hw1
                                     "ldr  x3, [%[f_0], #24]\n"
                                     "mov v12.16b, v23.16b\n"  // out_o1hw2
                                     "ins v19.d[1], x3\n"
                                     "mov v13.16b, v23.16b\n"  // out_o1hw3

                                     "0:\n"
                                     "cmp x0, #1\n"
                                     "ble 1f\n"
                                     "ldr  d1, [%[in_0], #8]\n"  // in_hw0
                                     "fmla  v2.8h, v18.8h, v0.h[0]\n"
                                     "ldr d20, [%[f_0], #32]\n"  // f_o0c0
                                     "fmla  v3.8h, v18.8h, v0.h[1]\n"
                                     "ldr  x2, [%[f_0], #40]\n"
                                     "fmla  v4.8h, v18.8h, v0.h[2]\n"
                                     "ins v20.d[1], x2\n"
                                     "fmla  v5.8h, v18.8h, v0.h[3]\n"
                                     "ldr d21, [%[f_0], #48]\n"  // f_o1c0
                                     "fmla v10.8h, v19.8h, v0.h[0]\n"
                                     "ldr  x3, [%[f_0], #56]\n"
                                     "fmla v11.8h, v19.8h, v0.h[1]\n"
                                     "ins v21.d[1], x3\n"
                                     "fmla v12.8h, v19.8h, v0.h[2]\n"
                                     "sub x0, x0, #2\n"
                                     "fmla v13.8h, v19.8h, v0.h[3]\n"

                                     "ldr  d0, [%[in_0], #16]\n"  // in_hw0
                                     "fmla  v2.8h, v20.8h, v1.h[0]\n"
                                     "ldr d18, [%[f_0], #64]\n"  // f_o0c0
                                     "fmla  v3.8h, v20.8h, v1.h[1]\n"
                                     "ldr  x2, [%[f_0], #72]\n"
                                     "fmla  v4.8h, v20.8h, v1.h[2]\n"
                                     "ldr d19, [%[f_0], #80]\n"  // f_o1c0
                                     "fmla  v5.8h, v20.8h, v1.h[3]\n"
                                     "ins v18.d[1], x2\n"
                                     "fmla v10.8h, v21.8h, v1.h[0]\n"
                                     "ldr  x3, [%[f_0], #88]\n"
                                     "fmla v11.8h, v21.8h, v1.h[1]\n"
                                     "ins v19.d[1], x3\n"
                                     "fmla v12.8h, v21.8h, v1.h[2]\n"
                                     "add %[in_0], %[in_0], #16\n"
                                     "fmla v13.8h, v21.8h, v1.h[3]\n"
                                     "add %[f_0], %[f_0], #64\n"
                                     "b 0b\n"

                                     "1:\n"
                                     "blt 2f\n"
                                     "fmla  v2.8h, v18.8h, v0.h[0]\n"
                                     "fmla  v3.8h, v18.8h, v0.h[1]\n"
                                     "fmla  v4.8h, v18.8h, v0.h[2]\n"
                                     "fmla  v5.8h, v18.8h, v0.h[3]\n"
                                     "add %[f_0], %[f_0], #32\n"
                                     "fmla v10.8h, v19.8h, v0.h[0]\n"
                                     "fmla v11.8h, v19.8h, v0.h[1]\n"
                                     "fmla v12.8h, v19.8h, v0.h[2]\n"
                                     "fmla v13.8h, v19.8h, v0.h[3]\n"
                                     "2:\n"
                                     : [in_0] "+r"(in_hw0), [f_0] "+r"(f_o0c0)
                                     : [ic] "r"(K), [b_0] "r"(b_o0), [b_1] "r"(b_o1)
                                     : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v10",
                                     "v11", "v12", "v13", "v18", "v19", "v20", "v21", "v22", "v23",
                                     "x0", "x1", "x2", "x3");
                switch (activationDesc.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("eor v1.16b, v1.16b, v1.16b\n"  // zero
                                             "fmax  v2.8h,  v2.8h, v1.8h\n"  // max(v2, 0)
                                             "fmax  v3.8h,  v3.8h, v1.8h\n"
                                             "fmax  v4.8h,  v4.8h, v1.8h\n"
                                             "fmax  v5.8h,  v5.8h, v1.8h\n"
                                             "fmax v10.8h, v10.8h, v1.8h\n"
                                             "fmax v11.8h, v11.8h, v1.8h\n"
                                             "fmax v12.8h, v12.8h, v1.8h\n"
                                             "fmax v13.8h, v13.8h, v1.8h\n"
                                             :
                                             :
                                             : "memory", "cc", "v1", "v2", "v3", "v4", "v5", "v10",
                                             "v11", "v12", "v13");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("eor v31.16b, v31.16b, v31.16b\n"  // zero
                                             "movi v30.8h, #0x46, lsl #8\n"     // six
                                             "fmax v2.8h, v2.8h, v31.8h\n"
                                             "fmax v3.8h, v3.8h, v31.8h\n"
                                             "fmax v4.8h, v4.8h, v31.8h\n"
                                             "fmax v5.8h, v5.8h, v31.8h\n"
                                             "fmax v10.8h, v10.8h, v31.8h\n"
                                             "fmax v11.8h, v11.8h, v31.8h\n"
                                             "fmax v12.8h, v12.8h, v31.8h\n"
                                             "fmax v13.8h, v13.8h, v31.8h\n"

                                             "fmin v2.8h, v2.8h, v30.8h\n"
                                             "fmin v3.8h, v3.8h, v30.8h\n"
                                             "fmin v4.8h, v4.8h, v30.8h\n"
                                             "fmin v5.8h, v5.8h, v30.8h\n"
                                             "fmin v10.8h, v10.8h, v30.8h\n"
                                             "fmin v11.8h, v11.8h, v30.8h\n"
                                             "fmin v12.8h, v12.8h, v30.8h\n"
                                             "fmin v13.8h, v13.8h, v30.8h\n"
                                             :
                                             :
                                             : "memory", "cc", "v2", "v3", "v4", "v5", "v10", "v11",
                                             "v12", "v13", "v30", "v31");
                        break;
                    }
                    default: {
                        ret = NOT_SUPPORTED;
                        break;
                    }
                }

                __asm__ __volatile__(
                    "str   q2, [%[out_0]]\n"       // out_o0hw0
                    "str   q3, [%[out_0], #16]\n"  // out_o0hw1
                    "str   q4, [%[out_0], #32]\n"  // out_o0hw2
                    "str   q5, [%[out_0], #48]\n"  // out_o0hw3
                    "str  q10, [%[out_1]]\n"       // out_o1hw0
                    "str  q11, [%[out_1], #16]\n"  // out_o1hw1
                    "str  q12, [%[out_1], #32]\n"  // out_o1hw2
                    "str  q13, [%[out_1], #48]\n"  // out_o1hw3
                    : [out_0] "+r"(out_o0hw0), [out_1] "+r"(out_o1hw0)
                    :
                    : "memory", "cc", "v2", "v3", "v4", "v5", "v10", "v11", "v12", "v13");
            }
            if (oc & 1) {
                // oc%2 != 0
                const F16 *f_r = filterArray + oc_1 * 8 * K;
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + ((n * oc + oc_1) * ohow + hw) * 8;
                // bias
                const F16 *b_o0 = biasArray + oc_1 * 8;
                __asm__ __volatile__("ldr d22, [%[b_0]]\n"  // b_o0
                                     "ldr  x1, [%[b_0], #8]\n"
                                     "ins v22.d[1], x1\n"
                                     "mov  x0, %[ic]\n"        // ic_blk
                                     "mov  v2.16b, v22.16b\n"  // out_o0hw0
                                     "ldr  d0, [%[in_0]]\n"    // in_hw0
                                     "mov  v3.16b, v22.16b\n"  // out_o0hw1
                                     "ldr d18, [%[f_0]]\n"     // f_o0c0
                                     "mov  v4.16b, v22.16b\n"  // out_o0hw2
                                     "ldr  x2, [%[f_0], #8]\n"
                                     "mov  v5.16b, v22.16b\n"  // out_o0hw3
                                     "ins v18.d[1], x2\n"

                                     "0:\n"
                                     "cmp x0, #1\n"
                                     "ble 1f\n"
                                     "ldr  d1, [%[in_0], #8]\n"  // in_hw0
                                     "fmla  v2.8h, v18.8h, v0.h[0]\n"
                                     "ldr d20, [%[f_0], #16]\n"  // f_o0c0
                                     "fmla  v3.8h, v18.8h, v0.h[1]\n"
                                     "ldr  x2, [%[f_0], #24]\n"
                                     "fmla  v4.8h, v18.8h, v0.h[2]\n"
                                     "ins v20.d[1], x2\n"
                                     "fmla  v5.8h, v18.8h, v0.h[3]\n"
                                     "sub x0, x0, #2\n"

                                     "ldr  d0, [%[in_0], #16]\n"  // in_hw0
                                     "fmla  v2.8h, v20.8h, v1.h[0]\n"
                                     "ldr d18, [%[f_0], #32]\n"  // f_o0c0
                                     "fmla  v3.8h, v20.8h, v1.h[1]\n"
                                     "ldr  x2, [%[f_0], #40]\n"
                                     "fmla  v4.8h, v20.8h, v1.h[2]\n"
                                     "ins v18.d[1], x2\n"
                                     "fmla  v5.8h, v20.8h, v1.h[3]\n"
                                     "add %[in_0], %[in_0], #16\n"
                                     "add %[f_0], %[f_0], #32\n"
                                     "b 0b\n"

                                     "1:\n"
                                     "blt 2f\n"
                                     "fmla  v2.8h, v18.8h, v0.h[0]\n"
                                     "fmla  v3.8h, v18.8h, v0.h[1]\n"
                                     "add %[f_0], %[f_0], #16\n"
                                     "fmla  v4.8h, v18.8h, v0.h[2]\n"
                                     "fmla  v5.8h, v18.8h, v0.h[3]\n"
                                     "2:\n"
                                     : [in_0] "+r"(in_hw0), [f_0] "+r"(f_r)
                                     : [ic] "r"(K), [b_0] "r"(b_o0)
                                     : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v18",
                                     "v20", "v22", "x0", "x1", "x2");
                switch (activationDesc.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("eor v1.16b, v1.16b, v1.16b\n"  // zero
                                             "fmax  v2.8h,  v2.8h, v1.8h\n"  // max(v2, 0)
                                             "fmax  v3.8h,  v3.8h, v1.8h\n"
                                             "fmax  v4.8h,  v4.8h, v1.8h\n"
                                             "fmax  v5.8h,  v5.8h, v1.8h\n"
                                             :
                                             :
                                             : "memory", "cc", "v1", "v2", "v3", "v4", "v5");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("eor v31.16b, v31.16b, v31.16b\n"  // zero
                                             "movi v30.8h, #0x46, lsl #8\n"     // six
                                             "fmax v2.8h, v2.8h, v31.8h\n"
                                             "fmax v3.8h, v3.8h, v31.8h\n"
                                             "fmax v4.8h, v4.8h, v31.8h\n"
                                             "fmax v5.8h, v5.8h, v31.8h\n"

                                             "fmin v2.8h, v2.8h, v30.8h\n"
                                             "fmin v3.8h, v3.8h, v30.8h\n"
                                             "fmin v4.8h, v4.8h, v30.8h\n"
                                             "fmin v5.8h, v5.8h, v30.8h\n"
                                             :
                                             :
                                             : "memory", "cc", "v2", "v3", "v4", "v5", "v30", "v31");
                        break;
                    }
                    default: {
                        ret = NOT_SUPPORTED;
                        break;
                    }
                }

                __asm__ __volatile__("str   q2, [%[out_0]]\n"       // out_o0hw0
                                     "str   q3, [%[out_0], #16]\n"  // out_o0hw1
                                     "str   q4, [%[out_0], #32]\n"  // out_o0hw2
                                     "str   q5, [%[out_0], #48]\n"  // out_o0hw3
                                     : [out_0] "+r"(out_o0hw0)
                                     :
                                     : "memory", "cc", "v2", "v3", "v4", "v5");
            }
        }
        // ohow_reminder % 4
        ohow_s = (ohow / 4) * 4;
        for (I32 hw = ohow_s; hw < ohow; hw++) {
            const F16 *f_o0c0 = filterArray;
            // pack input
            // NCHWc8 => NHWChw1 + im2col
            convolution_nchw_input_pack<F16, 1>(
                ic, it_pad, ih_pad, iw_pad, p, ft, fh, fw, ot, oh, ow, inArray_pad, hw, in_pack);

            // compute
            for (I32 o = 0; o < oc_1; o += 2) {
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + ((n * oc + o) * ohow + hw) * 8;
                F16 *out_o1hw0 = out_o0hw0 + ohow * 8;
                // bias
                const F16 *b_o0 = biasArray + o * 8;
                const F16 *b_o1 = b_o0 + 8;
                __asm__ __volatile__("ldr d22, [%[b_0]]\n"  // b_o0
                                     "ldr  x1, [%[b_0], #8]\n"
                                     "ins v22.d[1], x1\n"
                                     "ldr d23, [%[b_1]]\n"  // b_o1
                                     "mov  x0, %[ic]\n"     // ic_blk
                                     "ldr  x2, [%[b_1], #8]\n"
                                     "ins v23.d[1], x2\n"
                                     "ldr  h0, [%[in_0]]\n"    // in_hw0
                                     "mov  v2.16b, v22.16b\n"  // out_o0hw0
                                     "ldr d18, [%[f_0]]\n"     // f_o0c0
                                     "mov v10.16b, v23.16b\n"  // out_o1hw0
                                     "ldr  x2, [%[f_0], #8]\n"
                                     "ins v18.d[1], x2\n"
                                     "ldr d19, [%[f_0], #16]\n"  // f_o1c0
                                     "ldr  x3, [%[f_0], #24]\n"
                                     "ins v19.d[1], x3\n"

                                     "0:\n"
                                     "cmp x0, #1\n"
                                     "ble 1f\n"
                                     "ldr  h1, [%[in_0], #2]\n"  // in_hw0
                                     "fmla  v2.8h, v18.8h, v0.h[0]\n"
                                     "ldr d20, [%[f_0], #32]\n"  // f_o0c0
                                     "fmla v10.8h, v19.8h, v0.h[0]\n"
                                     "ldr  x2, [%[f_0], #40]\n"
                                     "ins v20.d[1], x2\n"
                                     "ldr d21, [%[f_0], #48]\n"  // f_o1c0
                                     "sub x0, x0, #2\n"
                                     "ldr  x3, [%[f_0], #56]\n"
                                     "ins v21.d[1], x3\n"

                                     "ldr  h0, [%[in_0], #4]\n"  // in_hw0
                                     "fmla  v2.8h, v20.8h, v1.h[0]\n"
                                     "ldr d18, [%[f_0], #64]\n"  // f_o0c0
                                     "fmla v10.8h, v21.8h, v1.h[0]\n"
                                     "ldr  x2, [%[f_0], #72]\n"
                                     "ins v18.d[1], x2\n"
                                     "ldr d19, [%[f_0], #80]\n"  // f_o1c0
                                     "add %[in_0], %[in_0], #4\n"
                                     "ldr  x3, [%[f_0], #88]\n"
                                     "ins v19.d[1], x3\n"
                                     "add %[f_0], %[f_0], #64\n"
                                     "b 0b\n"

                                     "1:\n"
                                     "blt 2f\n"
                                     "fmla  v2.8h, v18.8h, v0.h[0]\n"
                                     "add %[f_0], %[f_0], #32\n"
                                     "fmla v10.8h, v19.8h, v0.h[0]\n"
                                     "2:\n"
                                     : [in_0] "+r"(in_hw0), [f_0] "+r"(f_o0c0)
                                     : [ic] "r"(K), [b_0] "r"(b_o0), [b_1] "r"(b_o1)
                                     : "memory", "cc", "v0", "v1", "v2", "v10", "v18", "v19", "v20",
                                     "v21", "v22", "v23", "x0", "x1", "x2", "x3");
                switch (activationDesc.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("eor v1.16b, v1.16b, v1.16b\n"  // zero
                                             "fmax  v2.8h,  v2.8h, v1.8h\n"  // max(v2, 0)
                                             "fmax v10.8h, v10.8h, v1.8h\n"
                                             :
                                             :
                                             : "memory", "cc", "v1", "v2", "v10");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("eor v31.16b, v31.16b, v31.16b\n"  // zero
                                             "movi v30.8h, #0x46, lsl #8\n"     // six
                                             "fmax v2.8h, v2.8h, v31.8h\n"
                                             "fmax v10.8h, v10.8h, v31.8h\n"

                                             "fmin v2.8h, v2.8h, v30.8h\n"
                                             "fmin v10.8h, v10.8h, v30.8h\n"
                                             :
                                             :
                                             : "memory", "cc", "v2", "v10", "v30", "v31");
                        break;
                    }
                    default: {
                        ret = NOT_SUPPORTED;
                        break;
                    }
                }

                __asm__ __volatile__("str   q2, [%[out_0]]\n"  // out_o0hw0
                                     "str  q10, [%[out_1]]\n"  // out_o1hw0
                                     : [out_0] "+r"(out_o0hw0), [out_1] "+r"(out_o1hw0)
                                     :
                                     : "memory", "cc", "v2", "v10");
            }
            if (oc & 1) {
                // oc%2 != 0
                const F16 *f_r = filterArray + oc_1 * 8 * K;
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + ((n * oc + oc_1) * ohow + hw) * 8;
                // bias
                const F16 *b_o0 = biasArray + oc_1 * 8;
                __asm__ __volatile__("ldr d22, [%[b_0]]\n"  // b_o0
                                     "mov  x0, %[ic]\n"     // ic_blk
                                     "ldr  x1, [%[b_0], #8]\n"
                                     "ins v22.d[1], x1\n"
                                     "ldr  h0, [%[in_0]]\n"    // in_hw0
                                     "mov  v2.16b, v22.16b\n"  // out_o0hw0
                                     "ldr d18, [%[f_0]]\n"     // f_o0c0
                                     "ldr  x2, [%[f_0], #8]\n"
                                     "ins v18.d[1], x2\n"

                                     "0:\n"
                                     "cmp x0, #1\n"
                                     "ble 1f\n"
                                     "ldr  h1, [%[in_0], #2]\n"  // in_hw0
                                     "fmla  v2.8h, v18.8h, v0.h[0]\n"
                                     "ldr d20, [%[f_0], #16]\n"  // f_o0c0
                                     "sub x0, x0, #2\n"
                                     "ldr  x2, [%[f_0], #24]\n"
                                     "ins v20.d[1], x2\n"

                                     "ldr  h0, [%[in_0], #4]\n"  // in_hw0
                                     "fmla  v2.8h, v20.8h, v1.h[0]\n"
                                     "ldr d18, [%[f_0], #32]\n"  // f_o0c0
                                     "ldr  x2, [%[f_0], #40]\n"
                                     "ins v18.d[1], x2\n"
                                     "add %[in_0], %[in_0], #4\n"
                                     "add %[f_0], %[f_0], #32\n"
                                     "b 0b\n"

                                     "1:\n"
                                     "blt 2f\n"
                                     "fmla  v2.8h, v18.8h, v0.h[0]\n"
                                     "add %[f_0], %[f_0], #16\n"
                                     "2:\n"
                                     : [in_0] "+r"(in_hw0), [f_0] "+r"(f_r)
                                     : [ic] "r"(K), [b_0] "r"(b_o0)
                                     : "memory", "cc", "v0", "v1", "v2", "v10", "v18", "v20", "v22",
                                     "x0", "x1", "x2");
                switch (activationDesc.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("eor v1.16b, v1.16b, v1.16b\n"  // zero
                                             "fmax  v2.8h,  v2.8h, v1.8h\n"  // max(v2, 0)
                                             :
                                             :
                                             : "memory", "cc", "v1", "v2");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("eor v31.16b, v31.16b, v31.16b\n"  // zero
                                             "movi v30.8h, #0x46, lsl #8\n"     // six
                                             "fmax v2.8h, v2.8h, v31.8h\n"
                                             "fmin v2.8h, v2.8h, v30.8h\n"
                                             :
                                             :
                                             : "memory", "cc", "v2", "v30", "v31");
                        break;
                    }
                    default: {
                        ret = NOT_SUPPORTED;
                        break;
                    }
                }

                __asm__ __volatile__("str   q2, [%[out_0]]\n"  // out_o0hw0
                                     : [out_0] "+r"(out_o0hw0)
                                     :
                                     : "memory", "cc", "v2");
            }
        }
    }
    return ret;
}
