// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/fp32/tensor_computing_fp32.h"
#include "cpu/arm/transform_functions.h"
#include "thread_affinity.h"

EE convolution_gemm_V8(TensorDesc inputDesc,
    F32 *inArray,
    TensorDesc filterDesc,
    const F32 *filterArray,
    ConvolutionParamSpec p,
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
    U32 in, ic, it, ih, iw;
    U32 fn, fc, ft, fh, fw;
    U32 on, oc, ot, oh, ow;
    if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
        it = ft = ot = 1;
        p.dilatedRate_t = p.stride_t = 1;
        p.pad_before = p.pad_after = 0;
    } else if (tensorIs5d(inputDesc)) {
        CHECK_STATUS(tensor5dGet(inputDesc, &idt, &idf, &in, &ic, &it, &ih, &iw));
        CHECK_STATUS(tensor5dGet(filterDesc, &fdt, &fdf, &fn, &fc, &ft, &fh, &fw));
        CHECK_STATUS(tensor5dGet(outputDesc, &odt, &odf, &on, &oc, &ot, &oh, &ow));
    } else {
        return NOT_SUPPORTED;
    }

    if (fdf != DF_NHWCN8) {
        CHECK_STATUS(NOT_MATCH);
    }

    oc /= 8;
    U32 it_pad = it + p.pad_before + p.pad_after;
    U32 ih_pad = ih + p.pad_top + p.pad_bottom;
    U32 iw_pad = iw + p.pad_left + p.pad_right;
    I64 K = ic * ft * fh * fw;
    I32 ohow = ot * oh * ow;
    F32 *in_pack = ((F32 *)tmp) + ic * it_pad * ih_pad * iw_pad;
    EE ret = SUCCESS;
    ic /= 8;
#ifdef _USE_CACHE
    std::string key = tensorDesc2Str(inputDesc) + tensorDesc2Str(filterDesc) +
        tensorDesc2Str(outputDesc) + std::to_string(ih_pad) + std::to_string(iw_pad) +
        std::to_string(p.stride_h) + std::to_string(p.stride_w);
    U32 *padding_input_offsets =
        get_convolution_padding_input_offset<12, 8>(key, ih_pad, iw_pad, p, oh, ow, ohow);
#endif
    for (U32 n = 0; n < in; n++) {
        F32 *inArray_pad = convolution_input_padding_per_channel<F32, 8>(
            n, ic, it, ih, iw, p, inArray, (F32 *)tmp);

        // ohow / 12
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (I32 hw = 0; hw < ohow - 11; hw += 12) {
#ifdef _USE_OPENMP
            F32 *thread_in_pack = in_pack + 12 * K * omp_get_thread_num();
#else
            F32 *thread_in_pack = in_pack;
#endif
            // pack input
            // NCHWc8 => NHWChw12 + im2col
#ifdef _USE_CACHE
            U32 *padding_input_offset = padding_input_offsets + hw;
#else
            U32 padding_input_offset[12];
            convolution_padding_input_offset<12, 8>(
                ih_pad, iw_pad, p, oh, ow, hw, padding_input_offset);
#endif
            for (U32 c = 0; c < ic; c++) {
                for (U32 ft_idx = 0, id = 0; ft_idx < ft; ft_idx++) {
                    for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                        for (U32 fw_idx = 0; fw_idx < fw; fw_idx++, id++) {
                            F32 *in_hw12c8 = inArray_pad +
                                (((c * it_pad + ft_idx * p.dilatedRate_t) * ih_pad +
                                     fh_idx * p.dilatedRate_h) *
                                        iw_pad +
                                    p.dilatedRate_w * fw_idx) *
                                    8;
                            F32 *in_0 = in_hw12c8 + padding_input_offset[0];
                            F32 *in_1 = in_hw12c8 + padding_input_offset[1];
                            F32 *in_2 = in_hw12c8 + padding_input_offset[2];
                            F32 *in_3 = in_hw12c8 + padding_input_offset[3];
                            F32 *in_4 = in_hw12c8 + padding_input_offset[4];
                            F32 *in_5 = in_hw12c8 + padding_input_offset[5];
                            F32 *in_6 = in_hw12c8 + padding_input_offset[6];
                            F32 *in_7 = in_hw12c8 + padding_input_offset[7];
                            F32 *in_8 = in_hw12c8 + padding_input_offset[8];
                            F32 *in_9 = in_hw12c8 + padding_input_offset[9];
                            F32 *in_10 = in_hw12c8 + padding_input_offset[10];
                            F32 *in_11 = in_hw12c8 + padding_input_offset[11];

                            // NHWChw12
                            F32 *in_pack_c8hw12 = thread_in_pack + (id * ic + c) * 8 * 12;

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
                                : [pack] "r"(in_pack_c8hw12), [in_0] "r"(in_0), [in_1] "r"(in_1),
                                [in_2] "r"(in_2), [in_3] "r"(in_3), [in_4] "r"(in_4),
                                [in_5] "r"(in_5), [in_6] "r"(in_6), [in_7] "r"(in_7), [in_8] "r"(in_8),
                                [in_9] "r"(in_9), [in_10] "r"(in_10), [in_11] "r"(in_11)
                                : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                                "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
                                "v27");
                        }
                    }
                }
            }

            // compute
            for (U32 o = 0; o < oc; o++) {
                F32 *in_hw0 = thread_in_pack;
                const F32 *f_o0c0 = filterArray + o * 8 * K;
                F32 *out_o0hw0 = outArray + ((n * oc + o) * ohow + hw) * 8;
                // bias
                const F32 *b_o0 = biasArray + o * 8;
                const F32 *b_o1 = b_o0 + 4;
                __asm__ __volatile__("ldr q27, [%[b_0]]\n"
                                     "ldr q28, [%[b_1]]\n"
                                     // give in address to x3
                                     "mov x3, %[in_0]\n"

                                     // give f address to x0
                                     "mov x0, %[f_0]\n"

                                     "mov  x2, %[ic]\n"

                                     "mov  v5.16b, v27.16b\n"
                                     "ldr  q1, [%[in_0]]\n"  // in_hw0
                                     "mov  v7.16b, v27.16b\n"
                                     "mov  v9.16b, v27.16b\n"
                                     "mov  v11.16b, v27.16b\n"
                                     "ldr q0, [%[f_0]]\n"  // f_o0c0
                                     "mov  v13.16b, v27.16b\n"
                                     "mov  v15.16b, v27.16b\n"
                                     "mov  v17.16b, v27.16b\n"
                                     "ldr q3, [%[in_0], #16]\n"
                                     "mov  v19.16b, v27.16b\n"
                                     "mov v21.16b, v27.16b\n"
                                     "mov v23.16b, v27.16b\n"
                                     "mov v25.16b, v27.16b\n"

                                     "mov v6.16b, v28.16b\n"
                                     "mov v8.16b, v28.16b\n"
                                     "mov v10.16b, v28.16b\n"
                                     "mov v12.16b, v28.16b\n"
                                     "mov v14.16b, v28.16b\n"
                                     "mov v16.16b, v28.16b\n"
                                     "mov v18.16b, v28.16b\n"
                                     "mov v20.16b, v28.16b\n"
                                     "mov v22.16b, v28.16b\n"
                                     "mov v24.16b, v28.16b\n"
                                     "mov v26.16b, v28.16b\n"
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
                                     : [in_0] "+r"(in_hw0), [f_0] "+r"(f_o0c0)
                                     : [ic] "r"(K), [b_0] "r"(b_o0), [b_1] "r"(b_o1)
                                     : "memory", "cc", "v0", "v1", "v2", "v3", "v5", "v6", "v7",
                                     "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
                                     "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                                     "v26", "v27", "v28", "v29", "v30", "x0", "x1", "x2", "x3");

                switch (activationDesc.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("eor v1.16b, v1.16b, v1.16b\n"  // zero
                                             "fmax  v5.4s,  v5.4s, v1.4s\n"
                                             "fmax  v6.4s,  v6.4s, v1.4s\n"
                                             "fmax  v7.4s,  v7.4s, v1.4s\n"
                                             "fmax  v8.4s,  v8.4s, v1.4s\n"
                                             "fmax  v9.4s,  v9.4s, v1.4s\n"
                                             "fmax v10.4s, v10.4s, v1.4s\n"
                                             "fmax v11.4s, v11.4s, v1.4s\n"
                                             "fmax v12.4s, v12.4s, v1.4s\n"
                                             "fmax v13.4s, v13.4s, v1.4s\n"
                                             "fmax v14.4s, v14.4s, v1.4s\n"
                                             "fmax v15.4s, v15.4s, v1.4s\n"
                                             "fmax v16.4s, v16.4s, v1.4s\n"
                                             "fmax v17.4s, v17.4s, v1.4s\n"
                                             "fmax v18.4s, v18.4s, v1.4s\n"
                                             "fmax v19.4s, v19.4s, v1.4s\n"
                                             "fmax v20.4s, v20.4s, v1.4s\n"
                                             "fmax v21.4s, v21.4s, v1.4s\n"
                                             "fmax v22.4s, v22.4s, v1.4s\n"
                                             "fmax v23.4s, v23.4s, v1.4s\n"
                                             "fmax v24.4s, v24.4s, v1.4s\n"
                                             "fmax v25.4s, v25.4s, v1.4s\n"
                                             "fmax v26.4s, v26.4s, v1.4s\n"
                                             "fmax v27.4s, v27.4s, v1.4s\n"
                                             "fmax v28.4s, v28.4s, v1.4s\n"
                                             :
                                             :
                                             : "memory", "cc", "v1", "v5", "v6", "v7", "v8", "v9",
                                             "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                                             "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                                             "v26", "v27", "v28");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("eor v1.16b, v1.16b, v1.16b\n"  // zero
                                             "fmov v30.4s, 6.0\n"            // six
                                             "fmax  v5.4s,  v5.4s, v1.4s\n"
                                             "fmax  v6.4s,  v6.4s, v1.4s\n"
                                             "fmax  v7.4s,  v7.4s, v1.4s\n"
                                             "fmax  v8.4s,  v8.4s, v1.4s\n"
                                             "fmax  v9.4s,  v9.4s, v1.4s\n"
                                             "fmax v10.4s, v10.4s, v1.4s\n"
                                             "fmax v11.4s, v11.4s, v1.4s\n"
                                             "fmax v12.4s, v12.4s, v1.4s\n"
                                             "fmax v13.4s, v13.4s, v1.4s\n"
                                             "fmax v14.4s, v14.4s, v1.4s\n"
                                             "fmax v15.4s, v15.4s, v1.4s\n"
                                             "fmax v16.4s, v16.4s, v1.4s\n"
                                             "fmax v17.4s, v17.4s, v1.4s\n"
                                             "fmax v18.4s, v18.4s, v1.4s\n"
                                             "fmax v19.4s, v19.4s, v1.4s\n"
                                             "fmax v20.4s, v20.4s, v1.4s\n"
                                             "fmax v21.4s, v21.4s, v1.4s\n"
                                             "fmax v22.4s, v22.4s, v1.4s\n"
                                             "fmax v23.4s, v23.4s, v1.4s\n"
                                             "fmax v24.4s, v24.4s, v1.4s\n"
                                             "fmax v25.4s, v25.4s, v1.4s\n"
                                             "fmax v26.4s, v26.4s, v1.4s\n"
                                             "fmax v27.4s, v27.4s, v1.4s\n"
                                             "fmax v28.4s, v28.4s, v1.4s\n"

                                             "fmin v5.4s, v5.4s, v30.4s\n"
                                             "fmin v6.4s, v6.4s, v30.4s\n"
                                             "fmin v7.4s, v7.4s, v30.4s\n"
                                             "fmin v8.4s, v8.4s, v30.4s\n"
                                             "fmin v9.4s, v9.4s, v30.4s\n"
                                             "fmin v10.4s, v10.4s, v30.4s\n"
                                             "fmin v11.4s, v11.4s, v30.4s\n"
                                             "fmin v12.4s, v12.4s, v30.4s\n"
                                             "fmin v13.4s, v13.4s, v30.4s\n"
                                             "fmin v14.4s, v14.4s, v30.4s\n"
                                             "fmin v15.4s, v15.4s, v30.4s\n"
                                             "fmin v16.4s, v16.4s, v30.4s\n"
                                             "fmin v17.4s, v17.4s, v30.4s\n"
                                             "fmin v18.4s, v18.4s, v30.4s\n"
                                             "fmin v19.4s, v19.4s, v30.4s\n"
                                             "fmin v20.4s, v20.4s, v30.4s\n"
                                             "fmin v21.4s, v21.4s, v30.4s\n"
                                             "fmin v22.4s, v22.4s, v30.4s\n"
                                             "fmin v23.4s, v23.4s, v30.4s\n"
                                             "fmin v24.4s, v24.4s, v30.4s\n"
                                             "fmin v25.4s, v25.4s, v30.4s\n"
                                             "fmin v26.4s, v26.4s, v30.4s\n"
                                             "fmin v27.4s, v27.4s, v30.4s\n"
                                             "fmin v28.4s, v28.4s, v30.4s\n"
                                             :
                                             :
                                             : "memory", "cc", "v1", "v5", "v6", "v7", "v8", "v9",
                                             "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                                             "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                                             "v26", "v27", "v28", "v30");
                        break;
                    }
                    default: {
                        ret = NOT_SUPPORTED;
                        break;
                    }
                }

                __asm__ __volatile__("str   q5, [%[out_0]]\n"
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
                                     : [out_0] "+r"(out_o0hw0)
                                     :
                                     : "memory", "cc", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                                     "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                                     "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28");
            }
        }

        U32 ohow_s = (ohow / 12) * 12;
        U32 ohow_tail = ohow - ohow_s;

        if (ohow_tail >= 8) {
            I32 hw = ohow_s;
            // pack input
            // NCHWc8 => NHWChw8 + im2col
            U32 padding_input_offset[8];
            convolution_padding_input_offset<8, 8>(
                ih_pad, iw_pad, p, oh, ow, hw, padding_input_offset);
            for (U32 c = 0; c < ic; c++) {
                for (U32 ft_idx = 0, id = 0; ft_idx < ft; ft_idx++) {
                    for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                        for (U32 fw_idx = 0; fw_idx < fw; fw_idx++, id++) {
                            F32 *in_hw8c8 = inArray_pad +
                                (((c * it_pad + ft_idx * p.dilatedRate_t) * ih_pad +
                                     fh_idx * p.dilatedRate_h) *
                                        iw_pad +
                                    p.dilatedRate_w * fw_idx) *
                                    8;
                            F32 *in_0 = in_hw8c8 + padding_input_offset[0];
                            F32 *in_1 = in_hw8c8 + padding_input_offset[1];
                            F32 *in_2 = in_hw8c8 + padding_input_offset[2];
                            F32 *in_3 = in_hw8c8 + padding_input_offset[3];
                            F32 *in_4 = in_hw8c8 + padding_input_offset[4];
                            F32 *in_5 = in_hw8c8 + padding_input_offset[5];
                            F32 *in_6 = in_hw8c8 + padding_input_offset[6];
                            F32 *in_7 = in_hw8c8 + padding_input_offset[7];
                            F32 *in_pack_c8hw8 = in_pack + (id * ic + c) * 8 * 8;
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
                                : [pack] "r"(in_pack_c8hw8), [in_0] "r"(in_0), [in_1] "r"(in_1),
                                [in_2] "r"(in_2), [in_3] "r"(in_3), [in_4] "r"(in_4),
                                [in_5] "r"(in_5), [in_6] "r"(in_6), [in_7] "r"(in_7)
                                : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v24", "v25",
                                "v26", "v27");
                        }
                    }
                }
            }

            // compute
            for (U32 o = 0; o < oc; o++) {
                F32 *in_hw0 = in_pack;
                const F32 *f_o0c0 = filterArray + o * 8 * K;
                F32 *out_o0hw0 = outArray + ((n * oc + o) * ohow + hw) * 8;
                // bias
                const F32 *b_o0 = biasArray + o * 8;
                const F32 *b_o1 = b_o0 + 4;
                __asm__ __volatile__("ldr q27, [%[b_0]]\n"
                                     "ldr q28, [%[b_1]]\n"
                                     // give in address to x3
                                     "mov x3, %[in_0]\n"

                                     // give f address to x0
                                     "mov x0, %[f_0]\n"

                                     "mov  x2, %[ic]\n"

                                     "mov  v5.16b, v27.16b\n"
                                     "ldr  q1, [%[in_0]]\n"  // in_hw0
                                     "mov  v7.16b, v27.16b\n"
                                     "mov  v9.16b, v27.16b\n"
                                     "mov  v11.16b, v27.16b\n"
                                     "ldr q0, [%[f_0]]\n"  // f_o0c0
                                     "mov  v13.16b, v27.16b\n"
                                     "mov  v15.16b, v27.16b\n"
                                     "mov  v17.16b, v27.16b\n"
                                     "mov  v19.16b, v27.16b\n"

                                     "mov v6.16b, v28.16b\n"
                                     "mov v8.16b, v28.16b\n"
                                     "mov v10.16b, v28.16b\n"
                                     "mov v12.16b, v28.16b\n"
                                     "mov v14.16b, v28.16b\n"
                                     "mov v16.16b, v28.16b\n"
                                     "mov v18.16b, v28.16b\n"
                                     "mov v20.16b, v28.16b\n"
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
                                     : [in_0] "+r"(in_hw0), [f_0] "+r"(f_o0c0)
                                     : [ic] "r"(K), [b_0] "r"(b_o0), [b_1] "r"(b_o1)
                                     : "memory", "cc", "v0", "v1", "v3", "v5", "v6", "v7", "v8",
                                     "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                                     "v18", "v19", "v20", "v27", "v28", "v29", "x0", "x1", "x2",
                                     "x3");

                switch (activationDesc.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("eor v1.16b, v1.16b, v1.16b\n"  // zero
                                             "fmax  v5.4s,  v5.4s, v1.4s\n"
                                             "fmax  v6.4s,  v6.4s, v1.4s\n"
                                             "fmax  v7.4s,  v7.4s, v1.4s\n"
                                             "fmax  v8.4s,  v8.4s, v1.4s\n"
                                             "fmax  v9.4s,  v9.4s, v1.4s\n"
                                             "fmax v10.4s, v10.4s, v1.4s\n"
                                             "fmax v11.4s, v11.4s, v1.4s\n"
                                             "fmax v12.4s, v12.4s, v1.4s\n"
                                             "fmax v13.4s, v13.4s, v1.4s\n"
                                             "fmax v14.4s, v14.4s, v1.4s\n"
                                             "fmax v15.4s, v15.4s, v1.4s\n"
                                             "fmax v16.4s, v16.4s, v1.4s\n"
                                             "fmax v17.4s, v17.4s, v1.4s\n"
                                             "fmax v18.4s, v18.4s, v1.4s\n"
                                             "fmax v19.4s, v19.4s, v1.4s\n"
                                             "fmax v20.4s, v20.4s, v1.4s\n"
                                             :
                                             :
                                             : "memory", "cc", "v1", "v5", "v6", "v7", "v8", "v9",
                                             "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                                             "v18", "v19", "v20");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("eor v1.16b, v1.16b, v1.16b\n"  // zero
                                             "fmov v30.4s, 6.0\n"            // six
                                             "fmax  v5.4s,  v5.4s, v1.4s\n"
                                             "fmax  v6.4s,  v6.4s, v1.4s\n"
                                             "fmax  v7.4s,  v7.4s, v1.4s\n"
                                             "fmax  v8.4s,  v8.4s, v1.4s\n"
                                             "fmax  v9.4s,  v9.4s, v1.4s\n"
                                             "fmax v10.4s, v10.4s, v1.4s\n"
                                             "fmax v11.4s, v11.4s, v1.4s\n"
                                             "fmax v12.4s, v12.4s, v1.4s\n"
                                             "fmax v13.4s, v13.4s, v1.4s\n"
                                             "fmax v14.4s, v14.4s, v1.4s\n"
                                             "fmax v15.4s, v15.4s, v1.4s\n"
                                             "fmax v16.4s, v16.4s, v1.4s\n"
                                             "fmax v17.4s, v17.4s, v1.4s\n"
                                             "fmax v18.4s, v18.4s, v1.4s\n"
                                             "fmax v19.4s, v19.4s, v1.4s\n"
                                             "fmax v20.4s, v20.4s, v1.4s\n"

                                             "fmin v5.4s, v5.4s, v30.4s\n"
                                             "fmin v6.4s, v6.4s, v30.4s\n"
                                             "fmin v7.4s, v7.4s, v30.4s\n"
                                             "fmin v8.4s, v8.4s, v30.4s\n"
                                             "fmin v9.4s, v9.4s, v30.4s\n"
                                             "fmin v10.4s, v10.4s, v30.4s\n"
                                             "fmin v11.4s, v11.4s, v30.4s\n"
                                             "fmin v12.4s, v12.4s, v30.4s\n"
                                             "fmin v13.4s, v13.4s, v30.4s\n"
                                             "fmin v14.4s, v14.4s, v30.4s\n"
                                             "fmin v15.4s, v15.4s, v30.4s\n"
                                             "fmin v16.4s, v16.4s, v30.4s\n"
                                             "fmin v17.4s, v17.4s, v30.4s\n"
                                             "fmin v18.4s, v18.4s, v30.4s\n"
                                             "fmin v19.4s, v19.4s, v30.4s\n"
                                             "fmin v20.4s, v20.4s, v30.4s\n"
                                             :
                                             :
                                             : "memory", "cc", "v1", "v5", "v6", "v7", "v8", "v9",
                                             "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                                             "v18", "v19", "v20", "v30");
                        break;
                    }
                    default: {
                        ret = NOT_SUPPORTED;
                        break;
                    }
                }

                __asm__ __volatile__("str   q5, [%[out_0]]\n"
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
                                     : [out_0] "+r"(out_o0hw0)
                                     :
                                     : "memory", "cc", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                                     "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20");
            }
            ohow_s += 8;
            ohow_tail -= 8;
        }

        if (ohow_tail >= 4) {
            I32 hw = ohow_s;
            // pack input
            // NCHWc8 => NHWChw4 + im2col
            U32 padding_input_offset[4];
            convolution_padding_input_offset<4, 8>(
                ih_pad, iw_pad, p, oh, ow, hw, padding_input_offset);
            for (U32 c = 0; c < ic; c++) {
                for (U32 ft_idx = 0, id = 0; ft_idx < ft; ft_idx++) {
                    for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                        for (U32 fw_idx = 0; fw_idx < fw; fw_idx++, id++) {
                            F32 *in_hw4c8 = inArray_pad +
                                (((c * it_pad + ft_idx * p.dilatedRate_t) * ih_pad +
                                     fh_idx * p.dilatedRate_h) *
                                        iw_pad +
                                    p.dilatedRate_w * fw_idx) *
                                    8;
                            F32 *in_0 = in_hw4c8 + padding_input_offset[0];
                            F32 *in_1 = in_hw4c8 + padding_input_offset[1];
                            F32 *in_2 = in_hw4c8 + padding_input_offset[2];
                            F32 *in_3 = in_hw4c8 + padding_input_offset[3];
                            F32 *in_pack_c8hw4 = in_pack + (id * ic + c) * 8 * 4;

                            __asm__ __volatile__(
                                "ldp q0, q4, [%[in_0]]\n"
                                "ldp q1, q5, [%[in_1]]\n"
                                "ldp q2, q6, [%[in_2]]\n"
                                "ldp q3, q7, [%[in_3]]\n"

                                "st4 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[pack]], #64\n"
                                "st4 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[pack]]\n"
                                : [pack] "+r"(in_pack_c8hw4)
                                : [in_0] "r"(in_0), [in_1] "r"(in_1), [in_2] "r"(in_2), [in_3] "r"(in_3)
                                : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
                        }
                    }
                }
            }

            // compute
            for (U32 o = 0; o < oc; o++) {
                F32 *in_hw0 = in_pack;
                const F32 *f_o0c0 = filterArray + o * 8 * K;
                F32 *out_o0hw0 = outArray + ((n * oc + o) * ohow + hw) * 8;
                // bias
                const F32 *b_o0 = biasArray + o * 8;
                const F32 *b_o1 = b_o0 + 4;
                __asm__ __volatile__("ldr q27, [%[b_0]]\n"
                                     "ldr q28, [%[b_1]]\n"
                                     // give in address to x3
                                     "mov x3, %[in_0]\n"

                                     // give f address to x0
                                     "mov x0, %[f_0]\n"

                                     "mov  x2, %[ic]\n"

                                     "mov  v5.16b, v27.16b\n"
                                     "ldr  q1, [%[in_0]]\n"  // in_hw0
                                     "mov  v7.16b, v27.16b\n"
                                     "mov  v9.16b, v27.16b\n"
                                     "mov  v11.16b, v27.16b\n"
                                     "ldr q0, [%[f_0]]\n"  // f_o0c0

                                     "mov v6.16b, v28.16b\n"
                                     "mov v8.16b, v28.16b\n"
                                     "mov v10.16b, v28.16b\n"
                                     "mov v12.16b, v28.16b\n"
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
                                     : [in_0] "+r"(in_hw0), [f_0] "+r"(f_o0c0)
                                     : [ic] "r"(K), [b_0] "r"(b_o0), [b_1] "r"(b_o1)
                                     : "memory", "cc", "v0", "v1", "v3", "v5", "v6", "v7", "v8",
                                     "v9", "v10", "v11", "v12", "v27", "v28", "v29", "x0", "x1",
                                     "x2", "x3");

                switch (activationDesc.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("eor v1.16b, v1.16b, v1.16b\n"  // zero
                                             "fmax  v5.4s,  v5.4s, v1.4s\n"
                                             "fmax  v6.4s,  v6.4s, v1.4s\n"
                                             "fmax  v7.4s,  v7.4s, v1.4s\n"
                                             "fmax  v8.4s,  v8.4s, v1.4s\n"
                                             "fmax  v9.4s,  v9.4s, v1.4s\n"
                                             "fmax v10.4s, v10.4s, v1.4s\n"
                                             "fmax v11.4s, v11.4s, v1.4s\n"
                                             "fmax v12.4s, v12.4s, v1.4s\n"
                                             :
                                             :
                                             : "memory", "cc", "v1", "v5", "v6", "v7", "v8", "v9",
                                             "v10", "v11", "v12");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("eor v1.16b, v1.16b, v1.16b\n"  // zero
                                             "fmov v30.4s, 6.0\n"            // six
                                             "fmax  v5.4s,  v5.4s, v1.4s\n"
                                             "fmax  v6.4s,  v6.4s, v1.4s\n"
                                             "fmax  v7.4s,  v7.4s, v1.4s\n"
                                             "fmax  v8.4s,  v8.4s, v1.4s\n"
                                             "fmax  v9.4s,  v9.4s, v1.4s\n"
                                             "fmax v10.4s, v10.4s, v1.4s\n"
                                             "fmax v11.4s, v11.4s, v1.4s\n"
                                             "fmax v12.4s, v12.4s, v1.4s\n"

                                             "fmin v5.4s, v5.4s, v30.4s\n"
                                             "fmin v6.4s, v6.4s, v30.4s\n"
                                             "fmin v7.4s, v7.4s, v30.4s\n"
                                             "fmin v8.4s, v8.4s, v30.4s\n"
                                             "fmin v9.4s, v9.4s, v30.4s\n"
                                             "fmin v10.4s, v10.4s, v30.4s\n"
                                             "fmin v11.4s, v11.4s, v30.4s\n"
                                             "fmin v12.4s, v12.4s, v30.4s\n"
                                             :
                                             :
                                             : "memory", "cc", "v1", "v5", "v6", "v7", "v8", "v9",
                                             "v10", "v11", "v12", "v30");
                        break;
                    }
                    default: {
                        ret = NOT_SUPPORTED;
                        break;
                    }
                }

                __asm__ __volatile__(
                    "str   q5, [%[out_0]]\n"
                    "str   q6, [%[out_0], #16]\n"
                    "str   q7, [%[out_0], #32]\n"
                    "str   q8, [%[out_0], #48]\n"
                    "str   q9, [%[out_0], #64]\n"
                    "str   q10, [%[out_0], #80]\n"
                    "str   q11, [%[out_0], #96]\n"
                    "str   q12, [%[out_0], #112]\n"
                    : [out_0] "+r"(out_o0hw0)
                    :
                    : "memory", "cc", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12");
            }
            ohow_s += 4;
            ohow_tail -= 4;
        }

        for (I32 hw = ohow_s; hw < ohow; hw++) {
            // pack input
            // NCHW => NCHWc8hw1 + im2col
            convolution_nchwc8_input_pack_tile1<F32>(
                ic, it_pad, ih_pad, iw_pad, p, ft, fh, fw, ot, oh, ow, inArray_pad, hw, in_pack);

            // compute
            for (U32 o = 0; o < oc; o++) {
                F32 *in_hw0 = in_pack;
                const F32 *f_o0c0 = filterArray + o * 8 * K;
                F32 *out_o0hw0 = outArray + ((n * oc + o) * ohow + hw) * 8;
                // bias
                const F32 *b_o0 = biasArray + o * 8;
                const F32 *b_o1 = b_o0 + 4;
                __asm__ __volatile__("ldr q5, [%[b_0]]\n"
                                     "ldr q6, [%[b_1]]\n"
                                     // give in address to x3
                                     "mov x3, %[in_0]\n"

                                     // give f address to x0
                                     "mov x0, %[f_0]\n"

                                     "mov  x2, %[ic]\n"

                                     "ldr  s1, [%[in_0]]\n"     // in_hw0
                                     "ldp q0, q29, [%[f_0]]\n"  // f_o0c0

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
                                     : [in_0] "+r"(in_hw0), [f_0] "+r"(f_o0c0)
                                     : [ic] "r"(K), [b_0] "r"(b_o0), [b_1] "r"(b_o1)
                                     : "memory", "cc", "v0", "v1", "v3", "v5", "v6", "v28", "v29",
                                     "v30", "x0", "x1", "x2", "x3");

                switch (activationDesc.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("eor v1.16b, v1.16b, v1.16b\n"  // zero
                                             "fmax  v5.4s,  v5.4s, v1.4s\n"
                                             "fmax  v6.4s,  v6.4s, v1.4s\n"
                                             :
                                             :
                                             : "memory", "cc", "v1", "v5", "v6");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("eor v1.16b, v1.16b, v1.16b\n"  // zero
                                             "fmov v30.4s, 6.0\n"            // six
                                             "fmax  v5.4s,  v5.4s, v1.4s\n"
                                             "fmax  v6.4s,  v6.4s, v1.4s\n"

                                             "fmin v5.4s, v5.4s, v30.4s\n"
                                             "fmin v6.4s, v6.4s, v30.4s\n"
                                             :
                                             :
                                             : "memory", "cc", "v1", "v5", "v6", "v30");
                        break;
                    }
                    default: {
                        ret = NOT_SUPPORTED;
                        break;
                    }
                }

                __asm__ __volatile__("str   q5, [%[out_0]]\n"
                                     "str   q6, [%[out_0], #16]\n"
                                     : [out_0] "+r"(out_o0hw0)
                                     :
                                     : "memory", "cc", "v5", "v6");
            }
        }
    }
    return ret;
}
