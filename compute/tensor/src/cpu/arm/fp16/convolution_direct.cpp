// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <string.h>

#include "cpu/arm/fp16/convolution_direct.h"

EE convolution_direct(TensorDesc inputDesc,
    F16 *inArray,
    TensorDesc filterDesc,
    const F16 *filterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const F16 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F16 *outArray,
    ActivationParamSpec activationDesc,
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
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingB = convParamSpec.padding_bottom;
    U32 paddingL = convParamSpec.padding_left;
    U32 paddingR = convParamSpec.padding_right;

    if (fdf != DF_NCHWN16) {
        CHECK_STATUS(NOT_MATCH);
    }

    oc /= 8;
    ic /= 8;
    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;

    // naive, no blocking, in: NCHWc8, out: NOHWo8, filter: OCHWo16, no bias

    EE ret = SUCCESS;
    for (U32 n = 0; n < in; n++) {
        // copy input into a input with padding
        F16 *inArray_pad = (F16 *)tmp;
        F16 *inArray_pad_mov = inArray_pad;
        F16 *inArray_mov = inArray + n * ic * ih * iw * 8;
        for (U32 c = 0; c < ic; c++) {
            for (U32 h = 0; h < paddingT; h++) {
                memset(inArray_pad_mov, 0, iw_pad * 8 * bytesOf(idt));
                inArray_pad_mov += iw_pad * 8;
            }
            for (U32 h = paddingT; h < ih_pad - paddingB; h++) {
                memset(inArray_pad_mov, 0, paddingL * 8 * bytesOf(idt));
                inArray_pad_mov += paddingL * 8;
                memcpy(inArray_pad_mov, inArray_mov, iw * 8 * bytesOf(idt));
                inArray_pad_mov += iw * 8;
                inArray_mov += iw * 8;
                memset(inArray_pad_mov, 0, paddingR * 8 * bytesOf(idt));
                inArray_pad_mov += paddingR * 8;
            }
            for (U32 h = ih_pad - paddingB; h < ih_pad; h++) {
                memset(inArray_pad_mov, 0, iw_pad * 8 * bytesOf(idt));
                inArray_pad_mov += iw_pad * 8;
            }
        }

        // compute
        const F16 *f0 = filterArray;
        const F16 *f1 = f0 + fh * fw * 16;
        const F16 *f2 = f0 + fh * fw * 16 * 2;
        const F16 *f3 = f0 + fh * fw * 16 * 3;
        const F16 *f4 = f0 + fh * fw * 16 * 4;
        const F16 *f5 = f0 + fh * fw * 16 * 5;
        const F16 *f6 = f0 + fh * fw * 16 * 6;
        const F16 *f7 = f0 + fh * fw * 16 * 7;

        F16 *outo0h0 = outArray + n * oc * oh * ow * 8;
        F16 *outo1h0 = outo0h0 + oh * ow * 8;
        F16 *outo0h1 = outo0h0 + ow * 8;
        F16 *outo1h1 = outo1h0 + ow * 8;
        for (U32 o = 0; o < oc; o += 2) {
            for (U32 c = 0; c < ic; c++) {
                F16 *out_o0h0 = outo0h0;
                F16 *out_o1h0 = outo1h0;
                F16 *out_o0h1 = outo0h1;
                F16 *out_o1h1 = outo1h1;

                F16 *in_h0w0 = inArray_pad + n * ic * ih_pad * iw_pad * 8 + c * ih_pad * iw_pad * 8;
                F16 *in_h0w1 = in_h0w0 + strideW * 8;
                F16 *in_h0w2 = in_h0w0 + strideW * 8 * 2;
                F16 *in_h0w3 = in_h0w0 + strideW * 8 * 3;
                F16 *in_h1w0 = in_h0w0 + strideH * iw_pad * 8;
                F16 *in_h1w1 = in_h1w0 + strideW * 8;
                F16 *in_h1w2 = in_h1w0 + strideW * 8 * 2;
                F16 *in_h1w3 = in_h1w0 + strideW * 8 * 3;

                for (U32 h = 0; h < oh; h += 2) {
                    for (U32 w = 0; w < ow; w += 4) {
                        const F16 *f_c0 = f0;
                        const F16 *f_c1 = f1;
                        const F16 *f_c2 = f2;
                        const F16 *f_c3 = f3;
                        const F16 *f_c4 = f4;
                        const F16 *f_c5 = f5;
                        const F16 *f_c6 = f6;
                        const F16 *f_c7 = f7;
                        for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                            for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                                __asm__ __volatile__(
                                    "ldr d16, [%[f_c0]]\n"
                                    "ldr  x4, [%[f_c0], #8]\n"
                                    "ins v16.d[1], x4\n"
                                    "ldr  d0, [%[in_h0w0]]\n"
                                    "ldr  x0, [%[in_h0w0], #8]\n"
                                    "ins  v0.d[1], x0\n"
                                    "ldr  d1, [%[in_h0w1]]\n"
                                    "ldr  x1, [%[in_h0w1], #8]\n"
                                    "ins  v1.d[1], x1\n"
                                    "ldr  d2, [%[in_h0w2]]\n"
                                    "ldr  x2, [%[in_h0w2], #8]\n"
                                    "ins  v2.d[1], x2\n"
                                    "ldr  d3, [%[in_h0w3]]\n"
                                    "ldr  x3, [%[in_h0w3], #8]\n"
                                    "ins  v3.d[1], x3\n"
                                    "ldr  d4, [%[in_h1w0]]\n"
                                    "ldr  x0, [%[in_h1w0], #8]\n"
                                    "ins  v4.d[1], x0\n"
                                    "ldr  d5, [%[in_h1w1]]\n"
                                    "ldr  x1, [%[in_h1w1], #8]\n"
                                    "ins  v5.d[1], x1\n"
                                    "ldr  d6, [%[in_h1w2]]\n"
                                    "ldr  x2, [%[in_h1w2], #8]\n"
                                    "ins  v6.d[1], x2\n"
                                    "ldr  d7, [%[in_h1w3]]\n"
                                    "ldr  x3, [%[in_h1w3], #8]\n"
                                    "ins  v7.d[1], x3\n"
                                    "ldr  d8, [%[out_o0h0]]\n"
                                    "ldr  x0, [%[out_o0h0], #8]\n"
                                    "ins  v8.d[1], x0\n"
                                    "ldr  d9, [%[out_o0h0], #16]\n"
                                    "ldr  x1, [%[out_o0h0], #24]\n"
                                    "ins  v9.d[1], x1\n"
                                    "ldr d10, [%[out_o0h0], #32]\n"
                                    "ldr  x2, [%[out_o0h0], #40]\n"
                                    "ins v10.d[1], x2\n"
                                    "ldr d11, [%[out_o0h0], #48]\n"
                                    "ldr  x3, [%[out_o0h0], #56]\n"
                                    "ins v11.d[1], x3\n"
                                    "ldr d12, [%[out_o0h1]]\n"
                                    "ldr  x0, [%[out_o0h1], #8]\n"
                                    "ins v12.d[1], x0\n"
                                    "ldr d13, [%[out_o0h1], #16]\n"
                                    "ldr  x1, [%[out_o0h1], #24]\n"
                                    "ins v13.d[1], x1\n"
                                    "ldr d14, [%[out_o0h1], #32]\n"
                                    "ldr  x2, [%[out_o0h1], #40]\n"
                                    "ins v14.d[1], x2\n"
                                    "ldr d15, [%[out_o0h1], #48]\n"
                                    "ldr  x3, [%[out_o0h1], #56]\n"
                                    "ins v15.d[1], x3\n"

                                    "fmla  v8.8h,  v16.8h, v0.h[0]\n"
                                    "ldr d18, [%[out_o1h0]]\n"
                                    "fmla  v9.8h,  v16.8h, v1.h[0]\n"
                                    "ldr  x0, [%[out_o1h0], #8]\n"
                                    "fmla v10.8h,  v16.8h, v2.h[0]\n"
                                    "ldr d17, [%[f_c1]]\n"
                                    "fmla v11.8h,  v16.8h, v3.h[0]\n"
                                    "ldr  x5, [%[f_c1], #8]\n"
                                    "fmla v12.8h,  v16.8h, v4.h[0]\n"
                                    "ins v17.d[1], x5\n"
                                    "fmla v13.8h,  v16.8h, v5.h[0]\n"
                                    "ins v18.d[1], x0\n"
                                    "fmla v14.8h,  v16.8h, v6.h[0]\n"
                                    "ldr d19, [%[out_o1h0], #16]\n"
                                    "fmla v15.8h,  v16.8h, v7.h[0]\n"
                                    "ldr  x1, [%[out_o1h0], #24]\n"
                                    "fmla  v8.8h,  v17.8h, v0.h[1]\n"
                                    "ins v19.d[1], x1\n"
                                    "fmla  v9.8h,  v17.8h, v1.h[1]\n"
                                    "fmla v10.8h,  v17.8h, v2.h[1]\n"
                                    "ldr d16, [%[f_c2]]\n"
                                    "fmla v11.8h,  v17.8h, v3.h[1]\n"
                                    "ldr  x5, [%[f_c2], #8]\n"
                                    "fmla v12.8h,  v17.8h, v4.h[1]\n"
                                    "ins v16.d[1], x5\n"
                                    "fmla v13.8h,  v17.8h, v5.h[1]\n"
                                    "ldr d20, [%[out_o1h0], #32]\n"
                                    "fmla v14.8h,  v17.8h, v6.h[1]\n"
                                    "ldr  x2, [%[out_o1h0], #40]\n"
                                    "fmla v15.8h,  v17.8h, v7.h[1]\n"
                                    "ins v20.d[1], x2\n"
                                    "fmla  v8.8h,  v16.8h, v0.h[2]\n"
                                    "ldr d21, [%[out_o1h0], #48]\n"
                                    "fmla  v9.8h,  v16.8h, v1.h[2]\n"
                                    "fmla v10.8h,  v16.8h, v2.h[2]\n"
                                    "ldr d17, [%[f_c3]]\n"
                                    "fmla v11.8h,  v16.8h, v3.h[2]\n"
                                    "ldr  x5, [%[f_c3], #8]\n"
                                    "fmla v12.8h,  v16.8h, v4.h[2]\n"
                                    "ins v17.d[1], x5\n"
                                    "fmla v13.8h,  v16.8h, v5.h[2]\n"
                                    "ldr  x3, [%[out_o1h0], #56]\n"
                                    "fmla v14.8h,  v16.8h, v6.h[2]\n"
                                    "ins v21.d[1], x3\n"
                                    "fmla v15.8h,  v16.8h, v7.h[2]\n"
                                    "ldr d22, [%[out_o1h1]]\n"
                                    "fmla  v8.8h,  v17.8h, v0.h[3]\n"
                                    "ldr  x0, [%[out_o1h1], #8]\n"
                                    "fmla  v9.8h,  v17.8h, v1.h[3]\n"
                                    "ins v22.d[1], x0\n"
                                    "fmla v10.8h,  v17.8h, v2.h[3]\n"
                                    "ldr d16, [%[f_c4]]\n"
                                    "fmla v11.8h,  v17.8h, v3.h[3]\n"
                                    "ldr  x5, [%[f_c4], #8]\n"
                                    "fmla v12.8h,  v17.8h, v4.h[3]\n"
                                    "ins v16.d[1], x5\n"
                                    "fmla v13.8h,  v17.8h, v5.h[3]\n"
                                    "ldr d23, [%[out_o1h1], #16]\n"
                                    "fmla v14.8h,  v17.8h, v6.h[3]\n"
                                    "ldr  x1, [%[out_o1h1], #24]\n"
                                    "fmla v15.8h,  v17.8h, v7.h[3]\n"
                                    "ins v23.d[1], x1\n"
                                    "fmla  v8.8h,  v16.8h, v0.h[4]\n"
                                    "fmla  v9.8h,  v16.8h, v1.h[4]\n"
                                    "fmla v10.8h,  v16.8h, v2.h[4]\n"
                                    "ldr d17, [%[f_c5]]\n"
                                    "fmla v11.8h,  v16.8h, v3.h[4]\n"
                                    "ldr  x5, [%[f_c5], #8]\n"
                                    "fmla v12.8h,  v16.8h, v4.h[4]\n"
                                    "ins v17.d[1], x5\n"
                                    "fmla v13.8h,  v16.8h, v5.h[4]\n"
                                    "ldr d24, [%[out_o1h1], #32]\n"
                                    "fmla v14.8h,  v16.8h, v6.h[4]\n"
                                    "ldr  x2, [%[out_o1h1], #40]\n"
                                    "fmla v15.8h,  v16.8h, v7.h[4]\n"
                                    "ins v24.d[1], x2\n"
                                    "fmla  v8.8h,  v17.8h, v0.h[5]\n"
                                    "fmla  v9.8h,  v17.8h, v1.h[5]\n"
                                    "fmla v10.8h,  v17.8h, v2.h[5]\n"
                                    "ldr d16, [%[f_c6]]\n"
                                    "fmla v11.8h,  v17.8h, v3.h[5]\n"
                                    "ldr  x5, [%[f_c6], #8]\n"
                                    "fmla v12.8h,  v17.8h, v4.h[5]\n"
                                    "ins v16.d[1], x5\n"
                                    "fmla v13.8h,  v17.8h, v5.h[5]\n"
                                    "ldr d25, [%[out_o1h1], #48]\n"
                                    "fmla v14.8h,  v17.8h, v6.h[5]\n"
                                    "ldr  x3, [%[out_o1h1], #56]\n"
                                    "fmla v15.8h,  v17.8h, v7.h[5]\n"
                                    "ins v25.d[1], x3\n"
                                    "fmla  v8.8h,  v16.8h, v0.h[6]\n"
                                    "fmla  v9.8h,  v16.8h, v1.h[6]\n"
                                    "fmla v10.8h,  v16.8h, v2.h[6]\n"
                                    "ldr d17, [%[f_c7]]\n"
                                    "fmla v11.8h,  v16.8h, v3.h[6]\n"
                                    "ldr  x5, [%[f_c7], #8]\n"
                                    "fmla v12.8h,  v16.8h, v4.h[6]\n"
                                    "ins v17.d[1], x5\n"
                                    "fmla v13.8h,  v16.8h, v5.h[6]\n"
                                    "fmla v14.8h,  v16.8h, v6.h[6]\n"
                                    "fmla v15.8h,  v16.8h, v7.h[6]\n"
                                    "fmla  v8.8h,  v17.8h, v0.h[7]\n"
                                    "fmla  v9.8h,  v17.8h, v1.h[7]\n"
                                    "fmla v10.8h,  v17.8h, v2.h[7]\n"
                                    "ldr d16, [%[f_c0], #16]\n"
                                    "fmla v11.8h,  v17.8h, v3.h[7]\n"
                                    "ldr  x4, [%[f_c0], #24]\n"
                                    "fmla v12.8h,  v17.8h, v4.h[7]\n"
                                    "ins v16.d[1], x4\n"
                                    "fmla v13.8h,  v17.8h, v5.h[7]\n"
                                    "fmla v14.8h,  v17.8h, v6.h[7]\n"
                                    "fmla v15.8h,  v17.8h, v7.h[7]\n"

                                    "fmla v18.8h,  v16.8h, v0.h[0]\n"
                                    "fmla v19.8h,  v16.8h, v1.h[0]\n"
                                    "fmla v20.8h,  v16.8h, v2.h[0]\n"
                                    "ldr d17, [%[f_c1], #16]\n"
                                    "fmla v21.8h,  v16.8h, v3.h[0]\n"
                                    "ldr  x5, [%[f_c1], #24]\n"
                                    "fmla v22.8h,  v16.8h, v4.h[0]\n"
                                    "ins v17.d[1], x5\n"
                                    "fmla v23.8h,  v16.8h, v5.h[0]\n"
                                    "fmla v24.8h,  v16.8h, v6.h[0]\n"
                                    "fmla v25.8h,  v16.8h, v7.h[0]\n"
                                    "fmla v18.8h,  v17.8h, v0.h[1]\n"
                                    "fmla v19.8h,  v17.8h, v1.h[1]\n"
                                    "fmla v20.8h,  v17.8h, v2.h[1]\n"
                                    "ldr d16, [%[f_c2], #16]\n"
                                    "fmla v21.8h,  v17.8h, v3.h[1]\n"
                                    "ldr  x4, [%[f_c2], #24]\n"
                                    "fmla v22.8h,  v17.8h, v4.h[1]\n"
                                    "ins v16.d[1], x4\n"
                                    "fmla v23.8h,  v17.8h, v5.h[1]\n"
                                    "fmla v24.8h,  v17.8h, v6.h[1]\n"
                                    "fmla v25.8h,  v17.8h, v7.h[1]\n"
                                    "fmla v18.8h,  v16.8h, v0.h[2]\n"
                                    "fmla v19.8h,  v16.8h, v1.h[2]\n"
                                    "fmla v20.8h,  v16.8h, v2.h[2]\n"
                                    "ldr d17, [%[f_c3], #16]\n"
                                    "fmla v21.8h,  v16.8h, v3.h[2]\n"
                                    "ldr  x5, [%[f_c3], #24]\n"
                                    "fmla v22.8h,  v16.8h, v4.h[2]\n"
                                    "ins v17.d[1], x5\n"
                                    "fmla v23.8h,  v16.8h, v5.h[2]\n"
                                    "fmla v24.8h,  v16.8h, v6.h[2]\n"
                                    "fmla v25.8h,  v16.8h, v7.h[2]\n"
                                    "fmla v18.8h,  v17.8h, v0.h[3]\n"
                                    "fmla v19.8h,  v17.8h, v1.h[3]\n"
                                    "fmla v20.8h,  v17.8h, v2.h[3]\n"
                                    "ldr d16, [%[f_c4], #16]\n"
                                    "fmla v21.8h,  v17.8h, v3.h[3]\n"
                                    "ldr  x4, [%[f_c4], #24]\n"
                                    "fmla v22.8h,  v17.8h, v4.h[3]\n"
                                    "ins v16.d[1], x4\n"
                                    "fmla v23.8h,  v17.8h, v5.h[3]\n"
                                    "fmla v24.8h,  v17.8h, v6.h[3]\n"
                                    "fmla v25.8h,  v17.8h, v7.h[3]\n"
                                    "fmla v18.8h,  v16.8h, v0.h[4]\n"
                                    "fmla v19.8h,  v16.8h, v1.h[4]\n"
                                    "fmla v20.8h,  v16.8h, v2.h[4]\n"
                                    "ldr d17, [%[f_c5], #16]\n"
                                    "fmla v21.8h,  v16.8h, v3.h[4]\n"
                                    "ldr  x5, [%[f_c5], #24]\n"
                                    "fmla v22.8h,  v16.8h, v4.h[4]\n"
                                    "ins v17.d[1], x5\n"
                                    "fmla v23.8h,  v16.8h, v5.h[4]\n"
                                    "fmla v24.8h,  v16.8h, v6.h[4]\n"
                                    "fmla v25.8h,  v16.8h, v7.h[4]\n"
                                    "fmla v18.8h,  v17.8h, v0.h[5]\n"
                                    "fmla v19.8h,  v17.8h, v1.h[5]\n"
                                    "fmla v20.8h,  v17.8h, v2.h[5]\n"
                                    "ldr d16, [%[f_c6], #16]\n"
                                    "fmla v21.8h,  v17.8h, v3.h[5]\n"
                                    "ldr  x4, [%[f_c6], #24]\n"
                                    "fmla v22.8h,  v17.8h, v4.h[5]\n"
                                    "ins v16.d[1], x4\n"
                                    "fmla v23.8h,  v17.8h, v5.h[5]\n"
                                    "fmla v24.8h,  v17.8h, v6.h[5]\n"
                                    "fmla v25.8h,  v17.8h, v7.h[5]\n"
                                    "fmla v18.8h,  v16.8h, v0.h[6]\n"
                                    "fmla v19.8h,  v16.8h, v1.h[6]\n"
                                    "fmla v20.8h,  v16.8h, v2.h[6]\n"
                                    "ldr d17, [%[f_c7], #16]\n"
                                    "fmla v21.8h,  v16.8h, v3.h[6]\n"
                                    "ldr  x5, [%[f_c7], #24]\n"
                                    "fmla v22.8h,  v16.8h, v4.h[6]\n"
                                    "ins v17.d[1], x5\n"
                                    "fmla v23.8h,  v16.8h, v5.h[6]\n"
                                    "fmla v24.8h,  v16.8h, v6.h[6]\n"
                                    "fmla v25.8h,  v16.8h, v7.h[6]\n"
                                    "fmla v18.8h,  v17.8h, v0.h[7]\n"
                                    "fmla v19.8h,  v17.8h, v1.h[7]\n"
                                    "fmla v20.8h,  v17.8h, v2.h[7]\n"
                                    "fmla v21.8h,  v17.8h, v3.h[7]\n"
                                    "fmla v22.8h,  v17.8h, v4.h[7]\n"
                                    "fmla v23.8h,  v17.8h, v5.h[7]\n"
                                    "fmla v24.8h,  v17.8h, v6.h[7]\n"
                                    "fmla v25.8h,  v17.8h, v7.h[7]\n"
                                    "str  q8, [%[out_o0h0]]\n"
                                    "str  q9, [%[out_o0h0], #16]\n"
                                    "str q10, [%[out_o0h0], #32]\n"
                                    "str q11, [%[out_o0h0], #48]\n"
                                    "str q12, [%[out_o0h1]]\n"
                                    "str q13, [%[out_o0h1], #16]\n"
                                    "str q14, [%[out_o0h1], #32]\n"
                                    "str q15, [%[out_o0h1], #48]\n"
                                    "str q18, [%[out_o1h0]]\n"
                                    "str q19, [%[out_o1h0], #16]\n"
                                    "str q20, [%[out_o1h0], #32]\n"
                                    "str q21, [%[out_o1h0], #48]\n"
                                    "str q22, [%[out_o1h1]]\n"
                                    "str q23, [%[out_o1h1], #16]\n"
                                    "str q24, [%[out_o1h1], #32]\n"
                                    "str q25, [%[out_o1h1], #48]\n"

                                    : [out_o0h0] "+r"(out_o0h0), [out_o0h1] "+r"(out_o0h1),
                                    [out_o1h0] "+r"(out_o1h0), [out_o1h1] "+r"(out_o1h1)
                                    : [in_h0w0] "r"(in_h0w0), [in_h0w1] "r"(in_h0w1),
                                    [in_h0w2] "r"(in_h0w2), [in_h0w3] "r"(in_h0w3),
                                    [in_h1w0] "r"(in_h1w0), [in_h1w1] "r"(in_h1w1),
                                    [in_h1w2] "r"(in_h1w2), [in_h1w3] "r"(in_h1w3),
                                    [f_c0] "r"(f_c0), [f_c1] "r"(f_c1), [f_c2] "r"(f_c2),
                                    [f_c3] "r"(f_c3), [f_c4] "r"(f_c4), [f_c5] "r"(f_c5),
                                    [f_c6] "r"(f_c6), [f_c7] "r"(f_c7)
                                    : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                                    "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                                    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                                    "v25", "x0", "x1", "x2", "x3", "x4", "x5");
                                f_c0 += 16;
                                f_c1 += 16;
                                f_c2 += 16;
                                f_c3 += 16;
                                f_c4 += 16;
                                f_c5 += 16;
                                f_c6 += 16;
                                f_c7 += 16;
                                in_h0w0 += 8;
                                in_h0w1 += 8;
                                in_h0w2 += 8;
                                in_h0w3 += 8;
                                in_h1w0 += 8;
                                in_h1w1 += 8;
                                in_h1w2 += 8;
                                in_h1w3 += 8;
                            }
                            in_h0w0 += iw_pad * 8 - fw * 8;
                            in_h0w1 += iw_pad * 8 - fw * 8;
                            in_h0w2 += iw_pad * 8 - fw * 8;
                            in_h0w3 += iw_pad * 8 - fw * 8;
                            in_h1w0 += iw_pad * 8 - fw * 8;
                            in_h1w1 += iw_pad * 8 - fw * 8;
                            in_h1w2 += iw_pad * 8 - fw * 8;
                            in_h1w3 += iw_pad * 8 - fw * 8;
                        }
                        in_h0w0 = in_h0w0 + 4 * strideW * 8 - fh * iw_pad * 8;
                        in_h0w1 = in_h0w1 + 4 * strideW * 8 - fh * iw_pad * 8;
                        in_h0w2 = in_h0w2 + 4 * strideW * 8 - fh * iw_pad * 8;
                        in_h0w3 = in_h0w3 + 4 * strideW * 8 - fh * iw_pad * 8;
                        in_h1w0 = in_h1w0 + 4 * strideW * 8 - fh * iw_pad * 8;
                        in_h1w1 = in_h1w1 + 4 * strideW * 8 - fh * iw_pad * 8;
                        in_h1w2 = in_h1w2 + 4 * strideW * 8 - fh * iw_pad * 8;
                        in_h1w3 = in_h1w3 + 4 * strideW * 8 - fh * iw_pad * 8;
                        out_o0h0 += 32;
                        out_o1h0 += 32;
                        out_o0h1 += 32;
                        out_o1h1 += 32;
                    }
                    in_h0w0 += 2 * strideH * iw_pad * 8 - ow * strideW * 8;
                    in_h0w1 += 2 * strideH * iw_pad * 8 - ow * strideW * 8;
                    in_h0w2 += 2 * strideH * iw_pad * 8 - ow * strideW * 8;
                    in_h0w3 += 2 * strideH * iw_pad * 8 - ow * strideW * 8;
                    in_h1w0 += 2 * strideH * iw_pad * 8 - ow * strideW * 8;
                    in_h1w1 += 2 * strideH * iw_pad * 8 - ow * strideW * 8;
                    in_h1w2 += 2 * strideH * iw_pad * 8 - ow * strideW * 8;
                    in_h1w3 += 2 * strideH * iw_pad * 8 - ow * strideW * 8;
                    out_o0h0 += ow * 8;
                    out_o1h0 += ow * 8;
                    out_o0h1 += ow * 8;
                    out_o1h1 += ow * 8;
                }
                f0 += 8 * fh * fw * 16;
                f1 += 8 * fh * fw * 16;
                f2 += 8 * fh * fw * 16;
                f3 += 8 * fh * fw * 16;
                f4 += 8 * fh * fw * 16;
                f5 += 8 * fh * fw * 16;
                f6 += 8 * fh * fw * 16;
                f7 += 8 * fh * fw * 16;
            }
            outo0h0 += 2 * oh * ow * 8;
            outo1h0 += 2 * oh * ow * 8;
            outo0h1 += 2 * oh * ow * 8;
            outo1h1 += 2 * oh * ow * 8;
        }

        // bias
        F16 *out = outArray;
        float16x8_t v_0 = vmovq_n_f16(0);
        for (U32 o = 0; o < oc; o++) {
            float16x8_t v_b = vld1q_f16(biasArray + o * 8);
            for (U32 hw = 0; hw < oh * ow; hw++) {
                float16x8_t v = vld1q_f16(out);
                switch (activationDesc.mode) {
                    case ACTIVATION_NULL:
                        vst1q_f16(out, vaddq_f16(v, v_b));
                        break;
                    case ACTIVATION_RELU:
                        vst1q_f16(out, vmaxq_f16(vaddq_f16(v, v_b), v_0));
                        break;
                    default:
                        return NOT_SUPPORTED;
                }
                out += 8;
            }
        }
    }
    return ret;
}
