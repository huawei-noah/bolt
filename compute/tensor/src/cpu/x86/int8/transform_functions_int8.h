// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef CHEETAH_X86_TRANSFORM_FUNTIONS_INT8_H
#define CHEETAH_X86_TRANSFORM_FUNTIONS_INT8_H

#include "thread_affinity.h"

inline void PaddingNCHWC16(
    UINT8 *data, UINT8 *tmp, TensorDesc inputDesc, ConvolutionParamSpec convParamSpec)
{
    // NCHWC8
    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingB = convParamSpec.pad_bottom;
    U32 paddingL = convParamSpec.pad_left;
    U32 paddingR = convParamSpec.pad_right;

    U32 padih = paddingT + paddingB + ih;
    U32 padiw = paddingL + paddingR + iw;
    U32 simdW = 16, alignSize = 4;

    CHECK_REQUIREMENT(idf == DF_NCHWC16 && (ic % alignSize == 0));

    U32 icNum = ic / 16;
    for (U32 c = 0; c < icNum; ++c) {
        U32 coff = c * padih * padiw * simdW;
        UNI_MEMSET(tmp + coff, 128, padiw * paddingT * simdW);
        UNI_MEMSET(tmp + coff + (ih + paddingT) * padiw * simdW, 128, padiw * paddingB * simdW);
    }
    for (U32 hc = 0; hc < ih * icNum; ++hc) {
        U32 c = hc / ih;
        U32 coff = c * padih * padiw * simdW;
        U32 h = hc % ih;
        U32 hoff = (h + paddingT) * padiw;

        UNI_MEMSET(tmp + coff + hoff * simdW, 128, paddingL * simdW);
        UNI_MEMCPY(tmp + coff + (hoff + paddingL) * simdW,
            data + c * ih * iw * simdW + h * iw * simdW, iw * simdW);
        UNI_MEMSET(tmp + coff + (hoff + (paddingL + iw)) * simdW, 128, paddingR * simdW);
    }

    icNum *= 16;
    U32 resC = ic - icNum;
    while (resC > 0) {
        U32 cx = (resC == 12) ? 8 : resC;  // resC: 4, 8, 12, 16
        U32 coff = icNum * padih * padiw;
        UNI_MEMSET(tmp + coff, 128, padiw * paddingT * cx);
        UNI_MEMSET(tmp + coff + (ih + paddingT) * padiw * cx, 128, padiw * paddingB * cx);
        for (U32 h = 0; h < ih; ++h) {
            U32 hoff = (h + paddingT) * padiw;
            UNI_MEMSET(tmp + coff + hoff * cx, 128, paddingL * cx);
            UNI_MEMCPY(
                tmp + coff + (hoff + paddingL) * cx, data + icNum * ih * iw + h * iw * cx, iw * cx);
            UNI_MEMSET(tmp + coff + (hoff + (paddingL + iw)) * cx, 128, paddingR * cx);
        }
        resC -= cx;
    }
}

inline void PaddingNCHW2NCHWC16(
    UINT8 *data, UINT8 *tmp, TensorDesc inputDesc, ConvolutionParamSpec convParamSpec)
{
    // NCHWC8
    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingB = convParamSpec.pad_bottom;
    U32 paddingL = convParamSpec.pad_left;
    U32 paddingR = convParamSpec.pad_right;

    U32 padih = paddingT + paddingB + ih;
    U32 padiw = paddingL + paddingR + iw;
    U32 simdW = 16, alignSize = 4;
    U32 icPadding = (ic + 3) / 4 * 4;

    U32 icNum = ic / 16;
    for (U32 c = 0; c < icNum; ++c) {
        U32 coff = c * padih * padiw * simdW;
        UNI_MEMSET(tmp + coff, 128, padiw * paddingT * simdW);
        UNI_MEMSET(tmp + coff + (ih + paddingT) * padiw * simdW, 128, padiw * paddingB * simdW);
    }
    for (U32 hc = 0; hc < ih * icNum; ++hc) {
        U32 c = hc / ih;
        U32 coff = c * padih * padiw * simdW;
        U32 h = hc % ih;
        U32 hoff = (h + paddingT) * padiw;

        UNI_MEMSET(tmp + coff + hoff * simdW, 128, paddingL * simdW);
        for (U32 w = 0; w < iw; ++w) {
            for (U32 s = 0; s < simdW; ++s) {
                U32 iIdx = (c * simdW + s) * ih * iw + h * iw + w;
                U32 oIdx = coff + (hoff + paddingL) * simdW + w * simdW + s;
                tmp[oIdx] = data[iIdx];
            }
        }
        UNI_MEMSET(tmp + coff + (hoff + (paddingL + iw)) * simdW, 128, paddingR * simdW);
    }

    icNum *= 16;
    U32 resC = icPadding - icNum;
    while (resC > 0) {
        U32 icx = ic - icNum;
        U32 cx = (resC == 12) ? 8 : resC;  // resC: 4, 8, 12, 16
        U32 coff = icNum * padih * padiw;
        UNI_MEMSET(tmp + coff, 128, padiw * paddingT * cx);
        UNI_MEMSET(tmp + coff + (ih + paddingT) * padiw * cx, 128, padiw * paddingB * cx);
        for (U32 h = 0; h < ih; ++h) {
            U32 hoff = (h + paddingT) * padiw;
            UNI_MEMSET(tmp + coff + hoff * cx, 128, paddingL * cx);
            for (U32 w = 0; w < iw; ++w) {
                U32 woff = (hoff + paddingL) * cx + w * cx;
                for (U32 s = 0; s < icx; ++s) {
                    U32 iIdx = (icNum + s) * ih * iw + h * iw + w;
                    U32 oIdx = coff + woff + s;
                    tmp[oIdx] = data[iIdx];
                }
                UNI_MEMSET(tmp + coff + woff + icx, 128, cx - icx);
            }
            UNI_MEMSET(tmp + coff + (hoff + (paddingL + iw)) * cx, 128, paddingR * cx);
        }
        resC -= cx;
    }
}

inline void PaddingNCHWC8ToNCHWC16(
    UINT8 *data, UINT8 *tmp, TensorDesc inputDesc, ConvolutionParamSpec convParamSpec)
{
    // NCHWC8
    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingB = convParamSpec.pad_bottom;
    U32 paddingL = convParamSpec.pad_left;
    U32 paddingR = convParamSpec.pad_right;

    U32 padih = paddingT + paddingB + ih;
    U32 padiw = paddingL + paddingR + iw;
    U32 simdW = 16, alignSize = 8;

    CHECK_REQUIREMENT(idf == DF_NCHWC8);

    U32 icNum = ic / 16;
    if (paddingT != 0 || paddingB != 0) {
        for (U32 c = 0; c < icNum; ++c) {
            U32 coff = c * padih * padiw * simdW;
            UNI_MEMSET(tmp + coff, 128, padiw * paddingT * simdW);
            UNI_MEMSET(tmp + coff + (ih + paddingT) * padiw * simdW, 128, padiw * paddingB * simdW);
        }
    }
    for (U32 hc = 0; hc < ih * icNum; ++hc) {
        U32 c = hc / ih;
        U32 coff = c * padih * padiw * simdW;
        U32 h = hc % ih;
        U32 hoff = (h + paddingT) * padiw;

        UNI_MEMSET(tmp + coff + hoff * simdW, 128, paddingL * simdW);
        for (U32 w = 0; w < iw; ++w) {
            for (U32 s = 0; s < simdW; s += 8) {
                U32 iIdx = (c * simdW + s) * ih * iw + (h * iw + w) * 8;
                U32 oIdx = coff + (hoff + paddingL) * simdW + w * simdW + s;
                UNI_MEMCPY(tmp + oIdx, data + iIdx, 8);
            }
        }
        UNI_MEMSET(tmp + coff + (hoff + (paddingL + iw)) * simdW, 128, paddingR * simdW);
    }

    icNum *= 16;
    if (ic > icNum) {
        U32 cx = 8;
        U32 coff = icNum * padih * padiw;
        UNI_MEMSET(tmp + coff, 128, padiw * paddingT * cx);
        UNI_MEMSET(tmp + coff + (ih + paddingT) * padiw * cx, 128, padiw * paddingB * cx);
        for (U32 h = 0; h < ih; ++h) {
            U32 hoff = (h + paddingT) * padiw;
            UNI_MEMSET(tmp + coff + hoff * cx, 128, paddingL * cx);
            for (U32 w = 0; w < iw; ++w) {
                U32 iIdx = icNum * ih * iw + (h * iw + w) * 8;
                U32 oIdx = coff + (hoff + paddingL) * cx + w * cx;
                UNI_MEMCPY(tmp + oIdx, data + iIdx, 8);
            }
            UNI_MEMSET(tmp + coff + (hoff + (paddingL + iw)) * cx, 128, paddingR * cx);
        }
    }
}

#endif
