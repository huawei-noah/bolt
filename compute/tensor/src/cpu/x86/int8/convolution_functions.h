// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

struct ConvController {
    UINT8 *input;
    const INT8 *filter;
    void *output;
    F32 *eltwise;
    UINT8 *u8Output;
    const I32 *bias;
    I64 ic;
    I64 kw;
    I64 kh;
    I64 *stepC16;
    I64 dilateW;
    I64 dilateH;
    I64 ostepC16;
    I64 flags;
    I64 fStep;
    I64 f8Step;
    I64 f4Step;
    I64 stride;
    void *scale;
    bool cross;
    I64 idx;
    I64 off;
    void *max;
};

typedef void (*kernelFunc)(ConvController &c);

// clang-format off
#define clear1Regs(rtype) \
    "vxorps "#rtype"0, "#rtype"0, "#rtype"0    \n\t"

#define clear2Regs(rtype) \
    clear1Regs(rtype) \
    "vxorps "#rtype"1, "#rtype"1, "#rtype"1    \n\t"

#define clear3Regs(rtype) \
    clear2Regs(rtype) \
    "vxorps "#rtype"2, "#rtype"2, "#rtype"2    \n\t"

#define clear4Regs(rtype) \
    clear3Regs(rtype) \
    "vxorps "#rtype"3, "#rtype"3, "#rtype"3    \n\t"

#define clear5Regs(rtype) \
    clear4Regs(rtype) \
    "vxorps "#rtype"4, "#rtype"4, "#rtype"4    \n\t" \

#define clear6Regs(rtype) \
    clear5Regs(rtype) \
    "vxorps "#rtype"5, "#rtype"5, "#rtype"5    \n\t"

#define clear7Regs(rtype) \
    clear6Regs(rtype) \
    "vxorps "#rtype"6, "#rtype"6, "#rtype"6    \n\t" \

#define clear8Regs(rtype) \
    clear7Regs(rtype) \
    "vxorps "#rtype"7, "#rtype"7, "#rtype"7    \n\t"

#define clear9Regs(rtype) \
    clear8Regs(rtype) \
    "vxorps "#rtype"8, "#rtype"8, "#rtype"8    \n\t"

#define clear10Regs(rtype) \
    clear9Regs(rtype) \
    "vxorps "#rtype"9, "#rtype"9, "#rtype"9    \n\t"

#define clear11Regs(rtype) \
    clear10Regs(rtype) \
    "vxorps "#rtype"10, "#rtype"10, "#rtype"10 \n\t" \

#define clear12Regs(rtype) \
    clear11Regs(rtype) \
    "vxorps "#rtype"11, "#rtype"11, "#rtype"11 \n\t"

#define clear13Regs(rtype) \
    clear12Regs(rtype) \
    "vxorps "#rtype"12, "#rtype"12, "#rtype"12 \n\t" \

#define clear14Regs(rtype) \
    clear13Regs(rtype) \
    "vxorps "#rtype"13, "#rtype"13, "#rtype"13 \n\t"

#define clear15Regs(rtype) \
    clear14Regs(rtype) \
    "vxorps "#rtype"14, "#rtype"14, "#rtype"14 \n\t" \

#define clear16Regs(rtype) \
    clear15Regs(rtype) \
    "vxorps "#rtype"15, "#rtype"15, "#rtype"15 \n\t"

#define clear17Regs(rtype) \
    clear16Regs(rtype) \
    "vxorps "#rtype"16, "#rtype"16, "#rtype"16 \n\t" \

#define clear18Regs(rtype) \
    clear17Regs(rtype) \
    "vxorps "#rtype"17, "#rtype"17, "#rtype"17 \n\t"

#define clear19Regs(rtype) \
    clear18Regs(rtype) \
    "vxorps "#rtype"18, "#rtype"18, "#rtype"18 \n\t" \

#define clear20Regs(rtype) \
    clear19Regs(rtype) \
    "vxorps "#rtype"19, "#rtype"19, "#rtype"19 \n\t"

#define clear21Regs(rtype) \
    clear20Regs(rtype) \
    "vxorps "#rtype"20, "#rtype"20, "#rtype"20 \n\t" \

#define clear22Regs(rtype) \
    clear21Regs(rtype) \
    "vxorps "#rtype"21, "#rtype"21, "#rtype"21 \n\t"

#define clear23Regs(rtype) \
    clear22Regs(rtype) \
    "vxorps "#rtype"22, "#rtype"22, "#rtype"22 \n\t" \

#define clear24Regs(rtype) \
    clear23Regs(rtype) \
    "vxorps "#rtype"23, "#rtype"23, "#rtype"23 \n\t"

#define relu1Regs(rtype, pxor, max) \
    ""#pxor" "#rtype"15, "#rtype"15, "#rtype"15                  \n\t" \
    ""#max" "#rtype"15, "#rtype"0, "#rtype"0                    \n\t"

#define relu2Regs(rtype, pxor, max) \
    relu1Regs(rtype, pxor, max) \
    ""#max" "#rtype"15, "#rtype"1, "#rtype"1                    \n\t"

#define relu3Regs(rtype, pxor, max) \
    relu2Regs(rtype, pxor, max) \
    ""#max" "#rtype"15, "#rtype"2, "#rtype"2                    \n\t"

#define relu4Regs(rtype, pxor, max) \
    relu3Regs(rtype, pxor, max) \
    ""#max" "#rtype"15, "#rtype"3, "#rtype"3                    \n\t" \

#define relu5Regs(rtype, pxor, max) \
    relu4Regs(rtype, pxor, max) \
    ""#max" "#rtype"15, "#rtype"4, "#rtype"4                    \n\t" \

#define relu6Regs(rtype, pxor, max) \
    relu5Regs(rtype, pxor, max) \
    ""#max" "#rtype"15, "#rtype"5, "#rtype"5                    \n\t"

#define relu7Regs(rtype, pxor, max) \
    relu6Regs(rtype, pxor, max) \
    ""#max" "#rtype"15, "#rtype"6, "#rtype"6                    \n\t" \

#define relu8Regs(rtype, pxor, max) \
    relu7Regs(rtype, pxor, max) \
    ""#max" "#rtype"15, "#rtype"7, "#rtype"7                    \n\t"

#define relu9Regs(rtype, pxor, max) \
    relu8Regs(rtype, pxor, max) \
    ""#max" "#rtype"15, "#rtype"8, "#rtype"8                    \n\t" \

#define relu10Regs(rtype, pxor, max) \
    relu9Regs(rtype, pxor, max) \
    ""#max" "#rtype"15, "#rtype"9, "#rtype"9                    \n\t"

#define relu11Regs(rtype, pxor, max) \
    relu10Regs(rtype, pxor, max) \
    ""#max" "#rtype"15, "#rtype"10, "#rtype"10                    \n\t" \

#define relu12Regs(rtype, pxor, max) \
    relu11Regs(rtype, pxor, max) \
    ""#max" "#rtype"15, "#rtype"11, "#rtype"11                    \n\t"

#define relu13Regs(rtype, pxor, max) \
    relu12Regs(rtype, pxor, max) \
    ""#max" "#rtype"15, "#rtype"12, "#rtype"12                    \n\t" \

#define relu14Regs(rtype, pxor, max) \
    relu13Regs(rtype, pxor, max) \
    ""#max" "#rtype"15, "#rtype"13, "#rtype"13                    \n\t"

#define relu15Regs(rtype, pxor, max) \
    relu14Regs(rtype, pxor, max) \
    ""#max" "#rtype"15, "#rtype"14, "#rtype"14                    \n\t" \

#define relu16Regs(rtype, pxor, max) \
    ""#pxor" "#rtype"31, "#rtype"31, "#rtype"31                  \n\t" \
    ""#max" "#rtype"31, "#rtype"0, "#rtype"0                    \n\t"   \
    ""#max" "#rtype"31, "#rtype"1, "#rtype"1                    \n\t" \
    ""#max" "#rtype"31, "#rtype"2, "#rtype"2                    \n\t" \
    ""#max" "#rtype"31, "#rtype"3, "#rtype"3                    \n\t" \
    ""#max" "#rtype"31, "#rtype"4, "#rtype"4                    \n\t" \
    ""#max" "#rtype"31, "#rtype"5, "#rtype"5                    \n\t" \
    ""#max" "#rtype"31, "#rtype"6, "#rtype"6                    \n\t" \
    ""#max" "#rtype"31, "#rtype"7, "#rtype"7                    \n\t" \
    ""#max" "#rtype"31, "#rtype"8, "#rtype"8                    \n\t" \
    ""#max" "#rtype"31, "#rtype"9, "#rtype"9                    \n\t" \
    ""#max" "#rtype"31, "#rtype"10, "#rtype"10                    \n\t" \
    ""#max" "#rtype"31, "#rtype"11, "#rtype"11                    \n\t" \
    ""#max" "#rtype"31, "#rtype"12, "#rtype"12                    \n\t" \
    ""#max" "#rtype"31, "#rtype"13, "#rtype"13                    \n\t" \
    ""#max" "#rtype"31, "#rtype"14, "#rtype"14                    \n\t" \
    ""#max" "#rtype"31, "#rtype"15, "#rtype"15                    \n\t"

#define relu17Regs(rtype, pxor, max) \
    relu16Regs(rtype, pxor, max) \
    ""#max" "#rtype"31, "#rtype"16, "#rtype"16                    \n\t" \

#define relu18Regs(rtype, pxor, max) \
    relu17Regs(rtype, pxor, max) \
    ""#max" "#rtype"31, "#rtype"17, "#rtype"17                    \n\t"

#define relu19Regs(rtype, pxor, max) \
    relu18Regs(rtype, pxor, max) \
    ""#max" "#rtype"31, "#rtype"18, "#rtype"18                    \n\t" \

#define relu20Regs(rtype, pxor, max) \
    relu19Regs(rtype, pxor, max) \
    ""#max" "#rtype"31, "#rtype"19, "#rtype"19                    \n\t"

#define relu21Regs(rtype, pxor, max) \
    relu20Regs(rtype, pxor, max) \
    ""#max" "#rtype"31, "#rtype"20, "#rtype"20                    \n\t" \

#define relu22Regs(rtype, pxor, max) \
    relu21Regs(rtype, pxor, max) \
    ""#max" "#rtype"31, "#rtype"21, "#rtype"21                    \n\t"

#define relu23Regs(rtype, pxor, max) \
    relu22Regs(rtype, pxor, max) \
    ""#max" "#rtype"31, "#rtype"22, "#rtype"22                    \n\t" \

#define relu24Regs(rtype, pxor, max) \
    relu23Regs(rtype, pxor, max) \
    ""#max" "#rtype"31, "#rtype"23, "#rtype"23                    \n\t"

#define convert1RegsI32ToF32(scalePtr, rtype) \
    "vbroadcastss ("#scalePtr"), "#rtype"15               \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                       \n\t" \
    "vmulps "#rtype"0, "#rtype"15, "#rtype"0              \n\t" \

#define convert2RegsI32ToF32(scalePtr, rtype) \
    "vbroadcastss ("#scalePtr"), "#rtype"15               \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                       \n\t" \
    "vcvtdq2ps "#rtype"1, "#rtype"1                       \n\t" \
    "vmulps "#rtype"0, "#rtype"15, "#rtype"0              \n\t" \
    "vmulps "#rtype"1, "#rtype"15, "#rtype"1              \n\t" \

#define convert3RegsI32ToF32(scalePtr, rtype) \
    "vbroadcastss ("#scalePtr"), "#rtype"15               \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                       \n\t" \
    "vcvtdq2ps "#rtype"1, "#rtype"1                       \n\t" \
    "vcvtdq2ps "#rtype"2, "#rtype"2                       \n\t" \
    "vmulps "#rtype"0, "#rtype"15, "#rtype"0              \n\t" \
    "vmulps "#rtype"1, "#rtype"15, "#rtype"1              \n\t" \
    "vmulps "#rtype"2, "#rtype"15, "#rtype"2              \n\t" \

#define convert4RegsI32ToF32(scalePtr, rtype) \
    "vbroadcastss ("#scalePtr"), "#rtype"15               \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                       \n\t" \
    "vcvtdq2ps "#rtype"1, "#rtype"1                       \n\t" \
    "vcvtdq2ps "#rtype"2, "#rtype"2                       \n\t" \
    "vcvtdq2ps "#rtype"3, "#rtype"3                       \n\t" \
    "vmulps "#rtype"0, "#rtype"15, "#rtype"0              \n\t" \
    "vmulps "#rtype"1, "#rtype"15, "#rtype"1              \n\t" \
    "vmulps "#rtype"2, "#rtype"15, "#rtype"2              \n\t" \
    "vmulps "#rtype"3, "#rtype"15, "#rtype"3              \n\t" \

#define convert5RegsI32ToF32(scalePtr, rtype) \
    convert3RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"3, "#rtype"3                       \n\t" \
    "vcvtdq2ps "#rtype"4, "#rtype"4                       \n\t" \
    "vmulps "#rtype"3, "#rtype"15, "#rtype"3              \n\t" \
    "vmulps "#rtype"4, "#rtype"15, "#rtype"4              \n\t" \

#define convert6RegsI32ToF32(scalePtr, rtype) \
    convert3RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"3, "#rtype"3                       \n\t" \
    "vcvtdq2ps "#rtype"4, "#rtype"4                       \n\t" \
    "vcvtdq2ps "#rtype"5, "#rtype"5                       \n\t" \
    "vmulps "#rtype"3, "#rtype"15, "#rtype"3              \n\t" \
    "vmulps "#rtype"4, "#rtype"15, "#rtype"4              \n\t" \
    "vmulps "#rtype"5, "#rtype"15, "#rtype"5              \n\t" \

#define convert7RegsI32ToF32(scalePtr, rtype) \
    convert4RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"4, "#rtype"4                       \n\t" \
    "vcvtdq2ps "#rtype"5, "#rtype"5                       \n\t" \
    "vcvtdq2ps "#rtype"6, "#rtype"6                       \n\t" \
    "vmulps "#rtype"4, "#rtype"15, "#rtype"4              \n\t" \
    "vmulps "#rtype"5, "#rtype"15, "#rtype"5              \n\t" \
    "vmulps "#rtype"6, "#rtype"15, "#rtype"6              \n\t" \

#define convert8RegsI32ToF32(scalePtr, rtype) \
    convert4RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"4, "#rtype"4                       \n\t" \
    "vcvtdq2ps "#rtype"5, "#rtype"5                       \n\t" \
    "vcvtdq2ps "#rtype"6, "#rtype"6                       \n\t" \
    "vcvtdq2ps "#rtype"7, "#rtype"7                       \n\t" \
    "vmulps "#rtype"4, "#rtype"15, "#rtype"4              \n\t" \
    "vmulps "#rtype"5, "#rtype"15, "#rtype"5              \n\t" \
    "vmulps "#rtype"6, "#rtype"15, "#rtype"6              \n\t" \
    "vmulps "#rtype"7, "#rtype"15, "#rtype"7              \n\t" \

#define convert9RegsI32ToF32(scalePtr, rtype) \
    convert5RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"5, "#rtype"5                       \n\t" \
    "vcvtdq2ps "#rtype"6, "#rtype"6                       \n\t" \
    "vcvtdq2ps "#rtype"7, "#rtype"7                       \n\t" \
    "vcvtdq2ps "#rtype"8, "#rtype"8                       \n\t" \
    "vmulps "#rtype"5, "#rtype"15, "#rtype"5              \n\t" \
    "vmulps "#rtype"6, "#rtype"15, "#rtype"6              \n\t" \
    "vmulps "#rtype"7, "#rtype"15, "#rtype"7              \n\t" \
    "vmulps "#rtype"8, "#rtype"15, "#rtype"8              \n\t" \

#define convert10RegsI32ToF32(scalePtr, rtype) \
    convert6RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"6, "#rtype"6                       \n\t" \
    "vcvtdq2ps "#rtype"7, "#rtype"7                       \n\t" \
    "vcvtdq2ps "#rtype"8, "#rtype"8                       \n\t" \
    "vcvtdq2ps "#rtype"9, "#rtype"9                       \n\t" \
    "vmulps "#rtype"6, "#rtype"15, "#rtype"6              \n\t" \
    "vmulps "#rtype"7, "#rtype"15, "#rtype"7              \n\t" \
    "vmulps "#rtype"8, "#rtype"15, "#rtype"8              \n\t" \
    "vmulps "#rtype"9, "#rtype"15, "#rtype"9              \n\t" \

#define convert11RegsI32ToF32(scalePtr, rtype) \
    convert8RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"8, "#rtype"8                       \n\t" \
    "vcvtdq2ps "#rtype"9, "#rtype"9                       \n\t" \
    "vcvtdq2ps "#rtype"10, "#rtype"10                     \n\t" \
    "vmulps "#rtype"8,  "#rtype"15, "#rtype"8              \n\t" \
    "vmulps "#rtype"9,  "#rtype"15, "#rtype"9              \n\t" \
    "vmulps "#rtype"10, "#rtype"15, "#rtype"10            \n\t" \

#define convert12RegsI32ToF32(scalePtr, rtype) \
    convert8RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"8, "#rtype"8                       \n\t" \
    "vcvtdq2ps "#rtype"9, "#rtype"9                       \n\t" \
    "vcvtdq2ps "#rtype"10, "#rtype"10                     \n\t" \
    "vcvtdq2ps "#rtype"11, "#rtype"11                     \n\t" \
    "vmulps "#rtype"8,  "#rtype"15, "#rtype"8              \n\t" \
    "vmulps "#rtype"9,  "#rtype"15, "#rtype"9              \n\t" \
    "vmulps "#rtype"10, "#rtype"15, "#rtype"10            \n\t" \
    "vmulps "#rtype"11, "#rtype"15, "#rtype"11            \n\t" \

#define convert13RegsI32ToF32(scalePtr, rtype) \
    convert9RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"9, "#rtype"9                       \n\t" \
    "vcvtdq2ps "#rtype"10, "#rtype"10                     \n\t" \
    "vcvtdq2ps "#rtype"11, "#rtype"11                     \n\t" \
    "vcvtdq2ps "#rtype"12, "#rtype"12                     \n\t" \
    "vmulps "#rtype"9,  "#rtype"15, "#rtype"9              \n\t" \
    "vmulps "#rtype"10, "#rtype"15, "#rtype"10            \n\t" \
    "vmulps "#rtype"11, "#rtype"15, "#rtype"11            \n\t" \
    "vmulps "#rtype"12, "#rtype"15, "#rtype"12            \n\t" \

#define convert14RegsI32ToF32(scalePtr, rtype) \
    convert10RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"10, "#rtype"10                     \n\t" \
    "vcvtdq2ps "#rtype"11, "#rtype"11                     \n\t" \
    "vcvtdq2ps "#rtype"12, "#rtype"12                     \n\t" \
    "vcvtdq2ps "#rtype"13, "#rtype"13                     \n\t" \
    "vmulps "#rtype"10, "#rtype"15, "#rtype"10            \n\t" \
    "vmulps "#rtype"11, "#rtype"15, "#rtype"11            \n\t" \
    "vmulps "#rtype"12, "#rtype"15, "#rtype"12            \n\t" \
    "vmulps "#rtype"13, "#rtype"15, "#rtype"13            \n\t" \

#define convert15RegsI32ToF32(scalePtr, rtype) \
    convert11RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"11, "#rtype"11                     \n\t" \
    "vcvtdq2ps "#rtype"12, "#rtype"12                     \n\t" \
    "vcvtdq2ps "#rtype"13, "#rtype"13                     \n\t" \
    "vcvtdq2ps "#rtype"14, "#rtype"14                     \n\t" \
    "vmulps "#rtype"11, "#rtype"24, "#rtype"11            \n\t" \
    "vmulps "#rtype"12, "#rtype"24, "#rtype"12            \n\t" \
    "vmulps "#rtype"13, "#rtype"24, "#rtype"13            \n\t" \
    "vmulps "#rtype"14, "#rtype"24, "#rtype"14            \n\t" \

#define convert16RegsI32ToF32(scalePtr, rtype) \
    "vbroadcastss ("#scalePtr"), "#rtype"24               \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                       \n\t" \
    "vcvtdq2ps "#rtype"1, "#rtype"1                       \n\t" \
    "vcvtdq2ps "#rtype"2, "#rtype"2                       \n\t" \
    "vcvtdq2ps "#rtype"3, "#rtype"3                       \n\t" \
    "vcvtdq2ps "#rtype"4, "#rtype"4                       \n\t" \
    "vcvtdq2ps "#rtype"5, "#rtype"5                       \n\t" \
    "vcvtdq2ps "#rtype"6, "#rtype"6                       \n\t" \
    "vcvtdq2ps "#rtype"7, "#rtype"7                       \n\t" \
    "vcvtdq2ps "#rtype"8, "#rtype"8                       \n\t" \
    "vcvtdq2ps "#rtype"9, "#rtype"9                       \n\t" \
    "vcvtdq2ps "#rtype"10, "#rtype"10                     \n\t" \
    "vcvtdq2ps "#rtype"11, "#rtype"11                     \n\t" \
    "vcvtdq2ps "#rtype"12, "#rtype"12                     \n\t" \
    "vcvtdq2ps "#rtype"13, "#rtype"13                     \n\t" \
    "vcvtdq2ps "#rtype"14, "#rtype"14                     \n\t" \
    "vcvtdq2ps "#rtype"15, "#rtype"15                     \n\t" \
    "vmulps "#rtype"0, "#rtype"24, "#rtype"0              \n\t" \
    "vmulps "#rtype"1, "#rtype"24, "#rtype"1              \n\t" \
    "vmulps "#rtype"2, "#rtype"24, "#rtype"2              \n\t" \
    "vmulps "#rtype"3, "#rtype"24, "#rtype"3              \n\t" \
    "vmulps "#rtype"4, "#rtype"24, "#rtype"4              \n\t" \
    "vmulps "#rtype"5, "#rtype"24, "#rtype"5              \n\t" \
    "vmulps "#rtype"6, "#rtype"24, "#rtype"6              \n\t" \
    "vmulps "#rtype"7, "#rtype"24, "#rtype"7              \n\t" \
    "vmulps "#rtype"8, "#rtype"24, "#rtype"8              \n\t" \
    "vmulps "#rtype"9, "#rtype"24, "#rtype"9              \n\t" \
    "vmulps "#rtype"10, "#rtype"24, "#rtype"10            \n\t" \
    "vmulps "#rtype"11, "#rtype"24, "#rtype"11            \n\t" \
    "vmulps "#rtype"12, "#rtype"24, "#rtype"12            \n\t" \
    "vmulps "#rtype"13, "#rtype"24, "#rtype"13            \n\t" \
    "vmulps "#rtype"14, "#rtype"24, "#rtype"14            \n\t" \
    "vmulps "#rtype"15, "#rtype"24, "#rtype"15            \n\t" \

#define convert17RegsI32ToF32(scalePtr, rtype) \
    "vbroadcastss ("#scalePtr"), "#rtype"24               \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                       \n\t" \
    "vcvtdq2ps "#rtype"1, "#rtype"1                       \n\t" \
    "vcvtdq2ps "#rtype"2, "#rtype"2                       \n\t" \
    "vcvtdq2ps "#rtype"3, "#rtype"3                       \n\t" \
    "vcvtdq2ps "#rtype"4, "#rtype"4                       \n\t" \
    "vcvtdq2ps "#rtype"5, "#rtype"5                       \n\t" \
    "vcvtdq2ps "#rtype"6, "#rtype"6                       \n\t" \
    "vcvtdq2ps "#rtype"7, "#rtype"7                       \n\t" \
    "vcvtdq2ps "#rtype"8, "#rtype"8                       \n\t" \
    "vcvtdq2ps "#rtype"9, "#rtype"9                       \n\t" \
    "vcvtdq2ps "#rtype"10, "#rtype"10                     \n\t" \
    "vcvtdq2ps "#rtype"11, "#rtype"11                     \n\t" \
    "vcvtdq2ps "#rtype"12, "#rtype"12                     \n\t" \
    "vcvtdq2ps "#rtype"13, "#rtype"13                     \n\t" \
    "vcvtdq2ps "#rtype"14, "#rtype"14                     \n\t" \
    "vcvtdq2ps "#rtype"15, "#rtype"15                     \n\t" \
    "vcvtdq2ps "#rtype"16, "#rtype"16                     \n\t" \
    "vmulps "#rtype"0, "#rtype"24, "#rtype"0              \n\t" \
    "vmulps "#rtype"1, "#rtype"24, "#rtype"1              \n\t" \
    "vmulps "#rtype"2, "#rtype"24, "#rtype"2              \n\t" \
    "vmulps "#rtype"3, "#rtype"24, "#rtype"3              \n\t" \
    "vmulps "#rtype"4, "#rtype"24, "#rtype"4              \n\t" \
    "vmulps "#rtype"5, "#rtype"24, "#rtype"5              \n\t" \
    "vmulps "#rtype"6, "#rtype"24, "#rtype"6              \n\t" \
    "vmulps "#rtype"7, "#rtype"24, "#rtype"7              \n\t" \
    "vmulps "#rtype"8, "#rtype"24, "#rtype"8              \n\t" \
    "vmulps "#rtype"9, "#rtype"24, "#rtype"9              \n\t" \
    "vmulps "#rtype"10, "#rtype"24, "#rtype"10            \n\t" \
    "vmulps "#rtype"11, "#rtype"24, "#rtype"11            \n\t" \
    "vmulps "#rtype"12, "#rtype"24, "#rtype"12            \n\t" \
    "vmulps "#rtype"13, "#rtype"24, "#rtype"13            \n\t" \
    "vmulps "#rtype"14, "#rtype"24, "#rtype"14            \n\t" \
    "vmulps "#rtype"15, "#rtype"24, "#rtype"15            \n\t" \
    "vmulps "#rtype"16, "#rtype"24, "#rtype"16            \n\t" \

#define convert18RegsI32ToF32(scalePtr, rtype) \
    convert16RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"16, "#rtype"16                     \n\t" \
    "vcvtdq2ps "#rtype"17, "#rtype"17                     \n\t" \
    "vmulps "#rtype"16, "#rtype"24, "#rtype"16            \n\t" \
    "vmulps "#rtype"17, "#rtype"24, "#rtype"17            \n\t" \

#define convert19RegsI32ToF32(scalePtr, rtype) \
    convert16RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"16, "#rtype"16                     \n\t" \
    "vcvtdq2ps "#rtype"17, "#rtype"17                     \n\t" \
    "vcvtdq2ps "#rtype"18, "#rtype"18                     \n\t" \
    "vmulps "#rtype"16, "#rtype"24, "#rtype"16            \n\t" \
    "vmulps "#rtype"17, "#rtype"24, "#rtype"17            \n\t" \
    "vmulps "#rtype"18, "#rtype"24, "#rtype"18            \n\t" \

#define convert20RegsI32ToF32(scalePtr, rtype) \
    convert16RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"16, "#rtype"16                     \n\t" \
    "vcvtdq2ps "#rtype"17, "#rtype"17                     \n\t" \
    "vcvtdq2ps "#rtype"18, "#rtype"18                     \n\t" \
    "vcvtdq2ps "#rtype"19, "#rtype"19                     \n\t" \
    "vmulps "#rtype"16, "#rtype"24, "#rtype"16            \n\t" \
    "vmulps "#rtype"17, "#rtype"24, "#rtype"17            \n\t" \
    "vmulps "#rtype"18, "#rtype"24, "#rtype"18            \n\t" \
    "vmulps "#rtype"19, "#rtype"24, "#rtype"19            \n\t" \

#define convert21RegsI32ToF32(scalePtr, rtype) \
    convert17RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"17, "#rtype"17                     \n\t" \
    "vcvtdq2ps "#rtype"18, "#rtype"18                     \n\t" \
    "vcvtdq2ps "#rtype"19, "#rtype"19                     \n\t" \
    "vcvtdq2ps "#rtype"20, "#rtype"20                     \n\t" \
    "vmulps "#rtype"17, "#rtype"24, "#rtype"17            \n\t" \
    "vmulps "#rtype"18, "#rtype"24, "#rtype"18            \n\t" \
    "vmulps "#rtype"19, "#rtype"24, "#rtype"19            \n\t" \
    "vmulps "#rtype"20, "#rtype"24, "#rtype"20            \n\t" \

#define convert22RegsI32ToF32(scalePtr, rtype) \
    convert18RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"18, "#rtype"18                     \n\t" \
    "vcvtdq2ps "#rtype"19, "#rtype"19                     \n\t" \
    "vcvtdq2ps "#rtype"20, "#rtype"20                     \n\t" \
    "vcvtdq2ps "#rtype"21, "#rtype"21                     \n\t" \
    "vmulps "#rtype"18, "#rtype"24, "#rtype"18            \n\t" \
    "vmulps "#rtype"19, "#rtype"24, "#rtype"19            \n\t" \
    "vmulps "#rtype"20, "#rtype"24, "#rtype"20            \n\t" \
    "vmulps "#rtype"21, "#rtype"24, "#rtype"21            \n\t" \

#define convert23RegsI32ToF32(scalePtr, rtype) \
    convert19RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"19, "#rtype"19                     \n\t" \
    "vcvtdq2ps "#rtype"20, "#rtype"20                     \n\t" \
    "vcvtdq2ps "#rtype"21, "#rtype"21                     \n\t" \
    "vcvtdq2ps "#rtype"22, "#rtype"22                     \n\t" \
    "vmulps "#rtype"19, "#rtype"24, "#rtype"19            \n\t" \
    "vmulps "#rtype"20, "#rtype"24, "#rtype"20            \n\t" \
    "vmulps "#rtype"21, "#rtype"24, "#rtype"21            \n\t" \
    "vmulps "#rtype"22, "#rtype"24, "#rtype"22            \n\t" \

#define convert24RegsI32ToF32(scalePtr, rtype) \
    convert20RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"20, "#rtype"20                     \n\t" \
    "vcvtdq2ps "#rtype"21, "#rtype"21                     \n\t" \
    "vcvtdq2ps "#rtype"22, "#rtype"22                     \n\t" \
    "vcvtdq2ps "#rtype"23, "#rtype"23                     \n\t" \
    "vmulps "#rtype"20, "#rtype"24, "#rtype"20            \n\t" \
    "vmulps "#rtype"21, "#rtype"24, "#rtype"21            \n\t" \
    "vmulps "#rtype"22, "#rtype"24, "#rtype"22            \n\t" \
    "vmulps "#rtype"23, "#rtype"24, "#rtype"23            \n\t" \

#define load4BiasTo4ymmRegs(bias) \
    "vmovups ("#bias"), %%ymm0                       \n\t" \
    "vmovups 0x40("#bias"), %%ymm1                   \n\t" \
    "vmovups 0x80("#bias"), %%ymm2                   \n\t" \
    "vmovups 0xA0("#bias"), %%ymm3                   \n\t" \

#define load4BiasTo8Regs(bias, rtype) \
    load4BiasTo4##rtype##Regs(bias) \
    "vmovups %%"#rtype"0, %%"#rtype"4                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"5                   \n\t" \
    "vmovups %%"#rtype"2, %%"#rtype"6                   \n\t" \
    "vmovups %%"#rtype"3, %%"#rtype"7                   \n\t" \

#define load4BiasTo12Regs(bias, rtype) \
    load4BiasTo8Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"8                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"9                   \n\t" \
    "vmovups %%"#rtype"2, %%"#rtype"10                   \n\t" \
    "vmovups %%"#rtype"3, %%"#rtype"11                   \n\t" \

#define load3BiasTo3zmmRegs(bias) \
    "vmovups ("#bias"), %%zmm0                       \n\t" \
    "vmovups 0x40("#bias"), %%zmm1                   \n\t" \
    "vmovups 0x80("#bias"), %%zmm2                   \n\t" \

#define load3BiasTo3ymmRegs(bias) \
    "vmovups ("#bias"), %%ymm0                       \n\t" \
    "vmovups 0x20("#bias"), %%ymm1                   \n\t" \
    "vmovups 0x40("#bias"), %%ymm2                   \n\t" \

#define load3BiasTo3Regs(bias, rtype) \
    load3BiasTo3##rtype##Regs(bias)

#define load3BiasTo6Regs(bias, rtype) \
    load3BiasTo3Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"3                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"4                   \n\t" \
    "vmovups %%"#rtype"2, %%"#rtype"5                   \n\t" \

#define load3BiasTo9Regs(bias, rtype) \
    load3BiasTo6Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"6                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"7                   \n\t" \
    "vmovups %%"#rtype"2, %%"#rtype"8                   \n\t" \

#define load3BiasTo12Regs(bias, rtype) \
    load3BiasTo9Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"9                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"10                   \n\t" \
    "vmovups %%"#rtype"2, %%"#rtype"11                   \n\t"

#define load3BiasTo15Regs(bias, rtype) \
    load3BiasTo12Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"12                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"13                   \n\t" \
    "vmovups %%"#rtype"2, %%"#rtype"14                   \n\t" \

#define load3BiasTo18Regs(bias, rtype) \
    load3BiasTo15Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"15                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"16                   \n\t" \
    "vmovups %%"#rtype"2, %%"#rtype"17                   \n\t" \

#define load3BiasTo21Regs(bias, rtype) \
    load3BiasTo18Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"18                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"19                   \n\t" \
    "vmovups %%"#rtype"2, %%"#rtype"20                   \n\t" \

#define load3BiasTo24Regs(bias, rtype) \
    load3BiasTo21Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"21                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"22                   \n\t" \
    "vmovups %%"#rtype"2, %%"#rtype"23                   \n\t"

#define load2BiasTo2zmmRegs(bias) \
    "vmovups ("#bias"), %%zmm0                       \n\t" \
    "vmovups 0x40("#bias"), %%zmm1                   \n\t" \

#define load2BiasTo2ymmRegs(bias) \
    "vmovups ("#bias"), %%ymm0                       \n\t" \
    "vmovups 0x20("#bias"), %%ymm1                   \n\t" \

#define load2BiasTo2Regs(bias, rtype) \
    load2BiasTo2##rtype##Regs(bias)

#define load2BiasTo4Regs(bias, rtype) \
    load2BiasTo2Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"2                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"3                   \n\t" \

#define load2BiasTo6Regs(bias, rtype) \
    load2BiasTo4Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"4                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"5                   \n\t" \

#define load2BiasTo8Regs(bias, rtype) \
    load2BiasTo6Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"6                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"7                   \n\t" \

#define load2BiasTo10Regs(bias, rtype) \
    load2BiasTo8Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"8                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"9                   \n\t" \

#define load2BiasTo12Regs(bias, rtype) \
    load2BiasTo10Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"10                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"11                   \n\t"

#define load2BiasTo14Regs(bias, rtype) \
    load2BiasTo12Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"12                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"13                   \n\t"

#define load2BiasTo16Regs(bias, rtype) \
    load2BiasTo14Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"14                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"15                   \n\t"

#define load2BiasTo18Regs(bias, rtype) \
    load2BiasTo16Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"16                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"17                   \n\t"

#define load2BiasTo20Regs(bias, rtype) \
    load2BiasTo18Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"18                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"19                   \n\t"

#define load2BiasTo22Regs(bias, rtype) \
    load2BiasTo20Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"20                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"21                   \n\t"

#define load2BiasTo24Regs(bias, rtype) \
    load2BiasTo22Regs(bias, rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"22                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"23                   \n\t"

#define load1BiasTo1Regs(bias, rtype) \
    "vmovups ("#bias"), "#rtype"0                       \n\t"

#define load1BiasTo2Regs(bias, rtype) \
    load1BiasTo1Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"1                   \n\t" \

#define load1BiasTo3Regs(bias, rtype) \
    load1BiasTo2Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"2                   \n\t" \

#define load1BiasTo4Regs(bias, rtype) \
    load1BiasTo3Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"3                   \n\t" \

#define load1BiasTo5Regs(bias, rtype) \
    load1BiasTo4Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"4                   \n\t" \

#define load1BiasTo6Regs(bias, rtype) \
    load1BiasTo5Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"5                   \n\t" \

#define load1BiasTo7Regs(bias, rtype) \
    load1BiasTo6Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"6                   \n\t" \

#define load1BiasTo8Regs(bias, rtype) \
    load1BiasTo7Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"7                   \n\t" \

#define load1BiasTo9Regs(bias, rtype) \
    load1BiasTo8Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"8                   \n\t" \

#define load1BiasTo10Regs(bias, rtype) \
    load1BiasTo9Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"9                   \n\t" \

#define load1BiasTo11Regs(bias, rtype) \
    load1BiasTo10Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"10                   \n\t" \

#define load1BiasTo12Regs(bias, rtype) \
    load1BiasTo11Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"11                   \n\t"

#define load1BiasTo13Regs(bias, rtype) \
    load1BiasTo12Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"12                   \n\t" \

#define load1BiasTo14Regs(bias, rtype) \
    load1BiasTo13Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"13                   \n\t" \

#define load1BiasTo15Regs(bias, rtype) \
    load1BiasTo14Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"14                   \n\t" \

#define load1BiasTo16Regs(bias, rtype) \
    load1BiasTo15Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"15                   \n\t" \

#define load1BiasTo17Regs(bias, rtype) \
    load1BiasTo16Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"16                   \n\t" \

#define load1BiasTo18Regs(bias, rtype) \
    load1BiasTo17Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"17                   \n\t" \

#define load1BiasTo19Regs(bias, rtype) \
    load1BiasTo18Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"18                   \n\t" \

#define load1BiasTo20Regs(bias, rtype) \
    load1BiasTo19Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"19                   \n\t" \

#define load1BiasTo21Regs(bias, rtype) \
    load1BiasTo20Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"20                   \n\t" \

#define load1BiasTo22Regs(bias, rtype) \
    load1BiasTo21Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"21                   \n\t" \

#define load1BiasTo23Regs(bias, rtype) \
    load1BiasTo22Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"22                   \n\t" \

#define load1BiasTo24Regs(bias, rtype) \
    load1BiasTo23Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"23                   \n\t"

#define load3BiasToZmmRegs(nreg, bias) \
    load3BiasTo##nreg##Regs(bias, zmm)

#define load2BiasToZmmRegs(nreg, bias) \
    load2BiasTo##nreg##Regs(bias, zmm)

#define load4BiasToYmmRegs(nreg, bias) \
    load4BiasTo##nreg##Regs(bias, ymm)

#define load3BiasToYmmRegs(nreg, bias) \
    load3BiasTo##nreg##Regs(bias, ymm)

#define load2BiasToYmmRegs(nreg, bias) \
    load2BiasTo##nreg##Regs(bias, ymm)

#define load1BiasToRegs(nreg, bias, rtype) \
    load1BiasTo##nreg##Regs(bias, rtype)
