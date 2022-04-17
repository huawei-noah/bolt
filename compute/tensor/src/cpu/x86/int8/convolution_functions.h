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
    void *scale;
    bool cross;
};

typedef void (*kernelFunc)(ConvController &c);

// clang-format off
#define clear1Regs(rtype) \
    "vxorps "#rtype"0, "#rtype"0, "#rtype"0                     \n\t"

#define clear2Regs(rtype) \
    clear1Regs(rtype) \
    "vxorps "#rtype"1, "#rtype"1, "#rtype"1                     \n\t"

#define clear3Regs(rtype) \
    clear2Regs(rtype) \
    "vxorps "#rtype"2, "#rtype"2, "#rtype"2                     \n\t"

#define clear12Regs(rtype) \
    clear3Regs(rtype) \
    "vxorps "#rtype"3, "#rtype"3, "#rtype"3                     \n\t" \
    "vxorps "#rtype"4, "#rtype"4, "#rtype"4                     \n\t" \
    "vxorps "#rtype"5, "#rtype"5, "#rtype"5                     \n\t" \
    "vxorps "#rtype"6, "#rtype"6, "#rtype"6                     \n\t" \
    "vxorps "#rtype"7, "#rtype"7, "#rtype"7                     \n\t" \
    "vxorps "#rtype"8, "#rtype"8, "#rtype"8                     \n\t" \
    "vxorps "#rtype"9, "#rtype"9, "#rtype"9                     \n\t" \
    "vxorps "#rtype"10, "#rtype"10, "#rtype"10                  \n\t" \
    "vxorps "#rtype"11, "#rtype"11, "#rtype"11                  \n\t"

#define clear24Regs(rtype) \
    clear12Regs(rtype) \
    "vxorps "#rtype"12, "#rtype"12, "#rtype"12                  \n\t" \
    "vxorps "#rtype"13, "#rtype"13, "#rtype"13                  \n\t" \
    "vxorps "#rtype"14, "#rtype"14, "#rtype"14                  \n\t" \
    "vxorps "#rtype"15, "#rtype"15, "#rtype"15                  \n\t" \
    "vxorps "#rtype"16, "#rtype"16, "#rtype"16                  \n\t" \
    "vxorps "#rtype"17, "#rtype"17, "#rtype"17                  \n\t" \
    "vxorps "#rtype"18, "#rtype"18, "#rtype"18                  \n\t" \
    "vxorps "#rtype"19, "#rtype"19, "#rtype"19                  \n\t" \
    "vxorps "#rtype"20, "#rtype"20, "#rtype"20                  \n\t" \
    "vxorps "#rtype"21, "#rtype"21, "#rtype"21                  \n\t" \
    "vxorps "#rtype"22, "#rtype"22, "#rtype"22                  \n\t" \
    "vxorps "#rtype"23, "#rtype"23, "#rtype"23                  \n\t"

#define reluReg(rtype) \
    "vpxord "#rtype"31, "#rtype"31, "#rtype"31                  \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"0, "#rtype"0                    \n\t"

#define relu2Regs(rtype) \
    reluReg(rtype) \
    "vpmaxsd "#rtype"31, "#rtype"1, "#rtype"1                    \n\t"

#define relu3Regs(rtype) \
    relu2Regs(rtype) \
    "vpmaxsd "#rtype"31, "#rtype"2, "#rtype"2                    \n\t"

#define relu12Regs(rtype) \
    relu3Regs(rtype) \
    "vpmaxsd "#rtype"31, "#rtype"3, "#rtype"3                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"4, "#rtype"4                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"5, "#rtype"5                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"6, "#rtype"6                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"7, "#rtype"7                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"8, "#rtype"8                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"9, "#rtype"9                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"10, "#rtype"10                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"11, "#rtype"11                    \n\t"

#define relu24Regs(rtype) \
    relu12Regs(rtype) \
    "vpmaxsd "#rtype"31, "#rtype"12, "#rtype"12                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"13, "#rtype"13                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"14, "#rtype"14                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"15, "#rtype"15                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"16, "#rtype"16                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"17, "#rtype"17                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"18, "#rtype"18                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"19, "#rtype"19                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"20, "#rtype"20                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"21, "#rtype"21                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"22, "#rtype"22                    \n\t" \
    "vpmaxsd "#rtype"31, "#rtype"23, "#rtype"23                    \n\t"


#define reluRegPs(rtype) \
    "vpxord "#rtype"31, "#rtype"31, "#rtype"31                  \n\t" \
    "vmaxps "#rtype"31, "#rtype"0, "#rtype"0                    \n\t"

#define relu2RegsPs(rtype) \
    reluReg(rtype) \
    "vmaxps "#rtype"31, "#rtype"1, "#rtype"1                    \n\t"

#define relu3RegsPs(rtype) \
    relu2Regs(rtype) \
    "vmaxps "#rtype"31, "#rtype"2, "#rtype"2                    \n\t"

#define relu12RegsPs(rtype) \
    relu3Regs(rtype) \
    "vmaxps "#rtype"31, "#rtype"3, "#rtype"3                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"4, "#rtype"4                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"5, "#rtype"5                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"6, "#rtype"6                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"7, "#rtype"7                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"8, "#rtype"8                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"9, "#rtype"9                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"10, "#rtype"10                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"11, "#rtype"11                    \n\t"

#define relu24RegsPs(rtype) \
    relu12Regs(rtype) \
    "vmaxps "#rtype"31, "#rtype"12, "#rtype"12                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"13, "#rtype"13                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"14, "#rtype"14                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"15, "#rtype"15                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"16, "#rtype"16                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"17, "#rtype"17                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"18, "#rtype"18                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"19, "#rtype"19                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"20, "#rtype"20                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"21, "#rtype"21                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"22, "#rtype"22                    \n\t" \
    "vmaxps "#rtype"31, "#rtype"23, "#rtype"23                    \n\t"

#define convertRegI32ToF32(scalePtr, rtype) \
    "vbroadcastss ("#scalePtr"), "#rtype"24                        \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                       \n\t" \
    "vmulps "#rtype"0, "#rtype"24, "#rtype"0                       \n\t" \

#define convert2RegsI32ToF32(scalePtr, rtype) \
    "vbroadcastss ("#scalePtr"), "#rtype"24                        \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                       \n\t" \
    "vcvtdq2ps "#rtype"1, "#rtype"1                       \n\t" \
    "vmulps "#rtype"0, "#rtype"24, "#rtype"0                       \n\t" \
    "vmulps "#rtype"1, "#rtype"24, "#rtype"1                       \n\t" \

#define convert3RegsI32ToF32(scalePtr, rtype) \
    "vbroadcastss ("#scalePtr"), "#rtype"24                        \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                       \n\t" \
    "vcvtdq2ps "#rtype"1, "#rtype"1                       \n\t" \
    "vcvtdq2ps "#rtype"2, "#rtype"2                       \n\t" \
    "vmulps "#rtype"0, "#rtype"24, "#rtype"0                       \n\t" \
    "vmulps "#rtype"1, "#rtype"24, "#rtype"1                       \n\t" \
    "vmulps "#rtype"2, "#rtype"24, "#rtype"2                       \n\t"
#define convert12RegsI32ToF32(scalePtr, rtype) \
    "vbroadcastss ("#scalePtr"), "#rtype"24                        \n\t" \
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
    "vcvtdq2ps "#rtype"10, "#rtype"10                       \n\t" \
    "vcvtdq2ps "#rtype"11, "#rtype"11                       \n\t" \
    "vmulps "#rtype"0, "#rtype"24, "#rtype"0                       \n\t" \
    "vmulps "#rtype"1, "#rtype"24, "#rtype"1                       \n\t" \
    "vmulps "#rtype"2, "#rtype"24, "#rtype"2                       \n\t" \
    "vmulps "#rtype"3, "#rtype"24, "#rtype"3                       \n\t" \
    "vmulps "#rtype"4, "#rtype"24, "#rtype"4                       \n\t" \
    "vmulps "#rtype"5, "#rtype"24, "#rtype"5                       \n\t" \
    "vmulps "#rtype"6, "#rtype"24, "#rtype"6                       \n\t" \
    "vmulps "#rtype"7, "#rtype"24, "#rtype"7                       \n\t" \
    "vmulps "#rtype"8, "#rtype"24, "#rtype"8                       \n\t" \
    "vmulps "#rtype"9, "#rtype"24, "#rtype"9                       \n\t" \
    "vmulps "#rtype"10, "#rtype"24, "#rtype"10                     \n\t" \
    "vmulps "#rtype"11, "#rtype"24, "#rtype"11                     \n\t"

#define convert24RegsI32ToF32(scalePtr, rtype) \
    convert12RegsI32ToF32(scalePtr, rtype) \
    "vcvtdq2ps "#rtype"12, "#rtype"12                       \n\t" \
    "vcvtdq2ps "#rtype"13, "#rtype"13                       \n\t" \
    "vcvtdq2ps "#rtype"14, "#rtype"14                       \n\t" \
    "vcvtdq2ps "#rtype"15, "#rtype"15                       \n\t" \
    "vcvtdq2ps "#rtype"16, "#rtype"16                       \n\t" \
    "vcvtdq2ps "#rtype"17, "#rtype"17                       \n\t" \
    "vcvtdq2ps "#rtype"18, "#rtype"18                       \n\t" \
    "vcvtdq2ps "#rtype"19, "#rtype"19                       \n\t" \
    "vcvtdq2ps "#rtype"20, "#rtype"20                       \n\t" \
    "vcvtdq2ps "#rtype"21, "#rtype"21                       \n\t" \
    "vcvtdq2ps "#rtype"22, "#rtype"22                       \n\t" \
    "vcvtdq2ps "#rtype"23, "#rtype"23                       \n\t" \
    "vmulps "#rtype"12, "#rtype"24, "#rtype"12                     \n\t" \
    "vmulps "#rtype"13, "#rtype"24, "#rtype"13                     \n\t" \
    "vmulps "#rtype"14, "#rtype"24, "#rtype"14                     \n\t" \
    "vmulps "#rtype"15, "#rtype"24, "#rtype"15                     \n\t" \
    "vmulps "#rtype"16, "#rtype"24, "#rtype"16                     \n\t" \
    "vmulps "#rtype"17, "#rtype"24, "#rtype"17                     \n\t" \
    "vmulps "#rtype"18, "#rtype"24, "#rtype"18                     \n\t" \
    "vmulps "#rtype"19, "#rtype"24, "#rtype"19                     \n\t" \
    "vmulps "#rtype"20, "#rtype"24, "#rtype"20                     \n\t" \
    "vmulps "#rtype"21, "#rtype"24, "#rtype"21                     \n\t" \
    "vmulps "#rtype"22, "#rtype"24, "#rtype"22                     \n\t" \
    "vmulps "#rtype"23, "#rtype"24, "#rtype"23                     \n\t"

#define load48BiasTo3Regs(bias) \
    "vmovups ("#bias"), %%zmm0                       \n\t" \
    "vmovups 0x40("#bias"), %%zmm1                   \n\t" \
    "vmovups 0x80("#bias"), %%zmm2                   \n\t" \

#define load48BiasTo12Regs(bias) \
    load48BiasTo3Regs(bias) \
    "vmovups %%zmm0, %%zmm3                   \n\t" \
    "vmovups %%zmm1, %%zmm4                   \n\t" \
    "vmovups %%zmm2, %%zmm5                   \n\t" \
    "vmovups %%zmm0, %%zmm6                   \n\t" \
    "vmovups %%zmm1, %%zmm7                   \n\t" \
    "vmovups %%zmm2, %%zmm8                   \n\t" \
    "vmovups %%zmm0, %%zmm9                   \n\t" \
    "vmovups %%zmm1, %%zmm10                   \n\t" \
    "vmovups %%zmm2, %%zmm11                   \n\t"

#define load48BiasTo24Regs(bias) \
    load48BiasTo12Regs(bias) \
    "vmovups %%zmm0, %%zmm12                   \n\t" \
    "vmovups %%zmm1, %%zmm13                   \n\t" \
    "vmovups %%zmm2, %%zmm14                   \n\t" \
    "vmovups %%zmm0, %%zmm15                   \n\t" \
    "vmovups %%zmm1, %%zmm16                   \n\t" \
    "vmovups %%zmm2, %%zmm17                   \n\t" \
    "vmovups %%zmm0, %%zmm18                   \n\t" \
    "vmovups %%zmm1, %%zmm19                   \n\t" \
    "vmovups %%zmm2, %%zmm20                   \n\t" \
    "vmovups %%zmm0, %%zmm21                   \n\t" \
    "vmovups %%zmm1, %%zmm22                   \n\t" \
    "vmovups %%zmm2, %%zmm23                   \n\t"

#define load32BiasTo2Regs(bias) \
    "vmovups ("#bias"), %%zmm0                       \n\t" \
    "vmovups 0x40("#bias"), %%zmm1                   \n\t" \

#define load32BiasTo12Regs(bias) \
    load32BiasTo2Regs(bias) \
    "vmovups %%zmm0, %%zmm2                   \n\t" \
    "vmovups %%zmm1, %%zmm3                   \n\t" \
    "vmovups %%zmm0, %%zmm4                   \n\t" \
    "vmovups %%zmm1, %%zmm5                   \n\t" \
    "vmovups %%zmm0, %%zmm6                   \n\t" \
    "vmovups %%zmm1, %%zmm7                   \n\t" \
    "vmovups %%zmm0, %%zmm8                   \n\t" \
    "vmovups %%zmm1, %%zmm9                   \n\t" \
    "vmovups %%zmm0, %%zmm10                   \n\t" \
    "vmovups %%zmm1, %%zmm11                   \n\t"

#define load32BiasTo24Regs(bias) \
    load32BiasTo12Regs(bias) \
    "vmovups %%zmm0, %%zmm12                   \n\t" \
    "vmovups %%zmm1, %%zmm13                   \n\t" \
    "vmovups %%zmm0, %%zmm14                   \n\t" \
    "vmovups %%zmm1, %%zmm15                   \n\t" \
    "vmovups %%zmm0, %%zmm16                   \n\t" \
    "vmovups %%zmm1, %%zmm17                   \n\t" \
    "vmovups %%zmm0, %%zmm18                   \n\t" \
    "vmovups %%zmm1, %%zmm19                   \n\t" \
    "vmovups %%zmm0, %%zmm20                   \n\t" \
    "vmovups %%zmm1, %%zmm21                   \n\t" \
    "vmovups %%zmm0, %%zmm22                   \n\t" \
    "vmovups %%zmm1, %%zmm23                   \n\t"

#define load16BiasTo1Regs(bias, rtype) \
    "vmovups ("#bias"), "#rtype"0                       \n\t"

#define load16BiasTo12Regs(bias, rtype) \
    load16BiasTo1Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"1                   \n\t" \
    "vmovups "#rtype"0, "#rtype"2                   \n\t" \
    "vmovups "#rtype"0, "#rtype"3                   \n\t" \
    "vmovups "#rtype"0, "#rtype"4                   \n\t" \
    "vmovups "#rtype"0, "#rtype"5                   \n\t" \
    "vmovups "#rtype"0, "#rtype"6                   \n\t" \
    "vmovups "#rtype"0, "#rtype"7                   \n\t" \
    "vmovups "#rtype"0, "#rtype"8                   \n\t" \
    "vmovups "#rtype"0, "#rtype"9                   \n\t" \
    "vmovups "#rtype"0, "#rtype"10                   \n\t" \
    "vmovups "#rtype"0, "#rtype"11                   \n\t"

#define load16BiasTo24Regs(bias, rtype) \
    load16BiasTo12Regs(bias, rtype) \
    "vmovups "#rtype"0, "#rtype"12                   \n\t" \
    "vmovups "#rtype"0, "#rtype"13                   \n\t" \
    "vmovups "#rtype"0, "#rtype"14                   \n\t" \
    "vmovups "#rtype"0, "#rtype"15                   \n\t" \
    "vmovups "#rtype"0, "#rtype"16                   \n\t" \
    "vmovups "#rtype"0, "#rtype"17                   \n\t" \
    "vmovups "#rtype"0, "#rtype"18                   \n\t" \
    "vmovups "#rtype"0, "#rtype"19                   \n\t" \
    "vmovups "#rtype"0, "#rtype"20                   \n\t" \
    "vmovups "#rtype"0, "#rtype"21                   \n\t" \
    "vmovups "#rtype"0, "#rtype"22                   \n\t" \
    "vmovups "#rtype"0, "#rtype"23                   \n\t"
