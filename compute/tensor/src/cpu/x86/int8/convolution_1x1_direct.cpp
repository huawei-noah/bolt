// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "uni.h"
#include "data_type.h"
#include "sys.h"
#include "error.h"
#include "transform_functions_int8.h"
#include "cpu/x86/int8/tensor_computing_int8.h"
#include "cpu/x86/tensor_computing_x86.h"

#define SIMDW 16
#define BLOCK_IC_DIM 256
#define BLOCK_HW_DIM 768

struct ConvController {
    UINT8 *input;
    const INT8 *filter;
    void *output;
    UINT8 *u8Output;
    const I32 *bias;
    I64 ic;
    I64 kw;
    I64 kh;
    I64 stepC16;
    I64 dilateW;
    I64 dilateH;
    I64 ostepC16;
    I64 flags;
    I64 fStep;
    I64 f8Step;
    I64 f4Step;
    void *scale;
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

#ifdef _USE_AVX512_VNNI
#define convKernel8x48c4(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "vpbroadcastd ("#input"), %%zmm30                     \n\t" \
    "vpbroadcastd 0x10("#input"), %%zmm31                     \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm0              \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm1              \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm2              \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"                             \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm3              \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm4              \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm5              \n\t" \
    "vpbroadcastd 0x20("#input"), %%zmm30                     \n\t" \
    "vpbroadcastd 0x30("#input"), %%zmm31                     \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm6              \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm7              \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm8              \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"                             \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm9              \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm10              \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm11              \n\t" \
    "vpbroadcastd 0x40("#input"), %%zmm30                     \n\t" \
    "vpbroadcastd 0x50("#input"), %%zmm31                     \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm12              \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm13              \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm14              \n\t" \
    "vmovups "#off2"(%[filter]), "#preg2"                             \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm15              \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm16              \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm17              \n\t" \
    "vpbroadcastd 0x60("#input"), %%zmm30                     \n\t" \
    "vpbroadcastd 0x70("#input"), %%zmm31                     \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm18              \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm19              \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm20              \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm21              \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm22              \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm23              \n\t"

#define convKernel4x48c4(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "vpbroadcastd ("#input"), %%zmm30                     \n\t" \
    "vpbroadcastd 0x10("#input"), %%zmm31                     \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm0              \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm1              \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm2              \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"                             \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm3              \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm4              \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm5              \n\t" \
    "vpbroadcastd 0x20("#input"), %%zmm30                     \n\t" \
    "vpbroadcastd 0x30("#input"), %%zmm31                     \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm6              \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm7              \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm8              \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"                             \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm9              \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm10              \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm11              \n\t"

#define convKernel1x48c4(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "vpbroadcastd ("#input"), %%zmm30                     \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"                             \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"                             \n\t" \
    "vmovups "#off2"(%[filter]), "#preg2"                             \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm0              \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm1              \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm2              \n\t"
#else
#define convKernel8x48c4_3(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "vpbroadcastd ("#input"), %%zmm30                     \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x10("#input"), %%zmm30                     \n\t" \
    "vpaddd %%zmm0, "#preg0", %%zmm0              \n\t" \
    "vpaddd %%zmm1, "#preg1", %%zmm1              \n\t" \
    "vpaddd %%zmm2, "#preg2", %%zmm2              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x20("#input"), %%zmm30                     \n\t" \
    "vpaddd %%zmm3, "#preg0", %%zmm3              \n\t" \
    "vpaddd %%zmm4, "#preg1", %%zmm4              \n\t" \
    "vpaddd %%zmm5, "#preg2", %%zmm5              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x30("#input"), %%zmm30                     \n\t" \
    "vpaddd %%zmm6, "#preg0", %%zmm6              \n\t" \
    "vpaddd %%zmm7, "#preg1", %%zmm7              \n\t" \
    "vpaddd %%zmm8, "#preg2", %%zmm8              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x40("#input"), %%zmm30                     \n\t" \
    "vpaddd %%zmm9, "#preg0", %%zmm9              \n\t" \
    "vpaddd %%zmm10, "#preg1", %%zmm10              \n\t" \
    "vpaddd %%zmm11, "#preg2", %%zmm11             \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x50("#input"), %%zmm30                     \n\t" \
    "vpaddd %%zmm12, "#preg0", %%zmm12              \n\t" \
    "vpaddd %%zmm13, "#preg1", %%zmm13              \n\t" \
    "vpaddd %%zmm14, "#preg2", %%zmm14              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x60("#input"), %%zmm30                     \n\t" \
    "vpaddd %%zmm15, "#preg0", %%zmm15              \n\t" \
    "vpaddd %%zmm16, "#preg1", %%zmm16              \n\t" \
    "vpaddd %%zmm17, "#preg2", %%zmm17              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x70("#input"), %%zmm30                     \n\t" \
    "vpaddd %%zmm18, "#preg0", %%zmm18              \n\t" \
    "vpaddd %%zmm19, "#preg1", %%zmm19              \n\t" \
    "vpaddd %%zmm20, "#preg2", %%zmm20              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"                             \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vmovups "#off1"(%[filter]), "#freg1"                             \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vmovups "#off2"(%[filter]), "#freg2"                             \n\t" \
    "vpaddd %%zmm21, "#preg0", %%zmm21              \n\t" \
    "vpaddd %%zmm22, "#preg1", %%zmm22              \n\t" \
    "vpaddd %%zmm23, "#preg2", %%zmm23              \n\t"

#define convKernel4x48c4_3(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "vpbroadcastd ("#input"), %%zmm30                     \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x10("#input"), %%zmm30                     \n\t" \
    "vpaddd %%zmm0, "#preg0", %%zmm0              \n\t" \
    "vpaddd %%zmm1, "#preg1", %%zmm1              \n\t" \
    "vpaddd %%zmm2, "#preg2", %%zmm2              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x20("#input"), %%zmm30                     \n\t" \
    "vpaddd %%zmm3, "#preg0", %%zmm3              \n\t" \
    "vpaddd %%zmm4, "#preg1", %%zmm4              \n\t" \
    "vpaddd %%zmm5, "#preg2", %%zmm5              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x30("#input"), %%zmm30                     \n\t" \
    "vpaddd %%zmm6, "#preg0", %%zmm6              \n\t" \
    "vpaddd %%zmm7, "#preg1", %%zmm7              \n\t" \
    "vpaddd %%zmm8, "#preg2", %%zmm8              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"                             \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vmovups "#off1"(%[filter]), "#freg1"                             \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vmovups "#off2"(%[filter]), "#freg2"                             \n\t" \
    "vpaddd %%zmm9, "#preg0", %%zmm9              \n\t" \
    "vpaddd %%zmm10, "#preg1", %%zmm10              \n\t" \
    "vpaddd %%zmm11, "#preg2", %%zmm11             \n\t"

#define convKernel1x48c4_3(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "vpbroadcastd ("#input"), %%zmm30                     \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"                             \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vmovups "#off1"(%[filter]), "#freg1"                             \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vmovups "#off2"(%[filter]), "#freg2"                             \n\t" \
    "vpaddd %%zmm0, "#preg0", %%zmm0              \n\t" \
    "vpaddd %%zmm1, "#preg1", %%zmm1              \n\t" \
    "vpaddd %%zmm2, "#preg2", %%zmm2              \n\t"

#define convKernel8x48c4(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel8x48c4_3(input, %%zmm24, %%zmm25, %%zmm26, off0, off1, off2, %%zmm27, %%zmm28, %%zmm29)

#define convKernel4x48c4(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel4x48c4_3(input, %%zmm24, %%zmm25, %%zmm26, off0, off1, off2, %%zmm27, %%zmm28, %%zmm29)

#define convKernel1x48c4(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel1x48c4_3(input, %%zmm24, %%zmm25, %%zmm26, off0, off1, off2, %%zmm27, %%zmm28, %%zmm29)
#endif

#define convKernelForLoopXx48(rnum, wsize) \
     __asm__ __volatile__("vmovups (%[filter]), %%zmm24                             \n\t" \
                          "vmovups 0x40(%[filter]), %%zmm25                             \n\t" \
                          "vmovups 0x80(%[filter]), %%zmm26                             \n\t" \
                          "addq $0xC0, %[filter]                                    \n\t" \
                          "mov $1, %%eax \n\t" \
                          "vmovd %%eax, %%xmm0                    \n\t" \
                          "vpbroadcastw %%xmm0, %%zmm31            \n\t" \
                          "movq %[flags], %%rax          \n\t" \
                          "andq $0x1, %%rax          \n\t" \
                          "jne 0f                                         \n\t" \
                          load48BiasTo##rnum##Regs(%[bias]) \
                          "cmpq $0x10, %%rcx          \n\t" \
                          "jl 4f            \n\t" \
                          "jmp 1f          \n\t" \
                          ".align 16                                         \n\t" \
                          "0:                                                \n\t" \
                          clear##rnum##Regs(%%zmm) \
                          "cmpq $0x10, %%rcx          \n\t" \
                          "jl 4f            \n\t" \
                          ".align 16                                         \n\t" \
                          "1:                                                \n\t" \
                          "movq %[input], %%rax  \n\t" \
                          convKernel##wsize##x48c4(%%rax, %%zmm24, %%zmm25, %%zmm26, 0x0, 0x40, 0x80, %%zmm27, %%zmm28, %%zmm29) \
                          "addq $0x4, %%rax  \n\t" \
                          convKernel##wsize##x48c4(%%rax, %%zmm27, %%zmm28, %%zmm29, 0xC0, 0x100, 0x140, %%zmm24, %%zmm25, %%zmm26) \
                          "addq $0x4, %%rax  \n\t" \
                          convKernel##wsize##x48c4(%%rax, %%zmm24, %%zmm25, %%zmm26, 0x180, 0x1C0, 0x200, %%zmm27, %%zmm28, %%zmm29) \
                          "addq $0x4, %%rax  \n\t" \
                          convKernel##wsize##x48c4(%%rax, %%zmm27, %%zmm28, %%zmm29, 0x240, 0x280, 0x2C0, %%zmm24, %%zmm25, %%zmm26) \
                          "addq $0x300, %[filter]                                    \n\t" \
                          "addq %[fStep], %[input]                                    \n\t" \
                          "subq $0x10, %%rcx                                         \n\t" \
                          "cmpq $0x10, %%rcx                                         \n\t" \
                          "jge 1b                                             \n\t" \
                          "subq %[fStep], %[input]                                    \n\t" \
                          "addq %[f8Step], %[input]                                    \n\t" \
                          ".align 16                                         \n\t" \
                          "4:                                                \n\t" \
                          : "+c" (c.ic), [input] "+r" (c.input), [filter] "+r" (c.filter) \
                          : [bias] "r" (c.bias), [kh] "r" (c.kh), [kw] "r" (c.kw), \
                            [stepC16] "r" (c.stepC16), [fStep] "r" (c.fStep), [flags] "r" (c.flags),  \
                            [f8Step] "r" (c.f8Step) \
                          : "%rax", "%rbx", "%r9", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", \
                            "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",  \
                            "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", \
                            "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", \
                            "%zmm31", "memory", "cc"); \
     if (c.ic > 0) { \
         __asm__ __volatile__("cmpq $0x8, %%rcx          \n\t" \
                              "jl 2f            \n\t" \
                              "subq $0x8, %%rcx          \n\t" \
                              "movq %[input], %%rax  \n\t" \
                              convKernel##wsize##x48c4(%%rax, %%zmm24, %%zmm25, %%zmm26, 0x0, 0x40, 0x80, %%zmm27, %%zmm28, %%zmm29) \
                              "addq $0x4, %%rax  \n\t" \
                              convKernel##wsize##x48c4(%%rax, %%zmm27, %%zmm28, %%zmm29, 0xC0, 0x100, 0x140, %%zmm24, %%zmm25, %%zmm26) \
                              "addq $0x180, %[filter]                                    \n\t" \
                              "addq %[f4Step], %[input]                                    \n\t" \
                              ".align 16                                         \n\t" \
                              "2:                                                \n\t" \
                              "cmpq $0x4, %%rcx          \n\t" \
                              "jl 5f            \n\t" \
                              convKernel##wsize##x48c4(%%rax, %%zmm24, %%zmm25, %%zmm26, 0x0, 0x40, 0x80, %%zmm27, %%zmm28, %%zmm29) \
                              ".align 16                                         \n\t" \
                              "5:                                             \n\t" \
                              : "+c" (c.ic) \
                              : [input] "r" (c.input), [filter] "r" (c.filter), [bias] "r" (c.bias), [kh] "r" (c.kh), [kw] "r" (c.kw), \
                                [stepC16] "r" (c.stepC16), [f4Step] "r" (c.f4Step) \
                              : "%rax", "%rbx", "%r9", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", \
                                "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",  \
                                "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", \
                                "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", \
                                "%zmm31", "memory", "cc"); \
    }

void Avx512Conv1x1Kernel8x48(ConvController &c) {
     convKernelForLoopXx48(24, 8)

    __asm__ __volatile__("movq %[output], %%rax                                      \n\t"
                         "movq %[ostepC16], %%rbx                                      \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0x1, %%rcx                                  \n\t"
                         "je 0f                                             \n\t"
                         "vpaddd (%%rax), %%zmm0, %%zmm0                             \n\t"
                         "vpaddd 0x40(%%rax), %%zmm3, %%zmm3                         \n\t"
                         "vpaddd 0x80(%%rax), %%zmm6, %%zmm6                         \n\t"
                         "vpaddd 0xC0(%%rax), %%zmm9, %%zmm9                         \n\t"
                         "vpaddd 0x100(%%rax), %%zmm12, %%zmm12                         \n\t"
                         "vpaddd 0x140(%%rax), %%zmm15, %%zmm15                         \n\t"
                         "vpaddd 0x180(%%rax), %%zmm18, %%zmm18                         \n\t"
                         "vpaddd 0x1C0(%%rax), %%zmm21, %%zmm21                         \n\t"
                         "vpaddd (%%rax, %%rbx), %%zmm1, %%zmm1                             \n\t"
                         "vpaddd 0x40(%%rax, %%rbx), %%zmm4, %%zmm4                         \n\t"
                         "vpaddd 0x80(%%rax, %%rbx), %%zmm7, %%zmm7                         \n\t"
                         "vpaddd 0xC0(%%rax, %%rbx), %%zmm10, %%zmm10                         \n\t"
                         "vpaddd 0x100(%%rax, %%rbx), %%zmm13, %%zmm13                         \n\t"
                         "vpaddd 0x140(%%rax, %%rbx), %%zmm16, %%zmm16                         \n\t"
                         "vpaddd 0x180(%%rax, %%rbx), %%zmm19, %%zmm19                         \n\t"
                         "vpaddd 0x1C0(%%rax, %%rbx), %%zmm22, %%zmm22                         \n\t"
                         "vpaddd (%%rax, %%rbx, 2), %%zmm2, %%zmm2                             \n\t"
                         "vpaddd 0x40(%%rax, %%rbx, 2), %%zmm5, %%zmm5                         \n\t"
                         "vpaddd 0x80(%%rax, %%rbx, 2), %%zmm8, %%zmm8                         \n\t"
                         "vpaddd 0xC0(%%rax, %%rbx, 2), %%zmm11, %%zmm11                         \n\t"
                         "vpaddd 0x100(%%rax, %%rbx, 2), %%zmm14, %%zmm14                         \n\t"
                         "vpaddd 0x140(%%rax, %%rbx, 2), %%zmm17, %%zmm17                         \n\t"
                         "vpaddd 0x180(%%rax, %%rbx, 2), %%zmm20, %%zmm20                         \n\t"
                         "vpaddd 0x1C0(%%rax, %%rbx, 2), %%zmm23, %%zmm23                         \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                      \n\t"
                         "je 1f                                             \n\t"
                         relu24Regs(%%zmm)

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 2f      \n\t"
                         convert24RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         "vmovups %%zmm0, (%%rax)                             \n\t"
                         "vmovups %%zmm3, 0x40(%%rax)                         \n\t"
                         "vmovups %%zmm6, 0x80(%%rax)                         \n\t"
                         "vmovups %%zmm9, 0xC0(%%rax)                         \n\t"
                         "vmovups %%zmm12, 0x100(%%rax)                         \n\t"
                         "vmovups %%zmm15, 0x140(%%rax)                         \n\t"
                         "vmovups %%zmm18, 0x180(%%rax)                         \n\t"
                         "vmovups %%zmm21, 0x1C0(%%rax)                         \n\t"
                         "vmovups %%zmm1, (%%rax, %%rbx)                             \n\t"
                         "vmovups %%zmm4, 0x40(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm7, 0x80(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm10, 0xC0(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm13, 0x100(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm16, 0x140(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm19, 0x180(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm22, 0x1C0(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm2, (%%rax, %%rbx, 2)                             \n\t"
                         "vmovups %%zmm5, 0x40(%%rax, %%rbx, 2)                         \n\t"
                         "vmovups %%zmm8, 0x80(%%rax, %%rbx, 2)                         \n\t"
                         "vmovups %%zmm11, 0xC0(%%rax, %%rbx, 2)                         \n\t"
                         "vmovups %%zmm14, 0x100(%%rax, %%rbx, 2)                         \n\t"
                         "vmovups %%zmm17, 0x140(%%rax, %%rbx, 2)                         \n\t"
                         "vmovups %%zmm20, 0x180(%%rax, %%rbx, 2)                         \n\t"
                         "vmovups %%zmm23, 0x1C0(%%rax, %%rbx, 2)                         \n\t"
                         :
                         : [output] "r" (c.output), [ostepC16] "r" (c.ostepC16), [flags] "r" (c.flags), [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6",
                           "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",
                           "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22",
                           "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30",
                           "%zmm31", "memory", "cc");
}

void Avx512Conv1x1Kernel4x48(ConvController &c) {
    convKernelForLoopXx48(12, 4)

    __asm__ __volatile__("movq %[output], %%rax                                      \n\t"
                         "movq %[ostepC16], %%rbx                                      \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0x1, %%rcx                                  \n\t"
                         "je 0f                                             \n\t"
                         "vpaddd (%%rax), %%zmm0, %%zmm0                             \n\t"
                         "vpaddd 0x40(%%rax), %%zmm3, %%zmm3                         \n\t"
                         "vpaddd 0x80(%%rax), %%zmm6, %%zmm6                         \n\t"
                         "vpaddd 0xC0(%%rax), %%zmm9, %%zmm9                         \n\t"
                         "vpaddd (%%rax, %%rbx), %%zmm1, %%zmm1                             \n\t"
                         "vpaddd 0x40(%%rax, %%rbx), %%zmm4, %%zmm4                         \n\t"
                         "vpaddd 0x80(%%rax, %%rbx), %%zmm7, %%zmm7                         \n\t"
                         "vpaddd 0xC0(%%rax, %%rbx), %%zmm10, %%zmm10                         \n\t"
                         "vpaddd (%%rax, %%rbx, 2), %%zmm2, %%zmm2                             \n\t"
                         "vpaddd 0x40(%%rax, %%rbx, 2), %%zmm5, %%zmm5                         \n\t"
                         "vpaddd 0x80(%%rax, %%rbx, 2), %%zmm8, %%zmm8                         \n\t"
                         "vpaddd 0xC0(%%rax, %%rbx, 2), %%zmm11, %%zmm11                         \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                      \n\t"
                         "je 1f                                             \n\t"
                         relu12Regs(%%zmm)

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 2f      \n\t"
                         convert12RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         "vmovups %%zmm0, (%%rax)                             \n\t"
                         "vmovups %%zmm3, 0x40(%%rax)                         \n\t"
                         "vmovups %%zmm6, 0x80(%%rax)                         \n\t"
                         "vmovups %%zmm9, 0xC0(%%rax)                         \n\t"
                         "vmovups %%zmm1, (%%rax, %%rbx)                             \n\t"
                         "vmovups %%zmm4, 0x40(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm7, 0x80(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm10, 0xC0(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm2, (%%rax, %%rbx, 2)                             \n\t"
                         "vmovups %%zmm5, 0x40(%%rax, %%rbx, 2)                         \n\t"
                         "vmovups %%zmm8, 0x80(%%rax, %%rbx, 2)                         \n\t"
                         "vmovups %%zmm11, 0xC0(%%rax, %%rbx, 2)                         \n\t"
                         :
                         : [output] "r" (c.output), [ostepC16] "r" (c.ostepC16), [flags] "r" (c.flags), [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6",
                           "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",
                           "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22",
                           "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30",
                           "%zmm31", "memory", "cc");
}

void Avx512Conv1x1Kernel1x48(ConvController &c) {
    convKernelForLoopXx48(3, 1)

    __asm__ __volatile__("movq %[output], %%rax                                      \n\t"
                         "movq %[ostepC16], %%rbx                                      \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0x1, %%rcx                                  \n\t"
                         "je 0f                                             \n\t"
                         "vpaddd (%%rax), %%zmm0, %%zmm0                             \n\t"
                         "vpaddd (%%rax, %%rbx), %%zmm1, %%zmm1                             \n\t"
                         "vpaddd (%%rax, %%rbx, 2), %%zmm2, %%zmm2                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                      \n\t"
                         "je 1f                                             \n\t"
                         relu3Regs(%%zmm)

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 2f      \n\t"
                         convert3RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         "vmovups %%zmm0, (%%rax)                             \n\t"
                         "vmovups %%zmm1, (%%rax, %%rbx)                             \n\t"
                         "vmovups %%zmm2, (%%rax, %%rbx, 2)                             \n\t"
                         :
                         : [output] "r" (c.output), [ostepC16] "r" (c.ostepC16), [flags] "r" (c.flags), [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6",
                           "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",
                           "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22",
                           "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30",
                           "%zmm31", "memory", "cc");
}

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

#ifdef _USE_AVX512_VNNI
#define convKernel12x32c4(input, freg0, freg1, off0, off1, preg0, preg1) \
    "vpbroadcastd ("#input"), %%zmm28                     \n\t" \
    "vpbroadcastd 0x10("#input"), %%zmm29                     \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0              \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1              \n\t" \
    "vpbroadcastd 0x20("#input"), %%zmm30                     \n\t" \
    "vpbroadcastd 0x30("#input"), %%zmm31                     \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm2              \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm3              \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm4              \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm5              \n\t" \
    "vpbroadcastd 0x40("#input"), %%zmm28                     \n\t" \
    "vpbroadcastd 0x50("#input"), %%zmm29                     \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm6              \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm7              \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"                             \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm8              \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm9              \n\t" \
    "vpbroadcastd 0x60("#input"), %%zmm30                     \n\t" \
    "vpbroadcastd 0x70("#input"), %%zmm31                     \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm10              \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm11              \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"                             \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm12              \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm13              \n\t" \
    "vpbroadcastd 0x80("#input"), %%zmm28                     \n\t" \
    "vpbroadcastd 0x90("#input"), %%zmm29                     \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm14              \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm15              \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm16              \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm17              \n\t" \
    "vpbroadcastd 0xA0("#input"), %%zmm30                     \n\t" \
    "vpbroadcastd 0xB0("#input"), %%zmm31                     \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm18              \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm19              \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm20              \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm21              \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm22              \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm23              \n\t"

#define convKernel6x32c4(input, freg0, freg1, off0, off1, preg0, preg1) \
    "vpbroadcastd ("#input"), %%zmm28                     \n\t" \
    "vpbroadcastd 0x10("#input"), %%zmm29                     \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"                             \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0              \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1              \n\t" \
    "vpbroadcastd 0x20("#input"), %%zmm30                     \n\t" \
    "vpbroadcastd 0x30("#input"), %%zmm31                     \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm2              \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm3              \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm4              \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm5              \n\t" \
    "vpbroadcastd 0x40("#input"), %%zmm28                     \n\t" \
    "vpbroadcastd 0x50("#input"), %%zmm29                     \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm6              \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm7              \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"                             \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm8              \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm9              \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm10              \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm11              \n\t"

#define convKernel1x32c4(input, freg0, freg1, off0, off1, preg0, preg1) \
    "vpbroadcastd ("#input"), %%zmm28                     \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"                             \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"                             \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0              \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1              \n\t"
#else
#define convKernel12x32c4_3(input, freg0, freg1, off0, off1, preg0, preg1, preg2) \
    "vpbroadcastd ("#input"), %%zmm29                     \n\t" \
    "vpbroadcastd 0x10("#input"), %%zmm30                     \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg1"              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x20("#input"), %%zmm29                     \n\t" \
    "vpaddd %%zmm0, "#preg0", %%zmm0              \n\t" \
    "vpaddd %%zmm1, "#preg1", %%zmm1              \n\t" \
    "vpaddd %%zmm2, "#preg2", %%zmm2              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg1"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg2"              \n\t" \
    "vpbroadcastd 0x30("#input"), %%zmm30                     \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x40("#input"), %%zmm29                     \n\t" \
    "vpaddd %%zmm3, "#preg0", %%zmm3              \n\t" \
    "vpaddd %%zmm4, "#preg1", %%zmm4              \n\t" \
    "vpaddd %%zmm5, "#preg2", %%zmm5              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x50("#input"), %%zmm30                     \n\t" \
    "vpaddd %%zmm6, "#preg0", %%zmm6              \n\t" \
    "vpaddd %%zmm7, "#preg1", %%zmm7              \n\t" \
    "vpaddd %%zmm8, "#preg2", %%zmm8              \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg0"              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg2"              \n\t" \
    "vpbroadcastd 0x60("#input"), %%zmm29                     \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x70("#input"), %%zmm30                     \n\t" \
    "vpaddd %%zmm9, "#preg0", %%zmm9              \n\t" \
    "vpaddd %%zmm10, "#preg1", %%zmm10              \n\t" \
    "vpaddd %%zmm11, "#preg2", %%zmm11              \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg1"              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x80("#input"), %%zmm29                     \n\t" \
    "vpaddd %%zmm12, "#preg0", %%zmm12              \n\t" \
    "vpaddd %%zmm13, "#preg1", %%zmm13              \n\t" \
    "vpaddd %%zmm14, "#preg2", %%zmm14              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg1"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg2"              \n\t" \
    "vpbroadcastd 0x90("#input"), %%zmm30                     \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0xA0("#input"), %%zmm29                     \n\t" \
    "vpaddd %%zmm15, "#preg0", %%zmm15              \n\t" \
    "vpaddd %%zmm16, "#preg1", %%zmm16              \n\t" \
    "vpaddd %%zmm17, "#preg2", %%zmm17              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0xB0("#input"), %%zmm30                     \n\t" \
    "vpaddd %%zmm18, "#preg0", %%zmm18              \n\t" \
    "vpaddd %%zmm19, "#preg1", %%zmm19              \n\t" \
    "vpaddd %%zmm20, "#preg2", %%zmm20              \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg0"              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"                             \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vmovups "#off1"(%[filter]), "#freg1"                             \n\t" \
    "vpaddd %%zmm21, "#preg0", %%zmm21              \n\t" \
    "vpaddd %%zmm22, "#preg1", %%zmm22              \n\t" \
    "vpaddd %%zmm23, "#preg2", %%zmm23              \n\t"

#define convKernel6x32c4_3(input, freg0, freg1, off0, off1, preg0, preg1, preg2) \
    "vpbroadcastd ("#input"), %%zmm29                     \n\t" \
    "vpbroadcastd 0x10("#input"), %%zmm30                     \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg1"              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x20("#input"), %%zmm29                     \n\t" \
    "vpaddd %%zmm0, "#preg0", %%zmm0              \n\t" \
    "vpaddd %%zmm1, "#preg1", %%zmm1              \n\t" \
    "vpaddd %%zmm2, "#preg2", %%zmm2              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg1"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg2"              \n\t" \
    "vpbroadcastd 0x30("#input"), %%zmm30                     \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x40("#input"), %%zmm29                     \n\t" \
    "vpaddd %%zmm3, "#preg0", %%zmm3              \n\t" \
    "vpaddd %%zmm4, "#preg1", %%zmm4              \n\t" \
    "vpaddd %%zmm5, "#preg2", %%zmm5              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vpbroadcastd 0x50("#input"), %%zmm30                     \n\t" \
    "vpaddd %%zmm6, "#preg0", %%zmm6              \n\t" \
    "vpaddd %%zmm7, "#preg1", %%zmm7              \n\t" \
    "vpaddd %%zmm8, "#preg2", %%zmm8              \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg0"              \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg1"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg2"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"                             \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"              \n\t" \
    "vmovups "#off1"(%[filter]), "#freg1"                             \n\t" \
    "vpaddd %%zmm9, "#preg0", %%zmm9              \n\t" \
    "vpaddd %%zmm10, "#preg1", %%zmm10              \n\t" \
    "vpaddd %%zmm11, "#preg2", %%zmm11              \n\t"

#define convKernel1x32c4_3(input, freg0, freg1, off0, off1, preg0, preg1, preg2) \
    "vpbroadcastd ("#input"), %%zmm29                     \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg0"              \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg1"              \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"              \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"                             \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"              \n\t" \
    "vmovups "#off1"(%[filter]), "#freg1"                             \n\t" \
    "vpaddd %%zmm0, "#preg0", %%zmm0              \n\t" \
    "vpaddd %%zmm1, "#preg1", %%zmm1              \n\t"

#define convKernel12x32c4(input, freg0, freg1, off0, off1, preg0, preg1) \
    convKernel12x32c4_3(input, %%zmm24, %%zmm25, off0, off1, %%zmm26, %%zmm27, %%zmm28)

#define convKernel6x32c4(input, freg0, freg1, off0, off1, preg0, preg1) \
    convKernel6x32c4_3(input, %%zmm24, %%zmm25, off0, off1, %%zmm26, %%zmm27, %%zmm28)

#define convKernel1x32c4(input, freg0, freg1, off0, off1, preg0, preg1) \
    convKernel1x32c4_3(input, %%zmm24, %%zmm25, off0, off1, %%zmm26, %%zmm27, %%zmm28)
#endif

#define convKernelForLoopXx32(rnum, wsize) \
     __asm__ __volatile__("vmovups (%[filter]), %%zmm24                             \n\t" \
                          "vmovups 0x40(%[filter]), %%zmm25                             \n\t" \
                          "addq $0x80, %[filter]                                    \n\t" \
                          "mov $1, %%eax \n\t" \
                          "vmovd %%eax, %%xmm0                    \n\t" \
                          "vpbroadcastw %%xmm0, %%zmm31            \n\t" \
                          "movq %[flags], %%rax          \n\t" \
                          "andq $0x1, %%rax          \n\t" \
                          "jne 0f                                         \n\t" \
                          load32BiasTo##rnum##Regs(%[bias]) \
                          "cmpq $0x10, %%rcx          \n\t" \
                          "jl 4f            \n\t" \
                          "jmp 1f          \n\t" \
                          ".align 16                                         \n\t" \
                          "0:                                                \n\t" \
                          clear##rnum##Regs(%%zmm) \
                          "cmpq $0x10, %%rcx          \n\t" \
                          "jl 4f            \n\t" \
                          ".align 16                                         \n\t" \
                          "1:                                                \n\t" \
                          "movq %[input], %%rax  \n\t" \
                          convKernel##wsize##x32c4(%%rax, %%zmm24, %%zmm25, 0x0, 0x40, %%zmm26, %%zmm27) \
                          "addq $0x4, %%rax  \n\t" \
                          convKernel##wsize##x32c4(%%rax, %%zmm26, %%zmm27, 0x80, 0xC0, %%zmm24, %%zmm25) \
                          "addq $0x4, %%rax  \n\t" \
                          convKernel##wsize##x32c4(%%rax, %%zmm24, %%zmm25, 0x100, 0x140, %%zmm26, %%zmm27) \
                          "addq $0x4, %%rax  \n\t" \
                          convKernel##wsize##x32c4(%%rax, %%zmm26, %%zmm27, 0x180, 0x1C0, %%zmm24, %%zmm25) \
                          "addq $0x200, %[filter]                                    \n\t" \
                          "addq %[fStep], %[input]                                    \n\t" \
                          "subq $0x10, %%rcx                                         \n\t" \
                          "cmpq $0x10, %%rcx                                         \n\t" \
                          "jge 1b                                             \n\t" \
                          "subq %[fStep], %[input]                                    \n\t" \
                          "addq %[f8Step], %[input]                                    \n\t" \
                          ".align 16                                         \n\t" \
                          "4:                                                \n\t" \
                          : "+c" (c.ic), [input] "+r" (c.input), [filter] "+r" (c.filter) \
                          : [bias] "r" (c.bias), [stepC16] "r" (c.stepC16), [dilateW] "r" (c.dilateW), \
                            [dilateH] "r" (c.dilateH), [fStep] "r" (c.fStep), [flags] "r" (c.flags),  \
                            [f8Step] "r" (c.f8Step) \
                          : "%rax", "%rbx", "%r9", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", \
                            "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",  \
                            "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", \
                            "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", \
                            "%zmm31", "memory", "cc"); \
     if (c.ic > 0) { \
         __asm__ __volatile__("cmpq $0x8, %%rcx          \n\t" \
                              "jl 2f            \n\t" \
                              "subq $0x8, %%rcx          \n\t" \
                              "shr $1, %[dilateW]                                    \n\t" \
                              "shr $1, %[dilateH]                                    \n\t" \
                              "shr $1, %[fStep]                                    \n\t" \
                              "movq %[input], %%rax  \n\t" \
                              convKernel##wsize##x32c4(%%rax, %%zmm24, %%zmm25, 0x0, 0x40, %%zmm26, %%zmm27) \
                              "addq $0x4, %%rax  \n\t" \
                              convKernel##wsize##x32c4(%%rax, %%zmm26, %%zmm27, 0x80, 0xC0, %%zmm24, %%zmm25) \
                              "addq $0x100, %[filter]                                    \n\t" \
                              "addq %[f4Step], %[input]                                    \n\t" \
                              ".align 16                                         \n\t" \
                              "2:                                                \n\t" \
                              "cmpq $0x4, %%rcx          \n\t" \
                              "jl 5f            \n\t" \
                              "shr $1, %[dilateW]                                    \n\t" \
                              "shr $1, %[dilateH]                                    \n\t" \
                              convKernel##wsize##x32c4(%[input], %%zmm24, %%zmm25, 0x0, 0x40, %%zmm26, %%zmm27) \
                              "addq $0x80, %[filter]                                    \n\t" \
                              ".align 16                                         \n\t" \
                              "5:                                             \n\t" \
                              : "+c" (c.ic) \
                              : [input] "r" (c.input), [filter] "r" (c.filter), [bias] "r" (c.bias), \
                                [dilateW] "r" (c.dilateW), \
                                [dilateH] "r" (c.dilateH), [fStep] "r" (c.fStep), \
                                [f4Step] "r" (c.f4Step) \
                              : "%rax", "%rbx", "%r9", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", \
                                "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",  \
                                "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", \
                                "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", \
                                "%zmm31", "memory", "cc"); \
    }

void Avx512Conv1x1Kernel12x32(ConvController &c) {
    convKernelForLoopXx32(24, 12)

    __asm__ __volatile__("movq %[output], %%rax                                      \n\t"
                         "movq %[ostepC16], %%rbx                                      \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0x1, %%rcx                                  \n\t"
                         "je 0f                                             \n\t"
                         "vpaddd (%%rax), %%zmm0, %%zmm0                             \n\t"
                         "vpaddd 0x40(%%rax), %%zmm2, %%zmm2                         \n\t"
                         "vpaddd 0x80(%%rax), %%zmm4, %%zmm4                         \n\t"
                         "vpaddd 0xC0(%%rax), %%zmm6, %%zmm6                         \n\t"
                         "vpaddd 0x100(%%rax), %%zmm8, %%zmm8                         \n\t"
                         "vpaddd 0x140(%%rax), %%zmm10, %%zmm10                         \n\t"
                         "vpaddd 0x180(%%rax), %%zmm12, %%zmm12                         \n\t"
                         "vpaddd 0x1C0(%%rax), %%zmm14, %%zmm14                         \n\t"
                         "vpaddd 0x200(%%rax), %%zmm16, %%zmm16                         \n\t"
                         "vpaddd 0x240(%%rax), %%zmm18, %%zmm18                         \n\t"
                         "vpaddd 0x280(%%rax), %%zmm20, %%zmm20                         \n\t"
                         "vpaddd 0x2C0(%%rax), %%zmm22, %%zmm22                         \n\t"
                         "vpaddd (%%rax, %%rbx), %%zmm1, %%zmm1                             \n\t"
                         "vpaddd 0x40(%%rax, %%rbx), %%zmm3, %%zmm3                         \n\t"
                         "vpaddd 0x80(%%rax, %%rbx), %%zmm5, %%zmm5                         \n\t"
                         "vpaddd 0xC0(%%rax, %%rbx), %%zmm7, %%zmm7                         \n\t"
                         "vpaddd 0x100(%%rax, %%rbx), %%zmm9, %%zmm9                         \n\t"
                         "vpaddd 0x140(%%rax, %%rbx), %%zmm11, %%zmm11                         \n\t"
                         "vpaddd 0x180(%%rax, %%rbx), %%zmm13, %%zmm13                         \n\t"
                         "vpaddd 0x1C0(%%rax, %%rbx), %%zmm15, %%zmm15                         \n\t"
                         "vpaddd 0x200(%%rax, %%rbx), %%zmm17, %%zmm17                         \n\t"
                         "vpaddd 0x240(%%rax, %%rbx), %%zmm19, %%zmm19                         \n\t"
                         "vpaddd 0x280(%%rax, %%rbx), %%zmm21, %%zmm21                         \n\t"
                         "vpaddd 0x2C0(%%rax, %%rbx), %%zmm23, %%zmm23                         \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                      \n\t"
                         "je 1f                                             \n\t"
                         relu24Regs(%%zmm)

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 2f      \n\t"
                         convert24RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         "vmovups %%zmm0, (%%rax)                             \n\t"
                         "vmovups %%zmm2, 0x40(%%rax)                         \n\t"
                         "vmovups %%zmm4, 0x80(%%rax)                         \n\t"
                         "vmovups %%zmm6, 0xC0(%%rax)                         \n\t"
                         "vmovups %%zmm8, 0x100(%%rax)                         \n\t"
                         "vmovups %%zmm10, 0x140(%%rax)                         \n\t"
                         "vmovups %%zmm12, 0x180(%%rax)                         \n\t"
                         "vmovups %%zmm14, 0x1C0(%%rax)                         \n\t"
                         "vmovups %%zmm16, 0x200(%%rax)                         \n\t"
                         "vmovups %%zmm18, 0x240(%%rax)                         \n\t"
                         "vmovups %%zmm20, 0x280(%%rax)                         \n\t"
                         "vmovups %%zmm22, 0x2C0(%%rax)                         \n\t"
                         "vmovups %%zmm1, (%%rax, %%rbx)                             \n\t"
                         "vmovups %%zmm3, 0x40(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm5, 0x80(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm7, 0xC0(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm9, 0x100(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm11, 0x140(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm13, 0x180(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm15, 0x1C0(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm17, 0x200(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm19, 0x240(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm21, 0x280(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm23, 0x2C0(%%rax, %%rbx)                         \n\t"
                         :
                         : [output] "r" (c.output), [ostepC16] "r" (c.ostepC16), [flags] "r" (c.flags), [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6",
                           "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",
                           "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22",
                           "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30",
                           "%zmm31", "memory", "cc");
}

void Avx512Conv1x1Kernel6x32(ConvController &c) {
    convKernelForLoopXx32(12, 6)

    __asm__ __volatile__("movq %[output], %%rax                                      \n\t"
                         "movq %[ostepC16], %%rbx                                      \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0x1, %%rcx                                  \n\t"
                         "je 0f                                             \n\t"
                         "vpaddd (%%rax), %%zmm0, %%zmm0                             \n\t"
                         "vpaddd 0x40(%%rax), %%zmm2, %%zmm2                         \n\t"
                         "vpaddd 0x80(%%rax), %%zmm4, %%zmm4                         \n\t"
                         "vpaddd 0xC0(%%rax), %%zmm6, %%zmm6                         \n\t"
                         "vpaddd 0x100(%%rax), %%zmm8, %%zmm8                         \n\t"
                         "vpaddd 0x140(%%rax), %%zmm10, %%zmm10                         \n\t"
                         "vpaddd (%%rax, %%rbx), %%zmm1, %%zmm1                             \n\t"
                         "vpaddd 0x40(%%rax, %%rbx), %%zmm3, %%zmm3                         \n\t"
                         "vpaddd 0x80(%%rax, %%rbx), %%zmm5, %%zmm5                         \n\t"
                         "vpaddd 0xC0(%%rax, %%rbx), %%zmm7, %%zmm7                         \n\t"
                         "vpaddd 0x100(%%rax, %%rbx), %%zmm9, %%zmm9                         \n\t"
                         "vpaddd 0x140(%%rax, %%rbx), %%zmm11, %%zmm11                         \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                      \n\t"
                         "je 1f                                             \n\t"
                         relu12Regs(%%zmm)

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 2f      \n\t"
                         convert12RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         "vmovups %%zmm0, (%%rax)                             \n\t"
                         "vmovups %%zmm2, 0x40(%%rax)                         \n\t"
                         "vmovups %%zmm4, 0x80(%%rax)                         \n\t"
                         "vmovups %%zmm6, 0xC0(%%rax)                         \n\t"
                         "vmovups %%zmm8, 0x100(%%rax)                         \n\t"
                         "vmovups %%zmm10, 0x140(%%rax)                         \n\t"
                         "vmovups %%zmm1, (%%rax, %%rbx)                             \n\t"
                         "vmovups %%zmm3, 0x40(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm5, 0x80(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm7, 0xC0(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm9, 0x100(%%rax, %%rbx)                         \n\t"
                         "vmovups %%zmm11, 0x140(%%rax, %%rbx)                         \n\t"
                         :
                         : [output] "r" (c.output), [ostepC16] "r" (c.ostepC16), [flags] "r" (c.flags), [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6",
                           "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",
                           "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22",
                           "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30",
                           "%zmm31", "memory", "cc");
}

void Avx512Conv1x1Kernel1x32(ConvController &c) {
    convKernelForLoopXx32(2, 1)

    __asm__ __volatile__("movq %[output], %%rax                                      \n\t"
                         "movq %[ostepC16], %%rbx                                      \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0x1, %%rcx                                  \n\t"
                         "je 0f                                             \n\t"
                         "vpaddd (%%rax), %%zmm0, %%zmm0                             \n\t"
                         "vpaddd (%%rax, %%rbx), %%zmm1, %%zmm1                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                      \n\t"
                         "je 1f                                             \n\t"
                         relu2Regs(%%zmm)

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 2f      \n\t"
                         convert2RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         "vmovups %%zmm0, (%%rax)                             \n\t"
                         "vmovups %%zmm1, (%%rax, %%rbx)                             \n\t"
                         :
                         : [output] "r" (c.output), [ostepC16] "r" (c.ostepC16), [flags] "r" (c.flags), [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6",
                           "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",
                           "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22",
                           "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30",
                           "%zmm31", "memory", "cc");
}

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

#ifdef _USE_AVX512_VNNI
#define convKernel24x16c4(input, freg0, off0, preg0, rtype) \
    "vpbroadcastd ("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0x10("#input"), "#rtype"27         \n\t" \
    "vpbroadcastd 0x20("#input"), "#rtype"28                     \n\t" \
    "vpbroadcastd 0x30("#input"), "#rtype"29                     \n\t" \
    "vpbroadcastd 0x40("#input"), "#rtype"30                     \n\t" \
    "vpbroadcastd 0x50("#input"), "#rtype"31                     \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"                             \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"0              \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"1              \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"2              \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"3              \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"4              \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"5              \n\t" \
    "vpbroadcastd 0x60("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0x70("#input"), "#rtype"27         \n\t" \
    "vpbroadcastd 0x80("#input"), "#rtype"28                     \n\t" \
    "vpbroadcastd 0x90("#input"), "#rtype"29                     \n\t" \
    "vpbroadcastd 0xA0("#input"), "#rtype"30                     \n\t" \
    "vpbroadcastd 0xB0("#input"), "#rtype"31                     \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"6              \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"7              \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"8              \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"9              \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"10              \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"11              \n\t" \
    "vpbroadcastd 0xC0("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0xD0("#input"), "#rtype"27         \n\t" \
    "vpbroadcastd 0xE0("#input"), "#rtype"28                     \n\t" \
    "vpbroadcastd 0xF0("#input"), "#rtype"29                     \n\t" \
    "vpbroadcastd 0x100("#input"), "#rtype"30                     \n\t" \
    "vpbroadcastd 0x110("#input"), "#rtype"31                     \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"12              \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"13              \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"14              \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"15              \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"16              \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"17              \n\t" \
    "vpbroadcastd 0x120("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0x130("#input"), "#rtype"27         \n\t" \
    "vpbroadcastd 0x140("#input"), "#rtype"28                     \n\t" \
    "vpbroadcastd 0x150("#input"), "#rtype"29                     \n\t" \
    "vpbroadcastd 0x160("#input"), "#rtype"30                     \n\t" \
    "vpbroadcastd 0x170("#input"), "#rtype"31                     \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"18              \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"19              \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"20              \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"21              \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"22              \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"23              \n\t"

#define convKernel12x16c4(input, freg0, off0, preg0, rtype) \
    "vpbroadcastd ("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0x10("#input"), "#rtype"27         \n\t" \
    "vpbroadcastd 0x20("#input"), "#rtype"28                     \n\t" \
    "vpbroadcastd 0x30("#input"), "#rtype"29                     \n\t" \
    "vpbroadcastd 0x40("#input"), "#rtype"30                     \n\t" \
    "vpbroadcastd 0x50("#input"), "#rtype"31                     \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"                             \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"0              \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"1              \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"2              \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"3              \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"4              \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"5              \n\t" \
    "vpbroadcastd 0x60("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0x70("#input"), "#rtype"27         \n\t" \
    "vpbroadcastd 0x80("#input"), "#rtype"28                     \n\t" \
    "vpbroadcastd 0x90("#input"), "#rtype"29                     \n\t" \
    "vpbroadcastd 0xA0("#input"), "#rtype"30                     \n\t" \
    "vpbroadcastd 0xB0("#input"), "#rtype"31                     \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"6              \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"7              \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"8              \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"9              \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"10              \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"11              \n\t"

#define convKernel1x16c4(input, freg0, off0, preg0, rtype) \
    "vpbroadcastd ("#input"), "#rtype"26                     \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"                             \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"0              \n\t"
#else

#define convKernel24x16c4_3(input, freg0, off0, preg0, rtype) \
    "vpbroadcastd ("#input"), "#rtype"25                   \n\t" \
    "vpbroadcastd 0x10("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0x20("#input"), "#rtype"27                     \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30              \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28              \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29              \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30              \n\t" \
    "vpbroadcastd 0x30("#input"), "#rtype"25                   \n\t" \
    "vpbroadcastd 0x40("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0x50("#input"), "#rtype"27                     \n\t" \
    "vpaddd "#rtype"0, "#rtype"28, "#rtype"0              \n\t" \
    "vpaddd "#rtype"1, "#rtype"29, "#rtype"1              \n\t" \
    "vpaddd "#rtype"2, "#rtype"30, "#rtype"2              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30              \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28              \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29              \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30              \n\t" \
    "vpbroadcastd 0x60("#input"), "#rtype"25                   \n\t" \
    "vpbroadcastd 0x70("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0x80("#input"), "#rtype"27                     \n\t" \
    "vpaddd "#rtype"3, "#rtype"28, "#rtype"3              \n\t" \
    "vpaddd "#rtype"4, "#rtype"29, "#rtype"4              \n\t" \
    "vpaddd "#rtype"5, "#rtype"30, "#rtype"5              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30              \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28              \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29              \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30              \n\t" \
    "vpbroadcastd 0x90("#input"), "#rtype"25                   \n\t" \
    "vpbroadcastd 0xA0("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0xB0("#input"), "#rtype"27                     \n\t" \
    "vpaddd "#rtype"6, "#rtype"28, "#rtype"6              \n\t" \
    "vpaddd "#rtype"7, "#rtype"29, "#rtype"7              \n\t" \
    "vpaddd "#rtype"8, "#rtype"30, "#rtype"8              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30              \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28              \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29              \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30              \n\t" \
    "vpbroadcastd 0xC0("#input"), "#rtype"25                   \n\t" \
    "vpbroadcastd 0xD0("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0xE0("#input"), "#rtype"27                     \n\t" \
    "vpaddd "#rtype"9, "#rtype"28, "#rtype"9              \n\t" \
    "vpaddd "#rtype"10, "#rtype"29, "#rtype"10              \n\t" \
    "vpaddd "#rtype"11, "#rtype"30, "#rtype"11              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30              \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28              \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29              \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30              \n\t" \
    "vpbroadcastd 0xF0("#input"), "#rtype"25                   \n\t" \
    "vpbroadcastd 0x100("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0x110("#input"), "#rtype"27                     \n\t" \
    "vpaddd "#rtype"12, "#rtype"28, "#rtype"12              \n\t" \
    "vpaddd "#rtype"13, "#rtype"29, "#rtype"13              \n\t" \
    "vpaddd "#rtype"14, "#rtype"30, "#rtype"14              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30              \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28              \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29              \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30              \n\t" \
    "vpbroadcastd 0x120("#input"), "#rtype"25                   \n\t" \
    "vpbroadcastd 0x130("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0x140("#input"), "#rtype"27                     \n\t" \
    "vpaddd "#rtype"15, "#rtype"28, "#rtype"15              \n\t" \
    "vpaddd "#rtype"16, "#rtype"29, "#rtype"16              \n\t" \
    "vpaddd "#rtype"17, "#rtype"30, "#rtype"17              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30              \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28              \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29              \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30              \n\t" \
    "vpbroadcastd 0x150("#input"), "#rtype"25                   \n\t" \
    "vpbroadcastd 0x160("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0x170("#input"), "#rtype"27                     \n\t" \
    "vpaddd "#rtype"18, "#rtype"28, "#rtype"18              \n\t" \
    "vpaddd "#rtype"19, "#rtype"29, "#rtype"19              \n\t" \
    "vpaddd "#rtype"20, "#rtype"30, "#rtype"20              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30              \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28              \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29              \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30              \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"                             \n\t" \
    "vpaddd "#rtype"21, "#rtype"28, "#rtype"21              \n\t" \
    "vpaddd "#rtype"22, "#rtype"29, "#rtype"22              \n\t" \
    "vpaddd "#rtype"23, "#rtype"30, "#rtype"23              \n\t"

#define convKernel12x16c4_3(input, freg0, off0, preg0, rtype) \
    "vpbroadcastd ("#input"), "#rtype"25                   \n\t" \
    "vpbroadcastd 0x10("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0x20("#input"), "#rtype"27                     \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30              \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28              \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29              \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30              \n\t" \
    "vpbroadcastd 0x30("#input"), "#rtype"25                   \n\t" \
    "vpbroadcastd 0x40("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0x50("#input"), "#rtype"27                     \n\t" \
    "vpaddd "#rtype"0, "#rtype"28, "#rtype"0              \n\t" \
    "vpaddd "#rtype"1, "#rtype"29, "#rtype"1              \n\t" \
    "vpaddd "#rtype"2, "#rtype"30, "#rtype"2              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30              \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28              \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29              \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30              \n\t" \
    "vpbroadcastd 0x60("#input"), "#rtype"25                   \n\t" \
    "vpbroadcastd 0x70("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0x80("#input"), "#rtype"27                     \n\t" \
    "vpaddd "#rtype"3, "#rtype"28, "#rtype"3              \n\t" \
    "vpaddd "#rtype"4, "#rtype"29, "#rtype"4              \n\t" \
    "vpaddd "#rtype"5, "#rtype"30, "#rtype"5              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30              \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28              \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29              \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30              \n\t" \
    "vpbroadcastd 0x90("#input"), "#rtype"25                   \n\t" \
    "vpbroadcastd 0xA0("#input"), "#rtype"26                     \n\t" \
    "vpbroadcastd 0xB0("#input"), "#rtype"27                     \n\t" \
    "vpaddd "#rtype"6, "#rtype"28, "#rtype"6              \n\t" \
    "vpaddd "#rtype"7, "#rtype"29, "#rtype"7              \n\t" \
    "vpaddd "#rtype"8, "#rtype"30, "#rtype"8              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29              \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30              \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28              \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29              \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30              \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"                             \n\t" \
    "vpaddd "#rtype"9, "#rtype"28, "#rtype"9              \n\t" \
    "vpaddd "#rtype"10, "#rtype"29, "#rtype"10              \n\t" \
    "vpaddd "#rtype"11, "#rtype"30, "#rtype"11              \n\t"

#define convKernel1x16c4_3(input, freg0, off0, preg0, rtype) \
    "vpbroadcastd ("#input"), "#rtype"25                   \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28              \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28              \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"                             \n\t" \
    "vpaddd "#rtype"0, "#rtype"28, "#rtype"0              \n\t"

#define convKernel24x16c4(input, freg0, off0, preg0, rtype) \
    convKernel24x16c4_3(input, rtype##24, off0, rtype##25, rtype)

#define convKernel12x16c4(input, freg0, off0, preg0, rtype) \
    convKernel12x16c4_3(input, rtype##24, off0, rtype##25, rtype)

#define convKernel1x16c4(input, freg0, off0, preg0, rtype) \
    convKernel1x16c4_3(input, rtype##24, off0, rtype##25, rtype)
#endif

#define convKernelForLoopXx16(rnum, wsize, rtype, off0, off1, off2, off3, off4) \
     __asm__ __volatile__("vmovups (%[filter]), "#rtype"24                             \n\t" \
                          "addq $"#off1", %[filter]                                    \n\t" \
                          "mov $1, %%eax \n\t" \
                          "vmovd %%eax, %%xmm0                    \n\t" \
                          "vpbroadcastw %%xmm0, "#rtype"31            \n\t" \
                          "movq %[flags], %%rax          \n\t" \
                          "andq $0x1, %%rax          \n\t" \
                          "jne 0f                                         \n\t" \
                          load16BiasTo##rnum##Regs(%[bias], rtype) \
                          "cmpq $0x10, %%rcx          \n\t" \
                          "jl 4f            \n\t" \
                          "jmp 1f          \n\t" \
                          ".align 16                                         \n\t" \
                          "0:                                                \n\t" \
                          clear##rnum##Regs(rtype) \
                          "cmpq $0x10, %%rcx          \n\t" \
                          "jl 4f            \n\t" \
                          ".align 16                                         \n\t" \
                          "1:                                                \n\t" \
                          "movq %[input], %%rax  \n\t" \
                          convKernel##wsize##x16c4(%%rax, rtype##24, off0, rtype##25, rtype) \
                          "addq $0x4, %%rax  \n\t" \
                          convKernel##wsize##x16c4(%%rax, rtype##25, off1, rtype##24, rtype) \
                          "addq $0x4, %%rax  \n\t" \
                          convKernel##wsize##x16c4(%%rax, rtype##24, off2, rtype##25, rtype) \
                          "addq $0x4, %%rax  \n\t" \
                          convKernel##wsize##x16c4(%%rax, rtype##25, off3, rtype##24, rtype) \
                          "addq $"#off4", %[filter]                                    \n\t" \
                          "addq %[fStep], %[input]                                    \n\t" \
                          "subq $0x10, %%rcx                                         \n\t" \
                          "cmpq $0x10, %%rcx                                         \n\t" \
                          "jge 1b                                             \n\t" \
                          "subq %[fStep], %[input]                                    \n\t" \
                          "addq %[f8Step], %[input]                                    \n\t" \
                          ".align 16                                         \n\t" \
                          "4:                                                \n\t" \
                          : "+c" (c.ic), [input] "+r" (c.input), [filter] "+r" (c.filter) \
                          : [bias] "r" (c.bias), [kh] "r" (c.kh), [kw] "r" (c.kw), \
                            [fStep] "r" (c.fStep), [flags] "r" (c.flags),  \
                            [f8Step] "r" (c.f8Step) \
                          : "%rax", "%rbx", "%r9", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", \
                            "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",  \
                            "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", \
                            "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", \
                            "%zmm31", "memory", "cc"); \
     if (c.ic > 0) { \
         __asm__ __volatile__("cmpq $0x8, %%rcx          \n\t" \
                              "jl 2f            \n\t" \
                              "subq $0x8, %%rcx          \n\t" \
                              "movq %[input], %%rax  \n\t" \
                              convKernel##wsize##x16c4(%%rax, rtype##24, off0, rtype##25, rtype) \
                              "addq $0x4, %%rax  \n\t" \
                              convKernel##wsize##x16c4(%%rax, rtype##25, off1, rtype##24, rtype) \
                              "addq $"#off2", %[filter]                                    \n\t" \
                              "addq %[f4Step], %[input]                                    \n\t" \
                              ".align 16                                         \n\t" \
                              "2:                                                \n\t" \
                              "cmpq $0x4, %%rcx          \n\t" \
                              "jl 5f            \n\t" \
                              convKernel##wsize##x16c4(%[input], rtype##24, off0, rtype##25, rtype) \
                              ".align 16                                         \n\t" \
                              "5:                                             \n\t" \
                              : "+c" (c.ic) \
                              : [input] "r" (c.input), [filter] "r" (c.filter), [bias] "r" (c.bias), [kh] "r" (c.kh), [kw] "r" (c.kw), \
                                [stepC16] "r" (c.stepC16), [dilateW] "r" (c.dilateW), \
                                [dilateH] "r" (c.dilateH), [fStep] "r" (c.fStep), \
                                [f4Step] "r" (c.f4Step) \
                              : "%rax", "%rbx", "%r9", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", \
                                "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",  \
                                "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", \
                                "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", \
                                "%zmm31", "memory", "cc"); \
    }

void Avx512Conv1x1Kernel24x16(ConvController &c) {
    convKernelForLoopXx16(24, 24, %%zmm, 0x0, 0x40, 0x80, 0xC0, 0x100)

    __asm__ __volatile__("movq %[output], %%rax                                      \n\t"
                         "movq %[ostepC16], %%rbx                                      \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0x1, %%rcx                                  \n\t"
                         "je 0f                                             \n\t"
                         "vpaddd (%%rax),      %%zmm0, %%zmm0                             \n\t"
                         "vpaddd 0x40(%%rax),  %%zmm1, %%zmm1                             \n\t"
                         "vpaddd 0x80(%%rax),  %%zmm2, %%zmm2                         \n\t"
                         "vpaddd 0xC0(%%rax),  %%zmm3, %%zmm3                         \n\t"
                         "vpaddd 0x100(%%rax), %%zmm4, %%zmm4                         \n\t"
                         "vpaddd 0x140(%%rax), %%zmm5, %%zmm5                         \n\t"
                         "vpaddd 0x180(%%rax), %%zmm6, %%zmm6                         \n\t"
                         "vpaddd 0x1C0(%%rax), %%zmm7, %%zmm7                         \n\t"
                         "vpaddd 0x200(%%rax), %%zmm8, %%zmm8                         \n\t"
                         "vpaddd 0x240(%%rax), %%zmm9, %%zmm9                         \n\t"
                         "vpaddd 0x280(%%rax), %%zmm10, %%zmm10                         \n\t"
                         "vpaddd 0x2C0(%%rax), %%zmm11, %%zmm11                         \n\t"
                         "vpaddd 0x300(%%rax), %%zmm12, %%zmm12                         \n\t"
                         "vpaddd 0x340(%%rax), %%zmm13, %%zmm13                         \n\t"
                         "vpaddd 0x380(%%rax), %%zmm14, %%zmm14                         \n\t"
                         "vpaddd 0x3C0(%%rax), %%zmm15, %%zmm15                         \n\t"
                         "vpaddd 0x400(%%rax), %%zmm16, %%zmm16                         \n\t"
                         "vpaddd 0x440(%%rax), %%zmm17, %%zmm17                         \n\t"
                         "vpaddd 0x480(%%rax), %%zmm18, %%zmm18                         \n\t"
                         "vpaddd 0x4C0(%%rax), %%zmm19, %%zmm19                         \n\t"
                         "vpaddd 0x500(%%rax), %%zmm20, %%zmm20                         \n\t"
                         "vpaddd 0x540(%%rax), %%zmm21, %%zmm21                         \n\t"
                         "vpaddd 0x580(%%rax), %%zmm22, %%zmm22                         \n\t"
                         "vpaddd 0x5C0(%%rax), %%zmm23, %%zmm23                         \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                      \n\t"
                         "je 1f                                             \n\t"
                         relu24Regs(%%zmm)

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 2f      \n\t"
                         convert24RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         "vmovups %%zmm0,  (%%rax)                     \n\t"
                         "vmovups %%zmm1,  0x40(%%rax)                        \n\t"
                         "vmovups %%zmm2,  0x80(%%rax)                 \n\t"
                         "vmovups %%zmm3,  0xC0(%%rax)                        \n\t"
                         "vmovups %%zmm4,  0x100(%%rax)                 \n\t"
                         "vmovups %%zmm5,  0x140(%%rax)                        \n\t"
                         "vmovups %%zmm6,  0x180(%%rax)                 \n\t"
                         "vmovups %%zmm7,  0x1C0(%%rax)                        \n\t"
                         "vmovups %%zmm8,  0x200(%%rax)                  \n\t"
                         "vmovups %%zmm9,  0x240(%%rax)                         \n\t"
                         "vmovups %%zmm10, 0x280(%%rax)                  \n\t"
                         "vmovups %%zmm11, 0x2C0(%%rax)                         \n\t"
                         "vmovups %%zmm12, 0x300(%%rax)                  \n\t"
                         "vmovups %%zmm13, 0x340(%%rax)                         \n\t"
                         "vmovups %%zmm14, 0x380(%%rax)                  \n\t"
                         "vmovups %%zmm15, 0x3C0(%%rax)                         \n\t"
                         "vmovups %%zmm16, 0x400(%%rax)                  \n\t"
                         "vmovups %%zmm17, 0x440(%%rax)                         \n\t"
                         "vmovups %%zmm18, 0x480(%%rax)                  \n\t"
                         "vmovups %%zmm19, 0x4C0(%%rax)                         \n\t"
                         "vmovups %%zmm20, 0x500(%%rax)                  \n\t"
                         "vmovups %%zmm21, 0x540(%%rax)                         \n\t"
                         "vmovups %%zmm22, 0x580(%%rax)                  \n\t"
                         "vmovups %%zmm23, 0x5C0(%%rax)                         \n\t"
                         :
                         : [output] "r" (c.output), [ostepC16] "r" (c.ostepC16), [flags] "r" (c.flags), [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6",
                           "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",
                           "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22",
                           "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30",
                           "%zmm31", "memory", "cc");
}

void Avx512Conv1x1Kernel12x16(ConvController &c) {
     convKernelForLoopXx16(12, 12, %%zmm, 0x0, 0x40, 0x80, 0xC0, 0x100)

    __asm__ __volatile__("movq %[output], %%rax                                      \n\t"
                         "movq %[ostepC16], %%rbx                                      \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0x1, %%rcx                                  \n\t"
                         "je 0f                                             \n\t"
                         "vpaddd (%%rax),      %%zmm0, %%zmm0                             \n\t"
                         "vpaddd 0x40(%%rax),  %%zmm1, %%zmm1                             \n\t"
                         "vpaddd 0x80(%%rax),  %%zmm2, %%zmm2                         \n\t"
                         "vpaddd 0xC0(%%rax),  %%zmm3, %%zmm3                         \n\t"
                         "vpaddd 0x100(%%rax), %%zmm4, %%zmm4                         \n\t"
                         "vpaddd 0x140(%%rax), %%zmm5, %%zmm5                         \n\t"
                         "vpaddd 0x180(%%rax), %%zmm6, %%zmm6                         \n\t"
                         "vpaddd 0x1C0(%%rax), %%zmm7, %%zmm7                         \n\t"
                         "vpaddd 0x200(%%rax), %%zmm8, %%zmm8                         \n\t"
                         "vpaddd 0x240(%%rax), %%zmm9, %%zmm9                         \n\t"
                         "vpaddd 0x280(%%rax), %%zmm10, %%zmm10                         \n\t"
                         "vpaddd 0x2C0(%%rax), %%zmm11, %%zmm11                         \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                      \n\t"
                         "je 1f                                             \n\t"
                         relu12Regs(%%zmm)

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 2f      \n\t"
                         convert12RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         "vmovups %%zmm0,  (%%rax)                     \n\t"
                         "vmovups %%zmm1,  0x40(%%rax)                        \n\t"
                         "vmovups %%zmm2,  0x80(%%rax)                 \n\t"
                         "vmovups %%zmm3,  0xC0(%%rax)                        \n\t"
                         "vmovups %%zmm4,  0x100(%%rax)                 \n\t"
                         "vmovups %%zmm5,  0x140(%%rax)                        \n\t"
                         "vmovups %%zmm6,  0x180(%%rax)                 \n\t"
                         "vmovups %%zmm7,  0x1C0(%%rax)                        \n\t"
                         "vmovups %%zmm8,  0x200(%%rax)                  \n\t"
                         "vmovups %%zmm9,  0x240(%%rax)                         \n\t"
                         "vmovups %%zmm10, 0x280(%%rax)                  \n\t"
                         "vmovups %%zmm11, 0x2C0(%%rax)                         \n\t"
                         :
                         : [output] "r" (c.output), [ostepC16] "r" (c.ostepC16), [flags] "r" (c.flags), [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6",
                           "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",
                           "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22",
                           "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30",
                           "%zmm31", "memory", "cc");
}

void Avx512Conv1x1Kernel1x16(ConvController &c) {
    convKernelForLoopXx16(1, 1, %%zmm, 0x0, 0x40, 0x80, 0xC0, 0x100)

    __asm__ __volatile__("movq %[output], %%rax                                      \n\t"
                         "movq %[ostepC16], %%rbx                                      \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0x1, %%rcx                                  \n\t"
                         "je 0f                                             \n\t"
                         "vpaddd (%%rax),      %%zmm0, %%zmm0                             \n\t"
                         "vpaddd 0x40(%%rax),  %%zmm1, %%zmm1                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                      \n\t"
                         "je 1f                                             \n\t"
                         reluReg(%%zmm)

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 2f      \n\t"
                         convertRegI32ToF32(%[scale], %%zmm)

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         "vmovups %%zmm0,  (%%rax)                     \n\t"
                         :
                         : [output] "r" (c.output), [ostepC16] "r" (c.ostepC16), [flags] "r" (c.flags), [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6",
                           "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",
                           "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22",
                           "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30",
                           "%zmm31", "memory", "cc");
}

void Avx512Conv1x1Kernel24x8(ConvController &c) {
    convKernelForLoopXx16(24, 24, %%ymm, 0x0, 0x20, 0x40, 0x60, 0x80)

    __asm__ __volatile__("movq %[output], %%rax                                      \n\t"
                         "movq %[ostepC16], %%rbx                                      \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0x1, %%rcx                                  \n\t"
                         "je 0f                                             \n\t"
                         "vpaddd (%%rax),      %%ymm0,  %%ymm0                             \n\t"
                         "vpaddd 0x20(%%rax),  %%ymm1,  %%ymm1                             \n\t"
                         "vpaddd 0x40(%%rax),  %%ymm2,  %%ymm2                         \n\t"
                         "vpaddd 0x60(%%rax),  %%ymm3,  %%ymm3                         \n\t"
                         "vpaddd 0x80(%%rax),  %%ymm4,  %%ymm4                         \n\t"
                         "vpaddd 0xA0(%%rax),  %%ymm5,  %%ymm5                         \n\t"
                         "vpaddd 0xC0(%%rax),  %%ymm6,  %%ymm6                         \n\t"
                         "vpaddd 0xE0(%%rax),  %%ymm7,  %%ymm7                         \n\t"
                         "vpaddd 0x100(%%rax), %%ymm8,  %%ymm8                         \n\t"
                         "vpaddd 0x120(%%rax), %%ymm9,  %%ymm9                         \n\t"
                         "vpaddd 0x140(%%rax), %%ymm10, %%ymm10                         \n\t"
                         "vpaddd 0x160(%%rax), %%ymm11, %%ymm11                         \n\t"
                         "vpaddd 0x180(%%rax), %%ymm12, %%ymm12                         \n\t"
                         "vpaddd 0x1A0(%%rax), %%ymm13, %%ymm13                         \n\t"
                         "vpaddd 0x1C0(%%rax), %%ymm14, %%ymm14                         \n\t"
                         "vpaddd 0x1E0(%%rax), %%ymm15, %%ymm15                         \n\t"
                         "vpaddd 0x200(%%rax), %%ymm16, %%ymm16                         \n\t"
                         "vpaddd 0x220(%%rax), %%ymm17, %%ymm17                         \n\t"
                         "vpaddd 0x240(%%rax), %%ymm18, %%ymm18                         \n\t"
                         "vpaddd 0x260(%%rax), %%ymm19, %%ymm19                         \n\t"
                         "vpaddd 0x280(%%rax), %%ymm20, %%ymm20                         \n\t"
                         "vpaddd 0x2A0(%%rax), %%ymm21, %%ymm21                         \n\t"
                         "vpaddd 0x2C0(%%rax), %%ymm22, %%ymm22                         \n\t"
                         "vpaddd 0x2E0(%%rax), %%ymm23, %%ymm23                         \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                      \n\t"
                         "je 1f                                             \n\t"
                         relu24Regs(%%ymm)

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 2f      \n\t"
                         convert24RegsI32ToF32(%[scale], %%ymm)

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         "vmovups %%ymm0,  (%%rax)                    \n\t"
                         "vmovups %%ymm1,  0x20(%%rax)                       \n\t"
                         "vmovups %%ymm2,  0x40(%%rax)                \n\t"
                         "vmovups %%ymm3,  0x60(%%rax)                       \n\t"
                         "vmovups %%ymm4,  0x80(%%rax)                 \n\t"
                         "vmovups %%ymm5,  0xA0(%%rax)                        \n\t"
                         "vmovups %%ymm6,  0xC0(%%rax)                 \n\t"
                         "vmovups %%ymm7,  0xE0(%%rax)                        \n\t"
                         "vmovups %%ymm8,  0x100(%%rax)                  \n\t"
                         "vmovups %%ymm9,  0x120(%%rax)                         \n\t"
                         "vmovups %%ymm10, 0x140(%%rax)                  \n\t"
                         "vmovups %%ymm11, 0x160(%%rax)                         \n\t"
                         "vmovups %%ymm12, 0x180(%%rax)                  \n\t"
                         "vmovups %%ymm13, 0x1A0(%%rax)                         \n\t"
                         "vmovups %%ymm14, 0x1C0(%%rax)                  \n\t"
                         "vmovups %%ymm15, 0x1E0(%%rax)                         \n\t"
                         "vmovups %%ymm16, 0x200(%%rax)                  \n\t"
                         "vmovups %%ymm17, 0x220(%%rax)                         \n\t"
                         "vmovups %%ymm18, 0x240(%%rax)                  \n\t"
                         "vmovups %%ymm19, 0x260(%%rax)                         \n\t"
                         "vmovups %%ymm20, 0x280(%%rax)                  \n\t"
                         "vmovups %%ymm21, 0x2A0(%%rax)                         \n\t"
                         "vmovups %%ymm22, 0x2C0(%%rax)                  \n\t"
                         "vmovups %%ymm23, 0x2E0(%%rax)                         \n\t"
                         :
                         : [output] "r" (c.output), [ostepC16] "r" (c.ostepC16), [flags] "r" (c.flags), [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6",
                           "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14",
                           "%ymm15", "%ymm16", "%ymm17", "%ymm18", "%ymm19", "%ymm20", "%ymm21", "%ymm22",
                           "%ymm23", "%ymm24", "%ymm25", "%ymm26", "%ymm27", "%ymm28", "%ymm29", "%ymm30",
                           "%zmm31", "memory", "cc");
}

void Avx512Conv1x1Kernel12x8(ConvController &c) {
    convKernelForLoopXx16(12, 12, %%ymm, 0x0, 0x20, 0x40, 0x60, 0x80)

    __asm__ __volatile__("movq %[output], %%rax                                      \n\t"
                         "movq %[ostepC16], %%rbx                                      \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0x1, %%rcx                                  \n\t"
                         "je 0f                                             \n\t"
                         "vpaddd (%%rax),      %%ymm0,  %%ymm0                             \n\t"
                         "vpaddd 0x20(%%rax),  %%ymm1,  %%ymm1                             \n\t"
                         "vpaddd 0x40(%%rax),  %%ymm2,  %%ymm2                         \n\t"
                         "vpaddd 0x60(%%rax),  %%ymm3,  %%ymm3                         \n\t"
                         "vpaddd 0x80(%%rax),  %%ymm4,  %%ymm4                         \n\t"
                         "vpaddd 0xA0(%%rax),  %%ymm5,  %%ymm5                         \n\t"
                         "vpaddd 0xC0(%%rax),  %%ymm6,  %%ymm6                         \n\t"
                         "vpaddd 0xE0(%%rax),  %%ymm7,  %%ymm7                         \n\t"
                         "vpaddd 0x100(%%rax), %%ymm8,  %%ymm8                         \n\t"
                         "vpaddd 0x120(%%rax), %%ymm9,  %%ymm9                         \n\t"
                         "vpaddd 0x140(%%rax), %%ymm10, %%ymm10                         \n\t"
                         "vpaddd 0x160(%%rax), %%ymm11, %%ymm11                         \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                      \n\t"
                         "je 1f                                             \n\t"
                         relu12Regs(%%ymm)

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 2f      \n\t"
                         convert12RegsI32ToF32(%[scale], %%ymm)

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         "vmovups %%ymm0,  (%%rax)                    \n\t"
                         "vmovups %%ymm1,  0x20(%%rax)                       \n\t"
                         "vmovups %%ymm2,  0x40(%%rax)                \n\t"
                         "vmovups %%ymm3,  0x60(%%rax)                       \n\t"
                         "vmovups %%ymm4,  0x80(%%rax)                 \n\t"
                         "vmovups %%ymm5,  0xA0(%%rax)                        \n\t"
                         "vmovups %%ymm6,  0xC0(%%rax)                 \n\t"
                         "vmovups %%ymm7,  0xE0(%%rax)                        \n\t"
                         "vmovups %%ymm8,  0x100(%%rax)                  \n\t"
                         "vmovups %%ymm9,  0x120(%%rax)                         \n\t"
                         "vmovups %%ymm10, 0x140(%%rax)                  \n\t"
                         "vmovups %%ymm11, 0x160(%%rax)                         \n\t"
                         :
                         : [output] "r" (c.output), [ostepC16] "r" (c.ostepC16), [flags] "r" (c.flags), [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6",
                           "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14",
                           "%ymm15", "%ymm16", "%ymm17", "%ymm18", "%ymm19", "%ymm20", "%ymm21", "%ymm22",
                           "%ymm23", "%ymm24", "%ymm25", "%ymm26", "%ymm27", "%ymm28", "%ymm29", "%ymm30",
                           "%zmm31", "memory", "cc");
}

void Avx512Conv1x1Kernel1x8(ConvController &c) {
    convKernelForLoopXx16(1, 1, %%ymm, 0x0, 0x20, 0x40, 0x60, 0x80)

    __asm__ __volatile__("movq %[output], %%rax                                      \n\t"
                         "movq %[ostepC16], %%rbx                                      \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0x1, %%rcx                                  \n\t"
                         "je 0f                                             \n\t"
                         "vpaddd (%%rax), %%ymm0,  %%ymm0                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                      \n\t"
                         "je 1f                                             \n\t"
                         reluReg(%%ymm)

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 2f      \n\t"
                         convertRegI32ToF32(%[scale], %%ymm)

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         "vmovups %%ymm0,  (%%rax)                    \n\t"
                         :
                         : [output] "r" (c.output), [ostepC16] "r" (c.ostepC16), [flags] "r" (c.flags), [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6",
                           "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14",
                           "%ymm15", "%ymm16", "%ymm17", "%ymm18", "%ymm19", "%ymm20", "%ymm21", "%ymm22",
                           "%ymm23", "%ymm24", "%ymm25", "%ymm26", "%ymm27", "%ymm28", "%ymm29", "%ymm30",
                           "%zmm31", "memory", "cc");
}

// clang-format on
EE convolution_1x1_direct(TensorDesc inputDesc,
    UINT8 *inArray,
    TensorDesc filterDesc,
    const INT8 *filterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const I32 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *outArray,
    F32 *scale,
    ActivationParamSpec activationDesc)
{
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (idf == DF_MTK) {
        idf = DF_NCHW;
    }

    if (fdf != DF_NCHWC2NxC4 || (idf != DF_NCHWC16 && idf != DF_NCHW && idf != DF_NCHWC8)) {
        CHECK_STATUS(NOT_MATCH);
    }

    // get kernels
    U32 icSizeArray[3] = {4, 8, 16};
    U32 ocSizeArray[4] = {8, 16, 32, 48};
    U32 wSizeArray[4] = {24, 24, 12, 8};
    const kernelFunc kernel[4][3] = {
        {Avx512Conv1x1Kernel1x8, Avx512Conv1x1Kernel12x8, Avx512Conv1x1Kernel24x8},
        {Avx512Conv1x1Kernel1x16, Avx512Conv1x1Kernel12x16, Avx512Conv1x1Kernel24x16},
        {Avx512Conv1x1Kernel1x32, Avx512Conv1x1Kernel6x32, Avx512Conv1x1Kernel12x32},
        {Avx512Conv1x1Kernel1x48, Avx512Conv1x1Kernel4x48, Avx512Conv1x1Kernel8x48}};

    // get computing params
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingB = convParamSpec.padding_bottom;
    U32 paddingL = convParamSpec.padding_left;
    U32 paddingR = convParamSpec.padding_right;
    U32 dilateH = convParamSpec.dilatedRate_h;
    U32 dilateW = convParamSpec.dilatedRate_w;
    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;
    U32 ih_stride = (ih_pad + strideH - 1) / strideH;
    U32 iw_stride = (iw_pad + strideW - 1) / strideW;
    U32 ohow = oh * ow;
    UINT8 *output = (UINT8 *)outArray;

    CHECK_REQUIREMENT(paddingT == 0 && paddingB == 0 && paddingL == 0 && paddingR == 0);
    // infer block params

    // infer kernel params
    ConvController convCtl;
    convCtl.ostepC16 = oh * ow * 16 * 4;
    convCtl.dilateW = dilateW * SIMDW;
    convCtl.dilateH = (iw_stride - fw * dilateW + (dilateH - 1) * iw_stride) * SIMDW;
    convCtl.fStep = ih_stride * iw_stride * SIMDW;
    convCtl.stepC16 = 16;
    convCtl.kw = fw;
    convCtl.kh = fh;
    convCtl.scale = nullptr;
    U32 unrollOc = 48;
    if (fn % 32 == 0 && fn % 48 != 0) {
        unrollOc = 32;
    }

    // quantization
    F32 *scaleI = scale;
    F32 *scaleO = scale + 1;
    F32 *scaleF = scale + 2;
    if (idt != DT_U8_Q) {
        //quantize to U8_Q
        TensorDesc qDesc = inputDesc;
        qDesc.dt = DT_U8_Q;
        CHECK_STATUS(quantize_x86(inputDesc, (void *)inArray, &qDesc, tmp, scaleI));
        inArray = (UINT8 *)tmp;
        tmp = (void *)((U8 *)tmp + tensorNumBytes(qDesc));
    }
    *scaleO = scaleI[0] * scaleF[0];
    if (odt != DT_F32) {
        output = (UINT8 *)tmp;
        tmp = (void *)((U8 *)tmp + tensorNumElements(outputDesc) * bytesOf(DT_I32));
        outputDesc.dt = DT_I32;
    }
    F32 *factorPtr = nullptr;
    F32 factor = 0;
    if (scale != nullptr && odt == DT_F32) {
        factor = 1 / (*scaleO);
        factorPtr = &factor;
    }

    I32 *offsetC = (I32 *)tmp;
    tmp = (void *)((U8 *)tmp + oc * bytesOf(DT_I32));
    CHECK_STATUS(quantize_bias_offsetC((const void *)biasArray, biasDesc, DT_I32,
        (const void *)filterArray, filterDesc, scaleO, offsetC));
    filterArray += oc * 4;

    U32 oBytes = bytesOf(outputDesc.dt);
    UINT8 *tmpInput = (UINT8 *)tmp;
    if (idf != DF_NCHWC16) {
        tmp = (void *)((U8 *)tmp + ic * ih * iw);
    }
    UINT8 *useInput = (UINT8 *)tmp;
    for (U32 n = 0; n < in; ++n) {
        UINT8 *bInArray = inArray + n * ic * ih * iw;
        if (idf == DF_NCHWC16) {
            tmpInput = bInArray;
        } else if (idf == DF_NCHWC8) {
            PaddingNCHWC8ToNCHWC16(bInArray, tmpInput, inputDesc, convParamSpec);
        } else {
            PaddingNCHW2NCHWC16(bInArray, tmpInput, inputDesc, convParamSpec);
        }

        if (strideH > 1 || strideW > 1) {
            U32 ic16 = ic / 16;
            for (U32 hc = 0; hc < ih_stride * ic16; ++hc) {
                U32 c = hc / ih_stride;
                U32 h = hc % ih_stride;
                for (U32 w = 0; w < iw_stride; ++w) {
                    U32 nh = h * strideH;
                    U32 nw = w * strideW;
                    memcpy(
                        useInput + c * ih_stride * iw_stride * SIMDW + (h * iw_stride + w) * SIMDW,
                        tmpInput + c * ih_pad * iw_pad * SIMDW + (nh * iw_pad + nw) * SIMDW, SIMDW);
                }
            }
        } else {
            useInput = tmpInput;
        }

        U32 flags = 0;
        U32 icSize = 0;
        for (U32 icbb = 0; icbb < ic; icbb += icSize) {
            icSize = UNI_MIN(BLOCK_IC_DIM, ic - icbb);
            flags |= (icbb > 0);
            if (icbb == (int)ic - icSize) {
                flags |= U32(activationDesc.mode) << 2;
                convCtl.scale = factorPtr;
            }
            convCtl.flags = flags;
            U32 simdC = SIMDW;
            U32 simdOc = SIMDW;
            if (icSize < SIMDW) {
                simdC = icSizeArray[icSize >> 3];
            }
            U32 hwSize = 0;
            for (U32 hw = 0; hw < ohow; hw += hwSize) {
                U32 ocSize = 0;
                hwSize = UNI_MIN(BLOCK_HW_DIM, ohow - hw);
                for (U32 ocb = 0; ocb < oc; ocb += ocSize) {
                    ocSize = UNI_MIN(unrollOc, oc - ocb);
                    ocSize = ocSizeArray[ocSize >> 4];
                    simdOc = UNI_MIN(SIMDW, ocSize);
                    convCtl.bias = offsetC + ocb;
                    UINT8 *curI = useInput + icbb * ih_stride * iw_stride;
                    U32 wSize = 8;
                    U32 unrollW = wSizeArray[ocSize >> 4];
                    for (U32 ihw = hw; ihw < hw + hwSize; ihw += wSize) {
                        wSize = UNI_MIN(hw + hwSize - ihw, unrollW);
                        U32 idx = wSize * 2 / unrollW;
                        wSize = UNI_MAX(idx * unrollW / 2, 1);
                        U32 in_h = ihw / ow;
                        U32 in_w = ihw % ow;
                        convCtl.input = curI + in_h * iw_stride * simdC + in_w * simdC;
                        convCtl.output = output + ((n * oc + ocb) * ohow + ihw * simdOc) * oBytes;
                        convCtl.filter = filterArray + ocb * ic * fh * fw + ocSize * icbb * fh * fw;
                        if ((ic % 16 != 0) && (icbb == (int)ic - icSize)) {
                            U32 cx = (ic % 8 == 0) ? 8 : 4;
                            convCtl.f8Step =
                                convCtl.fStep - (in_h * iw_stride + in_w) * (SIMDW - cx);
                            convCtl.f4Step = convCtl.fStep / 2 - (in_h * iw_stride + in_w) * (8 - 4);
                        }
                        convCtl.ic = icSize;
                        kernel[ocSize >> 4][idx](convCtl);
                    }
                }
            }
        }
    }

    // quantization
    if (odt == DT_U8_Q) {
        F32 scales[2] = {-1, scaleO[0]};
        TensorDesc qDesc = outputDesc;
        qDesc.dt = DT_U8_Q;
        CHECK_STATUS(quantize_x86(outputDesc, (void *)output, &qDesc, (void *)outArray, scales));
        *scaleO = scales[0];
    }

    return SUCCESS;
}
