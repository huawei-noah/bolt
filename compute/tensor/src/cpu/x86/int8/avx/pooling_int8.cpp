// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/fp32/tensor_computing_fp32.h"

#define UNROLL_W 4

typedef void (*pooling_max_func)(
    const UINT8 **curI, UINT8 *curO, U32 kw, U32 kh, U32 iStep, U32 stride);
typedef void (*pooling_mean_func)(
    const UINT8 **curI, UINT8 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, I32 poolSize, UINT8 *index);

void pooling_c8_max_w4(const UINT8 **curI, UINT8 *curO, U32 kw, U32 kh, U32 iStep, U32 stride)
{
    __asm__ __volatile__("movq (%[input0]), %%xmm0                     \n\t"
                         "movq (%[input1]), %%xmm1                     \n\t"
                         "movq (%[input2]), %%xmm2                     \n\t"
                         "movq (%[input3]), %%xmm3                     \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"

                         "mov %[kw], %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "movq (%[input0]), %%xmm4                     \n\t"
                         "movq (%[input1]), %%xmm5                     \n\t"
                         "movq (%[input2]), %%xmm6                     \n\t"
                         "movq (%[input3]), %%xmm7                     \n\t"

                         "pmaxub %%xmm4, %%xmm0                     \n\t"
                         "pmaxub %%xmm5, %%xmm1                     \n\t"
                         "pmaxub %%xmm6, %%xmm2                     \n\t"
                         "pmaxub %%xmm7, %%xmm3                     \n\t"

                         "add $0x8, %[input0]                                   \n\t"
                         "add $0x8, %[input1]                                      \n\t"
                         "add $0x8, %[input2]                                     \n\t"
                         "add $0x8, %[input3]                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"

                         "add %[iStep], %[input0]                                   \n\t"
                         "add %[iStep], %[input1]                                      \n\t"
                         "add %[iStep], %[input2]                                     \n\t"
                         "add %[iStep], %[input3]                                      \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 0b                                             \n\t"

                         "movq %%xmm0, (%[output])                              \n\t"
                         "movq %%xmm1, 0x8(%[output])                          \n\t"
                         "movq %%xmm2, 0x10(%[output])                          \n\t"
                         "movq %%xmm3, 0x18(%[output])                          \n\t"
                         : [input0] "+r" (curI[0]),
                           [input1] "+r" (curI[1]),
                           [input2] "+r" (curI[2]),
                           [input3] "+r" (curI[3]),
                           "+b" (kh)
                         : [output] "r" (curO),
                           [kw] "r" (kw),
                           [iStep] "r" (int64_t(iStep))
                         : "%ecx", "%xmm0", "%xmm1", "%xmm2",
                           "%xmm3", "%xmm4", "%xmm5", "%xmm6",
                           "%xmm7", "memory", "cc");
}

void pooling_c8_max_w2(const UINT8 **curI, UINT8 *curO, U32 kw, U32 kh, U32 iStep, U32 stride)
{
    __asm__ __volatile__("movq (%[input0]), %%xmm0                     \n\t"
                         "movq (%[input1]), %%xmm1                     \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"

                         "mov %[kw], %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "movq (%[input0]), %%xmm4                     \n\t"
                         "movq (%[input1]), %%xmm5                     \n\t"

                         "pmaxub %%xmm4, %%xmm0                     \n\t"
                         "pmaxub %%xmm5, %%xmm1                     \n\t"

                         "add $0x8, %[input0]                                   \n\t"
                         "add $0x8, %[input1]                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"

                         "add %[iStep], %[input0]                                   \n\t"
                         "add %[iStep], %[input1]                                      \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 0b                                             \n\t"

                         "movq %%xmm0, (%[output])                              \n\t"
                         "movq %%xmm1, 0x8(%[output])                          \n\t"
                         : [input0] "+r" (curI[0]),
                           [input1] "+r" (curI[1]),
                           "+b" (kh)
                         : [output] "r" (curO),
                           [kw] "r" (kw),
                           [iStep] "r" (int64_t(iStep))
                         : "%ecx", "%xmm0", "%xmm1", "%xmm2",
                           "%xmm3", "%xmm4", "%xmm5", "%xmm6",
                           "%xmm7", "memory", "cc");
}

void pooling_c8_max_w1(const UINT8 **curI, UINT8 *curO, U32 kw, U32 kh, U32 iStep, U32 stride)
{
    __asm__ __volatile__("movq (%[input0]), %%xmm0                     \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"

                         "mov %[kw], %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "movq (%[input0]), %%xmm4                     \n\t"
                         "pmaxub %%xmm4, %%xmm0                     \n\t"

                         "add $0x8, %[input0]                                   \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"

                         "add %[iStep], %[input0]                                   \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 0b                                             \n\t"

                         "movq %%xmm0, (%[output])                              \n\t"
                         : [input0] "+r" (curI[0]),
                           "+b" (kh)
                         : [output] "r" (curO),
                           [kw] "r" (kw),
                           [iStep] "r" (int64_t(iStep))
                         : "%ecx", "%xmm0", "%xmm1", "%xmm2",
                           "%xmm3", "%xmm4", "%xmm5", "%xmm6",
                           "%xmm7", "memory", "cc");
}

void pooling_c8_mean_w4(
    const UINT8 **curI, UINT8 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, I32 poolSize, UINT8 *index)
{
    __asm__ __volatile__(
        "mov $-128, %%eax                                  \n\t"
        "imul %%ebx, %%eax                                  \n\t"
        "imul %[kw], %%eax                                  \n\t"
        "vmovd %%eax, %%xmm0              \n\t"
        "vpbroadcastd %%xmm0, %%ymm10              \n\t"
        "vpbroadcastd %%xmm0, %%ymm11              \n\t"
        "vpbroadcastd %%xmm0, %%ymm12              \n\t"
        "vpbroadcastd %%xmm0, %%ymm13              \n\t"

        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %[kw], %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "movq (%[input0]), %%xmm4              \n\t"
        "movq (%[input1]), %%xmm5              \n\t"
        "movq (%[input2]), %%xmm6              \n\t"
        "movq (%[input3]), %%xmm7              \n\t"
        "vpmovzxbd %%xmm4, %%ymm0                     \n\t"
        "vpmovzxbd %%xmm5, %%ymm1                     \n\t"
        "vpmovzxbd %%xmm6, %%ymm2                     \n\t"
        "vpmovzxbd %%xmm7, %%ymm3                     \n\t"
        "vpaddd %%ymm10, %%ymm0, %%ymm10                     \n\t"
        "vpaddd %%ymm11, %%ymm1, %%ymm11                     \n\t"
        "vpaddd %%ymm12, %%ymm2, %%ymm12                     \n\t"
        "vpaddd %%ymm13, %%ymm3, %%ymm13                     \n\t"
        "add $0x8, %[input0]                                   \n\t"
        "add $0x8, %[input1]                                      \n\t"
        "add $0x8, %[input2]                                     \n\t"
        "add $0x8, %[input3]                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %[iStep], %[input0]                                   \n\t"
        "add %[iStep], %[input1]                                      \n\t"
        "add %[iStep], %[input2]                                     \n\t"
        "add %[iStep], %[input3]                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vbroadcastss (%[poolSize]), %%ymm0                     \n\t"
        "vpmulld %%ymm0, %%ymm10, %%ymm10                     \n\t"
        "vpmulld %%ymm0, %%ymm11, %%ymm11                     \n\t"
        "vpmulld %%ymm0, %%ymm12, %%ymm12                     \n\t"
        "vpmulld %%ymm0, %%ymm13, %%ymm13                     \n\t"
        "vpsrld $16, %%ymm10, %%ymm10                     \n\t"
        "vpsrld $16, %%ymm11, %%ymm11                     \n\t"
        "vpsrld $16, %%ymm12, %%ymm12                     \n\t"
        "vpsrld $16, %%ymm13, %%ymm13                     \n\t"
        "mov $128, %%eax                                  \n\t"
        "vmovd %%eax, %%xmm0              \n\t"
        "vpbroadcastd %%xmm0, %%ymm4              \n\t"
        "vmovups (%[index]), %%ymm5                    \n\t"
        "vpaddd %%ymm10, %%ymm4, %%ymm10                     \n\t"
        "vpaddd %%ymm11, %%ymm4, %%ymm11                     \n\t"
        "vpaddd %%ymm12, %%ymm4, %%ymm12                     \n\t"
        "vpaddd %%ymm13, %%ymm4, %%ymm13                     \n\t"
        "vpshufb %%ymm10, %%ymm5, %%ymm0                 \n\t"
        "vpshufb %%ymm11, %%ymm5, %%ymm1                 \n\t"
        "vpshufb %%ymm12, %%ymm5, %%ymm2                 \n\t"
        "vpshufb %%ymm13, %%ymm5, %%ymm3                 \n\t"
        "vpermd  %%ymm0, %%ymm5, %%ymm10                 \n\t"
        "vpermd  %%ymm1, %%ymm5, %%ymm11                 \n\t"
        "vpermd  %%ymm2, %%ymm5, %%ymm12                 \n\t"
        "vpermd  %%ymm3, %%ymm5, %%ymm13                 \n\t"
        "movq  %%xmm10, (%[output])                 \n\t"
        "movq  %%xmm11, 0x8(%[output])                 \n\t"
        "movq  %%xmm12, 0x10(%[output])                 \n\t"
        "movq  %%xmm13, 0x18(%[output])                 \n\t"
        : [input0] "+r"(curI[0]),
          [input1] "+r"(curI[1]),
          [input2] "+r"(curI[2]),
          [input3] "+r"(curI[3]),
          "+b"(kh)
        : [output] "r"(curO),
          [kw] "r" (kw),
          [iStep] "r" (int64_t(iStep)),
          [poolSize] "r" (&poolSize),
          [index] "r" (index)
        : "%eax", "%ecx",
          "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
          "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm10",
          "%ymm11", "%ymm12", "%ymm13", "memory", "cc");
}

void pooling_c8_mean_w2(
    const UINT8 **curI, UINT8 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, I32 poolSize, UINT8 *index)
{
    __asm__ __volatile__(
        "mov $-128, %%eax                                  \n\t"
        "imul %%ebx, %%eax                                  \n\t"
        "imul %[kw], %%eax                                  \n\t"
        "vmovd %%eax, %%xmm0              \n\t"
        "vpbroadcastd %%xmm0, %%ymm10              \n\t"
        "vpbroadcastd %%xmm0, %%ymm11              \n\t"

        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %[kw], %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "movq (%[input0]), %%xmm4              \n\t"
        "movq (%[input1]), %%xmm5              \n\t"
        "vpmovzxbd %%xmm4, %%ymm0                     \n\t"
        "vpmovzxbd %%xmm5, %%ymm1                     \n\t"
        "vpaddd %%ymm10, %%ymm0, %%ymm10                     \n\t"
        "vpaddd %%ymm11, %%ymm1, %%ymm11                     \n\t"
        "add $0x8, %[input0]                                   \n\t"
        "add $0x8, %[input1]                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %[iStep], %[input0]                                   \n\t"
        "add %[iStep], %[input1]                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vbroadcastss (%[poolSize]), %%ymm0                     \n\t"
        "vpmulld %%ymm0, %%ymm10, %%ymm10                     \n\t"
        "vpmulld %%ymm0, %%ymm11, %%ymm11                     \n\t"
        "vpsrld $16, %%ymm10, %%ymm10                     \n\t"
        "vpsrld $16, %%ymm11, %%ymm11                     \n\t"
        "mov $128, %%eax                                  \n\t"
        "vmovd %%eax, %%xmm0              \n\t"
        "vpbroadcastd %%xmm0, %%ymm4              \n\t"
        "vmovups (%[index]), %%ymm5                    \n\t"
        "vpaddd %%ymm10, %%ymm4, %%ymm10                     \n\t"
        "vpaddd %%ymm11, %%ymm4, %%ymm11                     \n\t"
        "vpshufb %%ymm10, %%ymm5, %%ymm0                 \n\t"
        "vpshufb %%ymm11, %%ymm5, %%ymm1                 \n\t"
        "vpermd  %%ymm0, %%ymm5, %%ymm10                 \n\t"
        "vpermd  %%ymm1, %%ymm5, %%ymm11                 \n\t"
        "movq  %%xmm10, (%[output])                 \n\t"
        "movq  %%xmm11, 0x8(%[output])                 \n\t"
        : [input0] "+r"(curI[0]),
          [input1] "+r"(curI[1]),
          "+b"(kh)
        : [output] "r"(curO),
          [kw] "r" (kw),
          [iStep] "r" (int64_t(iStep)),
          [poolSize] "r" (&poolSize),
          [index] "r" (index)
        : "%eax", "%ecx",
          "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
          "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm10",
          "%ymm11", "%ymm12", "%ymm13", "memory", "cc");
}

void pooling_c8_mean_w1(
    const UINT8 **curI, UINT8 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, I32 poolSize, UINT8 *index)
{
    __asm__ __volatile__(
        "mov $-128, %%eax                                  \n\t"
        "imul %%ebx, %%eax                                  \n\t"
        "imul %[kw], %%eax                                  \n\t"
        "vmovd %%eax, %%xmm0              \n\t"
        "vpbroadcastd %%xmm0, %%ymm10              \n\t"

        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %[kw], %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "movq (%[input0]), %%xmm4              \n\t"
        "vpmovzxbd %%xmm4, %%ymm0                     \n\t"
        "vpaddd %%ymm10, %%ymm0, %%ymm10                     \n\t"
        "add $0x8, %[input0]                                   \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %[iStep], %[input0]                                   \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vbroadcastss (%[poolSize]), %%ymm0                     \n\t"
        "vpmulld %%ymm0, %%ymm10, %%ymm10                     \n\t"
        "vpsrld $16, %%ymm10, %%ymm10                     \n\t"
        "mov $128, %%eax                                  \n\t"
        "vmovd %%eax, %%xmm0              \n\t"
        "vpbroadcastd %%xmm0, %%ymm4              \n\t"
        "vmovups (%[index]), %%ymm5                    \n\t"
        "vpaddd %%ymm10, %%ymm4, %%ymm10                     \n\t"
        "vpshufb %%ymm10, %%ymm5, %%ymm0                 \n\t"
        "vpermd  %%ymm0, %%ymm5, %%ymm10                 \n\t"
        "movq  %%xmm10, (%[output])                 \n\t"
        : [input0] "+r"(curI[0]),
          "+b"(kh)
        : [output] "r"(curO),
          [kw] "r" (kw),
          [iStep] "r" (int64_t(iStep)),
          [poolSize] "r" (&poolSize),
          [index] "r" (index)
        : "%eax", "%ecx",
          "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
          "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm10",
          "%ymm11", "%ymm12", "%ymm13", "memory", "cc");
}

EE pooling_uint8(TensorDesc inputDesc,
    const UINT8 *input,
    PoolingParamSpec p,
    TensorDesc outputDesc,
    UINT8 *output,
    void *scale)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in = 0, ic = 0, ih = 0, iw = 0, on = 0, oc = 0, oh = 0, ow = 0;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (idt != odt || idt != DT_U8_Q) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (in != on || ic != oc) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (idf != DF_NCHWC8 || odf != idf) {
        CHECK_STATUS(NOT_MATCH);
    }

    PoolingMode pm = p.mode;
    U32 strideH = p.stride_h;
    U32 strideW = p.stride_w;
    U32 paddingT = p.pad_top;
    U32 paddingL = p.pad_left;
    U32 kernelSizeH = p.kernel_h;
    U32 kernelSizeW = p.kernel_w;
    U32 wSize, kh, kw, iStep;
    UINT8 *curO;
    const UINT8 *curI[4];
    if (paddingT >= kernelSizeH || paddingL >= kernelSizeW) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    if (ic % 8 != 0) {
        CHECK_STATUS(NOT_MATCH);
    }

    F32 *inputScale = (F32 *)scale;
    F32 *outputScale = inputScale + 1;
    I32 shift = 65536;
    I32 factor = shift / (kernelSizeH * kernelSizeW);
    if (factor < 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (pm == POOLING_MAX) {
        *outputScale = *inputScale;
    } else {
        *outputScale = *inputScale * factor * (kernelSizeW * kernelSizeH) / (F32)shift;
    }

    ic /= 8;
    U32 owInter = (iw + paddingL - kernelSizeW) / strideW + 1;
    U32 wSizes[3] = {1, 2, 4};
    pooling_max_func pooling_max[3] = {pooling_c8_max_w1, pooling_c8_max_w2, pooling_c8_max_w4};
    pooling_mean_func pooling_mean[3] = {
        pooling_c8_mean_w1, pooling_c8_mean_w2, pooling_c8_mean_w4};
    F32 poolSize = shift / (kernelSizeH * kernelSizeW);
    UINT8 index[32] = {0, 4, 8, 12, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
                       0, 4, 8, 12, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 h = 0; h < oh; h++) {
                for (U32 w = 0; w < ow; w += wSize) {
                    if (w < owInter) {
                        wSize = UNI_MIN(owInter - w, UNROLL_W);
                    } else {
                        wSize = 1;
                    }
                    wSize = wSizes[wSize >> 1];
                    int hstart = (int)h * (int)strideH - (int)paddingT;
                    int wstart = (int)w * (int)strideW - (int)paddingL;
                    int hend = UNI_MIN(hstart + kernelSizeH, ih);
                    int wend = UNI_MIN(wstart + kernelSizeW, iw);
                    hstart = UNI_MAX(hstart, 0);
                    wstart = UNI_MAX(wstart, 0);

                    for (U32 i = 0; i < wSize; ++i) {
                        curI[i] = input + (hstart * iw + wstart + wSize * (int)strideW) * 8;
                    }
                    curO = output + (h * ow + w) * 8;
                    kh = hend - hstart;
                    kw = wend - wstart;
                    iStep = (iw - kw) * 16;
                    if (!p.count_include_pad) {
                        poolSize = shift / (kh * kw);
                    }
                    if (kw < kernelSizeW) {
                        wSize = 1;
                    }
                    switch (pm) {
                        case POOLING_MAX: {
                            pooling_max[wSize >> 1](curI, curO, kw, kh, iStep, strideW * 16);
                            break;
                        }
                        case POOLING_MEAN: {
                            pooling_mean[wSize >> 1](
                                curI, curO, kw, kh, iStep, strideW * 16, poolSize, index);
                            break;
                        }
                        default:
                            return NOT_SUPPORTED;
                    }
                }
            }
            input += ih * iw * 16;
            output += oh * ow * 16;
        }
    }
    return SUCCESS;
}
