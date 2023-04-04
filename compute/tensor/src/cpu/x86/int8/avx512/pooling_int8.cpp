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
    const UINT8 *curI, UINT8 *curO, U32 kw, U32 kh, U32 iStep, U32 stride);
typedef void (*pooling_mean_func)(
    const UINT8 *curI, UINT8 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, I32 poolSize);

void pooling_c16_max_w4(const UINT8 *curI, UINT8 *curO, U32 kw, U32 kh, U32 iStep, U32 stride)
{
    __asm__ __volatile__("mov %%eax, %%eax                                  \n\t"
                         "mov %4, %%eax                                  \n\t"
                         "mov %%rax, %%rdi                                  \n\t"
                         "mov %%eax, %%eax                                  \n\t"
                         "mov %5, %%eax                                  \n\t"
                         "mov %%rax, %%r9                                  \n\t"
                         "add %%r9, %%r9                                  \n\t"
                         "mov %%rax, %%r10                                  \n\t"
                         "add %%r9, %%r10                                  \n\t"
                         "add %0, %%rax                                  \n\t"
                         "add %0, %%r9                                  \n\t"
                         "add %0, %%r10                                  \n\t"

                         "vmovups (%0), %%xmm0                     \n\t"
                         "vmovups (%%rax), %%xmm1                     \n\t"
                         "vmovups (%%r9), %%xmm2                     \n\t"
                         "vmovups (%%r10), %%xmm3                     \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"

                         "mov %2, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "vmovups (%0), %%xmm4                     \n\t"
                         "vmovups (%%rax), %%xmm5                     \n\t"
                         "vmovups (%%r9), %%xmm6                     \n\t"
                         "vmovups (%%r10), %%xmm7                     \n\t"

                         "vpmaxub %%xmm0, %%xmm4, %%xmm0                     \n\t"
                         "vpmaxub %%xmm1, %%xmm5, %%xmm1                     \n\t"
                         "vpmaxub %%xmm2, %%xmm6, %%xmm2                     \n\t"
                         "vpmaxub %%xmm3, %%xmm7, %%xmm3                     \n\t"

                         "add $0x10, %0                                      \n\t"
                         "add $0x10, %%rax                                      \n\t"
                         "add $0x10, %%r9                                      \n\t"
                         "add $0x10, %%r10                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"

                         "add %%rdi, %0                                      \n\t"
                         "add %%rdi, %%rax                                      \n\t"
                         "add %%rdi, %%r9                                      \n\t"
                         "add %%rdi, %%r10                                      \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 0b                                             \n\t"

                         "vmovups %%xmm0, (%1)                              \n\t"
                         "vmovups %%xmm1, 0x10(%1)                          \n\t"
                         "vmovups %%xmm2, 0x20(%1)                          \n\t"
                         "vmovups %%xmm3, 0x30(%1)                          \n\t"
                         :
                         : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride)
                         : "%eax", "%rax", "%ecx", "%r10", "%r9", "%rdi", "%xmm0", "%xmm1", "%xmm2",
                         "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "memory", "cc");
}

void pooling_c16_max_w2(const UINT8 *curI, UINT8 *curO, U32 kw, U32 kh, U32 iStep, U32 stride)
{
    __asm__ __volatile__(
        "mov %%eax, %%eax                                  \n\t"
        "mov %4, %%eax                                  \n\t"
        "mov %%rax, %%rdi                                  \n\t"
        "mov %%eax, %%eax                                  \n\t"
        "mov %5, %%eax                                  \n\t"
        "add %0, %%rax                                  \n\t"
        "vmovups (%0), %%xmm0                     \n\t"
        "vmovups (%%rax), %%xmm1                     \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %2, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vmovups (%0), %%xmm4                     \n\t"
        "vmovups (%%rax), %%xmm5                     \n\t"
        "vpmaxub %%xmm0, %%xmm4, %%xmm0                     \n\t"
        "vpmaxub %%xmm1, %%xmm5, %%xmm1                     \n\t"
        "add $0x10, %0                                      \n\t"
        "add $0x10, %%rax                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %%rdi, %0                                      \n\t"
        "add %%rdi, %%rax                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vmovups %%xmm0, (%1)                              \n\t"
        "vmovups %%xmm1, 0x10(%1)                          \n\t"
        :
        : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride)
        : "%eax", "%rax", "%ecx", "%rdi", "%xmm0", "%xmm1", "%xmm4", "%xmm5", "memory", "cc");
}

void pooling_c16_max_w1(const UINT8 *curI, UINT8 *curO, U32 kw, U32 kh, U32 iStep, U32 stride)
{
    __asm__ __volatile__("mov %%eax, %%eax                                  \n\t"
                         "mov %4, %%eax                                  \n\t"
                         "mov %%rax, %%rdi                                  \n\t"
                         "vmovups (%0), %%xmm0                     \n\t"
                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "mov %2, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "vmovups (%0), %%xmm4                     \n\t"
                         "vpmaxub %%xmm0, %%xmm4, %%xmm0                     \n\t"
                         "add $0x10, %0                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"
                         "add %%rdi, %0                                      \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 0b                                             \n\t"
                         "vmovups %%xmm0, (%1)                              \n\t"
                         :
                         : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride)
                         : "%eax", "%rax", "%ecx", "%rdi", "%xmm0", "%xmm4", "memory", "cc");
}

void pooling_c16_mean_w4(
    const UINT8 *curI, UINT8 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, I32 poolSize)
{
    __asm__ __volatile__(
        "mov $-128, %%eax                                  \n\t"
        "imul %%ebx, %%eax                                  \n\t"
        "imul %2, %%eax                                  \n\t"
        "vmovd %%eax, %%xmm0              \n\t"
        "vpbroadcastd %%xmm0, %%zmm10              \n\t"
        "vpbroadcastd %%xmm0, %%zmm11              \n\t"
        "vpbroadcastd %%xmm0, %%zmm12              \n\t"
        "vpbroadcastd %%xmm0, %%zmm13              \n\t"
        "mov $0x80, %%eax \n\t"
        "vmovd %%eax, %%xmm1                    \n\t"
        "vpbroadcastb %%xmm1, %%xmm8            \n\t"
        "mov %%eax, %%eax                                  \n\t"
        "mov %4, %%eax                                  \n\t"
        "mov %%rax, %%rdi                                  \n\t"
        "mov %5, %%eax                                  \n\t"
        "mov %%rax, %%r9                                  \n\t"
        "add %%r9, %%r9                                  \n\t"
        "mov %%rax, %%r10                                  \n\t"
        "add %%r9, %%r10                                  \n\t"
        "add %0, %%rax                                  \n\t"
        "add %0, %%r9                                  \n\t"
        "add %0, %%r10                                  \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %2, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vmovups (%0), %%xmm4              \n\t"
        "vmovups (%%rax), %%xmm5              \n\t"
        "vmovups (%%r9), %%xmm6              \n\t"
        "vmovups (%%r10), %%xmm7              \n\t"
        "vpmovzxbd %%xmm4, %%zmm0                     \n\t"
        "vpmovzxbd %%xmm5, %%zmm1                     \n\t"
        "vpmovzxbd %%xmm6, %%zmm2                     \n\t"
        "vpmovzxbd %%xmm7, %%zmm3                     \n\t"
        "vpaddd %%zmm10, %%zmm0, %%zmm10                     \n\t"
        "vpaddd %%zmm11, %%zmm1, %%zmm11                     \n\t"
        "vpaddd %%zmm12, %%zmm2, %%zmm12                     \n\t"
        "vpaddd %%zmm13, %%zmm3, %%zmm13                     \n\t"
        "add $0x10, %0                                      \n\t"
        "add $0x10, %%rax                                      \n\t"
        "add $0x10, %%r9                                      \n\t"
        "add $0x10, %%r10                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %%rdi, %0                                      \n\t"
        "add %%rdi, %%rax                                      \n\t"
        "add %%rdi, %%r9                                      \n\t"
        "add %%rdi, %%r10                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vbroadcastss (%6), %%zmm0                     \n\t"
        "vpmulld %%zmm0, %%zmm10, %%zmm10                     \n\t"
        "vpmulld %%zmm0, %%zmm11, %%zmm11                     \n\t"
        "vpmulld %%zmm0, %%zmm12, %%zmm12                     \n\t"
        "vpmulld %%zmm0, %%zmm13, %%zmm13                     \n\t"
        "vpsrld $16, %%zmm10, %%zmm10                     \n\t"
        "vpsrld $16, %%zmm11, %%zmm11                     \n\t"
        "vpsrld $16, %%zmm12, %%zmm12                     \n\t"
        "vpsrld $16, %%zmm13, %%zmm13                     \n\t"
        "mov $128, %%eax                                  \n\t"
        "vmovd %%eax, %%xmm0              \n\t"
        "vpbroadcastd %%xmm0, %%zmm4              \n\t"
        "vpaddd %%zmm10, %%zmm4, %%zmm10                     \n\t"
        "vpaddd %%zmm11, %%zmm4, %%zmm11                     \n\t"
        "vpaddd %%zmm12, %%zmm4, %%zmm12                     \n\t"
        "vpaddd %%zmm13, %%zmm4, %%zmm13                     \n\t"
        "vpmovusdb %%zmm10, (%1)                              \n\t"
        "vpmovusdb %%zmm11, 0x10(%1)                          \n\t"
        "vpmovusdb %%zmm12, 0x20(%1)                          \n\t"
        "vpmovusdb %%zmm13, 0x30(%1)                          \n\t"
        :
        : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride), "r"(&poolSize)
        : "%eax", "%rax", "%ecx", "%r10", "%r9", "%rdi", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4",
        "%zmm5", "%zmm6", "%zmm7", "%zmm8", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "memory", "cc");
}

void pooling_c16_mean_w2(
    const UINT8 *curI, UINT8 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, I32 poolSize)
{
    __asm__ __volatile__(
        "mov $-128, %%eax                                  \n\t"
        "imul %%ebx, %%eax                                  \n\t"
        "imul %2, %%eax                                  \n\t"
        "vmovd %%eax, %%xmm0              \n\t"
        "vpbroadcastd %%xmm0, %%zmm10              \n\t"
        "vpbroadcastd %%xmm0, %%zmm11              \n\t"
        "mov %%eax, %%eax                                  \n\t"
        "mov %4, %%eax                                  \n\t"
        "mov %%rax, %%rdi                                  \n\t"
        "mov %5, %%eax                                  \n\t"
        "add %0, %%rax                                  \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %2, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vmovups (%0), %%xmm4              \n\t"
        "vmovups (%%rax), %%xmm5              \n\t"
        "vpmovzxbd %%xmm4, %%zmm0                     \n\t"
        "vpmovzxbd %%xmm5, %%zmm1                     \n\t"
        "vpaddd %%zmm10, %%zmm0, %%zmm10                     \n\t"
        "vpaddd %%zmm11, %%zmm1, %%zmm11                     \n\t"
        "add $0x10, %0                                      \n\t"
        "add $0x10, %%rax                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %%rdi, %0                                      \n\t"
        "add %%rdi, %%rax                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vbroadcastss (%6), %%zmm0                     \n\t"
        "vpmulld %%zmm0, %%zmm10, %%zmm10                     \n\t"
        "vpmulld %%zmm0, %%zmm11, %%zmm11                     \n\t"
        "vpsrld $16, %%zmm10, %%zmm10                     \n\t"
        "vpsrld $16, %%zmm11, %%zmm11                     \n\t"
        "mov $128, %%eax                                  \n\t"
        "vmovd %%eax, %%xmm0              \n\t"
        "vpbroadcastd %%xmm0, %%zmm4              \n\t"
        "vpaddd %%zmm10, %%zmm4, %%zmm10                     \n\t"
        "vpaddd %%zmm11, %%zmm4, %%zmm11                     \n\t"
        "vpmovusdb %%zmm10, (%1)                              \n\t"
        "vpmovusdb %%zmm11, 0x10(%1)                          \n\t"
        :
        : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride), "r"(&poolSize)
        : "%eax", "%rax", "%ecx", "%rdi", "%zmm0", "%zmm1", "%zmm4", "%zmm5", "%zmm8", "%zmm10",
        "%zmm11", "memory", "cc");
}

void pooling_c16_mean_w1(
    const UINT8 *curI, UINT8 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, I32 poolSize)
{
    __asm__ __volatile__(
        "mov %%eax, %%eax                                  \n\t"
        "mov %4, %%eax                                  \n\t"
        "mov %%rax, %%rdi                                  \n\t"
        "mov $-128, %%eax                                  \n\t"
        "imul %%ebx, %%eax                                  \n\t"
        "imul %2, %%eax                                  \n\t"
        "vmovd %%eax, %%xmm0              \n\t"
        "vpbroadcastd %%xmm0, %%zmm10              \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %2, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vmovups (%0), %%xmm4              \n\t"
        "vpmovzxbd %%xmm4, %%zmm0                     \n\t"
        "vpaddd %%zmm10, %%zmm0, %%zmm10                     \n\t"
        "add $0x10, %0                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %%rdi, %0                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vbroadcastss (%6), %%zmm0                     \n\t"
        "vpmulld %%zmm0, %%zmm10, %%zmm10                     \n\t"
        "mov $128, %%eax                                  \n\t"
        "vmovd %%eax, %%xmm0              \n\t"
        "vpbroadcastd %%xmm0, %%zmm4              \n\t"
        "vpsrld $16, %%zmm10, %%zmm10                     \n\t"
        "vpaddd %%zmm10, %%zmm4, %%zmm10                     \n\t"
        "vpmovusdb %%zmm10, (%1)                              \n\t"
        :
        : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride), "r"(&poolSize)
        : "%eax", "%rax", "%ecx", "%rdi", "%zmm0", "%zmm1", "%zmm4", "%zmm8", "%zmm10", "memory",
        "cc");
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
    if (idf != DF_NCHWC16 || odf != idf) {
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
    const UINT8 *curI;
    if (paddingT >= kernelSizeH || paddingL >= kernelSizeW) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    if (ic % 16 != 0) {
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

    ic /= 16;
    U32 owInter = (iw + paddingL - kernelSizeW) / strideW + 1;
    U32 wSizes[3] = {1, 2, 4};
    pooling_max_func pooling_max[3] = {pooling_c16_max_w1, pooling_c16_max_w2, pooling_c16_max_w4};
    pooling_mean_func pooling_mean[3] = {
        pooling_c16_mean_w1, pooling_c16_mean_w2, pooling_c16_mean_w4};
    F32 poolSize = shift / (kernelSizeH * kernelSizeW);
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

                    curI = input + (hstart * iw + wstart) * 16;
                    curO = output + (h * ow + w) * 16;
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
                                curI, curO, kw, kh, iStep, strideW * 16, poolSize);
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
