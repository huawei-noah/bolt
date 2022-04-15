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

typedef void (*pooling_max_func)(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride);
typedef void (*pooling_mean_func)(
    const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize);

void pooling_max_w4(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride)
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

                         "vmovups (%0), %%ymm0                     \n\t"
                         "vmovups (%%rax), %%ymm1                     \n\t"
                         "vmovups (%%r9), %%ymm2                     \n\t"
                         "vmovups (%%r10), %%ymm3                     \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"

                         "mov %2, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "vmovups (%0), %%ymm4                     \n\t"
                         "vmovups (%%rax), %%ymm5                     \n\t"
                         "vmovups (%%r9), %%ymm6                     \n\t"
                         "vmovups (%%r10), %%ymm7                     \n\t"

                         "vmaxps %%ymm0, %%ymm4, %%ymm0                     \n\t"
                         "vmaxps %%ymm1, %%ymm5, %%ymm1                     \n\t"
                         "vmaxps %%ymm2, %%ymm6, %%ymm2                     \n\t"
                         "vmaxps %%ymm3, %%ymm7, %%ymm3                     \n\t"

                         "add $0x20, %0                                      \n\t"
                         "add $0x20, %%rax                                      \n\t"
                         "add $0x20, %%r9                                      \n\t"
                         "add $0x20, %%r10                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"

                         "add %%rdi, %0                                      \n\t"
                         "add %%rdi, %%rax                                      \n\t"
                         "add %%rdi, %%r9                                      \n\t"
                         "add %%rdi, %%r10                                      \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 0b                                             \n\t"

                         "vmovups %%ymm0, (%1)                              \n\t"
                         "vmovups %%ymm1, 0x20(%1)                          \n\t"
                         "vmovups %%ymm2, 0x40(%1)                          \n\t"
                         "vmovups %%ymm3, 0x60(%1)                          \n\t"
                         :
                         : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride)
                         : "%eax", "%rax", "%ecx", "%r10", "%r9", "%rdi", "%ymm0", "%ymm1", "%ymm2",
                         "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "memory", "cc");
}

void pooling_max_w2(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride)
{
    __asm__ __volatile__(
        "mov %%eax, %%eax                                  \n\t"
        "mov %4, %%eax                                  \n\t"
        "mov %%rax, %%rdi                                  \n\t"
        "mov %%eax, %%eax                                  \n\t"
        "mov %5, %%eax                                  \n\t"
        "add %0, %%rax                                  \n\t"
        "vmovups (%0), %%ymm0                     \n\t"
        "vmovups (%%rax), %%ymm1                     \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %2, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vmovups (%0), %%ymm4                     \n\t"
        "vmovups (%%rax), %%ymm5                     \n\t"
        "vmaxps %%ymm0, %%ymm4, %%ymm0                     \n\t"
        "vmaxps %%ymm1, %%ymm5, %%ymm1                     \n\t"
        "add $0x20, %0                                      \n\t"
        "add $0x20, %%rax                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %%rdi, %0                                      \n\t"
        "add %%rdi, %%rax                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vmovups %%ymm0, (%1)                              \n\t"
        "vmovups %%ymm1, 0x20(%1)                          \n\t"
        :
        : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride)
        : "%eax", "%rax", "%ecx", "%rdi", "%ymm0", "%ymm1", "%ymm4", "%ymm5", "memory", "cc");
}

void pooling_max_w1(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride)
{
    __asm__ __volatile__("mov %%eax, %%eax                                  \n\t"
                         "mov %4, %%eax                                  \n\t"
                         "mov %%rax, %%rdi                                  \n\t"
                         "vmovups (%0), %%ymm0                     \n\t"
                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "mov %2, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "vmovups (%0), %%ymm4                     \n\t"
                         "vmaxps %%ymm0, %%ymm4, %%ymm0                     \n\t"
                         "add $0x20, %0                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"
                         "add %%rdi, %0                                      \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 0b                                             \n\t"
                         "vmovups %%ymm0, (%1)                              \n\t"
                         :
                         : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride)
                         : "%eax", "%rax", "%ecx", "%rdi", "%ymm0", "%ymm4", "memory", "cc");
}

void pooling_mean_w4(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize)
{
    __asm__ __volatile__(
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
        "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
        "vxorps %%ymm2, %%ymm2, %%ymm2                     \n\t"
        "vxorps %%ymm3, %%ymm3, %%ymm3                     \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %2, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vmovups (%0), %%ymm4                     \n\t"
        "vmovups (%%rax), %%ymm5                     \n\t"
        "vmovups (%%r9), %%ymm6                     \n\t"
        "vmovups (%%r10), %%ymm7                     \n\t"
        "vaddps %%ymm0, %%ymm4, %%ymm0                     \n\t"
        "vaddps %%ymm1, %%ymm5, %%ymm1                     \n\t"
        "vaddps %%ymm2, %%ymm6, %%ymm2                     \n\t"
        "vaddps %%ymm3, %%ymm7, %%ymm3                     \n\t"
        "add $0x20, %0                                      \n\t"
        "add $0x20, %%rax                                      \n\t"
        "add $0x20, %%r9                                      \n\t"
        "add $0x20, %%r10                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %%rdi, %0                                      \n\t"
        "add %%rdi, %%rax                                      \n\t"
        "add %%rdi, %%r9                                      \n\t"
        "add %%rdi, %%r10                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vbroadcastss (%6), %%ymm4                     \n\t"
        "vdivps %%ymm4, %%ymm0, %%ymm0                     \n\t"
        "vdivps %%ymm4, %%ymm1, %%ymm1                     \n\t"
        "vdivps %%ymm4, %%ymm2, %%ymm2                     \n\t"
        "vdivps %%ymm4, %%ymm3, %%ymm3                     \n\t"
        "vmovups %%ymm0, (%1)                              \n\t"
        "vmovups %%ymm1, 0x20(%1)                          \n\t"
        "vmovups %%ymm2, 0x40(%1)                          \n\t"
        "vmovups %%ymm3, 0x60(%1)                          \n\t"
        :
        : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride), "r"(&poolSize)
        : "%eax", "%rax", "%ecx", "%r10", "%r9", "%rdi", "%ymm0", "%ymm1", "%ymm2", "%ymm3",
        "%ymm4", "%ymm5", "%ymm6", "%ymm7", "memory", "cc");
}

void pooling_mean_w2(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize)
{
    __asm__ __volatile__(
        "mov %%eax, %%eax                                  \n\t"
        "mov %4, %%eax                                  \n\t"
        "mov %%rax, %%rdi                                  \n\t"
        "mov %5, %%eax                                  \n\t"
        "add %0, %%rax                                  \n\t"
        "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %2, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vmovups (%0), %%ymm4                     \n\t"
        "vmovups (%%rax), %%ymm5                     \n\t"
        "vaddps %%ymm0, %%ymm4, %%ymm0                     \n\t"
        "vaddps %%ymm1, %%ymm5, %%ymm1                     \n\t"
        "add $0x20, %0                                      \n\t"
        "add $0x20, %%rax                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %%rdi, %0                                      \n\t"
        "add %%rdi, %%rax                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vbroadcastss (%6), %%ymm4                     \n\t"
        "vdivps %%ymm4, %%ymm0, %%ymm0                     \n\t"
        "vdivps %%ymm4, %%ymm1, %%ymm1                     \n\t"
        "vmovups %%ymm0, (%1)                              \n\t"
        "vmovups %%ymm1, 0x20(%1)                          \n\t"
        :
        : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride), "r"(&poolSize)
        : "%eax", "%rax", "%ecx", "%rdi", "%ymm0", "%ymm1", "%ymm4", "%ymm5", "memory", "cc");
}

void pooling_mean_w1(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize)
{
    __asm__ __volatile__(
        "mov %%eax, %%eax                                  \n\t"
        "mov %4, %%eax                                  \n\t"
        "mov %%rax, %%rdi                                  \n\t"
        "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %2, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vmovups (%0), %%ymm4                     \n\t"
        "vaddps %%ymm0, %%ymm4, %%ymm0                     \n\t"
        "add $0x20, %0                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %%rdi, %0                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vbroadcastss (%6), %%ymm4                     \n\t"
        "vdivps %%ymm4, %%ymm0, %%ymm0                     \n\t"
        "vmovups %%ymm0, (%1)                              \n\t"
        :
        : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride), "r"(&poolSize)
        : "%eax", "%rax", "%ecx", "%rdi", "%ymm0", "%ymm4", "memory", "cc");
}

EE pooling_fp32(TensorDesc inputDesc,
    const F32 *input,
    PoolingParamSpec poolingParamSpec,
    TensorDesc outputDesc,
    F32 *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in = 0, ic = 0, ih = 0, iw = 0, on = 0, oc = 0, oh = 0, ow = 0;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (idt != odt || idt != DT_F32) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (in != on || ic != oc) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (idf != DF_NCHWC8 || odf != idf) {
        CHECK_STATUS(NOT_MATCH);
    }

    PoolingMode pm = poolingParamSpec.mode;
    U32 strideH = poolingParamSpec.stride_h;
    U32 strideW = poolingParamSpec.stride_w;
    U32 paddingT = poolingParamSpec.padding_top;
    U32 paddingL = poolingParamSpec.padding_left;
    U32 kernelSizeH = poolingParamSpec.kernel_h;
    U32 kernelSizeW = poolingParamSpec.kernel_w;
    U32 wSize, kh, kw, iStep;
    F32 poolSize, *curO;
    const F32 *curI;
    if (paddingT >= kernelSizeH || paddingL >= kernelSizeW) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    ic /= 8;
    U32 owInter = (iw + paddingL - kernelSizeW) / strideW + 1;
    U32 wSizes[3] = {1, 2, 4};
    pooling_max_func pooling_max[3] = {pooling_max_w1, pooling_max_w2, pooling_max_w4};
    pooling_mean_func pooling_mean[3] = {pooling_mean_w1, pooling_mean_w2, pooling_mean_w4};
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

                    curI = input + (hstart * iw + wstart) * 8;
                    curO = output + (h * ow + w) * 8;
                    kh = hend - hstart;
                    kw = wend - wstart;
                    iStep = (iw - kw) * 32;
                    poolSize = kw * kh * 1.0f;
                    if (kw < kernelSizeW) {
                        wSize = 1;
                    }
                    switch (pm) {
                        case POOLING_MAX: {
                            pooling_max[wSize >> 1](curI, curO, kw, kh, iStep, strideW * 32);
                            break;
                        }
                        case POOLING_MEAN: {
                            pooling_mean[wSize >> 1](
                                curI, curO, kw, kh, iStep, strideW * 32, poolSize);
                            break;
                        }
                        default:
                            CHECK_STATUS(NOT_SUPPORTED);
                    }
                }
            }
            input += ih * iw * 8;
            output += oh * ow * 8;
        }
    }
    return SUCCESS;
}
