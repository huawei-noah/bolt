// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <math.h>
#include <set>

#include "uni.h"
#include "blas_enhance.h"
#include "cpu/x86/int8/tensor_computing_int8.h"

inline void getSymmetricQuantizeScale(U32 num16, U32 resMask, const F32 *data, F32 *scale)
{
    F32 maxVal = 0;
    __asm__ __volatile__("vxorps %%zmm0, %%zmm0, %%zmm0            \n\t"
                         "mov $0x7FFFFFFF, %%ebx \n\t"
                         "vmovd %%ebx, %%xmm1                      \n\t"
                         "vpbroadcastd %%xmm1, %%zmm2              \n\t"
                         "mov %2, %%ebx \n\t"
                         "cmp $0x0, %%ebx                         \n\t"
                         "je 1f                                    \n\t"
                         ".align 16                                \n\t"
                         "0:                                       \n\t"
                         "vpandd (%0), %%zmm2, %%zmm1              \n\t"
                         "vmaxps %%zmm1, %%zmm0, %%zmm0            \n\t"
                         "add $0x40, %0                            \n\t"
                         "dec %%ebx                                \n\t"
                         "jg 0b                                    \n\t"

                         ".align 16                                \n\t"
                         "1:                                       \n\t"
                         "cmp $0x0, %%eax                         \n\t"
                         "je 2f                                    \n\t"
                         "vxorps %%zmm1, %%zmm1, %%zmm1            \n\t"
                         "kmovw %%eax, %%k2                        \n\t"
                         "vmovups (%0), %%zmm1%{%%k2%}       \n\t"
                         "vpandd %%zmm1, %%zmm2, %%zmm1      \n\t"
                         "vmaxps %%zmm0, %%zmm1, %%zmm0            \n\t"
                         ".align 16                                \n\t"
                         "2:                                       \n\t"

                         "vextractf32x8 $0x1, %%zmm0, %%ymm1       \n\t"
                         "vmaxps %%ymm1, %%ymm0, %%ymm0            \n\t"
                         "vextractf32x4 $0x1, %%ymm0, %%xmm1       \n\t"
                         "vmaxps %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vpermilps $0b00001011, %%xmm0, %%xmm1    \n\t"
                         "vmaxps %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vpermilps $0b00000001, %%xmm0, %%xmm1    \n\t"
                         "vmaxps %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vmovd %%xmm0, %1                      \n\t"
                         : "+r"(data), "+r"(maxVal)
                         : "r"(num16), "a"(resMask)
                         : "%k2", "%ebx", "%zmm0", "%zmm1", "%zmm2", "memory", "cc");
    if (maxVal == 0) {
        *scale = 1;
    } else {
        *scale = 127 / maxVal;
    }
}

inline void getMaxValI32(U32 num16, U32 resMask, const I32 *data, I32 *maxVal)
{
    __asm__ __volatile__("vxorps %%zmm0, %%zmm0, %%zmm0            \n\t"
                         "mov %2, %%ebx \n\t"
                         "cmp $0x0, %%ebx                         \n\t"
                         "je 1f                                    \n\t"
                         ".align 16                                \n\t"
                         "0:                                       \n\t"
                         "vpabsd (%0), %%zmm1              \n\t"
                         "vpmaxsd %%zmm1, %%zmm0, %%zmm0            \n\t"
                         "add $0x40, %0                            \n\t"
                         "dec %%ebx                                \n\t"
                         "jg 0b                                    \n\t"

                         ".align 16                                \n\t"
                         "1:                                       \n\t"
                         "cmp $0x0, %%eax                         \n\t"
                         "je 2f                                    \n\t"
                         "vxorps %%zmm1, %%zmm1, %%zmm1            \n\t"
                         "kmovw %%eax, %%k2                        \n\t"
                         "vmovups (%0), %%zmm1%{%%k2%}       \n\t"
                         "vpabsd %%zmm1, %%zmm2              \n\t"
                         "vpmaxsd %%zmm0, %%zmm2, %%zmm0            \n\t"
                         ".align 16                                \n\t"
                         "2:                                       \n\t"

                         "vextractf32x8 $0x1, %%zmm0, %%ymm1       \n\t"
                         "vpmaxsd %%ymm1, %%ymm0, %%ymm0            \n\t"
                         "vextractf32x4 $0x1, %%ymm0, %%xmm1       \n\t"
                         "vpmaxsd %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vpermilps $0b00001011, %%xmm0, %%xmm1    \n\t"
                         "vpmaxsd %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vpermilps $0b00000001, %%xmm0, %%xmm1    \n\t"
                         "vpmaxsd %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vmovd %%xmm0, (%1)                     \n\t"
                         : "+r"(data), "+r"(maxVal)
                         : "r"(num16), "a"(resMask)
                         : "%k2", "%ebx", "%zmm0", "%zmm1", "%zmm2", "memory", "cc");
}

inline void getSymmetricQuantizeScaleI32(U32 num16, U32 resMask, const I32 *data, F32 *scale)
{
    I32 maxVal = 0;
    getMaxValI32(num16, resMask, data, &maxVal);
    if (maxVal == 0) {
        *scale = 1;
    } else {
        *scale = 127.0f / maxVal;
    }
}

EE quantizeF32ToU8(TensorDesc dDesc, const F32 *data, TensorDesc *qDesc, UINT8 *qData, F32 *scale, int mode)
{
    if (scale == nullptr) {
        return NOT_MATCH;
    }
    U32 dataNum = tensorNumElements(dDesc);
    U32 num16 = dataNum / 16;
    U32 resMask = pow(2, dataNum % 16) - 1;
    if (!mode || (*scale <= 0)) {
        F32 minScale = *scale;
        getSymmetricQuantizeScale(num16, resMask, data, scale);
        *scale = UNI_MAX(minScale, *scale);
    }

    __asm__ __volatile__(
        "mov $0x43000000, %%ebx \n\t"
        "vmovd %%ebx, %%xmm1                    \n\t"
        "vbroadcastss %%xmm1, %%zmm2            \n\t"
        "vbroadcastss (%2), %%zmm0              \n\t"
        "mov $255, %%ebx \n\t"
        "vmovd %%ebx, %%xmm1                    \n\t"
        "vbroadcastss %%xmm1, %%zmm4            \n\t"
        "mov $1, %%ebx \n\t"
        "vmovd %%ebx, %%xmm1                    \n\t"
        "vbroadcastss %%xmm1, %%zmm5            \n\t"
        "mov %3, %%ebx \n\t"
        "cmp $0x0, %%ebx                         \n\t"
        "je 1f                                    \n\t"
        ".align 16                              \n\t"
        "0:                                     \n\t"
        "vmulps (%0), %%zmm0, %%zmm1            \n\t"
        "vaddps %%zmm1, %%zmm2, %%zmm3          \n\t"
        "vcvtps2dq %{rn-sae%}, %%zmm3, %%zmm1   \n\t"
        "vpmaxsd %%zmm5, %%zmm1, %%zmm1          \n\t"
        "vpminsd %%zmm4, %%zmm1, %%zmm1          \n\t"
        "vpmovusdb %%zmm1, (%1)                 \n\t"
        "add $0x40, %0                          \n\t"
        "add $0x10, %1                          \n\t"
        "dec %%ebx                              \n\t"
        "jg 0b                                  \n\t"

        ".align 16                              \n\t"
        "1:                                     \n\t"
        "cmp $0x0, %%eax                       \n\t"
        "je 2f                                  \n\t"
        "kmovw %%eax, %%k1                      \n\t"
        "vmulps (%0), %%zmm0, %%zmm1 %{%%k1%}      \n\t"
        "vaddps %%zmm1, %%zmm2, %%zmm3          \n\t"
        "vcvtps2dq %{rn-sae%}, %%zmm3, %%zmm0              \n\t"
        "vpmaxsd %%zmm5, %%zmm0, %%zmm0          \n\t"
        "vpminsd %%zmm4, %%zmm0, %%zmm0          \n\t"
        "vpmovusdb %%zmm0, (%1) %{%%k1%}            \n\t"
        ".align 16                              \n\t"
        "2:                                     \n\t"
        : "+r"(data), "+r"(qData), "+r"(scale)
        : "r"(num16), "a"(resMask)
        : "%k1", "%ebx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "memory", "cc");
    return SUCCESS;
}

EE quantizeF32ToI8(TensorDesc dDesc, const F32 *data, TensorDesc *qDesc, INT8 *qData, F32 *scale, int mode)
{
    if (scale == nullptr) {
        return NOT_MATCH;
    }
    U32 dataNum = tensorNumElements(dDesc);
    U32 num16 = dataNum / 16;
    U32 resMask = pow(2, dataNum % 16) - 1;
    if (!mode || (*scale <= 0)) {
        F32 minScale = *scale;
        getSymmetricQuantizeScale(num16, resMask, data, scale);
        *scale = UNI_MAX(minScale, *scale);
    }

    __asm__ __volatile__(
        "vbroadcastss (%2), %%zmm0             \n\t"
        "mov $127, %%ebx \n\t"
        "vmovd %%ebx, %%xmm1                    \n\t"
        "vbroadcastss %%xmm1, %%zmm4            \n\t"
        "mov $-127, %%ebx \n\t"
        "vmovd %%ebx, %%xmm1                    \n\t"
        "vbroadcastss %%xmm1, %%zmm5            \n\t"
        "mov %3, %%ebx \n\t"
        "cmp $0x0, %%ebx                         \n\t"
        "je 1f                                    \n\t"
        ".align 16                             \n\t"
        "0:                                    \n\t"
        "vmulps (%0), %%zmm0, %%zmm1            \n\t"
        "vcvtps2dq %%zmm1, %%zmm2              \n\t"
        "vpmaxsd %%zmm5, %%zmm2, %%zmm3          \n\t"
        "vpminsd %%zmm4, %%zmm3, %%zmm1          \n\t"
        "vpmovsdb %%zmm1, (%1)                 \n\t"
        "add $0x40, %0                         \n\t"
        "add $0x10, %1                         \n\t"
        "dec %%ebx                             \n\t"
        "jg 0b                                 \n\t"

        ".align 16                             \n\t"
        "1:                                    \n\t"
        "cmp $0x0, %%eax                       \n\t"
        "je 2f                                 \n\t"
        "kmovw %%eax, %%k1                     \n\t"
        "vmulps (%0), %%zmm0, %%zmm1 %{%%k1%}      \n\t"
        "vcvtps2dq %%zmm1, %%zmm2              \n\t"
        "vpmaxsd %%zmm5, %%zmm2, %%zmm3          \n\t"
        "vpminsd %%zmm4, %%zmm3, %%zmm1          \n\t"
        "vpmovsdb %%zmm1, (%1) %{%%k1%}        \n\t"
        ".align 16                             \n\t"
        "2:                                    \n\t"
        : "+r"(data), "+r"(qData), "+r"(scale)
        : "r"(num16), "a"(resMask)
        : "%k1", "%ebx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "memory", "cc");
    return SUCCESS;
}

EE quantizeBiasOffsetCI32(const F32 *bias,
    TensorDesc biasDesc,
    INT8 *filter,
    TensorDesc filterDesc,
    const F32 *scale,
    I32 *offsetCBias)
{
    U32 N = tensorNumElements(biasDesc);
    std::set<DataFormat> nativeFormat = {DF_NCHW, DF_NHWC, DF_MTK, DF_NORMAL, DF_TRANSPOSE};
    I32 *offsetC = (I32 *)filter;
    if ((bias == nullptr) && (filter == nullptr)) {
        return SUCCESS;
    }
    if ((bias == nullptr) || (N == 0)) {
        N = UNI_MAX(filterDesc.dims[0], filterDesc.dims[1]);
        if (nativeFormat.count(filterDesc.df)) {
            UNI_MEMSET(offsetCBias, 0, N * bytesOf(DT_I32));
        } else {
            UNI_MEMCPY(offsetCBias, offsetC, N * bytesOf(DT_I32));
        }
        return SUCCESS;
    }

    if ((filter == nullptr) || nativeFormat.count(filterDesc.df)) {
        for (U32 i = 0; i < N; ++i) {
            offsetCBias[i] = round(bias[i] * scale[0]);
        }
    } else {
        for (U32 i = 0; i < N; ++i) {
            offsetCBias[i] = round(bias[i] * scale[0]) + offsetC[i];
        }
    }
    return SUCCESS;
}

EE transformU8ToI8(TensorDesc dDesc, const UINT8 *data, TensorDesc *qDesc, INT8 *qData)
{
    U32 dataNum = tensorNumElements(dDesc);
    U32 num16 = dataNum / 64;
    U64 resMask = dataNum % 64;
    if (resMask == 63) {
        resMask = 0xFFFFFFFFFFFFFFFF;
    } else {
        resMask = (1LL << resMask) - 1;
    }

    __asm__ __volatile__("mov $0x80, %%ebx \n\t"
                         "vmovd %%ebx, %%xmm1                    \n\t"
                         "vpbroadcastb %%xmm1, %%zmm2            \n\t"
                         "mov %2, %%ebx \n\t"
                         "cmp $0x0, %%ebx                         \n\t"
                         "je 1f                                    \n\t"
                         ".align 16                             \n\t"
                         "0:                                    \n\t"
                         "vpaddb (%0), %%zmm2, %%zmm0              \n\t"
                         "vmovups %%zmm0, (%1)              \n\t"
                         "add $0x40, %0                         \n\t"
                         "add $0x40, %1                         \n\t"
                         "dec %%ebx                             \n\t"
                         "jg 0b                                 \n\t"

                         ".align 16                             \n\t"
                         "1:                                    \n\t"
                         "cmp $0x0, %%rax                       \n\t"
                         "je 2f                                 \n\t"
                         "kmovq %%rax, %%k1                     \n\t"
                         "vxorps %%zmm0, %%zmm0, %%zmm0         \n\t"
                         "vxorps %%zmm1, %%zmm1, %%zmm1         \n\t"
                         "vmovdqu8 (%0), %%zmm1 %{%%k1%}  \n\t"
                         "vpaddb %%zmm1, %%zmm2, %%zmm0 %{%%k1%}   \n\t"
                         "vmovdqu8 %%zmm0, (%1) %{%%k1%}              \n\t"
                         ".align 16                             \n\t"
                         "2:                                    \n\t"
                         : "+r"(data), "+r"(qData)
                         : "r"(num16), "a"(resMask)
                         : "%k1", "%ebx", "%zmm0", "%zmm1", "%zmm2", "memory", "cc");
    return SUCCESS;
}

EE quantizeI32ToI8(TensorDesc dDesc, const I32 *data, TensorDesc *qDesc, INT8 *qData, F32 *scale)
{
    U32 dataNum = tensorNumElements(dDesc);
    U32 num16 = dataNum / 16;
    U32 resMask = pow(2, dataNum % 16) - 1;
    F32 scaleRaw = 0;
    if (*scale < 0 && scale[1] > 0) {
        getSymmetricQuantizeScaleI32(num16, resMask, data, &scaleRaw);
        scale[0] = scale[1] * scaleRaw;
    } else if (scale[0] > 0 && scale[1] > 0) {
        scaleRaw = scale[0] / scale[1];
    }

    __asm__ __volatile__(
        "vbroadcastss (%2), %%zmm0             \n\t"
        "mov %3, %%ebx \n\t"
        "cmp $0x0, %%ebx                         \n\t"
        "je 1f                                    \n\t"
        ".align 16                             \n\t"
        "0:                                    \n\t"
        "vmovups (%0), %%zmm3            \n\t"
        "vcvtdq2ps %%zmm3, %%zmm2               \n\t"
        "vmulps %%zmm2, %%zmm0, %%zmm1            \n\t"
        "vcvtps2dq %{rn-sae%}, %%zmm1, %%zmm2              \n\t"
        "vpmovsdb %%zmm2, (%1)                 \n\t"
        "add $0x40, %0                         \n\t"
        "add $0x10, %1                         \n\t"
        "dec %%ebx                             \n\t"
        "jg 0b                                 \n\t"

        ".align 16                             \n\t"
        "1:                                    \n\t"
        "cmp $0x0, %%eax                       \n\t"
        "je 2f                                 \n\t"
        "kmovw %%eax, %%k1                     \n\t"
        "vmovups (%0), %%zmm3 %{%%k1%}            \n\t"
        "vcvtdq2ps %%zmm3, %%zmm2               \n\t"
        "vmulps %%zmm2, %%zmm0, %%zmm1      \n\t"
        "vcvtps2dq %{rn-sae%}, %%zmm1, %%zmm2              \n\t"
        "vpmovsdb %%zmm2, (%1) %{%%k1%}        \n\t"
        ".align 16                             \n\t"
        "2:                                    \n\t"
        : "+r"(data), "+r"(qData)
        : "r"(&scaleRaw), "r"(num16), "a"(resMask)
        : "%k1", "%ebx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "memory", "cc");
    return SUCCESS;
}

#ifndef _USE_X86_ARM_CONSISTENCY
EE quantizeI32ToU8(TensorDesc dDesc, const I32 *data, TensorDesc *qDesc, UINT8 *qData, F32 *scale)
{
    U32 dataNum = tensorNumElements(dDesc);
    U32 num16 = dataNum / 16;
    U32 resMask = pow(2, dataNum % 16) - 1;
    F32 scaleRaw = 0;
    if (*scale < 0 && scale[1] > 0) {
        getSymmetricQuantizeScaleI32(num16, resMask, data, &scaleRaw);
        scale[0] = scale[1] * scaleRaw;
    } else if (scale[0] > 0 && scale[1] > 0) {
        scaleRaw = scale[0] / scale[1];
    }

    __asm__ __volatile__(
        "mov $0x43000000, %%ebx \n\t"
        "vmovd %%ebx, %%xmm1                    \n\t"
        "vbroadcastss %%xmm1, %%zmm4            \n\t"
        "vbroadcastss (%2), %%zmm0             \n\t"
        "mov %3, %%ebx \n\t"
        "cmp $0x0, %%ebx                         \n\t"
        "je 1f                                    \n\t"
        ".align 16                             \n\t"
        "0:                                    \n\t"
        "vmovups (%0), %%zmm3            \n\t"
        "vcvtdq2ps %%zmm3, %%zmm2               \n\t"
        "vmulps %%zmm2, %%zmm0, %%zmm1            \n\t"
        "vaddps %%zmm4, %%zmm1, %%zmm1            \n\t"
        "vcvtps2dq %{rn-sae%}, %%zmm1, %%zmm2              \n\t"
        "vpmovusdb %%zmm2, (%1)                 \n\t"
        "add $0x40, %0                         \n\t"
        "add $0x10, %1                         \n\t"
        "dec %%ebx                             \n\t"
        "jg 0b                                 \n\t"

        ".align 16                             \n\t"
        "1:                                    \n\t"
        "cmp $0x0, %%eax                       \n\t"
        "je 2f                                 \n\t"
        "kmovw %%eax, %%k1                     \n\t"
        "vmovups (%0), %%zmm3 %{%%k1%}            \n\t"
        "vcvtdq2ps %%zmm3, %%zmm2               \n\t"
        "vmulps %%zmm2, %%zmm0, %%zmm1      \n\t"
        "vaddps %%zmm4, %%zmm1, %%zmm1            \n\t"
        "vcvtps2dq %{rn-sae%}, %%zmm1, %%zmm2              \n\t"
        "vpmovusdb %%zmm2, (%1) %{%%k1%}        \n\t"
        ".align 16                             \n\t"
        "2:                                    \n\t"
        : "+r"(data), "+r"(qData)
        : "r"(&scaleRaw), "r"(num16), "a"(resMask)
        : "%k1", "%ebx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "memory", "cc");

    return SUCCESS;
}
#else
EE quantizeI32ToU8(TensorDesc dDesc, const I32 *data, TensorDesc *qDesc, UINT8 *qData, F32 *scale)
{
    U32 dataNum = tensorNumElements(dDesc);
    U32 num16 = dataNum / 16;
    U32 resMask = pow(2, dataNum % 16) - 1;
    F32 scaleRaw = 0;
    I32 factor = 0;
    if (*scale < 0 && scale[1] > 0) {
        I32 maxVal = 0;
        getMaxValI32(num16, resMask, data, &maxVal);
        if (maxVal == 0) {
            scaleRaw = 1;
            factor = 16777216;
        } else {
            scaleRaw = 127.0f / maxVal;
            factor = 16777216 * 127.0f / maxVal;
        }
        scale[0] = scale[1] * scaleRaw;
    } else if (scale[0] > 0 && scale[1] > 0) {
        scaleRaw = scale[0] / scale[1];
        factor = 16777216 * scaleRaw;
    }

    __asm__ __volatile__(
        "mov $0x80000000, %%ebx \n\t"
        "vmovd %%ebx, %%xmm1                    \n\t"
        "vbroadcastss %%xmm1, %%zmm4            \n\t"
        "vbroadcastss (%2), %%zmm0             \n\t"
        "mov %3, %%ebx \n\t"
        "cmp $0x0, %%ebx                         \n\t"
        "je 1f                                    \n\t"
        ".align 16                             \n\t"
        "0:                                    \n\t"
        "vmovups (%0), %%zmm3            \n\t"
        "vpmulld %%zmm3, %%zmm0, %%zmm1            \n\t"
        "vpaddd %%zmm4, %%zmm1, %%zmm1            \n\t"
        "vpsrld $24, %%zmm1, %%zmm2              \n\t"
        "vpmovusdb %%zmm2, (%1)                 \n\t"
        "add $0x40, %0                         \n\t"
        "add $0x10, %1                         \n\t"
        "dec %%ebx                             \n\t"
        "jg 0b                                 \n\t"

        ".align 16                             \n\t"
        "1:                                    \n\t"
        "cmp $0x0, %%eax                       \n\t"
        "je 2f                                 \n\t"
        "kmovw %%eax, %%k1                     \n\t"
        "vmovups (%0), %%zmm3 %{%%k1%}            \n\t"

        "vpmulld %%zmm3, %%zmm0, %%zmm1            \n\t"
        "vpaddd %%zmm4, %%zmm1, %%zmm1            \n\t"
        "vpsrld $24, %%zmm1, %%zmm2              \n\t"
        "vpmovusdb %%zmm2, (%1) %{%%k1%}                 \n\t"
        ".align 16                             \n\t"
        "2:                                    \n\t"
        : "+r"(data), "+r"(qData)
        : "r"(&factor), "r"(num16), "a"(resMask)
        : "%k1", "%ebx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "memory", "cc");

    return SUCCESS;
}
#endif