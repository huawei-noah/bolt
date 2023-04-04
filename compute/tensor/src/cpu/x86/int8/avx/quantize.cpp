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

inline void getSymmetricQuantizeScale(U32 num8, U32 resNum, const F32 *data, F32 *scale)
{
    F32 maxVal = 0;
    __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0            \n\t"
                         "mov $0x7FFFFFFF, %%ebx                   \n\t"
                         "vmovd %%ebx, %%xmm1                      \n\t"
                         "vpbroadcastd %%xmm1, %%ymm2              \n\t"
                         "mov %[num8], %%ebx                       \n\t"
                         "cmp $0x0, %%ebx                          \n\t"
                         "je 1f                                    \n\t"
                         ".align 16                                \n\t"
                         "0:                                       \n\t"
                         "vmovups (%[data]), %%ymm1         \n\t"
                         "vpand %%ymm1, %%ymm2, %%ymm1          \n\t"
                         "vpmaxsd %%ymm1, %%ymm0, %%ymm0            \n\t"
                         "add $0x20, %[data]                            \n\t"
                         "dec %%ebx                                \n\t"
                         "jg 0b                                    \n\t"
                         "vextractf128 $0x1, %%ymm0, %%xmm1       \n\t"
                         "vmaxps %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vpermilps $0b00001011, %%xmm0, %%xmm1    \n\t"
                         "vmaxps %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vpermilps $0b00000001, %%xmm0, %%xmm1    \n\t"
                         "vmaxps %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vmovd %%xmm0, %[maxVal]                  \n\t"
                         ".align 16                                \n\t"
                         "1:                                       \n\t"
                         : [data] "+r"(data),
                           [maxVal] "+r"(maxVal)
                         : [num8] "r"(num8)
                         : "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                           "memory", "cc");
    for (U32 i = 0; i < resNum; ++i) {
        maxVal = UNI_MAX(maxVal, UNI_ABS(data[i]));
    }
    if (maxVal == 0) {
        *scale = 1;
    } else {
        *scale = 127 / maxVal;
    }
}

inline void getMaxValI32(U32 num8, U32 resNum, const I32 *data, I32 *maxVal)
{
    __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0            \n\t"
                         "mov %[num8], %%ebx \n\t"
                         "cmp $0x0, %%ebx                         \n\t"
                         "je 1f                                    \n\t"
                         ".align 16                                \n\t"
                         "0:                                       \n\t"
                         "vpabsd (%[data]), %%ymm1              \n\t"
                         "vpmaxsd %%ymm1, %%ymm0, %%ymm0            \n\t"
                         "add $0x20, %0                            \n\t"
                         "dec %%ebx                                \n\t"
                         "jg 0b                                    \n\t"

                         ".align 16                                \n\t"
                         "1:                                       \n\t"

                         "vextractf128 $0x1, %%ymm0, %%xmm1       \n\t"
                         "vpmaxsd %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vpermilps $0b00001011, %%xmm0, %%xmm1    \n\t"
                         "vpmaxsd %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vpermilps $0b00000001, %%xmm0, %%xmm1    \n\t"
                         "vpmaxsd %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vmovd %%xmm0, (%[maxVal])                     \n\t"
                         : [data] "+r"(data),
                           [maxVal] "+r"(maxVal)
                         : [num8] "r"(num8)
                         : "%ebx", "%ymm0", "%ymm1", "%ymm2", "memory", "cc");
    for (U32 i = 0; i < resNum; ++i) {
        *maxVal = UNI_MAX(*maxVal, UNI_ABS(data[i]));
    }
}

inline void getSymmetricQuantizeScaleI32(U32 num8, U32 resNum, const I32 *data, F32 *scale)
{
    I32 maxVal = 0;
    getMaxValI32(num8, resNum, data, &maxVal);
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
    U32 num8 = dataNum / 8;
    U32 resNum = dataNum % 8;
    if (!mode || (*scale < 0)) {
        F32 minScale = *scale;
        getSymmetricQuantizeScale(num8, resNum, data, scale);
        *scale = UNI_MAX(minScale, *scale);
    }

    UINT8 index[32] = {0, 4, 8, 12, 4, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
                       0, 4, 8, 12, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

    __asm__ __volatile__(
        "vmovups (%4), %%ymm6                    \n\t"
        "mov $0x43000000, %%ebx \n\t"
        "vmovd %%ebx, %%xmm1                    \n\t"
        "vbroadcastss %%xmm1, %%ymm2            \n\t"
        "vbroadcastss (%2), %%ymm0              \n\t"
        "mov $255, %%ebx \n\t"
        "vmovd %%ebx, %%xmm1                    \n\t"
        "vbroadcastss %%xmm1, %%ymm4            \n\t"
        "mov $1, %%ebx \n\t"
        "vmovd %%ebx, %%xmm1                    \n\t"
        "vbroadcastss %%xmm1, %%ymm5            \n\t"
        "mov %3, %%ebx \n\t"
        "cmp $0x0, %%ebx                         \n\t"
        "je 1f                                    \n\t"
        ".align 16                              \n\t"
        "0:                                     \n\t"
        "vmulps (%0), %%ymm0, %%ymm1            \n\t"
        "vaddps %%ymm1, %%ymm2, %%ymm3          \n\t"
        "vcvtps2dq %%ymm3, %%ymm1   \n\t"
        "vpmaxsd %%ymm5, %%ymm1, %%ymm1          \n\t"
        "vpminsd %%ymm4, %%ymm1, %%ymm1          \n\t"
        "vpshufb %%ymm6, %%ymm1, %%ymm3                 \n\t"
        "vpermd  %%ymm3, %%ymm6, %%ymm1                 \n\t"
        "movq  %%xmm1, (%1)                 \n\t"
        "add $0x20, %0                          \n\t"
        "add $0x8, %1                          \n\t"
        "dec %%ebx                              \n\t"
        "jg 0b                                  \n\t"

        ".align 16                              \n\t"
        "1:                                     \n\t"
        : "+r"(data), "+r"(qData), "+r"(scale)
        : "r"(num8), "r"(index)
        : "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "memory", "cc");
    num8 *= 8;
    for (U32 i = 0; i < resNum; ++i) {
        qData[i] = UNI_MAX(UNI_MIN(round(data[i] * scale[0] + 128), 255), 1);
    }
    return SUCCESS;
}

EE quantizeF32ToI8(TensorDesc dDesc, const F32 *data, TensorDesc *qDesc, INT8 *qData, F32 *scale, int mode)
{
    if (scale == nullptr) {
        return NOT_MATCH;
    }
    U32 dataNum = tensorNumElements(dDesc);
    U32 num8 = dataNum / 8;
    U32 resNum = dataNum % 8;
    if (!mode || (*scale < 0)) {
        F32 minScale = *scale;
        getSymmetricQuantizeScale(num8, resNum, data, scale);
        *scale = UNI_MAX(minScale, *scale);
    }

    UINT8 index[32] = {0, 4, 8, 12, 4, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
                       0, 4, 8, 12, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

    __asm__ __volatile__(
        "vmovups (%4), %%ymm6                    \n\t"
        "vbroadcastss (%2), %%ymm0              \n\t"
        "mov $127, %%ebx \n\t"
        "vmovd %%ebx, %%xmm1                    \n\t"
        "vbroadcastss %%xmm1, %%ymm4            \n\t"
        "mov $-127, %%ebx \n\t"
        "vmovd %%ebx, %%xmm1                    \n\t"
        "vbroadcastss %%xmm1, %%ymm5            \n\t"
        "mov %3, %%ebx \n\t"
        "cmp $0x0, %%ebx                         \n\t"
        "je 1f                                    \n\t"
        ".align 16                              \n\t"
        "0:                                     \n\t"
        "vmulps (%0), %%ymm0, %%ymm3            \n\t"
        "vcvtps2dq %%ymm3, %%ymm1   \n\t"
        "vpmaxsd %%ymm5, %%ymm1, %%ymm1          \n\t"
        "vpminsd %%ymm4, %%ymm1, %%ymm1          \n\t"
        "vpshufb %%ymm6, %%ymm1, %%ymm3                 \n\t"
        "vpermd  %%ymm3, %%ymm6, %%ymm1                 \n\t"
        "movq  %%xmm1, (%1)                 \n\t"
        "add $0x20, %0                          \n\t"
        "add $0x8, %1                          \n\t"
        "dec %%ebx                              \n\t"
        "jg 0b                                  \n\t"

        ".align 16                              \n\t"
        "1:                                     \n\t"
        : "+r"(data), "+r"(qData), "+r"(scale)
        : "r"(num8), "r"(index)
        : "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "memory", "cc");
    for (U32 i = 0; i < resNum; ++i) {
        qData[i] = UNI_MAX(UNI_MIN(data[i] * scale[0], 127), -127);
    }
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
    U32 num8 = dataNum / 8;
    U32 resNum = dataNum % 8;

    __asm__ __volatile__("mov $0x80, %%ebx \n\t"
                         "vmovd %%ebx, %%xmm1                    \n\t"
                         "vpbroadcastb %%xmm1, %%ymm2            \n\t"
                         "mov %2, %%ebx \n\t"
                         "cmp $0x0, %%ebx                         \n\t"
                         "je 1f                                    \n\t"
                         ".align 16                             \n\t"
                         "0:                                    \n\t"
                         "vpaddb (%0), %%ymm2, %%ymm0              \n\t"
                         "vmovups %%ymm0, (%1)              \n\t"
                         "add $0x20, %0                         \n\t"
                         "add $0x20, %1                         \n\t"
                         "dec %%ebx                             \n\t"
                         "jg 0b                                 \n\t"

                         ".align 16                             \n\t"
                         "1:                                    \n\t"
                         : "+r"(data), "+r"(qData)
                         : "r"(num8)
                         : "%ebx", "%ymm0", "%ymm1", "%ymm2", "memory", "cc");
    for (U32 i = 0; i < resNum; ++i) {
        qData[i] = I32(data[i]) - 128;
    }
    return SUCCESS;
}

EE quantizeI32ToI8(TensorDesc dDesc, const I32 *data, TensorDesc *qDesc, INT8 *qData, F32 *scale)
{
    U32 dataNum = tensorNumElements(dDesc);
    U32 num8 = dataNum / 8;
    U32 resNum = dataNum % 8;
    F32 scaleRaw = 0;
    if (*scale < 0 && scale[1] > 0) {
        getSymmetricQuantizeScaleI32(num8, resNum, data, &scaleRaw);
        scale[0] = scale[1] * scaleRaw;
    } else if (scale[0] > 0 && scale[1] > 0) {
        scaleRaw = scale[0] / scale[1];
    }

    UINT8 index[32] = {0, 4, 8, 12, 4, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
                       0, 4, 8, 12, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

    __asm__ __volatile__(
        "vmovups (%4), %%ymm5                    \n\t"
        "mov $0x80, %%ebx \n\t"
        "vmovd %%ebx, %%xmm1                    \n\t"
        "vpbroadcastd %%xmm1, %%ymm4            \n\t"
        "vpbroadcastb %%xmm1, %%ymm6            \n\t"
        "vbroadcastss (%2), %%ymm0             \n\t"
        "mov %3, %%ebx \n\t"
        "cmp $0x0, %%ebx                         \n\t"
        "je 1f                                    \n\t"
        ".align 16                             \n\t"
        "0:                                    \n\t"
        "vmovups (%0), %%ymm3            \n\t"
        "vcvtdq2ps %%ymm3, %%ymm2               \n\t"
        "vmulps %%ymm2, %%ymm0, %%ymm1            \n\t"
        "vcvtps2dq %%ymm1, %%ymm2              \n\t"
        "vpaddd  %%ymm2, %%ymm4, %%ymm3                 \n\t"
        "vpshufb %%ymm3, %%ymm5, %%ymm1                 \n\t"
        "vpermd  %%ymm5, %%ymm1, %%ymm2                 \n\t"
        "vpaddb  %%ymm2, %%ymm6, %%ymm3                 \n\t"
        "movq  %%xmm3, (%1)                 \n\t"
        "add $0x20, %0                         \n\t"
        "add $0x8, %1                         \n\t"
        "dec %%ebx                             \n\t"
        "jg 0b                                 \n\t"

        ".align 16                             \n\t"
        "1:                                    \n\t"
        : "+r"(data), "+r"(qData)
        : "r"(&scaleRaw), "r"(num8), "r"(index)
        : "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "memory", "cc");
    for (U32 i = 0; i < resNum; ++i) {
        qData[i] = UNI_MIN(127, UNI_MAX(-127, round(data[i] * scaleRaw)));
    }
    return SUCCESS;
}

#ifndef _USE_X86_ARM_CONSISTENCY
EE quantizeI32ToU8(TensorDesc dDesc, const I32 *data, TensorDesc *qDesc, UINT8 *qData, F32 *scale)
{
    U32 dataNum = tensorNumElements(dDesc);
    U32 num8 = dataNum / 8;
    U32 resNum = dataNum % 8;
    F32 scaleRaw = 0;

    if (*scale < 0 && scale[1] > 0) {
        getSymmetricQuantizeScaleI32(num8, resNum, data, &scaleRaw);
        scale[0] = scale[1] * scaleRaw;
    } else if (scale[0] > 0 && scale[1] > 0) {
        scaleRaw = scale[0] / scale[1];
    }

    UINT8 index[32] = {0, 4, 8, 12, 4, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
                       0, 4, 8, 12, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

    __asm__ __volatile__(
        "vmovups (%4), %%ymm5                    \n\t"
        "mov $0x80, %%ebx \n\t"
        "vmovd %%ebx, %%xmm1                    \n\t"
        "vpbroadcastd %%xmm1, %%ymm4            \n\t"
        "vbroadcastss (%2), %%ymm0             \n\t"
        "mov %3, %%ebx \n\t"
        "cmp $0x0, %%ebx                         \n\t"
        "je 1f                                    \n\t"
        ".align 16                             \n\t"
        "0:                                    \n\t"
        "vmovups (%0), %%ymm3            \n\t"
        "vcvtdq2ps %%ymm3, %%ymm2               \n\t"
        "vmulps %%ymm2, %%ymm0, %%ymm1            \n\t"
        "vcvtps2dq %%ymm1, %%ymm2              \n\t"
        "vpaddd  %%ymm2, %%ymm4, %%ymm3                 \n\t"
        "vpshufb %%ymm5, %%ymm3, %%ymm1                 \n\t"
        "vpermd  %%ymm1, %%ymm5, %%ymm2                 \n\t"
        "movq  %%xmm2, (%1)                 \n\t"
        "add $0x20, %0                         \n\t"
        "add $0x8, %1                         \n\t"
        "dec %%ebx                             \n\t"
        "jg 0b                                 \n\t"

        ".align 16                             \n\t"
        "1:                                    \n\t"
        : "+r"(data), "+r"(qData)
        : "r"(&scaleRaw), "r"(num8), "r"(index)
        : "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "memory", "cc");

    for (U32 i = 0; i < resNum; ++i) {
        qData[i] = UNI_MIN(127, UNI_MAX(-127, round(data[i] * scaleRaw))) + 128;
    }

    return SUCCESS;
}
#else
EE quantizeI32ToU8(TensorDesc dDesc, const I32 *data, TensorDesc *qDesc, UINT8 *qData, F32 *scale)
{
    U32 dataNum = tensorNumElements(dDesc);
    U32 num8 = dataNum / 8;
    U32 resNum = dataNum % 8;
    F32 scaleRaw = 0;
    I32 factor = 0;

    UINT8 index[32] = {0, 4, 8, 12, 4, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
                       0, 4, 8, 12, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

    __asm__ __volatile__(
        "vmovups (%4), %%ymm5                    \n\t"
        "mov $0x80000000, %%ebx \n\t"
        "vmovd %%ebx, %%xmm1                    \n\t"
        "vbroadcastss %%xmm1, %%ymm4            \n\t"
        "vbroadcastss (%2), %%ymm0             \n\t"
        "mov %3, %%ebx \n\t"
        "cmp $0x0, %%ebx                         \n\t"
        "je 1f                                    \n\t"
        ".align 16                             \n\t"
        "0:                                    \n\t"
        "vmovups (%0), %%ymm3            \n\t"
        "vpmulld %%ymm3, %%ymm0, %%ymm1            \n\t"
        "vpaddd %%ymm4, %%ymm1, %%ymm1            \n\t"
        "vpsrld $24, %%ymm1, %%ymm2              \n\t"
        "vpshufb %%ymm5, %%ymm2, %%ymm3                 \n\t"
        "vpermd  %%ymm3, %%ymm5, %%ymm1                 \n\t"
        "movq  %%xmm1, (%1)                 \n\t"
        "add $0x20, %0                         \n\t"
        "add $0x8, %1                         \n\t"
        "dec %%ebx                             \n\t"
        "jg 0b                                 \n\t"

        ".align 16                             \n\t"
        "1:                                    \n\t"
        : "+r"(data), "+r"(qData)
        : "r"(&factor), "r"(num16), "r"(index)
        : "%k1", "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "memory", "cc");

    for (U32 i = 0; i < resNum; ++i) {
        qData[i] = UNI_MAX(255, UNI_MIN(1, (data[i] * factor + 0x80000000) >> 24));
    }
    return SUCCESS;
}
#endif