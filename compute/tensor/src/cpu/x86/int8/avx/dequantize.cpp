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

#include "cpu/x86/int8/tensor_computing_int8.h"

EE dequantizeI32ToF32(TensorDesc qDesc, I32 *qData, const F32 *scale, TensorDesc dDesc, F32 *data)
{
    U32 dataNum = tensorNumElements(dDesc);
    U32 num8 = dataNum / 8;
    F32 factor = 1 / *scale;
    __asm__ __volatile__("vbroadcastss (%2), %%ymm2              \n\t"
                         ".align 16                              \n\t"
                         "0:                                     \n\t"
                         "vmovups (%0), %%ymm0                   \n\t"
                         "vcvtdq2ps %%ymm0, %%ymm1               \n\t"
                         "vmulps %%ymm2, %%ymm1, %%ymm0          \n\t"
                         "vmovups %%ymm0, (%1)                   \n\t"
                         "add $0x20, %0                          \n\t"
                         "add $0x20, %1                          \n\t"
                         "dec %%ebx                              \n\t"
                         "jg 0b                                  \n\t"
                         : "+r"(qData), "+r"(data)
                         : "r"(&factor), "b"(num8)
                         : "%ymm0", "%ymm1", "%ymm2", "memory", "cc");
    num8 *= 8;
    for (U32 i = num8; i < dataNum; ++i) {
        data[i] = qData[i] * factor;
    }
    return SUCCESS;
}

EE dequantizeU8ToF32(TensorDesc qDesc, UINT8 *qData, const F32 *scale, TensorDesc dDesc, F32 *data)
{
    U32 dataNum = tensorNumElements(dDesc);
    // for (U32 i = 0; i < dataNum; ++i) {
    //     data[i] = (qData[i] - 128) / scale[0];
    // }
    U32 num8 = dataNum / 8;
    F32 factor = 1.0f / scale[0];

    __asm__ __volatile__(
        "mov $128, %%ebx \n\t"
        "vmovd %%ebx, %%xmm1                    \n\t"
        "vbroadcastss %%xmm1, %%ymm2            \n\t"
        "vbroadcastss (%[scale]), %%ymm0              \n\t"
        "mov %[num8], %%ebx \n\t"
        "cmp $0x0, %%ebx                         \n\t"
        "je 1f                                    \n\t"
        ".align 16                              \n\t"
        "0:                                     \n\t"
        "vpmovzxbd (%[qData]), %%ymm1            \n\t"
        "vpsubd %%ymm2, %%ymm1, %%ymm3          \n\t"
        "vcvtdq2ps %%ymm3, %%ymm4   \n\t"
        "vmulps %%ymm4, %%ymm0, %%ymm5          \n\t"
        "vmovups %%ymm5, (%[data])                 \n\t"
        "add $0x20, %[data]                          \n\t"
        "add $0x8, %[qData]                          \n\t"
        "dec %%ebx                              \n\t"
        "jg 0b                                  \n\t"

        ".align 16                              \n\t"
        "1:                                     \n\t"
        : [data] "+r"(data),
          [qData] "+r"(qData)
        : [scale] "r" (&factor),
          [num8] "r"(num8)
        : "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "memory", "cc");
    num8 *= 8;
    for (U32 i = num8; i < dataNum; ++i) {
        data[i] = (float(qData[i]) - 128) * factor;
    }
    return SUCCESS;
}
