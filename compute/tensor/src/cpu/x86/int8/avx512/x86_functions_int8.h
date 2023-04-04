// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef CHEETAH_X86_FUNCTIONS_INT8_H
#define CHEETAH_X86_FUNCTIONS_INT8_H

#include <math.h>
#include "parameter_spec.h"
#include "uni.h"
#include "thread_affinity.h"

inline EE activation_offset_int8(
    UINT8 *input, U32 len, ActivationParamSpec activationDesc, UINT8 *output, F32 *scale)
{
    U32 num32 = len / 32;
    U32 resMask = pow(2, len % 32) - 1;
    EE ret = SUCCESS;
    switch (activationDesc.mode) {
        case ACTIVATION_NULL: {
            break;
        }
        case ACTIVATION_RELU: {
            __asm__ __volatile__("mov $0x80, %%ebx \n\t"
                                 "vmovd %%ebx, %%xmm1                    \n\t"
                                 "vpbroadcastb %%xmm1, %%ymm2            \n\t"
                                 "mov %2, %%ebx \n\t"
                                 "cmp $0x0, %%ebx                         \n\t"
                                 "je 1f                                    \n\t"
                                 ".align 16                             \n\t"
                                 "0:                                    \n\t"
                                 "vpmaxub (%0), %%ymm2, %%ymm0              \n\t"
                                 "vmovups %%ymm0, (%1)              \n\t"
                                 "add $0x20, %0                         \n\t"
                                 "add $0x20, %1                         \n\t"
                                 "dec %%ebx                             \n\t"
                                 "jg 0b                                 \n\t"

                                 ".align 16                             \n\t"
                                 "1:                                    \n\t"
                                 "cmp $0x0, %%eax                       \n\t"
                                 "je 2f                                 \n\t"
                                 "kmovw %%eax, %%k1                     \n\t"
                                 "vmovdqu8 (%0), %%ymm1 %{%%k1%}              \n\t"
                                 "vpmaxub %%ymm1, %%ymm2, %%ymm0 %{%%k1%}              \n\t"
                                 "vmovdqu8 %%ymm0, (%1) %{%%k1%}              \n\t"
                                 ".align 16                             \n\t"
                                 "2:                                    \n\t"
                                 : "+r"(input), "+r"(output)
                                 : "r"(num32), "a"(resMask)
                                 : "%k1", "%ebx", "%zmm0", "%zmm1", "%zmm2", "memory", "cc");
            break;
        }
        case ACTIVATION_RELU6: {
            U8 maxVal = UNI_MIN(255, round(6 * *scale) + 128);
            __asm__ __volatile__("mov $0x80, %%ebx \n\t"
                                 "vmovd %%ebx, %%xmm1                    \n\t"
                                 "vpbroadcastb %%xmm1, %%ymm2            \n\t"
                                 "mov %4, %%ebx \n\t"
                                 "vmovd %%ebx, %%xmm1                    \n\t"
                                 "vpbroadcastb %%xmm1, %%ymm4            \n\t"
                                 "mov %2, %%ebx \n\t"
                                 "cmp $0x0, %%ebx                         \n\t"
                                 "je 1f                                    \n\t"
                                 ".align 16                             \n\t"
                                 "0:                                    \n\t"
                                 "vpmaxub (%0), %%ymm2, %%ymm0              \n\t"
                                 "vpminub %%ymm0, %%ymm4, %%ymm0              \n\t"
                                 "vmovups %%ymm0, (%1)              \n\t"
                                 "add $0x20, %0                         \n\t"
                                 "add $0x20, %1                         \n\t"
                                 "dec %%ebx                             \n\t"
                                 "jg 0b                                 \n\t"

                                 ".align 16                             \n\t"
                                 "1:                                    \n\t"
                                 "cmp $0x0, %%eax                       \n\t"
                                 "je 2f                                 \n\t"
                                 "kmovw %%eax, %%k1                     \n\t"
                                 "vmovdqu8 (%0), %%ymm1 %{%%k1%}              \n\t"
                                 "vpmaxub %%ymm1, %%ymm2, %%ymm0 %{%%k1%}              \n\t"
                                 "vpminub %%ymm0, %%ymm4, %%ymm0 %{%%k1%}              \n\t"
                                 "vmovdqu8 %%ymm0, (%1) %{%%k1%}              \n\t"
                                 ".align 16                             \n\t"
                                 "2:                                    \n\t"
                                 : "+r"(input), "+r"(output)
                                 : "r"(num32), "a"(resMask), "r"(U32(maxVal))
                                 : "%k1", "%ebx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "memory", "cc");
            break;
        }
        case ACTIVATION_SIGN: {
            __asm__ __volatile__("mov $0x80, %%ebx \n\t"
                                 "vmovd %%ebx, %%xmm1                    \n\t"
                                 "vpbroadcastb %%xmm1, %%ymm2            \n\t"
                                 "mov $129, %%ebx \n\t"
                                 "vmovd %%ebx, %%xmm1                    \n\t"
                                 "vpbroadcastb %%xmm1, %%ymm3            \n\t"
                                 "mov $127, %%ebx \n\t"
                                 "vmovd %%ebx, %%xmm1                    \n\t"
                                 "vpbroadcastb %%xmm1, %%ymm4            \n\t"
                                 "mov %2, %%ebx \n\t"
                                 "cmp $0x0, %%ebx                         \n\t"
                                 "je 1f                                    \n\t"
                                 ".align 16                             \n\t"
                                 "0:                                    \n\t"
                                 "vmovups (%0), %%ymm0              \n\t"

                                 "vpcmpltub %%ymm2, %%ymm0, %%k1              \n\t"
                                 "vpblendmb %%ymm3, %%ymm0, %%ymm0 %{%%k1%}              \n\t"
                                 "vpcmpltub %%ymm0, %%ymm2, %%k1              \n\t"
                                 "vpblendmb %%ymm4, %%ymm0, %%ymm0 %{%%k1%}              \n\t"
                                 "vmovups %%ymm0, (%1)              \n\t"
                                 "add $0x20, %0                         \n\t"
                                 "add $0x20, %1                         \n\t"
                                 "dec %%ebx                             \n\t"
                                 "jg 0b                                 \n\t"

                                 ".align 16                             \n\t"
                                 "1:                                    \n\t"
                                 "cmp $0x0, %%eax                       \n\t"
                                 "je 2f                                 \n\t"
                                 "kmovw %%eax, %%k2                     \n\t"
                                 "vmovdqu8 (%0), %%ymm0 %{%%k2%}              \n\t"
                                 "vpcmpltub %%ymm2, %%ymm0, %%k1              \n\t"
                                 "vpblendmb %%ymm3, %%ymm0, %%ymm0 %{%%k1%}              \n\t"
                                 "vpcmpltub %%ymm0, %%ymm2, %%k1              \n\t"
                                 "vpblendmb %%ymm4, %%ymm0, %%ymm0 %{%%k1%}              \n\t"
                                 "vmovdqu8 %%ymm0, (%1) %{%%k2%}              \n\t"
                                 ".align 16                             \n\t"
                                 "2:                                    \n\t"
                                 : "+r"(input), "+r"(output)
                                 : "r"(num32), "a"(resMask)
                                 : "%k1", "%k2", "%ebx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "memory", "cc");
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}

#endif  //CHEETAH_X86_FUNCTION_INT8_H
