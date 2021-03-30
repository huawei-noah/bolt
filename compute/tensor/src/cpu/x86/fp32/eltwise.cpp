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
#include "cpu/cpu_functions.h"

#define eltwise_kernel(velt, velts)                                                               \
    __asm__ __volatile__("vxorps %%ymm8, %%ymm8, %%ymm8       \n\t"                               \
                         "vxorps %%ymm9, %%ymm9, %%ymm9       \n\t"                               \
                         "vxorps %%ymm10, %%ymm10, %%ymm10    \n\t"                               \
                         "vxorps %%ymm11, %%ymm11, %%ymm11    \n\t"                               \
                         "mov %0, %%ecx                       \n\t"                               \
                         "cmp $32, %%ecx                      \n\t"                               \
                         "jl 1f                               \n\t"                               \
                         ".align 16                           \n\t"                               \
                         "0:                                  \n\t"                               \
                         "vmovups (%1), %%ymm0                \n\t"                               \
                         "vmovups 0x20(%1), %%ymm1            \n\t"                               \
                         "vmovups 0x40(%1), %%ymm2            \n\t"                               \
                         "vmovups 0x60(%1), %%ymm3            \n\t"                               \
                         "" #velt " (%2), %%ymm0, %%ymm8         \n\t"                            \
                         "" #velt " 0x20(%2), %%ymm1, %%ymm9     \n\t"                            \
                         "" #velt " 0x40(%2), %%ymm2, %%ymm10    \n\t"                            \
                         "" #velt " 0x60(%2), %%ymm3, %%ymm11    \n\t"                            \
                         "vmovups %%ymm8, (%3)                \n\t"                               \
                         "vmovups %%ymm9, 0x20(%3)            \n\t"                               \
                         "vmovups %%ymm10, 0x40(%3)           \n\t"                               \
                         "vmovups %%ymm11, 0x60(%3)           \n\t"                               \
                         "add $0x80, %2                       \n\t"                               \
                         "add $0x80, %1                       \n\t"                               \
                         "add $0x80, %3                       \n\t"                               \
                         "sub $32, %%ecx                      \n\t"                               \
                         "cmp $32, %%ecx                      \n\t"                               \
                         "jge 0b                              \n\t"                               \
                         ".align 16                           \n\t"                               \
                         "1:                                  \n\t"                               \
                         "cmp $16, %%ecx                      \n\t"                               \
                         "jl 2f                               \n\t"                               \
                         "vmovups (%1), %%ymm0                \n\t"                               \
                         "vmovups 0x20(%1), %%ymm1            \n\t"                               \
                         "" #velt " (%2), %%ymm0, %%ymm8         \n\t"                            \
                         "" #velt " 0x20(%2), %%ymm1, %%ymm9     \n\t"                            \
                         "vmovups %%ymm8, (%3)                \n\t"                               \
                         "vmovups %%ymm9, 0x20(%3)            \n\t"                               \
                         "add $0x40, %2                       \n\t"                               \
                         "add $0x40, %1                       \n\t"                               \
                         "add $0x40, %3                       \n\t"                               \
                         "sub $16, %%ecx                      \n\t"                               \
                         ".align 16                           \n\t"                               \
                         "2:                                  \n\t"                               \
                         "cmp $8, %%ecx                       \n\t"                               \
                         "jl 3f                               \n\t"                               \
                         "vmovups (%1), %%ymm0                \n\t"                               \
                         "" #velt " (%2), %%ymm0, %%ymm8         \n\t"                            \
                         "vmovups %%ymm8, (%3)                \n\t"                               \
                         "add $0x20, %2                       \n\t"                               \
                         "add $0x20, %1                       \n\t"                               \
                         "add $0x20, %3                       \n\t"                               \
                         "sub $8, %%ecx                       \n\t"                               \
                         ".align 16                           \n\t"                               \
                         "3:                                  \n\t"                               \
                         "cmp $4, %%ecx                       \n\t"                               \
                         "jl 4f                               \n\t"                               \
                         "vmovups (%1), %%xmm0                \n\t"                               \
                         "" #velt " (%2), %%xmm0, %%xmm8         \n\t"                            \
                         "vmovups %%xmm8, (%3)                \n\t"                               \
                         "add $0x10, %2                       \n\t"                               \
                         "add $0x10, %1                       \n\t"                               \
                         "add $0x10, %3                       \n\t"                               \
                         "sub $4, %%ecx                       \n\t"                               \
                         ".align 16                           \n\t"                               \
                         "4:                                  \n\t"                               \
                         "cmp $1, %%ecx                       \n\t"                               \
                         "jl 6f                               \n\t"                               \
                         ".align 16                           \n\t"                               \
                         "5:                                  \n\t"                               \
                         "vmovss (%1), %%xmm0                 \n\t"                               \
                         "" #velts " (%2), %%xmm0, %%xmm8         \n\t"                           \
                         "vmovss %%xmm8, (%3)                 \n\t"                               \
                         "add $0x4, %2                        \n\t"                               \
                         "add $0x4, %1                        \n\t"                               \
                         "add $0x4, %3                        \n\t"                               \
                         "sub $1, %%ecx                       \n\t"                               \
                         "cmp $1, %%ecx                       \n\t"                               \
                         "jge 5b                              \n\t"                               \
                         ".align 16                           \n\t"                               \
                         "6:                                  \n\t"                               \
                         :                                                                        \
                         : "r"(len), "r"(input[0]), "r"(input[1]), "r"(output)                    \
                         : "%ecx", "%xmm0", "%xmm8", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm8", \
                         "%ymm9", "%ymm10", "%ymm11", "memory");

EE eltwise_fp32(std::vector<void *> input,
    std::vector<int> inputSize,
    U32 num,
    U32 len,
    void *output,
    EltwiseMode eltwiseMode)
{
    if (((int)num == 2) && (inputSize[0] == (int)len) && (inputSize[0] == inputSize[1])) {
        switch (eltwiseMode) {
            case ELTWISE_SUM: {
                eltwise_kernel(vaddps, vaddss);
                break;
            }
            case ELTWISE_MAX: {
                eltwise_kernel(vmaxps, vmaxss);
                break;
            }
            case ELTWISE_PROD: {
                eltwise_kernel(vmulps, vmulss);
                break;
            }
            case ELTWISE_SUB: {
                eltwise_kernel(vsubps, vsubss);
                break;
            }
            case ELTWISE_DIV: {
                eltwise_kernel(vdivps, vdivss);
                break;
            }
            default:
                return NOT_SUPPORTED;
        }
        return SUCCESS;
    }

    F32 buffer[8];
    F32 *tmp = buffer;
    U32 len_tail = len % 8;
    U32 len_main = len - len_tail;
    F32 *output_ptr = (F32 *)output;
    for (U32 i = 0; i < len_main; i += 8) {
        get_vector<F32>((F32 *)input[0], inputSize[0], &tmp, 8, i, 8, buffer);
        __m256 tmp_v = _mm256_loadu_ps(tmp);
        for (U32 j = 1; j < num; j++) {
            get_vector<F32>((F32 *)input[j], inputSize[j], &tmp, 8, i, 8, buffer);
            __m256 value_v = _mm256_loadu_ps(tmp);
            switch (eltwiseMode) {
                case ELTWISE_SUM:
                    tmp_v = _mm256_add_ps(value_v, tmp_v);
                    break;
                case ELTWISE_MAX:
                    tmp_v = _mm256_max_ps(value_v, tmp_v);
                    break;
                case ELTWISE_PROD:
                    tmp_v = _mm256_mul_ps(value_v, tmp_v);
                    break;
                case ELTWISE_SUB:
                    tmp_v = _mm256_sub_ps(tmp_v, value_v);
                    break;
                case ELTWISE_DIV:
                    tmp_v = _mm256_div_ps(tmp_v, value_v);
                    break;
                default:
                    return NOT_SUPPORTED;
            }
        }
        _mm256_storeu_ps(output_ptr + i, tmp_v);
    }

    for (U32 i = len_main; i < len; i++) {
        get_vector<F32>((F32 *)input[0], inputSize[0], &tmp, 8, i, 1, buffer);
        F32 tmp_s = tmp[0];
        for (U32 j = 1; j < num; j++) {
            get_vector<F32>((F32 *)input[j], inputSize[j], &tmp, 8, i, 1, buffer);
            F32 value_s = tmp[0];
            switch (eltwiseMode) {
                case ELTWISE_SUM:
                    tmp_s = value_s + tmp_s;
                    break;
                case ELTWISE_MAX:
                    tmp_s = (value_s > tmp_s) ? value_s : tmp_s;
                    break;
                case ELTWISE_PROD:
                    tmp_s *= value_s;
                    break;
                default:
                    return NOT_SUPPORTED;
            }
        }
        output_ptr[i] = tmp_s;
    }
    return SUCCESS;
}
