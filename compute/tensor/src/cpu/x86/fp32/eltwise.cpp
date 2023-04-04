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

#define eltwise_kernel(velt, velts, block, in0, in1, out)                                         \
    __asm__ __volatile__("mov %0, %%ecx                       \n\t"                               \
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
                         : "r"(block), "r"(in0), "r"(in1), "r"(out)                               \
                         : "%ecx", "%xmm0", "%xmm8", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm8", \
                         "%ymm9", "%ymm10", "%ymm11", "memory");

EE eltwise_fp32(std::vector<void *> input,
    std::vector<int> inputSize,
    U32 num,
    U32 len,
    void *output,
    EltwiseMode eltwiseMode)
{
    EE ret = SUCCESS;
    if ((num == 2) && (inputSize[0] != 1) && (inputSize[1] != 1)) {
        F32 *in0 = (F32 *)input[0];
        F32 *in1 = (F32 *)input[1];
        F32 *out = (F32 *)output;
        len = UNI_MIN(inputSize[0], inputSize[1]);

#ifdef _USE_OPENMP
        U32 ompBlock = ((len + OMP_NUM_THREADS - 1) / OMP_NUM_THREADS + 7) / 8 * 8;
        U32 BLOCK = UNI_MAX(64, ompBlock);
        U32 blockNum = (len + BLOCK - 1) / BLOCK;
        int in_parallel = omp_in_parallel();
        if (in_parallel != 0) {
            BLOCK = len;
            blockNum = 1;
        }
#pragma omp parallel num_threads(OMP_NUM_THREADS) if (in_parallel == 0)
#endif
        {
            switch (eltwiseMode) {
                case ELTWISE_SUM: {
#ifdef _USE_OPENMP
#pragma omp for nowait
                    for (U32 ii = 0; ii < blockNum; ++ii) {
                        U32 off = ii * BLOCK;
                        U32 blockSize = UNI_MIN(len - off, BLOCK);
                        eltwise_kernel(vaddps, vaddss, blockSize, in0 + off, in1 + off, out + off);
                    }
#else
                    eltwise_kernel(vaddps, vaddss, len, in0, in1, out);
#endif
                    break;
                }
                case ELTWISE_MAX: {
#ifdef _USE_OPENMP
#pragma omp for nowait
                    for (U32 ii = 0; ii < blockNum; ++ii) {
                        U32 off = ii * BLOCK;
                        U32 blockSize = UNI_MIN(len - off, BLOCK);
                        eltwise_kernel(vmaxps, vmaxss, blockSize, in0 + off, in1 + off, out + off);
                    }
#else
                    eltwise_kernel(vmaxps, vmaxss, len, in0, in1, out);
#endif
                    break;
                }
                case ELTWISE_AND:
                case ELTWISE_PROD: {
#ifdef _USE_OPENMP
#pragma omp for nowait
                    for (U32 ii = 0; ii < blockNum; ++ii) {
                        U32 off = ii * BLOCK;
                        U32 blockSize = UNI_MIN(len - off, BLOCK);
                        eltwise_kernel(vmulps, vmulss, blockSize, in0 + off, in1 + off, out + off);
                    }
#else
                    eltwise_kernel(vmulps, vmulss, len, in0, in1, out);
#endif
                    break;
                }
                case ELTWISE_SUB: {
#ifdef _USE_OPENMP
#pragma omp for nowait
                    for (U32 ii = 0; ii < blockNum; ++ii) {
                        U32 off = ii * BLOCK;
                        U32 blockSize = UNI_MIN(len - off, BLOCK);
                        eltwise_kernel(vsubps, vsubss, blockSize, in0 + off, in1 + off, out + off);
                    }
#else
                    eltwise_kernel(vsubps, vsubss, len, in0, in1, out);
#endif
                    break;
                }
                case ELTWISE_DIV: {
#ifdef _USE_OPENMP
#pragma omp for nowait
                    for (U32 ii = 0; ii < blockNum; ++ii) {
                        U32 off = ii * BLOCK;
                        U32 blockSize = UNI_MIN(len - off, BLOCK);
                        eltwise_kernel(vdivps, vdivss, blockSize, in0 + off, in1 + off, out + off);
                    }
#else
                    eltwise_kernel(vdivps, vdivss, len, in0, in1, out);
#endif
                    break;
                }
                default:
                    ret = NOT_SUPPORTED;
                    break;
            }
        }
        return ret;
    }

    U32 len_tail = len % 8;
    U32 len_main = len - len_tail;
#ifdef _USE_OPENMP
    int in_parallel = omp_in_parallel();
#pragma omp parallel num_threads(OMP_NUM_THREADS) if (in_parallel == 0)
#endif
    {
        F32 buffer[8];
        F32 *tmp = buffer;
        F32 *output_ptr = (F32 *)output;
#ifdef _USE_OPENMP
#pragma omp for nowait
#endif
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
                        ret = NOT_SUPPORTED;
                        break;
                }
            }
            _mm256_storeu_ps(output_ptr + i, tmp_v);
        }

#ifdef _USE_OPENMP
#pragma omp for nowait
#endif
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
                    case ELTWISE_SUB:
                        tmp_s = tmp_s - value_s;
                        break;
                    case ELTWISE_DIV:
                        tmp_s = tmp_s / value_s;
                        break;
                    default:
                        ret = NOT_SUPPORTED;
                        break;
                }
            }
            output_ptr[i] = tmp_s;
        }
    }
    return ret;
}

EE eltwise_i32(std::vector<void *> input,
    std::vector<int> inputSize,
    U32 num,
    U32 len,
    void *output,
    EltwiseMode eltwiseMode)
{
    EE ret = SUCCESS;
    if ((num == 2) && (inputSize[0] != 1) && (inputSize[1] != 1)) {
        I32 *in0 = (I32 *)input[0];
        I32 *in1 = (I32 *)input[1];
        I32 *out = (I32 *)output;

#ifdef _USE_OPENMP
        U32 ompBlock = ((len + OMP_NUM_THREADS - 1) / OMP_NUM_THREADS + 7) / 8 * 8;
        U32 BLOCK = UNI_MAX(64, ompBlock);
        U32 blockNum = (len + BLOCK - 1) / BLOCK;
        int in_parallel = omp_in_parallel();
        if (in_parallel != 0) {
            BLOCK = len;
            blockNum = 1;
        }
#pragma omp parallel num_threads(OMP_NUM_THREADS) if (in_parallel == 0)
#endif
        {
            switch (eltwiseMode) {
                case ELTWISE_SUM: {
#ifdef _USE_OPENMP
#pragma omp for nowait
                    for (U32 ii = 0; ii < blockNum; ++ii) {
                        U32 off = ii * BLOCK;
                        U32 blockSize = UNI_MIN(len - off, BLOCK);
                        eltwise_kernel(vpaddd, vpaddd, blockSize, in0 + off, in1 + off, out + off);
                    }
#else
                    eltwise_kernel(vpaddd, vpaddd, len, in0, in1, out);
#endif
                    break;
                }
                case ELTWISE_MAX: {
#ifdef _USE_OPENMP
#pragma omp for nowait
                    for (U32 ii = 0; ii < blockNum; ++ii) {
                        U32 off = ii * BLOCK;
                        U32 blockSize = UNI_MIN(len - off, BLOCK);
                        eltwise_kernel(vpmaxsd, vpmaxsd, blockSize, in0 + off, in1 + off, out + off);
                    }
#else
                    eltwise_kernel(vpmaxsd, vpmaxsd, len, in0, in1, out);
#endif
                    break;
                }
                case ELTWISE_PROD: {
#ifdef _USE_OPENMP
#pragma omp for nowait
                    for (U32 ii = 0; ii < blockNum; ++ii) {
                        U32 off = ii * BLOCK;
                        U32 blockSize = UNI_MIN(len - off, BLOCK);
                        eltwise_kernel(vpmulld, vpmulld, blockSize, in0 + off, in1 + off, out + off);
                    }
#else
                    eltwise_kernel(vpmulld, vpmulld, len, in0, in1, out);
#endif
                    break;
                }
                case ELTWISE_SUB: {
#ifdef _USE_OPENMP
#pragma omp for nowait
                    for (U32 ii = 0; ii < blockNum; ++ii) {
                        U32 off = ii * BLOCK;
                        U32 blockSize = UNI_MIN(len - off, BLOCK);
                        eltwise_kernel(vpsubd, vpsubd, blockSize, in0 + off, in1 + off, out + off);
                    }
#else
                    eltwise_kernel(vpsubd, vpsubd, len, in0, in1, out);
#endif
                    break;
                }
                default:
                    ret = NOT_SUPPORTED;
                    break;
            }
        }
        return ret;
    }

    U32 len_tail = len % 8;
    U32 len_main = len - len_tail;
#ifdef _USE_OPENMP
    int in_parallel = omp_in_parallel();
#pragma omp parallel num_threads(OMP_NUM_THREADS) if (in_parallel == 0)
#endif
    {
        I32 buffer[8];
        I32 *tmp = buffer;
        I32 *output_ptr = (I32 *)output;
#ifdef _USE_OPENMP
#pragma omp for nowait
#endif
        for (U32 i = 0; i < len_main; i += 8) {
            get_vector<I32>((I32 *)input[0], inputSize[0], &tmp, 8, i, 8, buffer);
            __m256i tmp_v = _mm256_loadu_si256((const __m256i *)tmp);
            for (U32 j = 1; j < num; j++) {
                get_vector<I32>((I32 *)input[j], inputSize[j], &tmp, 8, i, 8, buffer);
                __m256i value_v = _mm256_loadu_si256((const __m256i *)tmp);
                switch (eltwiseMode) {
                    case ELTWISE_SUM:
                        tmp_v = _mm256_add_epi32(value_v, tmp_v);
                        break;
                    case ELTWISE_MAX:
                        tmp_v = _mm256_max_epi32(value_v, tmp_v);
                        break;
                    case ELTWISE_PROD:
                        tmp_v = _mm256_mullo_epi32(value_v, tmp_v);
                        break;
                    case ELTWISE_SUB:
                        tmp_v = _mm256_sub_epi32(tmp_v, value_v);
                        break;
                    default:
                        ret = NOT_SUPPORTED;
                }
            }
            _mm256_storeu_si256((__m256i *)(output_ptr + i), tmp_v);
        }

#ifdef _USE_OPENMP
#pragma omp for nowait
#endif
        for (U32 i = len_main; i < len; i++) {
            get_vector<I32>((I32 *)input[0], inputSize[0], &tmp, 8, i, 1, buffer);
            I32 tmp_s = tmp[0];
            for (U32 j = 1; j < num; j++) {
                get_vector<I32>((I32 *)input[j], inputSize[j], &tmp, 8, i, 1, buffer);
                I32 value_s = tmp[0];
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
                    case ELTWISE_SUB:
                        tmp_s = tmp_s - value_s;
                        break;
                    case ELTWISE_DIV:
                        tmp_s = tmp_s / value_s;
                        break;
                    default:
                        ret = NOT_SUPPORTED;
                }
            }
            output_ptr[i] = tmp_s;
        }
    }
    return ret;
}

EE eltwise_u8(std::vector<void *> input,
    std::vector<int> inputSize,
    U32 num,
    U32 len,
    void *output,
    EltwiseMode eltwiseMode)
{
    EE ret = SUCCESS;
    U32 len_tail = len % 32;
    U32 len_main = len - len_tail;
#ifdef _USE_OPENMP
    int in_parallel = omp_in_parallel();
#pragma omp parallel num_threads(OMP_NUM_THREADS) if (in_parallel == 0)
#endif
    {
        U8 buffer[32];
        U8 *tmp = buffer;
        U8 *output_ptr = (U8 *)output;
#ifdef _USE_OPENMP
#pragma omp for nowait
#endif
        for (U32 i = 0; i < len_main; i += 32) {
            get_vector<U8>((U8 *)input[0], inputSize[0], &tmp, 32, i, 32, buffer);
            __m256i tmp_v = _mm256_loadu_si256((__m256i const *)tmp);
            for (U32 j = 1; j < num; j++) {
                get_vector<U8>((U8 *)input[j], inputSize[j], &tmp, 32, i, 32, buffer);
                __m256i value_v = _mm256_loadu_si256((__m256i const *)tmp);
                switch (eltwiseMode) {
                    case ELTWISE_AND:
                        tmp_v = _mm256_and_si256(value_v, tmp_v);
                        break;
                    case ELTWISE_OR:
                        tmp_v = _mm256_or_si256(value_v, tmp_v);
                        break;
                    case ELTWISE_XOR:
                        tmp_v = _mm256_xor_si256(value_v, tmp_v);
                        break;
                    default:
                        ret = NOT_SUPPORTED;
                        break;
                }
            }
            _mm256_storeu_si256((__m256i *)(output_ptr + i), tmp_v);
        }

#ifdef _USE_OPENMP
#pragma omp for nowait
#endif
        for (U32 i = len_main; i < len; i++) {
            get_vector<U8>((U8 *)input[0], inputSize[0], &tmp, 32, i, 1, buffer);
            U8 tmp_s = tmp[0];
            for (U32 j = 1; j < num; j++) {
                get_vector<U8>((U8 *)input[j], inputSize[j], &tmp, 32, i, 1, buffer);
                U8 value_s = tmp[0];
                switch (eltwiseMode) {
                    case ELTWISE_AND:
                        tmp_s = value_s & tmp_s;
                        break;
                    case ELTWISE_OR:
                        tmp_s = value_s | tmp_s;
                        break;
                    case ELTWISE_XOR:
                        tmp_s = value_s ^ tmp_s;
                        break;
                    default:
                        ret = NOT_SUPPORTED;
                        break;
                }
            }
            output_ptr[i] = tmp_s;
        }
    }
    return ret;
}
