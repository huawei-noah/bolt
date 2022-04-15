// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/tensor_computing_x86.h"
#include "x86_avx2_expand.h"
#ifdef _USE_FP32
#include "cpu/x86/fp32/tensor_computing_fp32.h"
#endif

template <typename T>
EE check_u32(TensorDesc inputDescA,
    const T *inputA,
    TensorDesc inputDescB,
    const T *inputB,
    CheckMode checkMode,
    TensorDesc outputDesc,
    I32 *output)
{
    if (nullptr == inputA || nullptr == inputB || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }

    if (tensorNumElements(inputDescA) != tensorNumElements(inputDescB)) {
        CHECK_STATUS(NOT_MATCH);
    }

    U32 size = tensorNumElements(inputDescA);
    U32 loopOuter = inputDescA.dims[inputDescA.nDims - 1];
    if (tensorNumElements(outputDesc) != loopOuter) {
        CHECK_STATUS(NOT_MATCH);
    }
    I32 length = size / loopOuter;
    for (U32 j = 0; j < loopOuter; j++) {
        const T *arrayA = inputA + j * length;
        const T *arrayB = inputB + j * length;
        __m256i count_v = _mm256_set1_epi32(0);
        __m256i one_v = _mm256_set1_epi32(1);
        switch (checkMode) {
            case CHECK_GREAT: {
                I32 i = 0;
                for (; i < length - 7; i += 8) {
                    __m256i a = _mm256_loadu_si256((__m256i *)(arrayA + i));
                    __m256i b = _mm256_loadu_si256((__m256i *)(arrayB + i));
                    count_v = _mm256_add_epi32(
                        count_v, _mm256_and_si256(one_v, _mm256_cmpgt_epi32(a, b)));
                }
                I32 count = _mm256_hadd_u32(count_v);
                for (; i < length; i++) {
                    if (arrayA[i] == arrayB[i]) {
                        count++;
                    }
                }
                output[j] = (count == length);
                break;
            }
            case CHECK_GREATEQUAL: {
                I32 i = 0;
                for (; i < length - 7; i += 8) {
                    __m256i a = _mm256_loadu_si256((__m256i *)(arrayA + i));
                    __m256i b = _mm256_loadu_si256((__m256i *)(arrayB + i));
                    __m256i cmp =
                        _mm256_or_si256(_mm256_cmpeq_epi32(a, b), _mm256_cmpgt_epi32(a, b));
                    count_v = _mm256_add_epi32(count_v, _mm256_and_si256(one_v, cmp));
                }
                I32 count = _mm256_hadd_u32(count_v);
                for (; i < length; i++) {
                    if (arrayA[i] == arrayB[i]) {
                        count++;
                    }
                }
                output[j] = (count == length);
                break;
            }
            case CHECK_EQUAL: {
                I32 i = 0;
                for (; i < length - 7; i += 8) {
                    __m256i a = _mm256_loadu_si256((__m256i *)(arrayA + i));
                    __m256i b = _mm256_loadu_si256((__m256i *)(arrayB + i));
                    count_v = _mm256_add_epi32(
                        count_v, _mm256_and_si256(one_v, _mm256_cmpeq_epi32(a, b)));
                }
                I32 count = _mm256_hadd_u32(count_v);
                for (; i < length; i++) {
                    if (arrayA[i] == arrayB[i]) {
                        count++;
                    }
                }
                output[j] = (count == length);
                break;
            }
            default:
                return NOT_SUPPORTED;
                break;
        }
    }
    return SUCCESS;
}

template <typename TA, typename TB>
EE check_kernel(TensorDesc inputDescA,
    const TA *inputA,
    TensorDesc inputDescB,
    const TB *inputB,
    CheckMode checkMode,
    TensorDesc outputDesc,
    I32 *output)
{
    if (nullptr == inputA || nullptr == inputB || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }

    if (tensorNumElements(inputDescA) != tensorNumElements(inputDescB)) {
        CHECK_STATUS(NOT_MATCH);
    }

    U32 size = tensorNumElements(inputDescA);
    U32 loopOuter = inputDescA.dims[inputDescA.nDims - 1];
    if (tensorNumElements(outputDesc) != loopOuter) {
        CHECK_STATUS(NOT_MATCH);
    }
    I32 length = size / loopOuter;

    for (U32 j = 0; j < loopOuter; j++) {
        const TA *arrayA = inputA + j * length;
        const TB *arrayB = inputB + j * length;
        switch (checkMode) {
            case CHECK_GREAT: {
                output[j] = 1;
                for (I32 i = 0; i < length; i++) {
                    if (arrayA[i] <= (TA)arrayB[i]) {
                        output[j] = 0;
                        break;
                    }
                }
                break;
            }
            case CHECK_GREATEQUAL: {
                output[j] = 1;
                for (I32 i = 0; i < length; i++) {
                    if (arrayA[i] < (TA)arrayB[i]) {
                        output[j] = 0;
                        break;
                    }
                }
                break;
            }
            case CHECK_EQUAL: {
                output[j] = 1;
                for (I32 i = 0; i < length; i++) {
                    if (arrayA[i] != (TA)arrayB[i]) {
                        output[j] = 0;
                        break;
                    }
                }
                break;
            }
            default:
                return NOT_SUPPORTED;
                break;
        }
    }
    return SUCCESS;
}

template <typename TA>
EE check_wrapper(TensorDesc inputDescA,
    const TA *inputA,
    TensorDesc inputDescB,
    const void *inputB,
    CheckMode checkMode,
    TensorDesc outputDesc,
    I32 *output)
{
    EE ret = SUCCESS;
    switch (inputDescB.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = check_kernel<TA, F32>(
                inputDescA, inputA, inputDescB, (const F32 *)inputB, checkMode, outputDesc, output);
            break;
        }
#endif
        case DT_U32: {
            ret = check_kernel<TA, U32>(
                inputDescA, inputA, inputDescB, (const U32 *)inputB, checkMode, outputDesc, output);
            break;
        }
        case DT_I32: {
            ret = check_kernel<TA, I32>(
                inputDescA, inputA, inputDescB, (const I32 *)inputB, checkMode, outputDesc, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE check_x86(TensorDesc inputDescA,
    const void *inputA,
    TensorDesc inputDescB,
    const void *inputB,
    CheckParamSpec p,
    TensorDesc outputDesc,
    void *output)
{
    DataType idt = inputDescA.dt;
    EE ret = SUCCESS;

    if (idt != inputDescB.dt) {
        switch (idt) {
#ifdef _USE_FP32
            case DT_F32: {
                ret = check_wrapper<F32>(inputDescA, (const F32 *)inputA, inputDescB, inputB,
                    p.check_mode, outputDesc, (I32 *)output);
                break;
            }
#endif
            case DT_U32: {
                ret = check_wrapper<U32>(inputDescA, (const U32 *)inputA, inputDescB, inputB,
                    p.check_mode, outputDesc, (I32 *)output);
                break;
            }
            case DT_I32: {
                ret = check_wrapper<I32>(inputDescA, (const I32 *)inputA, inputDescB, inputB,
                    p.check_mode, outputDesc, (I32 *)output);
                break;
            }
            default:
                ret = NOT_SUPPORTED;
                break;
        }
        return ret;
    }

    switch (idt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = check_fp32(inputDescA, (const F32 *)inputA, inputDescB, (const F32 *)inputB,
                p.check_mode, outputDesc, (I32 *)output);
            break;
        }
#endif
        case DT_U32: {
            ret = check_u32<U32>(inputDescA, (const U32 *)inputA, inputDescB, (const U32 *)inputB,
                p.check_mode, outputDesc, (I32 *)output);
            break;
        }
        case DT_I32: {
            ret = check_u32<I32>(inputDescA, (const I32 *)inputA, inputDescB, (const I32 *)inputB,
                p.check_mode, outputDesc, (I32 *)output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}
