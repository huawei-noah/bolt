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
#include "x86_avx2_expand.h"

EE check_fp32(TensorDesc inputDescA,
    const F32 *inputA,
    TensorDesc inputDescB,
    const F32 *inputB,
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
    I32 length = size / loopOuter;
    if (tensorNumElements(outputDesc) != loopOuter) {
        CHECK_STATUS(NOT_MATCH);
    }
    for (U32 j = 0; j < loopOuter; j++) {
        const F32 *arrayA = inputA + j * length;
        const F32 *arrayB = inputB + j * length;
        switch (checkMode) {
            case CHECK_GREAT: {
                __m256i count_v = _mm256_set1_epi32(0);
                I32 i = 0;
                for (; i < length - 7; i += 8) {
                    __m256 a = _mm256_loadu_ps(arrayA + i);
                    __m256 b = _mm256_loadu_ps(arrayA + i);
                    count_v = _mm256_add_epi32(
                        count_v, _mm256_cvtps_epi32(_mm256_cmp_ps(a, b, _CMP_GT_OS)));
                }
                I32 count = _mm256_hadd_u32(count_v);
                for (; i < length; i++) {
                    if (arrayA[i] > arrayB[i]) {
                        count++;
                    }
                }
                output[j] = (count == length);
                break;
            }
            case CHECK_GREATEQUAL: {
                __m256i count_v = _mm256_set1_epi32(0);
                I32 i = 0;
                for (; i < length - 7; i += 8) {
                    __m256 a = _mm256_loadu_ps(arrayA + i);
                    __m256 b = _mm256_loadu_ps(arrayA + i);
                    count_v = _mm256_add_epi32(
                        count_v, _mm256_cvtps_epi32(_mm256_cmp_ps(a, b, _CMP_GE_OS)));
                }
                I32 count = _mm256_hadd_u32(count_v);
                for (; i < length; i++) {
                    if (arrayA[i] >= arrayB[i]) {
                        count++;
                    }
                }
                output[j] = (count == length);
                break;
            }
            case CHECK_EQUAL: {
                __m256i count_v = _mm256_set1_epi32(0);
                I32 i = 0;
                for (; i < length - 7; i += 8) {
                    __m256 a = _mm256_loadu_ps(arrayA + i);
                    __m256 b = _mm256_loadu_ps(arrayA + i);
                    count_v = _mm256_add_epi32(
                        count_v, _mm256_cvtps_epi32(_mm256_cmp_ps(a, b, _CMP_EQ_OS)));
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
                CHECK_STATUS(NOT_SUPPORTED);
                break;
        }
    }
    return SUCCESS;
}
