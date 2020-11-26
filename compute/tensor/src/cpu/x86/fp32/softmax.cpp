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
#include <string.h>
#include "cpu/x86/fp32/tensor_computing_fp32.h"

void softmax_lastAxis_fp32(const F32 *input, I32 loopOuter, I32 loops, F32 *output)
{
    for (I32 i = 0; i < loopOuter; i++) {
        const F32 *inputPtr = input + i * loops;
        F32 *outputPtr = output + i * loops;

        __m256 max_v, sub_v, sum_v, tmp_v;
        F32 max_s, tmp_s;
        max_s = array_max_value_f32(inputPtr, loops);
        max_v = _mm256_set1_ps(max_s);
        sum_v = _mm256_set1_ps(0.f);

        I32 j = 0;
        F32 sum_s = 0;
        for (j = 0; j < loops - 7; j += 8) {
            __m256 in = _mm256_loadu_ps(inputPtr + j);
            sub_v = _mm256_sub_ps(in, max_v);
            tmp_v = _mm256_exp_ps(sub_v);
            sum_v = _mm256_add_ps(sum_v, tmp_v);
            _mm256_storeu_ps(outputPtr + j, tmp_v);
        }
        sum_s += _mm256_sum_ps(sum_v);
        for (; j < loops; j++) {
            tmp_s = exp(inputPtr[j] - max_s);
            outputPtr[j] = tmp_s;
            sum_s += tmp_s;
        }
        array_scale_f32(outputPtr, outputPtr, loops, 1.0 / sum_s, 0);
    }
}

void softmax_anyAxis_fp32(const F32 *input, I32 loopOuter, I32 loops, I32 loopInner, F32 *output)
{
    std::vector<F32> buffer(loopInner * 2);
    F32 *maxBuffer = &buffer[0];
    F32 *sumBuffer = &buffer[loopInner];
    I32 k = 0;
    for (I32 i = 0; i < loopOuter; i++) {
        const F32 *inputPtrBase = input + i * loops * loopInner;
        F32 *outputPtrBase = output + i * loops * loopInner;

        memcpy(maxBuffer, inputPtrBase, loopInner * sizeof(F32));
        memset(sumBuffer, 0, loopInner * sizeof(F32));
        for (I32 j = 1; j < loops; j++) {
            const F32 *inputPtr = inputPtrBase + j * loopInner;
            for (k = 0; k < loopInner - 7; k += 8) {
                __m256 in_v = _mm256_loadu_ps(inputPtr + k);
                __m256 out_v = _mm256_loadu_ps(maxBuffer + k);
                __m256 max_v = _mm256_max_ps(in_v, out_v);
                _mm256_storeu_ps(maxBuffer + k, max_v);
            }
            for (; k < loopInner; k++) {
                maxBuffer[k] = UNI_MAX(maxBuffer[k], inputPtr[k]);
            }
        }
        for (I32 j = 0; j < loops; j++) {
            const F32 *inputPtr = inputPtrBase + j * loopInner;
            F32 *outputPtr = outputPtrBase + j * loopInner;
            for (k = 0; k < loopInner - 7; k += 8) {
                __m256 in_v = _mm256_loadu_ps(inputPtr + k);
                __m256 max_v = _mm256_loadu_ps(maxBuffer + k);
                __m256 sub_v = _mm256_sub_ps(in_v, max_v);
                __m256 exp_v = _mm256_exp_ps(sub_v);
                __m256 sum_v = _mm256_loadu_ps(sumBuffer + k);
                sum_v = _mm256_add_ps(sum_v, exp_v);
                _mm256_storeu_ps(sumBuffer + k, sum_v);
                _mm256_storeu_ps(outputPtr + k, exp_v);
            }
            for (; k < loopInner; k++) {
                outputPtr[k] = exp(inputPtr[k] - maxBuffer[k]);
                sumBuffer[k] += outputPtr[k];
            }
        }
        for (I32 j = 0; j < loops; j++) {
            F32 *outputPtr = outputPtrBase + j * loopInner;
            for (k = 0; k < loopInner - 7; k += 8) {
                __m256 out_v = _mm256_loadu_ps(outputPtr + k);
                __m256 sum_v = _mm256_loadu_ps(sumBuffer + k);
                out_v = _mm256_div_ps(out_v, sum_v);
                _mm256_storeu_ps(outputPtr + k, out_v);
            }
            for (; k < loopInner; k++) {
                outputPtr[k] /= sumBuffer[k];
            }
        }
    }
}

EE softmax_fp32(TensorDesc inputDesc, const F32 *input, int axis, TensorDesc outputDesc, F32 *output)
{
    UNUSED(outputDesc);
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }

    U32 size = tensorNumElements(inputDesc);
    axis = (axis + inputDesc.nDims) % inputDesc.nDims;
    axis = inputDesc.nDims - 1 - axis;
    I32 loops = inputDesc.dims[axis];

    I32 loopInner = 1;
    for (int i = 0; i < axis; i++) {
        loopInner *= inputDesc.dims[i];
    }
    U32 loopOuter = size / loops / loopInner;

    if (loopInner == 1) {
        if (DF_NCHWC8 == inputDesc.df && 4 == inputDesc.nDims &&
            (inputDesc.dims[1] != 1 || inputDesc.dims[0] != 1)) {
            CHECK_REQUIREMENT(2 != axis);
            loopInner *= 8;
            loopOuter /= 8;
            softmax_anyAxis_fp32(input, loopOuter, loops, loopInner, output);
        } else {
            softmax_lastAxis_fp32(input, loopOuter, loops, output);
        }
    } else {
        CHECK_REQUIREMENT(DF_NCHWC8 != inputDesc.df);
        softmax_anyAxis_fp32(input, loopOuter, loops, loopInner, output);
    }
    return SUCCESS;
}
