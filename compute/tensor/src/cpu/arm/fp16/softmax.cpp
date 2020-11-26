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
#include "cpu/arm/fp16/tensor_computing_fp16.h"

void softmax_lastAxis_fp16(const F16 *input, I32 loopOuter, I32 loops, F16 *output)
{
    for (I32 i = 0; i < loopOuter; i++) {
        const F16 *inputPtr = input + i * loops;
        F16 *outputPtr = output + i * loops;

        float16x8_t max_v, sub_v, sum_v, tmp_v;
        F32 max_s, tmp_s;
        max_s = array_max_value_f16(inputPtr, loops);
        max_v = vdupq_n_f16(max_s);
        sum_v = vdupq_n_f16(0);

        I32 j = 0;
        F32 sum_s = 0;
        for (j = 0; j < loops - 7; j += 8) {
            float16x8_t in = vld1q_f16(inputPtr + j);
            sub_v = vsubq_f16(in, max_v);
            tmp_v = vexpq_f16_f32(sub_v);
            sum_v = vaddq_f16(sum_v, tmp_v);
            vst1q_f16(outputPtr + j, tmp_v);
        }
        sum_s += vaddvq_f16(sum_v);
        for (; j < loops; j++) {
            tmp_s = exp(inputPtr[j] - max_s);
            outputPtr[j] = tmp_s;
            sum_s += tmp_s;
        }
        array_scale_f16(outputPtr, outputPtr, loops, 1.0 / sum_s, 0);
    }
}

void softmax_anyAxis_fp16(const F16 *input, I32 loopOuter, I32 loops, I32 loopInner, F16 *output)
{
    std::vector<F16> buffer(loopInner * 2);
    F16 *maxBuffer = &buffer[0];
    F16 *sumBuffer = &buffer[loopInner];
    I32 k = 0;
    for (I32 i = 0; i < loopOuter; i++) {
        const F16 *inputPtrBase = input + i * loops * loopInner;
        F16 *outputPtrBase = output + i * loops * loopInner;

        memcpy(maxBuffer, inputPtrBase, loopInner * sizeof(F16));
        memset(sumBuffer, 0, loopInner * sizeof(F16));
        for (I32 j = 1; j < loops; j++) {
            const F16 *inputPtr = inputPtrBase + j * loopInner;
            for (k = 0; k < loopInner - 7; k += 8) {
                float16x8_t in_v = vld1q_f16(inputPtr + k);
                float16x8_t out_v = vld1q_f16(maxBuffer + k);
                float16x8_t max_v = vmaxq_f16(in_v, out_v);
                vst1q_f16(maxBuffer + k, max_v);
            }
            for (; k < loopInner; k++) {
                maxBuffer[k] = UNI_MAX(maxBuffer[k], inputPtr[k]);
            }
        }
        for (I32 j = 0; j < loops; j++) {
            const F16 *inputPtr = inputPtrBase + j * loopInner;
            F16 *outputPtr = outputPtrBase + j * loopInner;
            for (k = 0; k < loopInner - 7; k += 8) {
                float16x8_t in_v = vld1q_f16(inputPtr + k);
                float16x8_t max_v = vld1q_f16(maxBuffer + k);
                float16x8_t sub_v = vsubq_f16(in_v, max_v);
                float16x8_t exp_v = vexpq_f16_f32(sub_v);
                float16x8_t sum_v = vld1q_f16(sumBuffer + k);
                sum_v = vaddq_f16(sum_v, exp_v);
                vst1q_f16(sumBuffer + k, sum_v);
                vst1q_f16(outputPtr + k, exp_v);
            }
            for (; k < loopInner; k++) {
                outputPtr[k] = exp(inputPtr[k] - maxBuffer[k]);
                sumBuffer[k] += outputPtr[k];
            }
        }
        for (I32 j = 0; j < loops; j++) {
            F16 *outputPtr = outputPtrBase + j * loopInner;
            for (k = 0; k < loopInner - 7; k += 8) {
                float16x8_t out_v = vld1q_f16(outputPtr + k);
                float16x8_t sum_v = vld1q_f16(sumBuffer + k);
                out_v = vdivq_f16(out_v, sum_v);
                vst1q_f16(outputPtr + k, out_v);
            }
            for (; k < loopInner; k++) {
                outputPtr[k] /= sumBuffer[k];
            }
        }
    }
}

EE softmax_fp16(TensorDesc inputDesc, const F16 *input, int axis, TensorDesc outputDesc, F16 *output)
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
            softmax_anyAxis_fp16(input, loopOuter, loops, loopInner, output);
        } else {
            softmax_lastAxis_fp16(input, loopOuter, loops, output);
        }
    } else {
        CHECK_REQUIREMENT(DF_NCHWC8 != inputDesc.df);
        softmax_anyAxis_fp16(input, loopOuter, loops, loopInner, output);
    }
    return SUCCESS;
}
