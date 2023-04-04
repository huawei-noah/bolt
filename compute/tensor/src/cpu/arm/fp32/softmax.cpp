// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/fp32/tensor_computing_fp32.h"
#include "tensor_transpose.h"

template <bool logsoftmax>
static void softmax_lastAxis_fp32(const F32 *input, I32 loopOuter, I32 loops, F32 *output)
{
    for (I32 i = 0; i < loopOuter; i++) {
        const F32 *inputPtr = input + i * loops;
        F32 *outputPtr = output + i * loops;

        float32x4_t max_v, tmp_v;
        F32 max_s, tmp_s;
        if (!logsoftmax) {
            array_minmax_value_f32(inputPtr, loops, 2, &max_s);
            max_v = vdupq_n_f32(max_s);
        }
        I32 j = 0;
        float32x4_t sum_v = vdupq_n_f32(0);
#pragma unroll(4)
        for (; j < loops - 3; j += 4) {
            float32x4_t in = vld1q_f32(inputPtr + j);
            if (!logsoftmax) {
                in = vsubq_f32(in, max_v);
            }
            tmp_v = vexpq_f32_03_percent_error(in);
            sum_v = vaddq_f32(sum_v, tmp_v);
            if (!logsoftmax) {
                vst1q_f32(outputPtr + j, tmp_v);
            }
        }
        F32 sum_s = vaddvq_f32(sum_v);
        for (; j < loops; j++) {
            if (logsoftmax) {
                tmp_s = exp(inputPtr[j]);
            } else {
                tmp_s = exp(inputPtr[j] - max_s);
                outputPtr[j] = tmp_s;
            }
            sum_s += tmp_s;
        }
        if (logsoftmax) {
            array_scale_f32(inputPtr, outputPtr, loops, 1.0, -log(sum_s));
        } else {
            array_scale_f32(outputPtr, outputPtr, loops, 1.0 / sum_s, 0);
        }
    }
}

template <bool logsoftmax>
void softmax_anyAxis_fp32(const F32 *input, I32 loopOuter, I32 loops, I32 loopInner, F32 *output)
{
    std::vector<F32> buffer(loopInner * 2);
    F32 *maxBuffer = &buffer[0];
    F32 *sumBuffer = &buffer[loopInner];
    I32 k = 0;
    F32 tmp_s;
    for (I32 i = 0; i < loopOuter; i++) {
        const F32 *inputPtrBase = input + i * loops * loopInner;
        F32 *outputPtrBase = output + i * loops * loopInner;

        UNI_MEMSET(sumBuffer, 0, loopInner * sizeof(F32));
        if (!logsoftmax) {
            UNI_MEMCPY(maxBuffer, inputPtrBase, loopInner * sizeof(F32));
            for (I32 j = 1; j < loops; j++) {
                const F32 *inputPtr = inputPtrBase + j * loopInner;
                for (k = 0; k < loopInner - 3; k += 4) {
                    float32x4_t in_v = vld1q_f32(inputPtr + k);
                    float32x4_t out_v = vld1q_f32(maxBuffer + k);
                    float32x4_t max_v = vmaxq_f32(in_v, out_v);
                    vst1q_f32(maxBuffer + k, max_v);
                }
                for (; k < loopInner; k++) {
                    maxBuffer[k] = UNI_MAX(maxBuffer[k], inputPtr[k]);
                }
            }
        }
        for (I32 j = 0; j < loops; j++) {
            const F32 *inputPtr = inputPtrBase + j * loopInner;
            F32 *outputPtr = outputPtrBase + j * loopInner;
            for (k = 0; k < loopInner - 3; k += 4) {
                float32x4_t in_v = vld1q_f32(inputPtr + k);
                if (!logsoftmax) {
                    in_v = vsubq_f32(in_v, vld1q_f32(maxBuffer + k));
                }
                float32x4_t exp_v = vexpq_f32_03_percent_error(in_v);
                float32x4_t sum_v = vld1q_f32(sumBuffer + k);
                sum_v = vaddq_f32(sum_v, exp_v);
                vst1q_f32(sumBuffer + k, sum_v);
                if (!logsoftmax) {
                    vst1q_f32(outputPtr + k, exp_v);
                }
            }
            for (; k < loopInner; k++) {
                if (logsoftmax) {
                    tmp_s = exp(inputPtr[k]);
                } else {
                    tmp_s = exp(inputPtr[k] - maxBuffer[k]);
                    outputPtr[k] = tmp_s;
                }
                sumBuffer[k] += tmp_s;
            }
        }
        if (logsoftmax) {
            for (k = 0; k < loopInner - 3; k += 4) {
                float32x4_t sum_v = vld1q_f32(sumBuffer + k);
                sum_v = vlogq_f32(sum_v);
                vst1q_f32(sumBuffer + k, sum_v);
            }
            for (; k < loopInner; k++) {
                sumBuffer[k] = log(sumBuffer[k]);
            }
            for (I32 j = 0; j < loops; j++) {
                const F32 *inputPtr = inputPtrBase + j * loopInner;
                F32 *outputPtr = outputPtrBase + j * loopInner;
                for (k = 0; k < loopInner - 3; k += 4) {
                    float32x4_t out_v = vld1q_f32(inputPtr + k);
                    float32x4_t sum_v = vld1q_f32(sumBuffer + k);
                    out_v = vsubq_f32(out_v, sum_v);
                    vst1q_f32(outputPtr + k, out_v);
                }
                for (; k < loopInner; k++) {
                    outputPtr[k] -= sumBuffer[k];
                }
            }
        } else {
            for (I32 j = 0; j < loops; j++) {
                F32 *outputPtr = outputPtrBase + j * loopInner;
                for (k = 0; k < loopInner - 3; k += 4) {
                    float32x4_t out_v = vld1q_f32(outputPtr + k);
                    float32x4_t sum_v = vld1q_f32(sumBuffer + k);
                    out_v = vdivq_f32(out_v, sum_v);
                    vst1q_f32(outputPtr + k, out_v);
                }
                for (; k < loopInner; k++) {
                    outputPtr[k] /= sumBuffer[k];
                }
            }
        }
    }
}

template <bool logsoftmax>
static EE softmax_kernel(
    TensorDesc inputDesc, const F32 *input, int axis, TensorDesc outputDesc, F32 *output)
{
    UNUSED(outputDesc);
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }

    U32 size = tensorNumElements(inputDesc);
    int channel_axis = inputDesc.nDims - 2;
    axis = (axis + inputDesc.nDims) % inputDesc.nDims;
    axis = inputDesc.nDims - 1 - axis;
    std::vector<F32> buffer;
    if (inputDesc.df == DF_NCHWC8) {
        if (axis == channel_axis) {
            U32 hw = 1;
            for (int i = 0; i < channel_axis; i++) {
                hw *= inputDesc.dims[i];
            }
            if (hw != 1) {
                buffer = std::vector<F32>(size);
                TensorDesc tmpInputDesc = inputDesc;
                tmpInputDesc.df = DF_NCHW;
                transformToNCHW(inputDesc, input, tmpInputDesc, buffer.data());
                input = (const F32 *)(buffer.data());
            }
        } else {
            for (I32 i = (int)inputDesc.nDims; i > 0; i--) {
                inputDesc.dims[i] = inputDesc.dims[i - 1];
            }
            inputDesc.dims[inputDesc.nDims - 1] /= 8;
            inputDesc.dims[0] = 8;
            inputDesc.nDims += 1;
            axis += 1;
        }
    }
    U32 loops = inputDesc.dims[axis];

    U32 loop_inner = 1;
    for (int i = 0; i < axis; i++) {
        loop_inner *= inputDesc.dims[i];
    }
    U32 loop_outer = size / loops / loop_inner;
    if (axis == 0) {
        softmax_lastAxis_fp32<logsoftmax>(input, loop_outer, loops, output);
    } else {
        softmax_anyAxis_fp32<logsoftmax>(input, loop_outer, loops, loop_inner, output);
    }
    return SUCCESS;
}

EE softmax_fp32(TensorDesc inputDesc, const F32 *input, int axis, TensorDesc outputDesc, F32 *output)
{
    return softmax_kernel<false>(inputDesc, input, axis, outputDesc, output);
}

EE logsoftmax_fp32(
    TensorDesc inputDesc, const F32 *input, int axis, TensorDesc outputDesc, F32 *output)
{
    return softmax_kernel<true>(inputDesc, input, axis, outputDesc, output);
}
