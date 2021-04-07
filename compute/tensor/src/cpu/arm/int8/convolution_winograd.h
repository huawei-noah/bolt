// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CONVOLUTION_WINOGRAD
#define _H_CONVOLUTION_WINOGRAD

#ifdef _USE_INT8
#include "sys.h"
#include "cpu/arm/int8/arm_functions_int8.h"

template <typename OT>
EE convolution_winograd_A55(TensorDesc inputDesc,
    const void *input,
    F16 *input_scale,
    TensorDesc filterDesc,
    const void *filter,
    F16 *filterScale,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    F16 *outputScale,
    ActivationParamSpec am);

template <typename OT>
EE convolution_winograd_A76(TensorDesc inputDesc,
    const void *input,
    F16 *input_scale,
    TensorDesc filterDesc,
    const void *filter,
    F16 *filterScale,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    F16 *outputScale,
    ActivationParamSpec am);

inline EE convolution_winograd(TensorDesc inputDesc,
    const void *input,
    F16 *input_scale,
    TensorDesc filterDesc,
    const void *filter,
    F16 *filterScale,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    F16 *outputScale,
    ActivationParamSpec am,
    Arch arch)
{
    EE ret = SUCCESS;
    switch (arch) {
        case ARM_A55:
            ret = convolution_winograd_A55<INT8>(inputDesc, input, input_scale, filterDesc, filter,
                filterScale, convParamSpec, biasDesc, bias, tmpBytes, tmp, outputDesc, output,
                outputScale, am);
            break;
        case ARM_A76:
            ret = convolution_winograd_A76<INT8>(inputDesc, input, input_scale, filterDesc, filter,
                filterScale, convParamSpec, biasDesc, bias, tmpBytes, tmp, outputDesc, output,
                outputScale, am);
            break;
        default:
            return NOT_SUPPORTED;
    }
    return ret;
}

inline void apply_scale_f16(U32 numData, F16 *array, F16 scale, INT8 *qArray, bool clamp = true)
{
    for (U32 i = 0; i < numData; i++) {
        F32 tmp = array[i] * scale;
        qArray[i] = round_towards_zero(tmp, clamp);
    }
}

inline void quantize_wino_input(F16 *itmArray, U32 len_per_36, INT8 *inQ, F32 *inputScale)
{
    U32 numData = len_per_36;
    F32 scale;

    for (U32 idx = 0; idx < 36; idx++) {
        F16 *in = itmArray + idx * numData;
        float16x8_t temp_v = vld1q_f16(in);
        float16x8_t max_v = temp_v;
        float16x8_t min_v = temp_v;

        for (U32 i = 8; i < numData; i += 8) {
            temp_v = vld1q_f16(in + i);
            max_v = vmaxq_f16(max_v, temp_v);
            min_v = vminq_f16(min_v, temp_v);
        }

        F16 max = vmaxvq_f16(max_v);
        F16 min = vminvq_f16(min_v);

        if (max == 0 && min == 0) {
            inputScale[idx] = 0.0;  // We can skip this dotprod later
            continue;
        }
        if (max > 0 && min < 0) {
            F32 scale_max = 127.0 / max;
            F32 scale_min = -127.0 / min;
            scale = (scale_max < scale_min) ? scale_max : scale_min;
        } else if (max < 0) {
            scale = -127.0 / min;
        } else {  // min > 0
            scale = 127.0 / max;
        }

        INT8 *base = inQ + idx * numData;
        apply_scale_f16(numData, in, scale, base);
        inputScale[idx] = scale;
    }
}

inline void quantize_wino_input_s16(
    short *itmArray, U32 len_per_36, INT8 *inQ, F32 *inputScale, F16 input_scale)
{
    U32 numData = len_per_36;
    F32 scale;

    for (U32 idx = 0; idx < 36; idx++) {
        short *in = itmArray + idx * numData;
        int16x8_t temp_v = vld1q_s16(in);
        int16x8_t max_v = temp_v;
        int16x8_t min_v = temp_v;

        for (U32 i = 8; i < numData; i += 8) {
            temp_v = vld1q_s16(in + i);
            max_v = vmaxq_s16(max_v, temp_v);
            min_v = vminq_s16(min_v, temp_v);
        }

        short max = vmaxvq_s16(max_v);
        short min = vminvq_s16(min_v);

        if (max == 0 && min == 0) {
            inputScale[idx] = 0.0;  // We can skip this dotprod later
            continue;
        }
        if (max > 0 && min < 0) {
            F32 scaleMax = 127.0 / max;
            F32 scaleMin = -127.0 / min;
            scale = (scaleMax < scaleMin) ? scaleMax : scaleMin;
        } else if (max < 0) {
            scale = -127.0 / min;
        } else {  // min > 0
            scale = 127.0 / max;
        }

        INT8 *base = inQ + idx * numData;
        for (U32 i = 0; i < numData; i++) {
            base[i] = round_towards_zero(scale * in[i], false);
        }
        inputScale[idx] = input_scale * scale;
    }
}
#endif
#endif
