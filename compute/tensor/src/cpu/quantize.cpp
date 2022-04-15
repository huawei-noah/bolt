// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <float.h>
#include "cpu/tensor_computing_cpu.h"
#include "cpu/cpu_functions.h"
#if defined(_USE_NEON) && defined(_USE_FP16) && defined(_USE_INT8)
#include "cpu/arm/int8/v8.2/convolution_gemm.h"
#endif
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif

typedef EE (*scaleFunc)(
    DataType dt, const void *input, INT8 *output, U32 length, F32 scale, bool clamp);

template <typename T>
inline static void apply_scale_round_template(
    const T *input, INT8 *output, U32 length, F32 scale, bool clamp)
{
    for (U32 i = 0; i < length; i++) {
        output[i] = round(input[i] * scale);
    }
}

template <typename T>
inline static void apply_scale_truncate_template(
    const T *input, INT8 *output, U32 length, F32 scale, bool clamp)
{
    for (U32 i = 0; i < length; i++) {
        output[i] = round_towards_zero(input[i] * scale, clamp);
    }
}

inline EE apply_scale_round(
    DataType dt, const void *input, INT8 *output, U32 length, F32 scale, bool clamp)
{
    EE ret = SUCCESS;
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32:
            apply_scale_round_template<F32>((const F32 *)input, output, length, scale, clamp);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            apply_scale_round_template<F16>((const F16 *)input, output, length, scale, clamp);
            break;
#endif
        case DT_I32:
            apply_scale_round_template<I32>((const I32 *)input, output, length, scale, clamp);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

inline EE apply_scale_truncate(
    DataType dt, const void *input, INT8 *output, U32 length, F32 scale, bool clamp)
{
    EE ret = SUCCESS;
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32:
            apply_scale_truncate_template<F32>((const F32 *)input, output, length, scale, clamp);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            apply_scale_truncate_template<F16>((const F16 *)input, output, length, scale, clamp);
            break;
#endif
        case DT_I32:
            apply_scale_truncate_template<I32>((const I32 *)input, output, length, scale, clamp);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE quantize_hwncn8c4_cpu(
    TensorDesc dDesc, const void *data, TensorDesc *qDesc, void *qData, F32 *scale, Arch arch)
{
    DataType dt;
    DataFormat df;
    U32 n, c, h, w;
    if (tensorIs2d(dDesc)) {
        CHECK_STATUS(tensor2dGet(dDesc, &dt, &df, &n, &w));
        c = 1;
        h = 1;
    } else if (tensorIs3d(dDesc)) {
        CHECK_STATUS(tensor3dGet(dDesc, &dt, &df, &n, &h, &w));
        c = 1;
    } else if (tensorIs4d(dDesc)) {
        CHECK_STATUS(tensor4dGet(dDesc, &dt, &df, &n, &c, &h, &w));
    } else {
        return NOT_SUPPORTED;
    }

    F32 minmax[2] = {1, -1};
    ArrayMinMaxValueFunction minmax_value_func = get_array_minmax_value_function(arch);
    CHECK_REQUIREMENT(h * w == 36);
    for (U32 idx = 0; idx < 36; idx++) {
        F32 min = FLT_MAX;
        F32 max = -FLT_MAX;
        for (U32 o = 0; o < n; o += 8) {
            U32 base = o * 36 * c + idx * 8 * c;
            const U8 *input = (const U8 *)data + base * bytesOf(dt);
            CHECK_STATUS(minmax_value_func(dt, input, 8 * c, 3, minmax));
            min = UNI_MIN(min, minmax[0]);
            max = UNI_MAX(max, minmax[1]);
        }
        if (max == 0 && min == 0) {
            return NOT_SUPPORTED;
        } else {
            F32 absMax = UNI_MAX(UNI_ABS(max), UNI_ABS(min));
            scale[idx] = 127.0 / absMax;
            INT8 *qArray = (INT8 *)qData;
            for (U32 o = 0; o < n; o += 8) {
                U32 base = o * 36 * c + idx * 8 * c;
                const U8 *input = (const U8 *)data + base * bytesOf(dt);
                CHECK_STATUS(apply_scale_round(dt, input, qArray + base, 8 * c, scale[idx], false));
            }
        }
    }
    return SUCCESS;
}

EE quantize_cpu(
    TensorDesc dDesc, const void *data, TensorDesc *qDesc, void *qData, F32 *scale, Arch arch)
{
    if (nullptr == data || nullptr == qDesc || nullptr == qData || nullptr == scale) {
        CHECK_STATUS(NULL_POINTER);
    }

#if defined(_USE_X86) && defined(_USE_INT8)
    if (IS_X86_AVX512(arch)) {
        return quantize_x86(dDesc, data, qDesc, qData, scale);
    }
#endif

    DataType dt = dDesc.dt;
    *qDesc = dDesc;
    qDesc->dt = DT_I8;
    if (dDesc.df == DF_HWNCN8C4) {
        return quantize_hwncn8c4_cpu(dDesc, data, qDesc, qData, scale, arch);
    }
    ArrayMinMaxValueFunction minmax_value_func = get_array_minmax_value_function(arch);
    F32 minmax[2] = {1, -1};
    U32 numData = tensorNumElements(dDesc);
    CHECK_STATUS(minmax_value_func(dt, data, numData, 3, minmax));

    F32 min = minmax[0];
    F32 max = minmax[1];
    EE ret = SUCCESS;
    scaleFunc arrayScale = apply_scale_round;

    if (max == 0 && min == 0) {
        *scale = 1;
        UNI_MEMSET(qData, 0, tensorNumBytes(*qDesc));
    } else {
        F32 absMax = UNI_MAX(UNI_ABS(max), UNI_ABS(min));
        F32 scaleRaw = 127.0 / absMax;

        bool clamp = false;
        INT8 *qArray = (INT8 *)qData;
        if (dt == DT_I32) {
            if (*scale < 0 && scale[1] > 0) {
                scale[0] = scale[1] * scaleRaw;
            } else if (scale[0] > 0 && scale[1] > 0) {
                scaleRaw = scale[0] / scale[1];
            }
            const I32 *array = (const I32 *)data;
            I32 factor = 127 * 16777216 / (int)absMax;

            U32 main = 0;
#if defined(_USE_NEON) && defined(_USE_FP16) && defined(_USE_INT8)
            if (arch == ARM_A76 || arch == ARM_A55) {
                main = numData / 16;
                ret = quantize_I32(main * 4, (I32 *)data, factor, scaleRaw, qArray);
            }
#endif
            for (U32 i = main * 16; i < numData; i++) {
                qArray[i] = round(array[i] * scaleRaw);
            }
        } else {
            if (*scale < scaleRaw) {
                *scale = scaleRaw;
            }
            ret = arrayScale(dt, data, qArray, numData, *scale, (*scale) != scaleRaw);
        }
    }
    UNI_DEBUG_LOG("tensor min value is %f, max value is %f, scale value is %f.\n", min, max, *scale);
    return ret;
}
