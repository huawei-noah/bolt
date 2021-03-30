// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_ARM_FUNCTIONS
#define _H_ARM_FUNCTIONS

#include "cpu/cpu_functions_template.h"
#ifdef _USE_FP16
#include "cpu/arm/fp16/arm_functions_fp16.h"
#endif
#ifdef _USE_FP32
#include "cpu/arm/fp32/arm_functions_fp32.h"
#endif
#ifdef _USE_INT8
#include "cpu/arm/int8/arm_functions_int8.h"
#endif

// array sum
inline F32 array_sum_arm(DataType dt, const void *data, I32 len)
{
    F32 result = 0;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            result = array_sum_f16((const F16 *)data, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            result = array_sum_f32((const F32 *)data, len);
            break;
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    return result;
}

// array mean
inline F32 array_mean_arm(DataType dt, const void *data, I32 len)
{
    F32 result = 0;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            result = array_mean_f16((const F16 *)data, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            result = array_mean_f32((const F32 *)data, len);
            break;
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    return result;
}

// array var
inline F32 array_var_arm(DataType dt, const void *data, I32 len, F32 mean)
{
    F32 result = 0;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            result = array_var_f16((const F16 *)data, len, mean);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            result = array_var_f32((const F32 *)data, len, mean);
            break;
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    return result;
}

// array max
inline F32 array_max_value_arm(DataType dt, const void *data, I32 len)
{
    F32 result = 0;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            result = array_max_value_f16((const F16 *)data, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            result = array_max_value_f32((const F32 *)data, len);
            break;
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    return result;
}

inline F32 array_maxabs_arm(DataType dt, const void *data, I32 len)
{
    F32 result = 0;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            result = array_maxabs_f16((const F16 *)data, len);
            break;
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    return result;
}

inline void array_scale_arm(
    DataType dt, const void *input, void *output, I32 len, F32 alpha, F32 beta)
{
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            array_scale_f16((const F16 *)input, (F16 *)output, len, alpha, beta);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            array_scale_f32((const F32 *)input, (F32 *)output, len, alpha, beta);
            break;
#endif
        case DT_I32:
            array_scale_template<I32>((const I32 *)input, (I32 *)output, len, alpha, beta);
            break;
        case DT_U32:
            array_scale_template<U32>((const U32 *)input, (U32 *)output, len, alpha, beta);
            break;
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
}

inline void array_power_arm(DataType dt, void *input, void *output, I32 len, F32 power)
{
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            array_power_f16((F16 *)input, (F16 *)output, len, power);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            array_power_f32((F32 *)input, (F32 *)output, len, power);
            break;
#endif
        case DT_I32:
            array_power_template<I32>((I32 *)input, (I32 *)output, len, power);
            break;
        case DT_U32:
            array_power_template<U32>((U32 *)input, (U32 *)output, len, power);
            break;
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
}

inline EE array_activation_arm(
    DataType dt, void *input, U32 len, ActivationParamSpec activationDesc, void *output)
{
    EE ret = SUCCESS;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            ret = activation_fp16((F16 *)input, len, activationDesc, (F16 *)output);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            ret = activation_fp32((F32 *)input, len, activationDesc, (F32 *)output);
            break;
#endif
#ifdef _USE_INT8
        case DT_I8: {
            ret = activation_int8((INT8 *)input, len, activationDesc, (INT8 *)output);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

inline void array_mul_arm(DataType dt, const void *inputA, const void *inputB, void *output, I32 len)
{
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            array_mul_f16((const F16 *)inputA, (const F16 *)inputB, (F16 *)output, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            array_mul_f32((const F32 *)inputA, (const F32 *)inputB, (F32 *)output, len);
            break;
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
}

inline void array_add_arm(DataType dt, const void *inputA, const void *inputB, void *output, I32 len)
{
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            array_add_f16((const F16 *)inputA, (const F16 *)inputB, (F16 *)output, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            array_add_f32((const F32 *)inputA, (const F32 *)inputB, (F32 *)output, len);
            break;
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
}

inline void array_square_and_add_arm(
    DataType dt, const void *inputA, const void *inputB, void *output, I32 len)
{
    array_mul_arm(dt, inputB, inputB, output, len);
    array_add_arm(dt, inputA, output, output, len);
}

inline void array_max_arm(DataType dt, const void *inputA, const void *inputB, void *output, I32 len)
{
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            array_max_f16((const F16 *)inputA, (const F16 *)inputB, (F16 *)output, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            array_max_f32((const F32 *)inputA, (const F32 *)inputB, (F32 *)output, len);
            break;
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
}
#endif
