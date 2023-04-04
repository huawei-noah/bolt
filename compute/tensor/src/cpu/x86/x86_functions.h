// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_X86_FUNCTIONS
#define _H_X86_FUNCTIONS
#include "cpu/cpu_functions_template.h"
#include "cpu/general/general_functions.h"
#ifdef _USE_FP32
#include "cpu/x86/fp32/x86_functions_fp32.h"
#endif
#ifdef _USE_INT8
#ifdef _USE_AVX512_VNNI
#include "cpu/x86/int8/avx512/x86_functions_int8.h"
#else
#include "cpu/x86/int8/avx/x86_functions_int8.h"
#endif
#endif

inline void array_add_x86(DataType dt, const void *inputA, const void *inputB, void *output, I32 len)
{
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32:
            array_add_f32((const F32 *)inputA, (const F32 *)inputB, (F32 *)output, len);
            break;
#endif
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
}

inline void array_mul_x86(DataType dt, const void *inputA, const void *inputB, void *output, I32 len)
{
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32:
            array_mul_f32((const F32 *)inputA, (const F32 *)inputB, (F32 *)output, len);
            break;
#endif
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
}

inline void array_mul_and_add_x86(
    DataType dt, const void *inputA, const void *inputB, const void *inputC, void *output, I32 len)
{
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32:
            array_mul_and_add_f32(
                (const F32 *)inputA, (const F32 *)inputB, (const F32 *)inputC, (F32 *)output, len);
            break;
#endif
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
}

// array mean
inline F32 array_mean_x86(DataType dt, const void *data, I32 len)
{
    F32 result = 0;
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32:
            result = array_mean_f32((const F32 *)data, len);
            break;
#endif
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
    return result;
}

inline void array_power_x86(DataType dt, void *input, void *output, I32 len, F32 power)
{
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32:
            array_power_f32((F32 *)input, (F32 *)output, len, power);
            break;
#endif
        case DT_I64:
            array_power_template<I64>((I64 *)input, (I64 *)output, len, power);
            break;
        case DT_I32:
            array_power_template<I32>((I32 *)input, (I32 *)output, len, power);
            break;
        case DT_U32:
            array_power_template<U32>((U32 *)input, (U32 *)output, len, power);
            break;
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
}

inline F32 array_sum_x86(DataType dt, const void *data, I32 len)
{
    F32 result = 0;
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32:
            result = array_sum_f32((const F32 *)data, len);
            break;
#endif
        case DT_U32:
        case DT_I32:
            result = array_sum_i32((const I32 *)data, len);
            break;
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
    return result;
}

inline void array_scale_x86(
    DataType dt, const void *input, void *output, I32 len, F32 alpha, F32 beta)
{
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32:
            array_scale_f32((const F32 *)input, (F32 *)output, len, alpha, beta);
            break;
#endif
        case DT_I64:
            array_scale_template<I64>((const I64 *)input, (I64 *)output, len, alpha, beta);
            break;
        case DT_I32:
            array_scale_template<I32>((const I32 *)input, (I32 *)output, len, alpha, beta);
            break;
        case DT_U32:
            array_scale_template<U32>((const U32 *)input, (U32 *)output, len, alpha, beta);
            break;
        case DT_I8:
            array_scale_template<INT8>((const INT8 *)input, (INT8 *)output, len, alpha, beta);
            break;
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
}

// array var
inline F32 array_var_x86(DataType dt, const void *data, I32 len, F32 mean)
{
    F32 result = 0;
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32:
            result = array_var_f32((const F32 *)data, len, mean);
            break;
#endif
	case DT_I32:
	        result = array_var_i32((const I32 *)data, len);
	        break;    
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
    return result;
}

inline EE array_activation_x86(
    DataType dt, void *input, U32 len, ActivationParamSpec activationDesc, void *output, F32 *scale)
{
    EE result = SUCCESS;
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32:
            result = activation_fp32((F32 *)input, len, activationDesc, (F32 *)output);
            break;
#endif
#ifdef _USE_INT8
        case DT_U8_Q:
            result = activation_offset_int8((UINT8 *)input, len, activationDesc, (UINT8 *)output, scale);
            break;
#endif
        default:
            result = array_activation_general(dt, input, len, activationDesc, output, nullptr);
            break;
    }
    return result;
}

inline EE array_minmax_value_x86(DataType dt, const void *data, I32 len, int mode, F32 *result)
{
    EE ret = SUCCESS;
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32:
            ret = array_minmax_value_f32((const F32 *)data, len, mode, result);
            break;
#endif
        case DT_U32:
            ret = array_minmax_value_general(dt, data, len, mode, result);
            break;
        case DT_I32:
            ret = array_minmax_value_i32((const I32 *)data, len, mode, result);
            break;
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

inline void array_max_x86(DataType dt, const void *inputA, const void *inputB, void *output, I32 len)
{
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32:
            array_max_f32((const F32 *)inputA, (const F32 *)inputB, (F32 *)output, len);
            break;
#endif
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
}

inline void array_norm_scalar_scale_x86(
    DataType dt, void *input, void *output, I32 len, F32 mean, F32 var, void *alpha, void *beta)
{
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32:
            array_norm_scalar_scale_fp32(
                (F32 *)input, (F32 *)output, len, mean, var, (F32 *)alpha, (F32 *)beta);
            break;
#endif
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
}

#endif  // _H_X86_FUNCTIONS
