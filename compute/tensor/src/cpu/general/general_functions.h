// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_GENERAL_FUNCTIONS
#define _H_GENERAL_FUNCTIONS

#include "error.h"
#include "cpu/cpu_functions_template.h"

inline F32 array_mean_general(DataType dt, const void *data, I32 len)
{
    F32 result = 0;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            result = array_mean_template<F16>((const F16 *)data, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            result = array_mean_template<F32>((const F32 *)data, len);
            break;
#endif
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
    return result;
}

inline F32 array_var_general(DataType dt, const void *data, I32 len, F32 mean)
{
    F32 result = 0;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            result = array_var_template<F16>((const F16 *)data, len, mean);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            result = array_var_template<F32>((const F32 *)data, len, mean);
            break;
#endif
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
    return result;
}

inline void array_power_general(DataType dt, void *input, void *output, I32 len, F32 power)
{
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            array_power_template<F16>((F16 *)input, (F16 *)output, len, power);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            array_power_template<F32>((F32 *)input, (F32 *)output, len, power);
            break;
#endif
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

inline void array_mul_general(
    DataType dt, const void *inputA, const void *inputB, void *output, I32 len)
{
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            array_mul_template<F16>((const F16 *)inputA, (const F16 *)inputB, (F16 *)output, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            array_mul_template<F32>((const F32 *)inputA, (const F32 *)inputB, (F32 *)output, len);
            break;
#endif
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
}

inline void array_add_general(
    DataType dt, const void *inputA, const void *inputB, void *output, I32 len)
{
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            array_add_template<F16>((const F16 *)inputA, (const F16 *)inputB, (F16 *)output, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            array_add_template<F32>((const F32 *)inputA, (const F32 *)inputB, (F32 *)output, len);
            break;
#endif
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
}

inline void array_mul_and_add_general(
    DataType dt, const void *inputA, const void *inputB, const void *inputC, void *output, I32 len)
{
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            array_mul_and_add_template<F16>(
                (const F16 *)inputA, (const F16 *)inputB, (const F16 *)inputC, (F16 *)output, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            array_mul_and_add_template<F32>(
                (const F32 *)inputA, (const F32 *)inputB, (const F32 *)inputC, (F32 *)output, len);
            break;
#endif
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
}

inline void array_scale_general(
    DataType dt, const void *input, void *output, I32 len, F32 alpha, F32 beta)
{
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            array_scale_template<F16>((const F16 *)input, (F16 *)output, len, alpha, beta);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            array_scale_template<F32>((const F32 *)input, (F32 *)output, len, alpha, beta);
            break;
#endif
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

inline EE array_scale_round_general(
    DataType dt, const void *input, INT8 *output, I32 len, F32 scale, bool clamp)
{
    EE ret = SUCCESS;
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32:
            array_scale_round_template<F32>((const F32 *)input, output, len, scale, clamp);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            array_scale_round_template<F16>((const F16 *)input, output, len, scale, clamp);
            break;
#endif
        case DT_I32:
            array_scale_round_template<I32>((const I32 *)input, output, len, scale, clamp);
            break;
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

inline F32 array_sum_general(DataType dt, const void *data, I32 len)
{
    F32 result = 0;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            result = array_sum_template<F16>((const F16 *)data, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            result = array_sum_template<F32>((const F32 *)data, len);
            break;
#endif
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
    return result;
}

inline EE array_activation_general(
    DataType dt, void *input, U32 len, ActivationParamSpec activationDesc, void *output, F32 *scale)
{
    EE ret = SUCCESS;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16: {
            F16 *inPtr = (F16 *)input;
            F16 *outPtr = (F16 *)output;
            for (U32 i = 0; i < len; i++) {
                activation_template<F16>(activationDesc, inPtr[i], &outPtr[i]);
            }
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            F32 *inPtr = (F32 *)input;
            F32 *outPtr = (F32 *)output;
            for (U32 i = 0; i < len; i++) {
                activation_template<F32>(activationDesc, inPtr[i], &outPtr[i]);
            }
            break;
        }
#endif
        case DT_U32: {
            U32 *inPtr = (U32 *)input;
            U32 *outPtr = (U32 *)output;
            for (U32 i = 0; i < len; i++) {
                activation_template<U32>(activationDesc, inPtr[i], &outPtr[i]);
            }
            break;
        }
        case DT_I32: {
            I32 *inPtr = (I32 *)input;
            I32 *outPtr = (I32 *)output;
            for (U32 i = 0; i < len; i++) {
                activation_template<I32>(activationDesc, inPtr[i], &outPtr[i]);
            }
            break;
        }
        case DT_I8: {
            INT8 *inPtr = (INT8 *)input;
            INT8 *outPtr = (INT8 *)output;
            for (U32 i = 0; i < len; i++) {
                activation_template<INT8>(activationDesc, inPtr[i], &outPtr[i]);
            }
            break;
        }
        case DT_U8: {
            U8 *inPtr = (U8 *)input;
            U8 *outPtr = (U8 *)output;
            for (U32 i = 0; i < len; i++) {
                activation_template<U8>(activationDesc, inPtr[i], &outPtr[i]);
            }
            break;
        }
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
    return ret;
}

inline void array_max_general(
    DataType dt, const void *inputA, const void *inputB, void *output, I32 len)
{
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            array_max_template<F16>((const F16 *)inputA, (const F16 *)inputB, (F16 *)output, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            array_max_template<F32>((const F32 *)inputA, (const F32 *)inputB, (F32 *)output, len);
            break;
#endif
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
}

inline EE array_minmax_value_general(DataType dt, const void *data, I32 len, int mode, F32 *result)
{
    EE ret = SUCCESS;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            ret = array_minmax_value_template<F16>((const F16 *)data, len, mode, result);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            ret = array_minmax_value_template<F32>((const F32 *)data, len, mode, result);
            break;
#endif
        case DT_I32:
            ret = array_minmax_value_template<I32>((const I32 *)data, len, mode, result);
            break;
        case DT_U32:
            ret = array_minmax_value_template<U32>((const U32 *)data, len, mode, result);
            break;
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

inline void array_norm_scalar_scale_general(
    DataType dt, void *input, void *output, I32 len, F32 mean, F32 var, void *alpha, void *beta)
{
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            array_norm_scalar_scale_template<F16>(
                (F16 *)input, (F16 *)output, len, mean, var, (F16 *)alpha, (F16 *)beta);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            array_norm_scalar_scale_template<F32>(
                (F32 *)input, (F32 *)output, len, mean, var, (F32 *)alpha, (F32 *)beta);
            break;
#endif
        default:
            UNI_ERROR_LOG("%s not support %s data.\n", __func__, DataTypeName()[dt]);
            break;
    }
}
#endif
