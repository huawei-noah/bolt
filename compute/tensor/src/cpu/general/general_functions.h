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

template <typename T>
inline EE from_nchwc8_to_nchw(TensorDesc *desc, T *data)
{
    if (desc == nullptr || data == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }

    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(*desc, &idt, &idf, &in, &ic, &ih, &iw));
    if (idf != DF_NCHWC8) {
        CHECK_STATUS(NOT_MATCH);
    }

    *desc = tensor4df(idt, DF_NCHW, in, ic, ih, iw);

    T *tmp = (T *)malloc(tensorNumBytes(*desc));
    ic /= 8;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 hw = 0; hw < ih * iw; hw++) {
                for (U32 c8 = 0; c8 < 8; c8++) {
                    tmp[n * ic * 8 * ih * iw + (c * 8 + c8) * ih * iw + hw] =
                        data[n * ic * ih * iw * 8 + c * ih * iw * 8 + hw * 8 + c8];
                }
            }
        }
    }
    memcpy(data, tmp, tensorNumBytes(*desc));
    free(tmp);
    return SUCCESS;
}

template <typename T>
inline EE from_nchw_to_nchwc8(TensorDesc *desc, T *data)
{
    if (desc == nullptr || data == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }

    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(*desc, &idt, &idf, &in, &ic, &ih, &iw));
    if (idf != DF_NCHW) {
        CHECK_STATUS(NOT_MATCH);
    }

    *desc = tensor4df(idt, DF_NCHWC8, in, ic, ih, iw);

    T *tmp = (T *)malloc(tensorNumBytes(*desc));
    ic /= 8;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 hw = 0; hw < ih * iw; hw++) {
                for (U32 c8 = 0; c8 < 8; c8++) {
                    tmp[n * ic * ih * iw * 8 + c * ih * iw * 8 + hw * 8 + c8] =
                        data[n * ic * 8 * ih * iw + (c * 8 + c8) * ih * iw + hw];
                }
            }
        }
    }
    memcpy(data, tmp, tensorNumBytes(*desc));
    free(tmp);
    return SUCCESS;
}

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
            CHECK_STATUS(NOT_SUPPORTED);
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
            CHECK_STATUS(NOT_SUPPORTED);
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
            CHECK_STATUS(NOT_SUPPORTED);
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
            CHECK_STATUS(NOT_SUPPORTED);
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
            CHECK_STATUS(NOT_SUPPORTED);
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
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
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
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    return result;
}

inline void array_square_and_add_general(
    DataType dt, const void *inputA, const void *inputB, void *output, I32 len)
{
    array_mul_general(dt, inputB, inputB, output, len);
    array_add_general(dt, inputA, output, output, len);
}

inline EE array_activation_general(
    DataType dt, void *input, U32 len, ActivationParamSpec activationDesc, void *output)
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
        default:
            ret = NOT_SUPPORTED;
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
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
}

inline F32 array_max_value_general(DataType dt, const void *data, I32 len)
{
    F32 result = 0;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            result = array_max_value_template<F16>((const F16 *)data, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            result = array_max_value_template<F32>((const F32 *)data, len);
            break;
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    return result;
}
#endif
