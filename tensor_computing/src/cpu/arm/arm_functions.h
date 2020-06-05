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

#include "cpu/arm/fp16/arm_functions_fp16.h"
#include "cpu/arm/fp32/arm_functions_fp32.h"

// array sum
inline F32 array_sum(DataType dt, const void *data, I32 len) {
    F32 result = 0;
    switch(dt) {
#ifdef _USE_FP16
        case DT_F16:
            result = array_sum_f16((const F16*)data, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            result = array_sum_f32((const F32*)data, len);
            break;
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    return result;
}

// array mean
inline F32 array_mean(DataType dt, const void *data, I32 len) {
    F32 result = 0;
    switch(dt) {
#ifdef _USE_FP16
        case DT_F16:
            result = array_mean_f16((const F16*)data, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            result = array_mean_f32((const F32*)data, len);
            break;
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    return result;
}

// array var
inline F32 array_var(DataType dt, const void *data, I32 len, F32 mean) {
    F32 result = 0;
    switch(dt) {
#ifdef _USE_FP16
        case DT_F16:
            result = array_var_f16((const F16*)data, len, mean);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            result = array_var_f32((const F32*)data, len, mean);
            break;
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    return result;
}

// array max
inline F32 array_max(DataType dt, const void* data, I32 len) {
    F32 result = 0;
    switch(dt) {
#ifdef _USE_FP16
        case DT_F16:
            result = array_max_f16((const F16*)data, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            result = array_max_f32((const F32*)data, len);
            break;
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    return result;
}

inline F32 array_maxabs(DataType dt, const void* data, I32 len)
{
    F32 result = 0;
    switch(dt) {
#ifdef _USE_FP16
        case DT_F16:
            result = array_maxabs_f16((const F16*)data, len);
            break;
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    return result;
}

template<typename T>
inline void array_scale_template(T *input, T *output, I32 len, F32 alpha, F32 beta) {
    for (I32 i = 0; i < len; i++) {
        output[i] = alpha * input[i] + beta;
    }
}
    
inline void array_scale(DataType dt, void *input, void *output, I32 len, F32 alpha, F32 beta) {
    switch(dt) {
#ifdef _USE_FP16
        case DT_F16:
            array_scale_f16((F16*)input, (F16*)output, len, alpha, beta);
            break;                             
#endif                                         
#ifdef _USE_FP32                               
        case DT_F32:                           
            array_scale_f32((F32*)input, (F32*)output, len, alpha, beta);
            break;
#endif
        case DT_I32: {
            array_scale_template<I32>((I32 *)input, (I32 *)output, len, alpha, beta);
            break;
        }
        case DT_U32: {
            array_scale_template<U32>((U32 *)input, (U32 *)output, len, alpha, beta);
            break;
        }
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
}

inline EE array_activation(DataType dt, void* input, U32 len, ActivationDesc activationDesc, void* output)
{
    EE result = SUCCESS;
    switch(dt) {
#ifdef _USE_FP16
        case DT_F16:
            result = activation_fp16((F16*)input, len, activationDesc, (F16*)output);
            break;                                                           
#endif                                                                       
#ifdef _USE_FP32                                                             
        case DT_F32:                                                         
            result = activation_fp32((F32*)input, len, activationDesc, (F32*)output);
            break;
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
    return result;
}

inline void array_add(DataType dt, const void *inputA, const void *inputB, void *output, I32 len) {
    switch(dt) {
#ifdef _USE_FP16
        case DT_F16:
            array_add_f16((const F16*)inputA, (const F16*)inputB, (F16*)output, len);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            array_add_f32((const F32*)inputA, (const F32*)inputB, (F32*)output, len);
            break;
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }
}

#endif
