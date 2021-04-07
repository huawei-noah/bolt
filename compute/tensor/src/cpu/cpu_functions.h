// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CPU_FUNCTIONS
#define _H_CPU_FUNCTIONS

#ifdef _USE_GENERAL
#include "cpu/general/general_functions.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/arm_functions.h"
#endif
#ifdef _USE_X86
#include "cpu/x86/x86_functions.h"
#endif

typedef void (*ArrayScaleFunction)(
    DataType dt, const void *input, void *output, I32 len, F32 alpha, F32 beta);
typedef void (*ArrayMulFunction)(
    DataType dt, const void *inputA, const void *inputB, void *output, I32 len);
typedef void (*ArrayAddFunction)(
    DataType dt, const void *inputA, const void *inputB, void *output, I32 len);
typedef F32 (*ArraySumFunction)(DataType dt, const void *data, I32 len);
typedef F32 (*ArrayMeanFunction)(DataType dt, const void *data, I32 len);
typedef F32 (*ArrayVarFunction)(DataType dt, const void *data, I32 len, F32 mean);
typedef void (*ArrayPowerFunction)(DataType dt, void *input, void *output, I32 len, F32 power);
typedef void (*ArraySquareAndAddFunction)(
    DataType dt, const void *inputA, const void *inputB, void *output, I32 len);
typedef EE (*ArrayActivationFunction)(
    DataType dt, void *input, U32 len, ActivationParamSpec activationDesc, void *output);
typedef F32 (*ArrayMaxValueFunction)(DataType dt, const void *data, I32 len);
typedef void (*ArrayMaxFunction)(
    DataType dt, const void *inputA, const void *inputB, void *output, I32 len);

inline ArrayScaleFunction get_array_scale_function(Arch arch)
{
    ArrayScaleFunction func = nullptr;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_scale_general;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_scale_arm;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_scale_x86;
#endif
    }
    CHECK_REQUIREMENT(func != nullptr);
    return func;
}

inline ArrayMulFunction get_array_mul_function(Arch arch)
{
    ArrayMulFunction func = nullptr;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_mul_general;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_mul_arm;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_mul_x86;
#endif
    }
    CHECK_REQUIREMENT(func != nullptr);
    return func;
}

inline ArrayAddFunction get_array_add_function(Arch arch)
{
    ArrayAddFunction func = nullptr;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_add_general;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_add_arm;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_add_x86;
#endif
    }
    CHECK_REQUIREMENT(func != nullptr);
    return func;
}

inline ArrayMeanFunction get_array_mean_function(Arch arch)
{
    ArrayMeanFunction func = nullptr;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_mean_general;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_mean_arm;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_mean_x86;
#endif
    }
    CHECK_REQUIREMENT(func != nullptr);
    return func;
}

inline ArrayVarFunction get_array_var_function(Arch arch)
{
    ArrayVarFunction func = nullptr;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_var_general;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_var_arm;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_var_x86;
#endif
    }
    CHECK_REQUIREMENT(func != nullptr);
    return func;
}

inline ArrayPowerFunction get_array_power_function(Arch arch)
{
    ArrayPowerFunction func = nullptr;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_power_general;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_power_arm;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_power_x86;
#endif
    }
    CHECK_REQUIREMENT(func != nullptr);
    return func;
}

inline ArraySumFunction get_array_sum_function(Arch arch)
{
    ArraySumFunction func = nullptr;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_sum_general;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_sum_arm;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_sum_x86;
#endif
    }
    CHECK_REQUIREMENT(func != nullptr);
    return func;
}

inline ArraySquareAndAddFunction get_array_square_and_add_function(Arch arch)
{
    ArraySquareAndAddFunction func = nullptr;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_square_and_add_general;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_square_and_add_arm;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_square_and_add_x86;
#endif
    }
    CHECK_REQUIREMENT(func != nullptr);
    return func;
}

inline ArrayActivationFunction get_array_activation_function(Arch arch)
{
    ArrayActivationFunction func = nullptr;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_activation_general;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_activation_arm;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_activation_x86;
#endif
    }
    CHECK_REQUIREMENT(func != nullptr);
    return func;
}

inline ArrayMaxValueFunction get_array_max_value_function(Arch arch)
{
    ArrayMaxValueFunction func = nullptr;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_max_value_general;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_max_value_arm;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_max_value_x86;
#endif
    }
    CHECK_REQUIREMENT(func != nullptr);
    return func;
}

inline ArrayMaxFunction get_array_max_function(Arch arch)
{
    ArrayMaxFunction func = nullptr;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_max_general;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_max_arm;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_max_x86;
#endif
    }
    CHECK_REQUIREMENT(func != nullptr);
    return func;
}
#endif
