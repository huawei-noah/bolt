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

inline ArrayScaleFunction get_array_scale_function(Arch arch)
{
    ArrayScaleFunction func;
    bool find = false;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_scale_general;
        find = true;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_scale_arm;
        find = true;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_scale_x86;
        find = true;
#endif
    }
    CHECK_REQUIREMENT(find);
    return func;
}

inline ArrayAddFunction get_array_add_function(Arch arch)
{
    ArrayAddFunction func;
    bool find = false;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_add_general;
        find = true;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_add_arm;
        find = true;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_add_x86;
        find = true;
#endif
    }
    CHECK_REQUIREMENT(find);
    return func;
}

inline ArrayMeanFunction get_array_mean_function(Arch arch)
{
    ArrayMeanFunction func;
    bool find = false;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_mean_general;
        find = true;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_mean_arm;
        find = true;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_mean_x86;
        find = true;
#endif
    }
    CHECK_REQUIREMENT(find);
    return func;
}

inline ArrayVarFunction get_array_var_function(Arch arch)
{
    ArrayVarFunction func;
    bool find = false;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_var_general;
        find = true;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_var_arm;
        find = true;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_var_x86;
        find = true;
#endif
    }
    CHECK_REQUIREMENT(find);
    return func;
}

inline ArrayPowerFunction get_array_power_function(Arch arch)
{
    ArrayPowerFunction func;
    bool find = false;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_power_general;
        find = true;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_power_arm;
        find = true;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_power_x86;
        find = true;
#endif
    }
    CHECK_REQUIREMENT(find);
    return func;
}

inline ArraySumFunction get_array_sum_function(Arch arch)
{
    ArraySumFunction func;
    bool find = false;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_sum_general;
        find = true;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_sum_arm;
        find = true;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_sum_x86;
        find = true;
#endif
    }
    CHECK_REQUIREMENT(find);
    return func;
}

inline ArraySquareAndAddFunction get_array_square_and_add_function(Arch arch)
{
    ArraySquareAndAddFunction func;
    bool find = false;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_square_and_add_general;
        find = true;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_square_and_add_arm;
        find = true;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_square_and_add_x86;
        find = true;
#endif
    }
    CHECK_REQUIREMENT(find);
    return func;
}

inline ArrayActivationFunction get_array_activation_function(Arch arch)
{
    ArrayActivationFunction func;
    bool find = false;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        func = array_activation_general;
        find = true;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        func = array_activation_arm;
        find = true;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        func = array_activation_x86;
        find = true;
#endif
    }
    CHECK_REQUIREMENT(find);
    return func;
}
#endif
