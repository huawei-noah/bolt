// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_CPU_FUNCTIONS_TEMPLATE
#define _H_CPU_FUNCTIONS_TEMPLATE

#include "data_type.h"
#include "parameter_spec.h"
#include "uni.h"

// copy input[index]~input[index+length] to output buffer
template <typename T>
void get_vector(T *input, int lda, T **output, int ldb, int index, int length, T *buffer)
{
    UNUSED(ldb);
    int local = index % lda;
    if (length == 1) {
        *output = input + local;
    } else if (lda == 1) {
        *output = buffer;
        for (int i = 0; i < length; i++) {
            (*output)[i] = input[local];
        }
    } else {
        int remain = lda - local;
        if (remain >= length) {
            *output = input + local;
        } else {
            *output = buffer;
            UNI_MEMCPY(*output, input + local, sizeof(T) * remain);
            for (int i = 0; i < length - remain; i++) {
                (*output)[remain + i] = input[i % lda];
            }
        }
    }
}

template <typename T>
inline void array_scale_template(const T *input, T *output, I32 len, F32 alpha, F32 beta)
{
    for (I32 i = 0; i < len; i++) {
        output[i] = alpha * input[i] + beta;
    }
}

template <typename T>
inline void array_power_template(T *input, T *output, I32 len, F32 power)
{
    for (I32 i = 0; i < len; i++) {
        output[i] = powf(input[i], power);
    }
}

template <typename T>
EE activation_template(ActivationParamSpec activationDesc, F32 input, T *output)
{
    F32 value, result = 0;
    EE ret = SUCCESS;
    switch (activationDesc.mode) {
        case ACTIVATION_NULL: {
            result = input;
            break;
        }
        case ACTIVATION_RELU: {
            value = input;
            F32 tmp = activationDesc.value[0] * value;
            if (value < tmp) {
                value = tmp;
            }
            result = value;
            break;
        }
        case ACTIVATION_RELU6: {
            value = input;
            if (value < 0) {
                value = 0;
            }
            if (value > 6) {
                value = 6;
            }
            result = value;
            break;
        }
        case ACTIVATION_H_SIGMOID: {
            value = input + 3;
            if (value < 0) {
                value = 0;
            }
            if (value > 6) {
                value = 6;
            }
            result = value / 6;
            break;
        }
        case ACTIVATION_H_SWISH: {
            value = input + 3;
            if (value < 0) {
                value = 0;
            }
            if (value > 6) {
                value = 6;
            }
            result = input * (value / 6);
            break;
        }
        case ACTIVATION_H_SWISH_NODIV: {
            value = input + 3;
            if (value < 0) {
                value = 0;
            }
            if (value > 6) {
                value = 6;
            }
            result = input * value;
            break;
        }
        case ACTIVATION_GELU: {
            value = input;
            F32 two_div_PI_sqrt = sqrt(2 / 3.14159265358979323846);
            value = two_div_PI_sqrt * (value + 0.044715 * powf(value, 3));
            value = 1.0 - 2.0 / (exp(2.0 * value) + 1.0);
            value = 0.5 * (1.0 + value);
            value = input * value;
            result = value;
            break;
        }
        case ACTIVATION_TANH: {
            value = 1.0 - 2.0 / (exp(2.0 * input) + 1.0);
            result = value;
            break;
        }
        case ACTIVATION_SIGMOID: {
            value = 1.0 / (1.0 + exp(-1.0 * input));
            result = value;
            break;
        }
        case ACTIVATION_MISH: {
            value = input;
            F32 mish_threshold = 20;
            if (value < -mish_threshold) {
                value = exp(value);
            } else if (!(value > mish_threshold || value < -mish_threshold)) {
                value = log(exp(value) + 1.0);
            }
            value = input * tanh(value);
            result = value;
            break;
        }
        case ACTIVATION_SOFTPLUS: {
            result = log(1 + exp(input));
            break;
        }
        case ACTIVATION_EXP: {
            result = exp(input);
            break;
        }
        case ACTIVATION_ABS: {
            result = UNI_ABS(input);
            break;
        }
        case ACTIVATION_SIGN: {
            result = UNI_SIGN(input);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    *output = result;
    return ret;
}

template <typename T>
F32 array_sum_template(const T *array, U32 length)
{
    F32 sum = 0;
    for (U32 i = 0; i < length; i++) {
        sum += array[i];
    }
    return sum;
}

// array mean
template <typename T>
F32 array_mean_template(const T *data, I32 len)
{
    if (len <= 0) {
        return 0;
    }
    return array_sum_template<T>(data, len) / len;
}

template <typename T>
F32 array_var_template(const T *data, I32 len, F32 mean)
{
    F32 sum_s = 0;
    for (I32 i = 0; i < len; i++) {
        F32 in = data[i];
        F32 tmp = in - mean;
        sum_s += tmp * tmp;
    }
    return sum_s / len;
}

template <typename T>
inline void array_add_template(const T *inputA, const T *inputB, T *output, I32 len)
{
    for (I32 i = 0; i < len; i++) {
        output[i] = inputA[i] + inputB[i];
    }
}

template <typename T>
inline F32 array_sum_template(const T *data, I32 len)
{
    if (len <= 0) {
        return 0;
    }

    F32 sum_s = 0;
    for (I32 i = 0; i < len; i++) {
        sum_s += data[i];
    }
    return sum_s;
}

template <typename T>
inline void array_mul_template(const T *inputA, const T *inputB, T *output, I32 len)
{
    for (I32 i = 0; i < len; i++) {
        output[i] = inputA[i] * inputB[i];
    }
}

template <typename T>
inline void array_max_template(const T *inputA, const T *inputB, T *output, I32 len)
{
    for (I32 i = 0; i < len; i++) {
        output[i] = UNI_MAX(inputA[i], inputB[i]);
    }
}

template <typename T>
F32 array_max_value_template(const T *data, I32 len)
{
    if (len <= 0) {
        return 0;
    }
    F32 max_s = data[0];
    for (I32 i = 1; i < len; i++) {
        max_s = UNI_MAX(data[i], max_s);
    }
    return max_s;
}
#endif
