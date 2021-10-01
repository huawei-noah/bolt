// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef CHEETAH_X86_FUNCTIONS_FP32_H
#define CHEETAH_X86_FUNCTIONS_FP32_H
#include <math.h>
#include "x86_avx2_expand.h"
#include "parameter_spec.h"
#include "uni.h"
#include "thread_affinity.h"

inline EE activation_fp32(F32 *input, U32 len, ActivationParamSpec activationDesc, F32 *output)
{
    __m256 in, out;
    __m256 zero = _mm256_set1_ps(0.);
    __m256 one = _mm256_set1_ps(1.);
    __m256 three = _mm256_set1_ps(3.);
    __m256 six = _mm256_set1_ps(6.);
    __m256 signm = _mm256_set1_ps(-0.0);
    U32 len_main = len / 8;
    U32 len_tail = len % 8;

    F32 value;
    EE ret = SUCCESS;

    switch (activationDesc.mode) {
        case ACTIVATION_NULL: {
            break;
        }
        case ACTIVATION_RELU: {
            U32 main_len = len - len_tail;
            if (activationDesc.value[0] == 0) {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
                for (U32 i = 0; i < len_main; i++) {
                    _mm256_storeu_ps(
                        output + i * 8, _mm256_max_ps(zero, _mm256_loadu_ps(input + i * 8)));
                }
                for (U32 i = 0; i < len_tail; i++) {
                    output[main_len + i] = (input[main_len + i] < 0) ? 0 : input[main_len + i];
                }
            } else {
                __m256 scale = _mm256_set1_ps(activationDesc.value[0]);
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
                for (U32 i = 0; i < len_main; i++) {
                    __m256 tmp = _mm256_loadu_ps(input + i * 8);
                    _mm256_storeu_ps(output + i * 8, _mm256_max_ps(_mm256_mul_ps(scale, tmp), tmp));
                }
                for (U32 i = 0; i < len_tail; i++) {
                    float tmp = activationDesc.value[0] * input[main_len + i];
                    output[main_len + i] = (input[main_len + i] < tmp) ? tmp : input[main_len + i];
                }
            }
            break;
        }
        case ACTIVATION_RELU6: {
            for (U32 i = 0; i < len_main; i++) {
                in = _mm256_loadu_ps(input);
                out = _mm256_max_ps(zero, in);
                out = _mm256_min_ps(six, out);
                _mm256_storeu_ps(output, out);
                input += 8;
                output += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                value = (input[i] < 0) ? 0 : input[i];
                if (value > 6) {
                    value = 6;
                }
                output[i] = value;
            }
            break;
        }
        case ACTIVATION_H_SIGMOID: {
            for (U32 i = 0; i < len_main; i++) {
                in = _mm256_loadu_ps(input);
                out = _mm256_add_ps(in, three);
                out = _mm256_max_ps(out, zero);
                out = _mm256_min_ps(out, six);
                out = _mm256_div_ps(out, six);
                _mm256_storeu_ps(output, out);
                input += 8;
                output += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                value = input[i] + 3;
                value = (value < 0) ? 0 : value;
                value = (value > 6) ? 6 : value;
                value = value / 6;
                output[i] = value;
            }
            break;
        }
        case ACTIVATION_H_SWISH: {
            for (U32 i = 0; i < len_main; i++) {
                in = _mm256_loadu_ps(input);
                out = _mm256_add_ps(in, three);
                out = _mm256_max_ps(out, zero);
                out = _mm256_min_ps(out, six);
                out = _mm256_div_ps(out, six);
                out = _mm256_mul_ps(out, in);
                _mm256_storeu_ps(output, out);
                input += 8;
                output += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                value = input[i] + 3;
                value = (value < 0) ? 0 : value;
                value = (value > 6) ? 6 : value;
                value = input[i] * value;
                value = value / 6;
                output[i] = value;
            }
            break;
        }
        case ACTIVATION_H_SWISH_NODIV: {
            for (U32 i = 0; i < len_main; i++) {
                in = _mm256_loadu_ps(input);
                out = _mm256_add_ps(in, three);
                out = _mm256_max_ps(out, zero);
                out = _mm256_min_ps(out, six);
                out = _mm256_mul_ps(out, in);
                _mm256_storeu_ps(output, out);
                input += 8;
                output += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                value = input[i] + 3;
                value = (value < 0) ? 0 : value;
                value = (value > 6) ? 6 : value;
                value = input[i] * value;
                output[i] = value;
            }
            break;
        }
        case ACTIVATION_GELU: {
            F32 two_div_PI_sqrt = sqrt(2 / 3.14159265358979323846);
            __m256 vec0 = _mm256_set1_ps(two_div_PI_sqrt);
            __m256 vec1 = _mm256_set1_ps(0.044715);
            __m256 vec2 = _mm256_set1_ps(0.5);
            for (U32 i = 0; i < len_main; i++) {
                in = _mm256_loadu_ps(input);
                out = _mm256_mul_ps(in, in);
                out = _mm256_mul_ps(out, in);
                out = _mm256_fmadd_ps(vec1, out, in);
                out = _mm256_mul_ps(vec0, out);
                out = _mm256_tanh_ps(out);
                out = _mm256_add_ps(one, out);
                out = _mm256_mul_ps(vec2, out);
                out = _mm256_mul_ps(in, out);
                _mm256_storeu_ps(output, out);
                input += 8;
                output += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                value = input[i];
                value = two_div_PI_sqrt * (value + 0.044715 * powf(value, 3));
                value = 1.0 - 2.0 / (exp(2.0 * value) + 1.0);
                value = 0.5 * (1.0 + value);
                value = input[i] * value;
                output[i] = value;
            }
            break;
        }
        case ACTIVATION_TANH: {
            for (U32 i = 0; i < len_main; i++) {
                in = _mm256_loadu_ps(input);
                out = _mm256_tanh_ps(in);
                _mm256_storeu_ps(output, out);
                input += 8;
                output += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                value = 1.0 - 2.0 / (exp(2.0 * input[i]) + 1.0);
                output[i] = value;
            }
            break;
        }
        case ACTIVATION_SIGMOID: {
            for (U32 i = 0; i < len_main; i++) {
                in = _mm256_loadu_ps(input);
                out = _mm256_sigmod_ps(in);
                _mm256_storeu_ps(output, out);
                input += 8;
                output += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                value = 1.0 / (1.0 + exp(-1.0 * input[i]));
                output[i] = value;
            }
            break;
        }
        case ACTIVATION_MISH: {
            for (U32 i = 0; i < len_main; i++) {
                in = _mm256_loadu_ps(input);
                out = _mm256_mul_ps(
                    in, _mm256_tanh_ps(_mm256_log_ps(_mm256_add_ps(_mm256_exp_ps(in), one))));
                _mm256_storeu_ps(output, out);
                input += 8;
                output += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                value = input[i] * tanh(log(exp(input[i]) + 1.0));
                output[i] = value;
            }
            break;
        }
        case ACTIVATION_SOFTPLUS: {
            for (U32 i = 0; i < len_main; i++) {
                in = _mm256_loadu_ps(input);
                out = _mm256_log_ps(_mm256_add_ps(_mm256_exp_ps(in), one));
                _mm256_storeu_ps(output, out);
                input += 8;
                output += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                output[i] = log(1 + exp(input[i]));
            }
            break;
        }
        case ACTIVATION_EXP: {
            for (U32 i = 0; i < len_main; i++) {
                in = _mm256_loadu_ps(input);
                out = _mm256_exp_ps(in);
                _mm256_storeu_ps(output, out);
                input += 8;
                output += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                output[i] = exp(input[i]);
            }
            break;
        }
        case ACTIVATION_ABS: {
            for (U32 i = 0; i < len_main; i++) {
                in = _mm256_loadu_ps(input);
                out = _mm256_andnot_ps(signm, in);
                _mm256_storeu_ps(output, out);
                input += 8;
                output += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                output[i] = UNI_ABS(input[i]);
            }
            break;
        }
        case ACTIVATION_SIGN: {
            for (U32 i = 0; i < len; i++) {
                output[i] = UNI_SIGN(input[i]);
            }
            break;
        }
        case ACTIVATION_LOG: {
            for (U32 i = 0; i < len; i++) {
                output[i] = log(input[i]);
            }
            break;
        }
        case ACTIVATION_NOT: {
            for (U32 i = 0; i < len; i++) {
                output[i] = (input[i] > 0) ? 0 : 1;
            }
            break;
        }
        case ACTIVATION_GREATER: {
            for (U32 i = 0; i < len; i++) {
                output[i] = input[i] > 1 ? 1 : 0;
            }
            break;
        }
        case ACTIVATION_NEG: {
            for (U32 i = 0; i < len; i++) {
                output[i] = -input[i];
            }
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

inline void array_scale_f32(const F32 *input, F32 *output, I32 len, F32 alpha, F32 beta)
{
    __m256 alpha_v = _mm256_set1_ps(alpha);
    __m256 beta_v = _mm256_set1_ps(beta);
    I32 i = 0;
    for (i = 0; i < len - 7; i += 8) {
        __m256 in = _mm256_loadu_ps(input + i);
        __m256 tmp_v = _mm256_add_ps(beta_v, _mm256_mul_ps(alpha_v, in));
        _mm256_storeu_ps(output + i, tmp_v);
    }
    for (; i < len; i++) {
        output[i] = alpha * input[i] + beta;
    }
}

inline void array_power_f32(F32 *input, F32 *output, I32 len, F32 power)
{
    I32 i = 0;
    if (power == -1) {
        __m256 one_v = _mm256_set1_ps(1);
        for (i = 0; i < len - 7; i += 8) {
            __m256 in = _mm256_loadu_ps(input + i);
            __m256 tmp_v = _mm256_div_ps(one_v, in);
            _mm256_storeu_ps(output + i, tmp_v);
        }
    } else if (power == -0.5) {
        __m256 one_v = _mm256_set1_ps(1);
        for (i = 0; i < len - 7; i += 8) {
            __m256 in = _mm256_loadu_ps(input + i);
            __m256 tmp_v = _mm256_div_ps(one_v, _mm256_sqrt_ps(in));
            _mm256_storeu_ps(output + i, tmp_v);
        }
    } else if (power == 0.5) {
        for (i = 0; i < len - 7; i += 8) {
            __m256 in = _mm256_loadu_ps(input + i);
            __m256 tmp_v = _mm256_sqrt_ps(in);
            _mm256_storeu_ps(output + i, tmp_v);
        }
    } else if (power == 1) {
        if (input != output) {
            memcpy(output, input, len * sizeof(F32));
        }
        i = len;
    } else if (power == 2) {
        for (i = 0; i < len - 7; i += 8) {
            __m256 in = _mm256_loadu_ps(input + i);
            __m256 tmp_v = _mm256_mul_ps(in, in);
            _mm256_storeu_ps(output + i, tmp_v);
        }
    }
    for (; i < len; i++) {
        output[i] = powf(input[i], power);
    }
}

template <int mode>
inline static void array_minmax_value_f32_template(const F32 *data, I32 len, F32 *result)
{
    F32 min_s = data[0];
    F32 max_s = data[0];
    I32 i = 0;
    if (len >= 8) {
        __m256 min_v, max_v, tmp_v;
        min_v = max_v = _mm256_loadu_ps(data);
        for (i = 8; i < len - 7; i += 8) {
            tmp_v = _mm256_loadu_ps(data + i);
            if (mode & 1)
                min_v = _mm256_min_ps(tmp_v, min_v);
            if (mode & 2)
                max_v = _mm256_max_ps(tmp_v, max_v);
        }
        if (mode & 1)
            max_s = _mm256_hmax_ps(min_v);
        if (mode & 2)
            max_s = _mm256_hmax_ps(max_v);
    }
    for (; i < len; i++) {
        if (data[i] < min_s) {
            min_s = data[i];
        }
        if (data[i] > max_s) {
            max_s = data[i];
        }
    }
    int id = 0;
    if (mode & 1)
        result[id++] = min_s;
    if (mode & 2)
        result[id++] = max_s;
}

inline EE array_minmax_value_f32(const F32 *data, I32 len, int mode, F32 *result)
{
    EE ret = SUCCESS;
    switch (mode) {
        case 1:
            array_minmax_value_f32_template<1>(data, len, result);
            break;
        case 2:
            array_minmax_value_f32_template<2>(data, len, result);
            break;
        case 3:
            array_minmax_value_f32_template<3>(data, len, result);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

template <int mode>
inline static void array_minmax_value_i32_template(const I32 *data, I32 len, F32 *result)
{
    I32 min_s = data[0];
    I32 max_s = data[0];
    I32 i = 0;
    if (len >= 8) {
        __m256i min_v, max_v, tmp_v;
        min_v = max_v = _mm256_loadu_si256((__m256i const *)data);
        for (i = 8; i < len - 7; i += 8) {
            tmp_v = _mm256_loadu_si256((__m256i const *)(data + i));
            if (mode & 1)
                min_v = _mm256_min_epu32(tmp_v, min_v);
            if (mode & 2)
                max_v = _mm256_max_epu32(tmp_v, max_v);
        }
        if (mode & 1)
            max_s = _mm256_hmax_epu32(min_v);
        if (mode & 2)
            max_s = _mm256_hmax_epu32(max_v);
    }
    for (; i < len; i++) {
        if (data[i] < min_s) {
            min_s = data[i];
        }
        if (data[i] > max_s) {
            max_s = data[i];
        }
    }
    int id = 0;
    if (mode & 1)
        result[id++] = min_s;
    if (mode & 2)
        result[id++] = max_s;
}

inline EE array_minmax_value_i32(const I32 *data, I32 len, int mode, F32 *result)
{
    EE ret = SUCCESS;
    switch (mode) {
        case 1:
            array_minmax_value_i32_template<1>(data, len, result);
            break;
        case 2:
            array_minmax_value_i32_template<2>(data, len, result);
            break;
        case 3:
            array_minmax_value_i32_template<3>(data, len, result);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

// array var
inline F32 array_var_f32(const F32 *data, I32 len, F32 mean)
{
    if (len <= 0) {
        return 0;
    }

    I32 i = 0;
    F32 sum_s = 0;
    __m256 mean_v = _mm256_set1_ps(mean);
    for (i = 0; i < len - 7; i += 8) {
        __m256 in = _mm256_loadu_ps(data + i);
        __m256 tmp_v = _mm256_sub_ps(in, mean_v);
        __m256 sum_v = _mm256_mul_ps(tmp_v, tmp_v);
        sum_s += _mm256_sum_ps(sum_v);
    }
    for (; i < len; i++) {
        F32 in = data[i];
        F32 tmp = in - mean;
        sum_s += tmp * tmp;
    }
    return sum_s / len;
}

// array sum
inline F32 array_sum_f32(const F32 *data, I32 len)
{
    if (len <= 0) {
        return 0;
    }

    I32 i = 0;
    F32 sum_s = 0;
    __m256 sum_v = _mm256_set1_ps(0);
    for (i = 0; i < len - 7; i += 8) {
        __m256 in = _mm256_loadu_ps(data + i);
        sum_v = _mm256_add_ps(sum_v, in);
    }
    sum_s += _mm256_sum_ps(sum_v);
    for (; i < len; i++) {
        sum_s += data[i];
    }
    return sum_s;
}

// array mean
inline F32 array_mean_f32(const F32 *data, I32 len)
{
    if (len <= 0) {
        return 0;
    }
    return array_sum_f32(data, len) / len;
}

inline void array_add_f32(const F32 *inputA, const F32 *inputB, F32 *output, I32 len)
{
    I32 i = 0;
    for (i = 0; i < len - 7; i += 8) {
        __m256 a = _mm256_loadu_ps(inputA + i);
        __m256 b = _mm256_loadu_ps(inputB + i);
        __m256 c = _mm256_add_ps(a, b);
        _mm256_storeu_ps(output + i, c);
    }

    for (; i < len; i++) {
        output[i] = inputA[i] + inputB[i];
    }
}

inline void array_mul_f32(const F32 *inputA, const F32 *inputB, F32 *output, I32 len)
{
    I32 i = 0;
    for (i = 0; i < len - 7; i += 8) {
        __m256 a = _mm256_loadu_ps(inputA + i);
        __m256 b = _mm256_loadu_ps(inputB + i);
        __m256 c = _mm256_mul_ps(a, b);
        _mm256_storeu_ps(output + i, c);
    }

    for (; i < len; i++) {
        output[i] = inputA[i] * inputB[i];
    }
}

inline void array_mul_and_add_f32(
    const F32 *inputA, const F32 *inputB, const F32 *inputC, F32 *output, I32 len)
{
    I32 i = 0;
    for (i = 0; i < len - 7; i += 8) {
        __m256 a = _mm256_loadu_ps(inputA + i);
        __m256 b;
        if (inputA == inputB) {
            b = a;
        } else {
            b = _mm256_loadu_ps(inputB + i);
        }
        __m256 c = _mm256_add_ps(_mm256_mul_ps(a, b), _mm256_loadu_ps(inputC + i));
        _mm256_storeu_ps(output + i, c);
    }

    for (; i < len; i++) {
        output[i] = inputA[i] * inputB[i] + inputC[i];
    }
}

inline void array_max_f32(const F32 *inputA, const F32 *inputB, F32 *output, I32 len)
{
    I32 i = 0;
    for (; i < len - 7; i += 8) {
        __m256 a = _mm256_loadu_ps(inputA + i);
        __m256 b = _mm256_loadu_ps(inputB + i);
        _mm256_storeu_ps(output + i, _mm256_max_ps(a, b));
    }
    for (; i < len; i++) {
        output[i] = UNI_MAX(inputA[i], inputB[i]);
    }
}

inline void array_norm_scalar_scale_fp32(
    F32 *input, F32 *output, I32 len, F32 mean, F32 var, F32 *alpha, F32 *beta)
{
    F32 eps = 1e-6;
    F32 std_value = sqrt(var + eps);
    __m256 mean_v = _mm256_set1_ps(mean);
    __m256 std_v = _mm256_set1_ps(std_value);
    __m256 alpha_v = _mm256_set1_ps(*alpha);
    __m256 beta_v = _mm256_set1_ps(*beta);

    I32 i = 0;
    for (i = 0; i < len - 7; i += 8) {
        __m256 in = _mm256_loadu_ps(input + i);
        __m256 tmp_v = _mm256_sub_ps(in, mean_v);
        tmp_v = _mm256_div_ps(tmp_v, std_v);
        tmp_v = _mm256_fmadd_ps(alpha_v, tmp_v, beta_v);
        _mm256_storeu_ps(output + i, tmp_v);
    }
    for (; i < len; i++) {
        output[i] = *alpha * (input[i] - mean) / std_value + *beta;
    }
}

#endif  //CHEETAH_X86_FUNCTION_FP32_H
