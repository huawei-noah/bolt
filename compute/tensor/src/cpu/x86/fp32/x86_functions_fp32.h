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

#include "cpu/cpu_functions_template.h"
#include "x86_avx2_expand.h"
#include "thread_affinity.h"

inline EE activation_fp32(F32 *input, U32 len, ActivationParamSpec activationDesc, F32 *output)
{
    __m256 zero = _mm256_set1_ps(0.);
    __m256 one = _mm256_set1_ps(1.);
    __m256 three = _mm256_set1_ps(3.);
    __m256 six = _mm256_set1_ps(6.);
    __m256 signm = _mm256_set1_ps(-0.0);
    U32 loops = len / 8 * 8;
    EE ret = SUCCESS;
    switch (activationDesc.mode) {
        case ACTIVATION_NULL: {
            if (output != input) {
                UNI_MEMCPY(output, input, sizeof(float) * len);
            }
            loops = len;
            break;
        }
        case ACTIVATION_RELU: {
            if (activationDesc.value[0] == 0) {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
                for (U32 i = 0; i < loops; i += 8) {
                    _mm256_storeu_ps(output + i, _mm256_max_ps(zero, _mm256_loadu_ps(input + i)));
                }
            } else {
                __m256 scale = _mm256_set1_ps(activationDesc.value[0]);
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
                for (U32 i = 0; i < loops; i += 8) {
                    __m256 tmp = _mm256_loadu_ps(input + i);
                    _mm256_storeu_ps(output + i, _mm256_max_ps(_mm256_mul_ps(scale, tmp), tmp));
                }
            }
            break;
        }
        case ACTIVATION_ELU: {
            __m256 alpha = _mm256_set1_ps(activationDesc.value[0]);
            __m256 beta = _mm256_set1_ps(-activationDesc.value[0]);
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_fmadd_ps(alpha, _mm256_exp_ps(in), beta);
                __m256 mask = _mm256_cmp_ps(in, zero, _CMP_GE_OS);
                _mm256_storeu_ps(output + i, _mm256_blendv_ps(out, in, mask));
            }
            break;
        }
        case ACTIVATION_RELU6: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_max_ps(zero, in);
                out = _mm256_min_ps(six, out);
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_H_SIGMOID: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_add_ps(in, three);
                out = _mm256_max_ps(out, zero);
                out = _mm256_min_ps(out, six);
                out = _mm256_div_ps(out, six);
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_H_SWISH: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_add_ps(in, three);
                out = _mm256_max_ps(out, zero);
                out = _mm256_min_ps(out, six);
                out = _mm256_div_ps(out, six);
                out = _mm256_mul_ps(out, in);
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_H_SWISH_NODIV: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_add_ps(in, three);
                out = _mm256_max_ps(out, zero);
                out = _mm256_min_ps(out, six);
                out = _mm256_mul_ps(out, in);
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_GELU: {
            __m256 vec0 = _mm256_set1_ps(sqrt(2 / 3.14159265358979323846) * 2);
            __m256 vec1 = _mm256_set1_ps(0.044715);
            __m256 vec2 = _mm256_set1_ps(0.5);
            __m256 two_v = _mm256_set1_ps(2);
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_mul_ps(in, in);
                __m256 tmp0 = _mm256_mul_ps(vec0, in);
                __m256 x = _mm256_fmadd_ps(vec1, out, one);
                __m256 tmp1 = _mm256_mul_ps(vec2, in);
                out = _mm256_mul_ps(x, tmp0);
                __m256 e_2G_v = _mm256_exp_ps(out);
                __m256 result_v = _mm256_sub_ps(two_v,
                    _mm256_div_ps(two_v, _mm256_add_ps(one, e_2G_v)));
                out = _mm256_mul_ps(result_v, tmp1);
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_TANH: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_tanh_ps(in);
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_SIGMOID: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_sigmod_ps(in);
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_SWISH: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_mul_ps(in, _mm256_sigmod_ps(in));
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_MISH: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_mul_ps(
                    in, _mm256_tanh_ps(_mm256_log_ps(_mm256_add_ps(_mm256_exp_ps(in), one))));
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_SOFTPLUS: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_log_ps(_mm256_add_ps(_mm256_exp_ps(in), one));
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_EXP: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_exp_ps(in);
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_ABS: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_andnot_ps(signm, in);
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_LOG: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_log_ps(in);
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_ROUND: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_round_ps(in, _MM_FROUND_TO_NEAREST_INT);
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_CEIL: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_ceil_ps(in);
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_FLOOR: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_floor_ps(in);
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_RECIPROCAL: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 8) {
                __m256 in = _mm256_loadu_ps(input + i);
                __m256 out = _mm256_div_ps(one, in);
                _mm256_storeu_ps(output + i, out);
            }
            break;
        }
        case ACTIVATION_SIGN:
        case ACTIVATION_NOT:
        case ACTIVATION_GREATER:
        case ACTIVATION_SIN:
        case ACTIVATION_COS:
        case ACTIVATION_NEG: {
            loops = 0;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    if (ret == SUCCESS) {
        for (U32 i = loops; i < len; i++) {
            ret = activation_template<F32>(activationDesc, input[i], output + i);
        }
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
            UNI_MEMCPY(output, input, len * sizeof(F32));
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
        F32 tmp = data[i] - mean;
        sum_s += tmp * tmp;
    }
    return sum_s / len;
}

inline F32 array_var_i32(const I32 *data, I32 len)
{
    if (len <= 0) {
        return 0;
    }
    I32 i = 0;
    F32 sum_s = 1;
    for (i = 0; i < len; i++) {
        sum_s *= data[i];
    }
    return sum_s;
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

inline I32 array_sum_i32(const I32 *data, I32 len)
{
    if (len <= 0) {
        return 0;
    }

    I32 i = 0;
    I32 sum_s = 0;
    __m256i sum_v = _mm256_set1_epi32(0);
    for (i = 0; i < len - 7; i += 8) {
        __m256i in = _mm256_loadu_si256((const __m256i *)(data + i));
        sum_v = _mm256_add_epi32(sum_v, in);
    }
    sum_s += _mm256_sum_epi32(sum_v);
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
