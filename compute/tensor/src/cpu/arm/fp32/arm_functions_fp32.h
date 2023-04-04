// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_ARM_FUNCTIONS_FP32
#define _H_ARM_FUNCTIONS_FP32

#include "cpu/cpu_functions_template.h"
#include "arm_neon_expand.h"

// array sum
inline F32 array_sum_f32(const F32 *data, I32 len)
{
    if (len <= 0) {
        return 0;
    }

    I32 i = 0;
    F32 sum_s = 0;
    float32x4_t sum_v = vdupq_n_f32(0);
#pragma unroll(4)
    for (i = 0; i < len - 3; i += 4) {
        float32x4_t in = vld1q_f32(data + i);
        sum_v = vaddq_f32(sum_v, in);
    }
    sum_s += vaddvq_f32(sum_v);
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

// array var
inline F32 array_var_f32(const F32 *data, I32 len, F32 mean)
{
    if (len <= 0) {
        return 0;
    }

    I32 i = 0;
    F32 sum_s = 0;
    float32x4_t mean_v = vdupq_n_f32(mean);
#pragma unroll(4)
    for (i = 0; i < len - 3; i += 4) {
        float32x4_t in = vld1q_f32(data + i);
        float32x4_t tmp_v = vsubq_f32(in, mean_v);
        float32x4_t sum_v = vmulq_f32(tmp_v, tmp_v);
        sum_s += vaddvq_f32(sum_v);
    }
    for (; i < len; i++) {
        F32 in = data[i];
        F32 tmp = in - mean;
        sum_s += tmp * tmp;
    }
    return sum_s / len;
}

template <int mode>
inline static void array_minmax_value_f32_template(const F32 *data, I32 len, F32 *result)
{
    F32 min_s = data[0];
    F32 max_s = data[0];
    I32 i = 0;
    if (len >= 4) {
        float32x4_t min_v, max_v, tmp_v;
        min_v = max_v = vld1q_f32(data);
#pragma unroll(4)
        for (i = 4; i < len - 3; i += 4) {
            tmp_v = vld1q_f32(data + i);
            if (mode & 1)
                min_v = vminq_f32(tmp_v, min_v);
            if (mode & 2)
                max_v = vmaxq_f32(tmp_v, max_v);
        }
        if (mode & 1)
            min_s = vminvq_f32(min_v);
        if (mode & 2)
            max_s = vmaxvq_f32(max_v);
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
    if (len >= 4) {
        int32x4_t min_v, max_v, tmp_v;
        min_v = max_v = vld1q_s32(data);
#pragma unroll(4)
        for (i = 4; i < len - 3; i += 4) {
            tmp_v = vld1q_s32(data + i);
            if (mode & 1)
                min_v = vminq_s32(tmp_v, min_v);
            if (mode & 2)
                max_v = vmaxq_s32(tmp_v, max_v);
        }
        if (mode & 1)
            min_s = vminvq_s32(min_v);
        if (mode & 2)
            max_s = vmaxvq_s32(max_v);
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

inline void array_scale_f32(const F32 *input, F32 *output, I32 len, F32 alpha, F32 beta)
{
    float32x4_t alpha_v = vdupq_n_f32(alpha);
    float32x4_t beta_v = vdupq_n_f32(beta);
    I32 i = 0;
#pragma unroll(4)
    for (i = 0; i < len - 3; i += 4) {
        float32x4_t in = vld1q_f32(input + i);
        float32x4_t tmp_v = vfmaq_f32(beta_v, alpha_v, in);
        vst1q_f32(output + i, tmp_v);
    }
    for (; i < len; i++) {
        output[i] = alpha * input[i] + beta;
    }
}

inline void array_scale_round_f32(const F32 *input, INT8 *output, I32 len, F32 alpha, bool clamp)
{
    float32x4_t alpha_v = vdupq_n_f32(alpha);
    I32 i = 0;
#pragma unroll(4)
    for (i = 0; i < len - 7; i += 8) {
        float32x4_t in0 = vld1q_f32(input + i);
        float32x4_t in1 = vld1q_f32(input + i + 4);
        int16x4_t t0 = vmovn_s32(vcvtaq_s32_f32(vmulq_f32(alpha_v, in0)));
        int16x4_t t1 = vmovn_s32(vcvtaq_s32_f32(vmulq_f32(alpha_v, in1)));
        int8x8_t t2 = vmovn_s16(vcombine_s16(t0, t1));
        vst1_s8(output + i, t2);
    }
    for (; i < len; i++) {
        output[i] = round(alpha * input[i]);
    }
}

inline void array_scale_round_f32_i32(const F32 *input, I32 *output, I32 len, F32 alpha, bool clamp)
{
    float32x4_t alpha_v = vdupq_n_f32(alpha);
    I32 i = 0;
#pragma unroll(4)
    for (i = 0; i < len - 3; i += 4) {
        float32x4_t in0 = vld1q_f32(input + i);
        int32x4_t t0 = vcvtaq_s32_f32(vmulq_f32(alpha_v, in0));
        vst1q_s32(output + i, t0);
    }
    for (; i < len; i++) {
        output[i] = round(alpha * input[i]);
    }
}

inline void array_power_f32(F32 *input, F32 *output, I32 len, F32 power)
{
    I32 i = 0;
    if (power == -1) {
        float32x4_t one_v = vdupq_n_f32(1);
#pragma unroll(4)
        for (i = 0; i < len - 3; i += 4) {
            float32x4_t in = vld1q_f32(input + i);
            float32x4_t tmp_v = vdivq_f32(one_v, in);
            vst1q_f32(output + i, tmp_v);
        }
    } else if (power == -0.5) {
#ifdef __aarch64__
        float32x4_t one_v = vdupq_n_f32(1);
#pragma unroll(4)
        for (i = 0; i < len - 3; i += 4) {
            float32x4_t in = vld1q_f32(input + i);
            float32x4_t tmp_v = vdivq_f32(one_v, vsqrtq_f32(in));
            vst1q_f32(output + i, tmp_v);
        }
#endif
    } else if (power == 0.5) {
#ifdef __aarch64__
#pragma unroll(4)
        for (i = 0; i < len - 3; i += 4) {
            float32x4_t in = vld1q_f32(input + i);
            float32x4_t tmp_v = vsqrtq_f32(in);
            vst1q_f32(output + i, tmp_v);
        }
#endif
    } else if (power == 1) {
        if (input != output) {
            UNI_MEMCPY(output, input, len * sizeof(F32));
        }
        i = len;
    } else if (power == 2) {
#pragma unroll(4)
        for (i = 0; i < len - 3; i += 4) {
            float32x4_t in = vld1q_f32(input + i);
            float32x4_t tmp_v = vmulq_f32(in, in);
            vst1q_f32(output + i, tmp_v);
        }
    }
    for (; i < len; i++) {
        output[i] = powf(input[i], power);
    }
}

inline EE activation_fp32(F32 *input, U32 len, ActivationParamSpec activationDesc, F32 *output)
{
    float32x4_t zero = vdupq_n_f32(0.);
    float32x4_t one = vdupq_n_f32(1.);
    float32x4_t three = vdupq_n_f32(3.);
    float32x4_t six = vdupq_n_f32(6.);
    U32 loops = len / 4 * 4;
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
                for (U32 i = 0; i < loops; i += 4) {
                    float32x4_t in = vld1q_f32(input + i);
                    float32x4_t out = vmaxq_f32(zero, in);
                    vst1q_f32(output + i, out);
                }
            } else {
                float32x4_t scale = vdupq_n_f32(activationDesc.value[0]);
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
                for (U32 i = 0; i < loops; i += 4) {
                    float32x4_t in = vld1q_f32(input + i);
                    float32x4_t tmp = vmulq_f32(in, scale);
                    float32x4_t out = vmaxq_f32(tmp, in);
                    vst1q_f32(output + i, out);
                }
            }
            break;
        }
        case ACTIVATION_ELU: {
            float32x4_t alpha = vdupq_n_f32(activationDesc.value[0]);
            float32x4_t beta = vdupq_n_f32(-activationDesc.value[0]);
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                uint32x4_t mask = vcgtq_f32(in, zero);
                float32x4_t t = vfmaq_f32(beta, alpha, vexpq_f32_03_percent_error(in));
                float32x4_t out = vbslq_f32(mask, in, t);
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_RELU6: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vmaxq_f32(zero, in);
                out = vminq_f32(six, out);
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_H_SIGMOID: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vaddq_f32(in, three);
                out = vmaxq_f32(out, zero);
                out = vminq_f32(out, six);
                out = vdivq_f32(out, six);
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_H_SWISH: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vaddq_f32(in, three);
                out = vmaxq_f32(out, zero);
                out = vminq_f32(out, six);
                out = vdivq_f32(out, six);
                out = vmulq_f32(out, in);
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_H_SWISH_NODIV: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vaddq_f32(in, three);
                out = vmaxq_f32(out, zero);
                out = vminq_f32(out, six);
                out = vmulq_f32(out, in);
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_GELU: {
            float32x4_t vec0 = vdupq_n_f32(sqrt(2 / 3.14159265358979323846));
            float32x4_t vec1 = vdupq_n_f32(0.044715);
            float32x4_t vec2 = vdupq_n_f32(0.5);
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vmulq_f32(in, in);
                out = vmulq_f32(out, in);
                out = vfmaq_f32(in, vec1, out);
                out = vmulq_f32(vec0, out);
                out = vtanhq_f32(out);
                out = vaddq_f32(one, out);
                out = vmulq_f32(vec2, out);
                out = vmulq_f32(in, out);
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_TANH: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vtanhq_f32(in);
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_SIGMOID: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vsigmoidq_f32(in);
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_SWISH: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#else
#pragma unroll(4)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vmulq_f32(in, vsigmoidq_f32(in));
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_MISH: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vmulq_f32(
                    in, vtanhq_f32(vlogq_f32(vaddq_f32(vexpq_f32_03_percent_error(in), one))));
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_SOFTPLUS: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vlogq_f32(vaddq_f32(vexpq_f32_03_percent_error(in), one));
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_EXP: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vexpq_f32_03_percent_error(in);
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_ABS: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vabsq_f32(in);
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_RECIPROCAL: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vdivq_f32(one, in);
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_SIN: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vsinq_f32(in);
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_COS: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vcosq_f32(in);
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_ROUND: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vrndaq_f32(in);
                vst1q_f32(output + i, out);
            }
            break;
        }
        case ACTIVATION_CEIL: {
#ifdef __aarch64__
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vrndpq_f32(in);
                vst1q_f32(output + i, out);
            }
            break;
#endif
        }
        case ACTIVATION_FLOOR: {
#ifdef __aarch64__
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
            for (U32 i = 0; i < loops; i += 4) {
                float32x4_t in = vld1q_f32(input + i);
                float32x4_t out = vrndmq_f32(in);
                vst1q_f32(output + i, out);
            }
            break;
#endif
        }
        case ACTIVATION_SIGN:
        case ACTIVATION_LOG:
        case ACTIVATION_NOT:
        case ACTIVATION_GREATER:
        case ACTIVATION_NEG: {
            loops = 0;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    if (ret == SUCCESS) {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
        for (U32 i = loops; i < len; i++) {
            ret = activation_template<F32>(activationDesc, input[i], output + i);
        }
    }
    return ret;
}

inline void array_mul_f32(const F32 *inputA, const F32 *inputB, F32 *output, I32 len)
{
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
    for (I32 i = 0; i < len - 3; i += 4) {
        float32x4_t a = vld1q_f32(inputA + i);
        float32x4_t b = vld1q_f32(inputB + i);
        float32x4_t c = vmulq_f32(a, b);
        vst1q_f32(output + i, c);
    }
    for (I32 i = len / 4 * 4; i < len; i++) {
        output[i] = inputA[i] * inputB[i];
    }
}

inline void array_mul_and_add_f32(
    const F32 *inputA, const F32 *inputB, const F32 *inputC, F32 *output, I32 len)
{
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#else
#pragma unroll(4)
#endif
    for (I32 i = 0; i < len - 3; i += 4) {
        float32x4_t a = vld1q_f32(inputA + i);
        float32x4_t b = vld1q_f32(inputB + i);
        float32x4_t c = vld1q_f32(inputC + i);
        c = vfmaq_f32(c, a, b);
        vst1q_f32(output + i, c);
    }
    for (I32 i = len / 4 * 4; i < len; i++) {
        output[i] = inputA[i] * inputB[i] + inputC[i];
    }
}

inline void array_add_f32(const F32 *inputA, const F32 *inputB, F32 *output, I32 len)
{
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#else
#pragma unroll(4)
#endif
    for (I32 i = 0; i < len - 3; i += 4) {
        float32x4_t a = vld1q_f32(inputA + i);
        float32x4_t b = vld1q_f32(inputB + i);
        float32x4_t c = vaddq_f32(a, b);
        vst1q_f32(output + i, c);
    }
    for (I32 i = len / 4 * 4; i < len; i++) {
        output[i] = inputA[i] + inputB[i];
    }
}

inline void array_max_f32(const F32 *inputA, const F32 *inputB, F32 *output, I32 len)
{
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#else
#pragma unroll(4)
#endif
    for (I32 i = 0; i < len - 3; i += 4) {
        float32x4_t a = vld1q_f32(inputA + i);
        float32x4_t b = vld1q_f32(inputB + i);
        vst1q_f32(output + i, vmaxq_f32(a, b));
    }
    for (I32 i = len / 4 * 4; i < len; i++) {
        output[i] = UNI_MAX(inputA[i], inputB[i]);
    }
}

inline void array_norm_scalar_scale_fp32(
    F32 *input, F32 *output, I32 len, F32 mean, F32 var, F32 *alpha, F32 *beta)
{
    F32 eps = 1e-6;
    F32 std_value = sqrt(var + eps);
    float32x4_t mean_v = vdupq_n_f32(mean);
    float32x4_t std_v = vdupq_n_f32(std_value);
    float32x4_t alpha_v = vdupq_n_f32(*alpha);
    float32x4_t beta_v = vdupq_n_f32(*beta);

#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#else
#pragma unroll(4)
#endif
    for (I32 i = 0; i < len - 3; i += 4) {
        float32x4_t in = vld1q_f32(input + i);
        float32x4_t tmp_v = vsubq_f32(in, mean_v);
        tmp_v = vdivq_f32(tmp_v, std_v);
        tmp_v = vfmaq_f32(beta_v, alpha_v, tmp_v);
        vst1q_f32(output + i, tmp_v);
    }
    for (I32 i = len / 4 * 4; i < len; i++) {
        output[i] = *alpha * (input[i] - mean) / std_value + *beta;
    }
}

#endif
