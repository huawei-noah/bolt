// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/tensor_computing_x86.h"
#include "cpu/x86/fp32/tensor_computing_fp32.h"

EE instance_norm_fp32(TensorDesc inputDesc,
    F32 *input,
    F32 *tmp,
    F32 *scale,
    F32 *bias,
    InstanceNormParamSpec p,
    F32 *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.df != DF_NCHWC8) {
        CHECK_STATUS(NOT_MATCH);
    }

    I32 axis = (p.axis + inputDesc.nDims) % inputDesc.nDims;
    axis = inputDesc.nDims - 1 - axis;


    // support axisDim != inputDesc.dims[axis]
    I32 axisDim = (p.axis_dim > 0) ? p.axis_dim : inputDesc.dims[axis];

    I32 loopInner = 1;
    for (I32 i = 0; i < axis; ++i) {
        loopInner *= inputDesc.dims[i];
    }

    I32 loopOuter = 1;
    for (U32 i = axis; i < inputDesc.nDims; ++i) {
        loopOuter *= inputDesc.dims[i];
    }

    if (axisDim == (I32)inputDesc.dims[axis]) {
        __m256 epsv = _mm256_set1_ps(1e-6);
        __m256 loopInner_v = _mm256_set1_ps((float)loopInner);
        F32 *scaleTmp = scale;
        F32 *biasTmp = bias;
        if (axisDim < loopOuter) {
            scaleTmp = tmp;
            biasTmp = scaleTmp + loopOuter;
            for (I32 i = 0; i < loopOuter; i += 8) {
                scaleTmp[i] = scale[i % axisDim];
                biasTmp[i] = bias[i % axisDim];
            }
        }

        for (I32 i = 0; i < loopOuter; i += 8) {
            __m256 m1 = _mm256_setzero_ps();
            __m256 m = _mm256_setzero_ps();
            for (I32 j = 0; j < loopInner; ++j) {
                m1 = _mm256_add_ps(m1, _mm256_loadu_ps(input + i * loopInner + j * 8));
                if (((j + 1) % 1024 == 0) || (j == loopInner - 1)) {
                    m = _mm256_add_ps(m, _mm256_div_ps(m1, loopInner_v));
                    m1 = _mm256_setzero_ps();
                }
            }
            __m256 v = _mm256_setzero_ps();
            for (I32 j = 0; j < loopInner; ++j) {
                __m256 t = _mm256_sub_ps(_mm256_loadu_ps(input + i * loopInner + j * 8), m);
                v = _mm256_add_ps(v, _mm256_mul_ps(t, t));
            }
            v = _mm256_sqrt_ps(_mm256_add_ps(_mm256_div_ps(v, loopInner_v), epsv));

            __m256 s = _mm256_loadu_ps(scaleTmp + i);
            __m256 b = _mm256_loadu_ps(biasTmp + i);
            for (I32 j = 0; j < loopInner; ++j) {
                __m256 t = _mm256_div_ps(
                    _mm256_sub_ps(_mm256_loadu_ps(input + i * loopInner + j * 8), m), v);
                t = _mm256_add_ps(_mm256_mul_ps(t, s), b);
                _mm256_storeu_ps(output + i * loopInner + j * 8, t);
            }
        }
    } else {
        I32 loopInnerIn = loopInner * inputDesc.dims[axis] * 1.0f / axisDim;
        I32 loopOuterIn = loopOuter * axisDim * 1.0f / inputDesc.dims[axis];

        if (loopOuterIn == 1) {
            F32 mean = 0;
            F32 var = 0;

            I32 j = 0;
            __m256 tmp_v = _mm256_set1_ps(0.f);
            for (j = 0; j < loopInnerIn - 7; j += 8) {
                tmp_v = _mm256_add_ps(tmp_v, _mm256_loadu_ps(input + j));
            }
            mean = _mm256_sum_ps(tmp_v);
            for (; j < loopInnerIn; j++) {
                mean += input[j];
            }
            mean = mean / loopInnerIn;

            tmp_v = _mm256_set1_ps(0.f);
            __m256 val = _mm256_set1_ps(0.f);
            __m256 mean_v = _mm256_set1_ps(mean);
            for (j = 0; j < loopInnerIn - 7; j += 8) {
                val = _mm256_sub_ps(_mm256_loadu_ps(input + j), mean_v);
                tmp_v = _mm256_add_ps(tmp_v, _mm256_mul_ps(val, val));
            }
            var = _mm256_sum_ps(tmp_v);
            for (; j < loopInnerIn; j++) {
                F32 tmpVal = input[j] - mean;
                var += tmpVal * tmpVal;
            }
            var = scale[0] / sqrt(var / loopInnerIn + 1e-6);

            __m256 bias_v = _mm256_set1_ps(bias[0]);
            __m256 var_v = _mm256_set1_ps(var);
            for (j = 0; j < loopInnerIn - 7; j += 8) {
                _mm256_storeu_ps(output + j, _mm256_add_ps(bias_v, _mm256_mul_ps(var_v, _mm256_sub_ps(_mm256_loadu_ps(input + j), mean_v))));
            }
            for (; j < loopInnerIn; ++j) {
                output[j] = (input[j] - mean) * var + bias[0];
            }
            return SUCCESS;
        }

        I32 maxLoopOuter = UNI_MAX(loopOuter, loopOuterIn);
        F32 *mean = tmp;
        F32 *var = tmp + maxLoopOuter;
        F32 *scaleTmp = var + maxLoopOuter;
        F32 *biasTmp = scaleTmp + maxLoopOuter;

        CHECK_REQUIREMENT(loopOuterIn % 8 == 0 && loopOuter % 8 == 0);

#ifdef _USE_LINUX_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
#endif
        {
#ifdef _USE_LINUX_OPENMP
#pragma omp for schedule(static, 2)
#endif
            for (I32 i = 0; i < loopOuter; i += 8) {
                __m256 m = _mm256_setzero_ps();
                for (I32 j = 0; j < loopInner; ++j) {
                    m = _mm256_add_ps(m, _mm256_loadu_ps(input + i * loopInner + j * 8));
                }
                _mm256_storeu_ps(mean + i, m);
            }
            I32 s = inputDesc.dims[axis] / axisDim;
#ifdef _USE_LINUX_OPENMP
#pragma omp for schedule(static, 2)
#endif
            for (I32 i = 0; i < loopOuter; i += s) {
                F32 sm = 0.0f;
                for (I32 j = i; j < i + s; ++j) {
                    sm += mean[j];
                }
                sm = sm * 1.0f / loopInnerIn;
                for (I32 j = i; j < i + s; ++j) {
                    mean[j] = sm;
                }
            }
#ifdef _USE_LINUX_OPENMP
#pragma omp for schedule(static, 2)
#endif
            for (I32 i = 0; i < loopOuter; i += 8) {
                __m256 v = _mm256_setzero_ps();
                __m256 m = _mm256_loadu_ps(mean + i);
                for (I32 j = 0; j < loopInner; ++j) {
                    __m256 t = _mm256_sub_ps(_mm256_loadu_ps(input + i * loopInner + j * 8), m);
                    v = _mm256_add_ps(v, _mm256_mul_ps(t, t));
                }
                _mm256_storeu_ps(var + i, v);
            }
#ifdef _USE_LINUX_OPENMP
#pragma omp for schedule(static, 2)
#endif
            for (I32 i = 0; i < loopOuter; i += s) {
                F32 sv = 0;
                for (I32 j = i; j < i + s; ++j) {
                    sv += var[j];
                }
                sv = sqrt(sv / loopInnerIn + 1e-6);
                for (I32 j = i; j < i + s; ++j) {
                    var[j] = sv;
                }
            }

            //compute
            if (loopOuter >= loopOuterIn) {
#ifdef _USE_LINUX_OPENMP
#pragma omp for schedule(static, 2)
#endif
                for (I32 i = loopOuter - 1; i >= 0; --i) {
                    I32 idx = i * loopInner / loopInnerIn;
                    scaleTmp[i] = scale[idx % axisDim];
                    biasTmp[i] = bias[idx % axisDim];
                }

#ifdef _USE_LINUX_OPENMP
#pragma omp for schedule(static)
#endif
                for (I32 i = 0; i < loopOuter; i += 8) {
                    __m256 m = _mm256_loadu_ps(mean + i);
                    __m256 v = _mm256_loadu_ps(var + i);
                    __m256 s = _mm256_loadu_ps(scaleTmp + i);
                    __m256 b = _mm256_loadu_ps(biasTmp + i);
                    for (I32 j = 0; j < loopInner; ++j) {
                        I32 ioff = i * loopInner + j * 8;
                        I32 ooff = i * loopInner + j * 8;
                        __m256 t = _mm256_div_ps(_mm256_sub_ps(_mm256_loadu_ps(input + ioff), m), v);
                        t = _mm256_add_ps(_mm256_mul_ps(t, s), b);
                        _mm256_storeu_ps(output + ooff, t);
                    }
                }
            } else {
                for (I32 i = 0; i < loopOuter; i += 8) {
                    for (I32 j = 0; j < loopInner; ++j) {
                        for (I32 ii = 0; ii < 8; ++ii) {
                            I32 idx = ((i + ii) * loopInner + j) / loopInnerIn;
                            output[i * loopInner + j * 8 + ii] = scale[idx % axisDim] *
                                    (input[i * loopInner + j * 8 + ii] - mean[i + ii]) /
                                    var[i + ii] +
                                bias[idx % axisDim];
                        }
                    }
                }
            }
        }
    }
    return SUCCESS;
}
