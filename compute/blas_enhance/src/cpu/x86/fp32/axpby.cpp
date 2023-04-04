// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/fp32/blas_fp32.h"

EE axpby_fp32(I32 len, F32 a, const F32 *x, F32 b, F32 *y)
{
    __m256 alpha = _mm256_set1_ps(a);
    __m256 beta = _mm256_set1_ps(b);
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
    for (I32 i = 0; i < len - 7; i += 8) {
        __m256 in = _mm256_loadu_ps(x + i);
        __m256 out = _mm256_loadu_ps(y + i);
        out = _mm256_mul_ps(out, beta);
        out = _mm256_fmadd_ps(alpha, in, out);
        _mm256_storeu_ps(y + i, out);
    }
    for (I32 i = len / 8 * 8; i < len; i++) {
        y[i] = a * x[i] + b * y[i];
    }
    return SUCCESS;
}
