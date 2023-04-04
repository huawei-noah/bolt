// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/fp32/tensor_computing_fp32.h"

EE decode_priorbox_fp32(const F32 *location,
    const F32 *priorbox,
    const F32 *variance,
    I32 num_total_priorbox,
    F32 *xmin,
    F32 *ymin,
    F32 *xmax,
    F32 *ymax)
{
    __m256 loc[4], pb[4], var[4];
    __m256i index = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
    __m256 v0 = _mm256_set1_ps(0.5f);
    __m256 v1 = _mm256_set1_ps(-0.5f);
    I32 i = 0;
    for (; i < num_total_priorbox - 7; i += 8) {
        for (int j = 0; j < 4; j++) {
            loc[j] = _mm256_i32gather_ps(location + i * 4 + j, index, 4);
            pb[j] = _mm256_i32gather_ps(priorbox + i * 4 + j, index, 4);
            var[j] = _mm256_i32gather_ps(variance + i * 4 + j, index, 4);
        }

        __m256 pb_w = _mm256_sub_ps(pb[2], pb[0]);
        __m256 pb_h = _mm256_sub_ps(pb[3], pb[1]);
        __m256 pb_cx = _mm256_mul_ps(_mm256_add_ps(pb[0], pb[2]), v0);
        __m256 pb_cy = _mm256_mul_ps(_mm256_add_ps(pb[1], pb[3]), v0);

        __m256 box_cx = _mm256_fmadd_ps(var[0], _mm256_mul_ps(loc[0], pb_w), pb_cx);
        __m256 box_cy = _mm256_fmadd_ps(var[1], _mm256_mul_ps(loc[1], pb_h), pb_cy);
        __m256 box_w = _mm256_mul_ps(_mm256_exp_ps(_mm256_mul_ps(var[2], loc[2])), pb_w);
        __m256 box_h = _mm256_mul_ps(_mm256_exp_ps(_mm256_mul_ps(var[3], loc[3])), pb_h);

        _mm256_storeu_ps(xmin + i, _mm256_fmadd_ps(box_w, v1, box_cx));
        _mm256_storeu_ps(ymin + i, _mm256_fmadd_ps(box_h, v1, box_cy));
        _mm256_storeu_ps(xmax + i, _mm256_fmadd_ps(box_w, v0, box_cx));
        _mm256_storeu_ps(ymax + i, _mm256_fmadd_ps(box_h, v0, box_cy));
    }
    for (; i < num_total_priorbox; i++) {
        const F32 *loc = location + i * 4;
        const F32 *pb = priorbox + i * 4;
        const F32 *var = variance + i * 4;

        F32 pb_w = pb[2] - pb[0];
        F32 pb_h = pb[3] - pb[1];
        F32 pb_cx = (pb[0] + pb[2]) * 0.5f;
        F32 pb_cy = (pb[1] + pb[3]) * 0.5f;

        F32 box_cx = var[0] * loc[0] * pb_w + pb_cx;
        F32 box_cy = var[1] * loc[1] * pb_h + pb_cy;
        F32 box_w = static_cast<F32>(exp(var[2] * loc[2]) * pb_w);
        F32 box_h = static_cast<F32>(exp(var[3] * loc[3]) * pb_h);

        xmin[i] = box_cx + box_w * -0.5f;
        ymin[i] = box_cy + box_h * -0.5f;
        xmax[i] = box_cx + box_w * 0.5f;
        ymax[i] = box_cy + box_h * 0.5f;
    }
    return SUCCESS;
}
