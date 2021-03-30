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

#define UNROLL_W 4

typedef void (*pooling_bp_func)(
    const F32 *input, int hstart, int hend, int wstart, int wend, F32 *output, U32 ow, U32 strideW);

void pooling_bp_c8_w4_fp32(
    const F32 *input, int hstart, int hend, int wstart, int wend, F32 *output, U32 ow, U32 strideW)
{
    __m256 poolSize = _mm256_set1_ps((hend - hstart) * (wend - wstart) * 1.0f);
    __m256 in0 = _mm256_div_ps(_mm256_loadu_ps(input), poolSize);
    __m256 in1 = _mm256_div_ps(_mm256_loadu_ps(input + 8), poolSize);
    __m256 in2 = _mm256_div_ps(_mm256_loadu_ps(input + 16), poolSize);
    __m256 in3 = _mm256_div_ps(_mm256_loadu_ps(input + 24), poolSize);
    for (int kernelH = hstart; kernelH < hend; kernelH++) {
        for (int kernelW = wstart; kernelW < wend; kernelW++) {
            U32 index0 = (kernelH * ow + kernelW) * 8;
            U32 index1 = (kernelH * ow + kernelW + strideW) * 8;
            U32 index2 = (kernelH * ow + kernelW + strideW * 2) * 8;
            U32 index3 = (kernelH * ow + kernelW + strideW * 3) * 8;
            __m256 out0 = _mm256_add_ps(_mm256_loadu_ps(output + index0), in0);
            __m256 out1 = _mm256_add_ps(_mm256_loadu_ps(output + index1), in1);
            __m256 out2 = _mm256_add_ps(_mm256_loadu_ps(output + index2), in2);
            __m256 out3 = _mm256_add_ps(_mm256_loadu_ps(output + index3), in3);
            _mm256_storeu_ps(output + index0, out0);
            _mm256_storeu_ps(output + index1, out1);
            _mm256_storeu_ps(output + index2, out2);
            _mm256_storeu_ps(output + index3, out3);
        }
    }
}

void pooling_bp_c8_w2_fp32(
    const F32 *input, int hstart, int hend, int wstart, int wend, F32 *output, U32 ow, U32 strideW)
{
    __m256 poolSize = _mm256_set1_ps((hend - hstart) * (wend - wstart) * 1.0f);
    __m256 in0 = _mm256_div_ps(_mm256_loadu_ps(input), poolSize);
    __m256 in1 = _mm256_div_ps(_mm256_loadu_ps(input + 8), poolSize);
    for (int kernelH = hstart; kernelH < hend; kernelH++) {
        for (int kernelW = wstart; kernelW < wend; kernelW++) {
            U32 index0 = (kernelH * ow + kernelW) * 8;
            U32 index1 = (kernelH * ow + kernelW + strideW) * 8;
            __m256 out0 = _mm256_add_ps(_mm256_loadu_ps(output + index0), in0);
            __m256 out1 = _mm256_add_ps(_mm256_loadu_ps(output + index1), in1);
            _mm256_storeu_ps(output + index0, out0);
            _mm256_storeu_ps(output + index1, out1);
        }
    }
}

void pooling_bp_c8_w1_fp32(
    const F32 *input, int hstart, int hend, int wstart, int wend, F32 *output, U32 ow, U32 strideW)
{
    __m256 poolSize = _mm256_set1_ps((hend - hstart) * (wend - wstart) * 1.0f);
    __m256 in0 = _mm256_div_ps(_mm256_loadu_ps(input), poolSize);
    for (int kernelH = hstart; kernelH < hend; kernelH++) {
        for (int kernelW = wstart; kernelW < wend; kernelW++) {
            U32 index = (kernelH * ow + kernelW) * 8;
            __m256 out0 = _mm256_add_ps(_mm256_loadu_ps(output + index), in0);
            _mm256_storeu_ps(output + index, out0);
        }
    }
}

EE pooling_bp_fp32(
    TensorDesc inputDesc, const F32 *input, PoolingParamSpec p, TensorDesc outputDesc, F32 *output)
{
    EE ret = SUCCESS;
    if (nullptr == input || nullptr == output) {
        ret = NULL_POINTER;
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in = 0, ic = 0, ih = 0, iw = 0, on = 0, oc = 0, oh = 0, ow = 0;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (idt != odt) {
        ret = NOT_MATCH;
    }
    if (in != on || ic != oc) {
        ret = NOT_MATCH;
    }
    if (idf != DF_NCHWC8 || odf != idf) {
        ret = NOT_MATCH;
    }
    if (p.padding_top >= p.kernel_h || p.padding_left >= p.kernel_w) {
        ret = NOT_SUPPORTED;
    }
    PoolingMode pm = p.mode;
    if (pm != POOLING_MEAN) {
        ret = NOT_SUPPORTED;
    }

    ic /= 8;
    U32 wSize = 0;
    U32 iwInter = (ow + p.padding_left - p.kernel_w) / p.stride_w + 1;
    const F32 *curI = input;
    F32 *curO = output;
    pooling_bp_func pooling_bp[3] = {
        pooling_bp_c8_w1_fp32, pooling_bp_c8_w2_fp32, pooling_bp_c8_w4_fp32};
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 h = 0; h < ih; h++) {
                for (U32 w = 0; w < iw; w += wSize) {
                    if (w < iwInter) {
                        wSize = UNI_MIN(iwInter - w, UNROLL_W);
                    } else {
                        wSize = 1;
                    }
                    int hstart = (int)h * (int)p.stride_h - (int)p.padding_top;
                    int wstart = (int)w * (int)p.stride_w - (int)p.padding_left;
                    int hend = UNI_MIN(hstart + p.kernel_h, oh);
                    int wend = UNI_MIN(wstart + p.kernel_w, ow);
                    hstart = UNI_MAX(hstart, 0);
                    wstart = UNI_MAX(wstart, 0);
                    if (wend < wstart + (int)p.kernel_w) {
                        wSize = 1;
                    }
                    pooling_bp[wSize >> 1](curI, hstart, hend, wstart, wend, curO, ow, p.stride_w);
                    curI += wSize * 8;
                }
            }
            curO += oh * ow * 8;
        }
    }
    return ret;
}