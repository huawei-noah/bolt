// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/image_x86.h"
#include "cpu/cpu_functions.h"

typedef void (*copy_func)(
    F32 *input, F32 *output, U32 icStep, U32 ocStep, I32 oc, U32 inStep, U32 onStep, U32 on);
typedef void (*compute_linear_func)(F32 *input0,
    F32 *input1,
    F32 *output,
    F32 r0,
    F32 r1,
    U32 icStep,
    U32 ocStep,
    I32 oc,
    U32 inStep,
    U32 onStep,
    U32 on);
typedef void (*compute_bilinear_func)(F32 *input0,
    F32 *input1,
    F32 *input2,
    F32 *input3,
    F32 *output,
    F32 r0,
    F32 r1,
    F32 r2,
    F32 r3,
    U32 icStep,
    U32 ocStep,
    I32 oc,
    U32 inStep,
    U32 onStep,
    U32 on);

inline void copy_batch_nchwc8_fp32(
    F32 *input, F32 *output, U32 icStep, U32 ocStep, I32 oc, U32 inStep, U32 onStep, U32 on)
{
    for (U32 n = 0; n < on; ++n) {
        for (I32 c = 0; c < oc; c += 8) {
            _mm256_storeu_ps(output, _mm256_loadu_ps(input));
            input += icStep;
            output += ocStep;
        }
    }
}

inline void compute_linear_nchwc8_fp32(F32 *input0,
    F32 *input1,
    F32 *output,
    F32 r0,
    F32 r1,
    U32 icStep,
    U32 ocStep,
    I32 ic,
    U32 inStep,
    U32 onStep,
    U32 on)
{
    __m256 r0_256 = _mm256_set1_ps(r0);
    __m256 r1_256 = _mm256_set1_ps(r1);
    __m256 tmp0, tmp1;
    for (U32 n = 0; n < on; ++n) {
        for (I32 c = 0; c < ic; c += 8) {
            tmp0 = _mm256_mul_ps(_mm256_loadu_ps(input0), r0_256);
            tmp1 = _mm256_mul_ps(_mm256_loadu_ps(input1), r1_256);
            _mm256_storeu_ps(output, _mm256_add_ps(tmp0, tmp1));
            input0 += icStep;
            input1 += icStep;
            output += ocStep;
        }
    }
}

inline void copy_batch_nchw_fp32(
    F32 *input, F32 *output, U32 icStep, U32 ocStep, I32 ic, U32 inStep, U32 onStep, U32 on)
{
    U32 oStep = ocStep / 8;
    U32 iStep = icStep;
    __m256i v256index = _mm256_set_epi32(
        iStep * 7, iStep * 6, iStep * 5, iStep * 4, iStep * 3, iStep * 2, iStep, 0);
    __m128i v128index = _mm_set_epi32(iStep * 3, iStep * 2, iStep, 0);
    for (U32 n = 0; n < on; ++n) {
        I32 c = 0;
        for (; c < ic - 7; c += 8) {
            _mm256_storeu_ps(
                output + c * oStep, _mm256_i32gather_ps(input + c * iStep, v256index, 4));
        }
        I32 mainc = c;
        for (; c < ic - 3; c += 4) {
            _mm_storeu_ps(output + mainc * oStep, _mm_i32gather_ps(input + c * iStep, v128index, 4));
        }
        for (; c < ic; ++c) {
            *(output + mainc * oStep + c - mainc) = *(input + c * iStep);
        }
        input += inStep;
        output += onStep;
    }
}

inline void compute_linear_nchw_fp32(F32 *input0,
    F32 *input1,
    F32 *output,
    F32 r0,
    F32 r1,
    U32 icStep,
    U32 ocStep,
    I32 ic,
    U32 inStep,
    U32 onStep,
    U32 on)
{
    U32 oStep = ocStep / 8;
    U32 iStep = icStep;
    __m256 r0_256 = _mm256_set1_ps(r0);
    __m256 r1_256 = _mm256_set1_ps(r1);
    __m256i v256index = _mm256_set_epi32(
        iStep * 7, iStep * 6, iStep * 5, iStep * 4, iStep * 3, iStep * 2, iStep, 0);
    __m128 r0_128 = _mm_set1_ps(r0);
    __m128 r1_128 = _mm_set1_ps(r1);
    __m128i v128index = _mm_set_epi32(iStep * 3, iStep * 2, iStep, 0);
    for (U32 n = 0; n < on; ++n) {
        I32 c = 0;
        for (c = 0; c < ic - 7; c += 8) {
            __m256 tmp0 =
                _mm256_mul_ps(_mm256_i32gather_ps(input0 + c * iStep, v256index, 4), r0_256);
            __m256 tmp1 =
                _mm256_mul_ps(_mm256_i32gather_ps(input1 + c * iStep, v256index, 4), r1_256);
            _mm256_storeu_ps(output + c * oStep, _mm256_add_ps(tmp0, tmp1));
        }
        I32 mainc = c;
        for (; c < ic - 3; c += 4) {
            __m128 tmp0 = _mm_mul_ps(_mm_i32gather_ps(input0 + c * iStep, v128index, 4), r0_128);
            __m128 tmp1 = _mm_mul_ps(_mm_i32gather_ps(input1 + c * iStep, v128index, 4), r1_128);
            _mm_storeu_ps(output + mainc * oStep, _mm_add_ps(tmp0, tmp1));
        }
        for (; c < ic; ++c) {
            *(output + mainc * oStep + c - mainc) =
                *(input0 + c * iStep) * r0 + *(input1 + c * iStep) * r1;
        }
        input0 += inStep;
        input1 += inStep;
        output += onStep;
    }
}

inline void compute_bilinear_nchwc8_fp32(F32 *input0,
    F32 *input1,
    F32 *input2,
    F32 *input3,
    F32 *output,
    F32 r0,
    F32 r1,
    F32 r2,
    F32 r3,
    U32 icStep,
    U32 ocStep,
    I32 oc,
    U32 inStep,
    U32 onStep,
    U32 on)
{
    __m256 r0_256 = _mm256_set1_ps(r0);
    __m256 r1_256 = _mm256_set1_ps(r1);
    __m256 r2_256 = _mm256_set1_ps(r2);
    __m256 r3_256 = _mm256_set1_ps(r3);
    __m256 tmp0, tmp1, tmp2, tmp3;
    for (U32 n = 0; n < on; ++n) {
        for (I32 c = 0; c < oc; c += 8) {
            tmp0 = _mm256_mul_ps(_mm256_loadu_ps(input0), r0_256);
            tmp1 = _mm256_mul_ps(_mm256_loadu_ps(input1), r1_256);
            tmp2 = _mm256_mul_ps(_mm256_loadu_ps(input2), r2_256);
            tmp3 = _mm256_mul_ps(_mm256_loadu_ps(input3), r3_256);
            tmp0 = _mm256_add_ps(_mm256_add_ps(tmp0, tmp1), _mm256_add_ps(tmp2, tmp3));
            _mm256_storeu_ps(output, tmp0);
            input0 += icStep;
            input1 += icStep;
            input2 += icStep;
            input3 += icStep;
            output += ocStep;
        }
    }
}

inline void compute_bilinear_nchw_fp32(F32 *input0,
    F32 *input1,
    F32 *input2,
    F32 *input3,
    F32 *output,
    F32 r0,
    F32 r1,
    F32 r2,
    F32 r3,
    U32 icStep,
    U32 ocStep,
    I32 ic,
    U32 inStep,
    U32 onStep,
    U32 on)
{
    U32 oStep = ocStep / 8;
    U32 iStep = icStep;
    __m256 r0_256 = _mm256_set1_ps(r0);
    __m256 r1_256 = _mm256_set1_ps(r1);
    __m256 r2_256 = _mm256_set1_ps(r2);
    __m256 r3_256 = _mm256_set1_ps(r3);
    __m256i v256index = _mm256_set_epi32(
        iStep * 7, iStep * 6, iStep * 5, iStep * 4, iStep * 3, iStep * 2, iStep, 0);
    __m128 r0_128 = _mm_set1_ps(r0);
    __m128 r1_128 = _mm_set1_ps(r1);
    __m128 r2_128 = _mm_set1_ps(r2);
    __m128 r3_128 = _mm_set1_ps(r3);
    __m128i v128index = _mm_set_epi32(iStep * 3, iStep * 2, iStep, 0);
    for (U32 n = 0; n < on; ++n) {
        I32 c = 0;
        for (; c < ic - 7; c += 8) {
            __m256 tmp0 =
                _mm256_mul_ps(_mm256_i32gather_ps(input0 + c * iStep, v256index, 4), r0_256);
            __m256 tmp1 =
                _mm256_mul_ps(_mm256_i32gather_ps(input1 + c * iStep, v256index, 4), r1_256);
            __m256 tmp2 =
                _mm256_mul_ps(_mm256_i32gather_ps(input2 + c * iStep, v256index, 4), r2_256);
            __m256 tmp3 =
                _mm256_mul_ps(_mm256_i32gather_ps(input3 + c * iStep, v256index, 4), r3_256);
            tmp0 = _mm256_add_ps(_mm256_add_ps(tmp0, tmp1), _mm256_add_ps(tmp2, tmp3));
            _mm256_storeu_ps(output + c * oStep, tmp0);
        }
        I32 mainc = c;
        for (; c < ic - 3; c += 4) {
            __m128 tmp0 = _mm_mul_ps(_mm_i32gather_ps(input0 + c * iStep, v128index, 4), r0_128);
            __m128 tmp1 = _mm_mul_ps(_mm_i32gather_ps(input1 + c * iStep, v128index, 4), r1_128);
            __m128 tmp2 = _mm_mul_ps(_mm_i32gather_ps(input2 + c * iStep, v128index, 4), r2_128);
            __m128 tmp3 = _mm_mul_ps(_mm_i32gather_ps(input3 + c * iStep, v128index, 4), r3_128);
            tmp0 = _mm_add_ps(_mm_add_ps(tmp0, tmp1), _mm_add_ps(tmp2, tmp3));
            _mm_storeu_ps(output + mainc * oStep, tmp0);
        }
        for (; c < ic; ++c) {
            *(output + mainc * oStep + c - mainc) = *(input0 + c * iStep) * r0 +
                *(input1 + c * iStep) * r1 + *(input2 + c * iStep) * r2 + *(input3 + c * iStep) * r3;
        }
        input0 += inStep;
        input1 += inStep;
        input2 += inStep;
        input3 += inStep;
        output += onStep;
    }
}

inline void copy_batch_nchwc16_fp32(
    F32 *input, F32 *output, U32 icStep, U32 ocStep, I32 oc, U32 inStep, U32 onStep, U32 on)
{
#ifdef _USE_AVX512_VNNI
    for (U32 n = 0; n < on; ++n) {
        I32 c = 0;
        for (; c < oc - 15; c += 16) {
            _mm512_storeu_ps(output, _mm512_loadu_ps(input));
            input += icStep;
            output += ocStep;
        }
        if ((c + 8) == oc) {
            _mm256_storeu_ps(output, _mm256_loadu_ps(input));
            input += icStep / 2;
            output += ocStep / 2;
        }
    }
#endif
}

inline void compute_linear_nchwc16_fp32(F32 *input0,
    F32 *input1,
    F32 *output,
    F32 r0,
    F32 r1,
    U32 icStep,
    U32 ocStep,
    I32 ic,
    U32 inStep,
    U32 onStep,
    U32 on)
{
#ifdef _USE_AVX512_VNNI
    __m512 r0_512 = _mm512_set1_ps(r0);
    __m512 r1_512 = _mm512_set1_ps(r1);
    __m512 tmp0, tmp1;
    __m256 r0_256 = _mm256_set1_ps(r0);
    __m256 r1_256 = _mm256_set1_ps(r1);
    for (U32 n = 0; n < on; ++n) {
        I32 c = 0;
        for (; c < ic - 15; c += 16) {
            tmp0 = _mm512_mul_ps(_mm512_loadu_ps(input0), r0_512);
            tmp1 = _mm512_mul_ps(_mm512_loadu_ps(input1), r1_512);
            _mm512_storeu_ps(output, _mm512_add_ps(tmp0, tmp1));
            input0 += icStep;
            input1 += icStep;
            output += ocStep;
        }
        if ((c + 8) == ic) {
            r0_256 = _mm256_mul_ps(_mm256_loadu_ps(input0), r0_256);
            r1_256 = _mm256_mul_ps(_mm256_loadu_ps(input1), r1_256);
            _mm256_storeu_ps(output, _mm256_add_ps(r0_256, r1_256));
            input0 += icStep / 2;
            input1 += icStep / 2;
            output += ocStep / 2;
        }
    }
#endif
}

inline void compute_bilinear_nchwc16_fp32(F32 *input0,
    F32 *input1,
    F32 *input2,
    F32 *input3,
    F32 *output,
    F32 r0,
    F32 r1,
    F32 r2,
    F32 r3,
    U32 icStep,
    U32 ocStep,
    I32 oc,
    U32 inStep,
    U32 onStep,
    U32 on)
{
#ifdef _USE_AVX512_VNNI
    __m512 r0_512 = _mm512_set1_ps(r0);
    __m512 r1_512 = _mm512_set1_ps(r1);
    __m512 r2_512 = _mm512_set1_ps(r2);
    __m512 r3_512 = _mm512_set1_ps(r3);
    __m256 r0_256 = _mm256_set1_ps(r0);
    __m256 r1_256 = _mm256_set1_ps(r1);
    __m256 r2_256 = _mm256_set1_ps(r2);
    __m256 r3_256 = _mm256_set1_ps(r3);
    __m512 tmp0, tmp1, tmp2, tmp3;
    for (U32 n = 0; n < on; ++n) {
        I32 c = 0;
        for (; c < oc - 15; c += 16) {
            tmp0 = _mm512_mul_ps(_mm512_loadu_ps(input0), r0_512);
            tmp1 = _mm512_mul_ps(_mm512_loadu_ps(input1), r1_512);
            tmp2 = _mm512_mul_ps(_mm512_loadu_ps(input2), r2_512);
            tmp3 = _mm512_mul_ps(_mm512_loadu_ps(input3), r3_512);
            tmp0 = _mm512_add_ps(_mm512_add_ps(tmp0, tmp1), _mm512_add_ps(tmp2, tmp3));
            _mm512_storeu_ps(output, tmp0);
            input0 += icStep;
            input1 += icStep;
            input2 += icStep;
            input3 += icStep;
            output += ocStep;
        }
        if ((c + 8) == oc) {
            r0_256 = _mm256_mul_ps(_mm256_loadu_ps(input0), r0_256);
            r1_256 = _mm256_mul_ps(_mm256_loadu_ps(input1), r1_256);
            r2_256 = _mm256_mul_ps(_mm256_loadu_ps(input2), r2_256);
            r3_256 = _mm256_mul_ps(_mm256_loadu_ps(input3), r3_256);
            _mm256_storeu_ps(output,
                _mm256_add_ps(_mm256_add_ps(r0_256, r1_256), _mm256_add_ps(r2_256, r3_256)));
            input0 += icStep / 2;
            input1 += icStep / 2;
            input2 += icStep / 2;
            input3 += icStep / 2;
            output += ocStep / 2;
        }
    }
#endif
}

template <CoordinateTransMode trans_mode>
EE resize_bilinear_x86_fp32_nchw(
    TensorDesc inputDesc, F32 *input, ResizeParamSpec p, F32 *tmp, TensorDesc outputDesc, F32 *output)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    CHECK_REQUIREMENT(odf == DF_NCHW || idf == DF_NCHW);
    float r0_w = iw / (float)ow;
    float r0_h = ih / (float)oh;
    float r1_w = (iw - 1.0f) / (ow - 1.0f);
    float r1_h = (ih - 1.0f) / (oh - 1.0f);

    U32 loops = on * oc * oh;

#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (U32 l = 0; l < loops; ++l) {
        U32 c = l / oh;
        U32 h = l % oh;
        F32 *outp = output + c * oh * ow;
        F32 *inp = input + c * ih * iw;
        F32 hC = coordinate_trans<trans_mode>(h, ih, oh, r0_h, r1_h);
        hC = UNI_MIN(ih - 1, UNI_MAX(0, hC));
        I32 hT = floor(hC);
        I32 hB = ceil(hC);
        F32 h1 = hB - hC;
        F32 h2 = hC - hT;

        for (U32 w = 0; w < ow; ++w) {
            F32 wC = coordinate_trans<trans_mode>(w, iw, ow, r0_w, r1_w);
            wC = UNI_MIN(iw - 1, UNI_MAX(0, wC));
            I32 wL = floor(wC);
            I32 wR = ceil(wC);
            F32 w1 = wR - wC;
            F32 w2 = wC - wL;

            U32 output_idx = h * ow + w;
            if (hB == hT && wL == wR) {
                outp[output_idx] = inp[hT * iw + wL];
            } else if (hB == hT) {
                outp[output_idx] = w1 * inp[hT * iw + wL] + w2 * inp[hT * iw + wR];
            } else if (wL == wR) {
                outp[output_idx] = h1 * inp[hT * iw + wL] + h2 * inp[hB * iw + wL];
            } else {
                outp[output_idx] = h1 * w1 * inp[hT * iw + wL] + h1 * w2 * inp[hT * iw + wR] +
                    h2 * w1 * inp[hB * iw + wL] + h2 * w2 * inp[hB * iw + wR];
            }
        }
    }
    return SUCCESS;
}

template <CoordinateTransMode trans_mode>
EE resize_bilinear_x86_fp32(
    TensorDesc inputDesc, F32 *input, ResizeParamSpec p, F32 *tmp, TensorDesc outputDesc, F32 *output)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    CHECK_REQUIREMENT(idf == DF_NCHWC8 || idf == DF_NCHW || idf == DF_NCHWC16);
    CHECK_REQUIREMENT(odf == DF_NCHWC8 || odf == DF_NCHWC16);
    float r0_w = iw / (float)ow;
    float r0_h = ih / (float)oh;
    float r1_w = (iw - 1.0f) / (ow - 1.0f);
    float r1_h = (ih - 1.0f) / (oh - 1.0f);

#ifndef _USE_AVX512_VNNI
    if (idf == DF_NCHWC16 || odf == DF_NCHWC16) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
#endif

    EE ret = SUCCESS;

    U32 icx = 8;
    U32 ocx = 8;
    if ((idf == DF_NCHWC16) && (ic > 8)) {
        icx = 16;
        ocx = 16;
    } else if (idf == DF_NCHWC8) {
        icx = 8;
    } else if (idf == DF_NCHW) {
        icx = 1;
    }

    U32 ocStep = oh * ow * ocx;
    U32 onStep = oh * ow * oc;
    U32 icStep = ih * iw * icx;
    U32 inStep = ih * iw * ic;
    copy_func copy[3] = {copy_batch_nchw_fp32, copy_batch_nchwc8_fp32, copy_batch_nchwc16_fp32};
    compute_linear_func compute_linear[3] = {
        compute_linear_nchw_fp32, compute_linear_nchwc8_fp32, compute_linear_nchwc16_fp32};
    compute_bilinear_func compute_bilinear[3] = {
        compute_bilinear_nchw_fp32, compute_bilinear_nchwc8_fp32, compute_bilinear_nchwc16_fp32};
    U32 func_idx = icx >> 3;

#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (U32 h = 0; h < oh; ++h) {
        for (U32 w = 0; w < ow; ++w) {
            F32 hC = coordinate_trans<trans_mode>(h, ih, oh, r0_h, r1_h);
            F32 wC = coordinate_trans<trans_mode>(w, iw, ow, r0_w, r1_w);
            U32 output_idx = h * ow * ocx + w * ocx;

            // process edge pixel, linear
            hC = UNI_MIN(ih - 1, UNI_MAX(0, hC));
            wC = UNI_MIN(iw - 1, UNI_MAX(0, wC));

            I32 hT = floor(hC);
            I32 hB = ceil(hC);
            I32 wL = floor(wC);
            I32 wR = ceil(wC);

            if (hB == hT && wL == wR) {
                copy[func_idx](input + (hT * iw + wL) * icx, output + output_idx, icStep, ocStep,
                    ic, inStep, onStep, on);
            } else if (hB == hT) {
                compute_linear[func_idx](input + (hT * iw + wL) * icx, input + (hT * iw + wR) * icx,
                    output + output_idx, wR - wC, wC - wL, icStep, ocStep, ic, inStep, onStep, on);
            } else if (wL == wR) {
                compute_linear[func_idx](input + (hT * iw + wL) * icx, input + (hB * iw + wL) * icx,
                    output + output_idx, hB - hC, hC - hT, icStep, ocStep, ic, inStep, onStep, on);
            } else {
                compute_bilinear[func_idx](input + (hT * iw + wL) * icx,
                    input + (hT * iw + wR) * icx, input + (hB * iw + wL) * icx,
                    input + (hB * iw + wR) * icx, output + output_idx, (hB - hC) * (wR - wC),
                    (hB - hC) * (wC - wL), (hC - hT) * (wR - wC), (hC - hT) * (wC - wL), icStep,
                    ocStep, ic, inStep, onStep, on);
            }
        }
    }

    return ret;
}

inline static EE resize_bilinear_x86_fp32_nchw_wrapper(
    TensorDesc inputDesc, F32 *input, ResizeParamSpec p, F32 *tmp, TensorDesc outputDesc, F32 *output)
{
    EE ret = SUCCESS;
    switch (p.trans_mode) {
        case COORDINATE_TRANS_HALF_PIXEL:
            ret = resize_bilinear_x86_fp32_nchw<COORDINATE_TRANS_HALF_PIXEL>(
                inputDesc, input, p, tmp, outputDesc, output);
            break;
        case COORDINATE_TRANS_ALIGN_CORNERS:
            ret = resize_bilinear_x86_fp32_nchw<COORDINATE_TRANS_ALIGN_CORNERS>(
                inputDesc, input, p, tmp, outputDesc, output);
            break;
        case COORDINATE_TRANS_PYTORCH_HALF_PIXEL:
            ret = resize_bilinear_x86_fp32_nchw<COORDINATE_TRANS_PYTORCH_HALF_PIXEL>(
                inputDesc, input, p, tmp, outputDesc, output);
            break;
        case COORDINATE_TRANS_ASYMMETRIC:
            ret = resize_bilinear_x86_fp32_nchw<COORDINATE_TRANS_ASYMMETRIC>(
                inputDesc, input, p, tmp, outputDesc, output);
            break;
        default:
            ret = NOT_SUPPORTED;
            UNI_ERROR_LOG("X86 Resize currently not support this coordinate transformation "
                          "mode.\n");
            break;
    }
    return ret;
}

inline static EE resize_bilinear_x86_fp32_wrapper(
    TensorDesc inputDesc, F32 *input, ResizeParamSpec p, F32 *tmp, TensorDesc outputDesc, F32 *output)
{
    EE ret = SUCCESS;
    switch (p.trans_mode) {
        case COORDINATE_TRANS_HALF_PIXEL:
            ret = resize_bilinear_x86_fp32<COORDINATE_TRANS_HALF_PIXEL>(
                inputDesc, input, p, tmp, outputDesc, output);
            break;
        case COORDINATE_TRANS_ALIGN_CORNERS:
            ret = resize_bilinear_x86_fp32<COORDINATE_TRANS_ALIGN_CORNERS>(
                inputDesc, input, p, tmp, outputDesc, output);
            break;
        case COORDINATE_TRANS_PYTORCH_HALF_PIXEL:
            ret = resize_bilinear_x86_fp32<COORDINATE_TRANS_PYTORCH_HALF_PIXEL>(
                inputDesc, input, p, tmp, outputDesc, output);
            break;
        case COORDINATE_TRANS_ASYMMETRIC:
            ret = resize_bilinear_x86_fp32<COORDINATE_TRANS_ASYMMETRIC>(
                inputDesc, input, p, tmp, outputDesc, output);
            break;
        default:
            ret = NOT_SUPPORTED;
            UNI_ERROR_LOG("X86 Resize currently not support this coordinate transformation "
                          "mode.\n");
            break;
    }
    return ret;
}

EE resize_bilinear_x86(TensorDesc inputDesc,
    void *input,
    ResizeParamSpec p,
    void *tmp,
    TensorDesc outputDesc,
    void *output)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    EE ret = NOT_SUPPORTED;
    switch (idt) {
        case DT_F32:
            if (idf == DF_NCHW && odf == DF_NCHW) {
                ret = resize_bilinear_x86_fp32_nchw_wrapper(
                    inputDesc, (F32 *)input, p, (F32 *)tmp, outputDesc, (F32 *)output);
            } else {
                ret = resize_bilinear_x86_fp32_wrapper(
                    inputDesc, (F32 *)input, p, (F32 *)tmp, outputDesc, (F32 *)output);
            }
        default:
            break;
    }
    return ret;
}
