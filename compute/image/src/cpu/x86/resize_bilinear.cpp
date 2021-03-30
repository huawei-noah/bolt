// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <math.h>
#include "tensor_desc.h"
#include "image.h"
#include "cpu/x86/image_x86.h"
#include "thread_affinity.h"

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

inline F32 infer_src(I32 x, I32 iw, I32 ow, ResizeCoordinateTransMode trans_mode)
{
    switch (trans_mode) {
        case HALF_PIXEL:
            return (x + 0.5f) * 1.0f * iw / ow - 0.5;
        case ALIGN_CORNERS:
        default:
            return x * 1.0f * (iw - 1) / (ow - 1);
    }
}

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
    I32 oc,
    U32 inStep,
    U32 onStep,
    U32 on)
{
    __m256 r0_256 = _mm256_set1_ps(r0);
    __m256 r1_256 = _mm256_set1_ps(r1);
    __m256 tmp0, tmp1;
    for (U32 n = 0; n < on; ++n) {
        for (I32 c = 0; c < oc; c += 8) {
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
    F32 *input, F32 *output, U32 icStep, U32 ocStep, I32 oc, U32 inStep, U32 onStep, U32 on)
{
    U32 oStep = ocStep / 8;
    U32 iStep = icStep / 8;
    __m256i v256index = _mm256_set_epi32(
        iStep * 7, iStep * 6, iStep * 5, iStep * 4, iStep * 3, iStep * 2, iStep, 0);
    __m128i v128index = _mm_set_epi32(iStep * 3, iStep * 2, iStep, 0);
    for (U32 n = 0; n < on; ++n) {
        I32 c = 0;
        for (; c < oc - 7; c += 8) {
            _mm256_storeu_ps(
                output + c * oStep, _mm256_i32gather_ps(input + c * iStep, v256index, 4));
        }
        for (; c < oc - 3; c += 4) {
            _mm_storeu_ps(output + c * oStep, _mm_i32gather_ps(input + c * iStep, v128index, 4));
        }
        for (; c < oc; ++c) {
            *(output + c * oStep) = *(input + c * iStep);
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
    I32 oc,
    U32 inStep,
    U32 onStep,
    U32 on)
{
    U32 oStep = ocStep / 8;
    U32 iStep = icStep / 8;
    __m256 r0_256 = _mm256_set1_ps(r0);
    __m256 r1_256 = _mm256_set1_ps(r1);
    __m256i v256index = _mm256_set_epi32(
        iStep * 7, iStep * 6, iStep * 5, iStep * 4, iStep * 3, iStep * 2, iStep, 0);
    __m128 r0_128 = _mm_set1_ps(r0);
    __m128 r1_128 = _mm_set1_ps(r1);
    __m128i v128index = _mm_set_epi32(iStep * 3, iStep * 2, iStep, 0);
    for (U32 n = 0; n < on; ++n) {
        I32 c = 0;
        for (c = 0; c < oc - 7; c += 8) {
            __m256 tmp0 =
                _mm256_mul_ps(_mm256_i32gather_ps(input0 + c * iStep, v256index, 4), r0_256);
            __m256 tmp1 =
                _mm256_mul_ps(_mm256_i32gather_ps(input1 + c * iStep, v256index, 4), r1_256);
            _mm256_storeu_ps(output + c * oStep, _mm256_add_ps(tmp0, tmp1));
        }
        for (; c < oc - 3; c += 4) {
            __m128 tmp0 = _mm_mul_ps(_mm_i32gather_ps(input0 + c * iStep, v128index, 4), r0_128);
            __m128 tmp1 = _mm_mul_ps(_mm_i32gather_ps(input1 + c * iStep, v128index, 4), r1_128);
            _mm_storeu_ps(output + c * oStep, _mm_add_ps(tmp0, tmp1));
        }
        for (; c < oc; ++c) {
            *(output + c * oStep) = *(input0 + c * iStep) * r0 + *(input1 + c * iStep) * r1;
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
    I32 oc,
    U32 inStep,
    U32 onStep,
    U32 on)
{
    U32 oStep = ocStep / 8;
    U32 iStep = icStep / 8;
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
        for (; c < oc - 7; c += 8) {
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
        for (; c < oc - 3; c += 4) {
            __m128 tmp0 = _mm_mul_ps(_mm_i32gather_ps(input0 + c * iStep, v128index, 4), r0_128);
            __m128 tmp1 = _mm_mul_ps(_mm_i32gather_ps(input1 + c * iStep, v128index, 4), r1_128);
            __m128 tmp2 = _mm_mul_ps(_mm_i32gather_ps(input2 + c * iStep, v128index, 4), r2_128);
            __m128 tmp3 = _mm_mul_ps(_mm_i32gather_ps(input3 + c * iStep, v128index, 4), r3_128);
            tmp0 = _mm_add_ps(_mm_add_ps(tmp0, tmp1), _mm_add_ps(tmp2, tmp3));
            _mm_storeu_ps(output + c * oStep, tmp0);
        }
        for (; c < oc; ++c) {
            *(output + c * oStep) = *(input0 + c * iStep) * r0 + *(input1 + c * iStep) * r1 +
                *(input2 + c * iStep) * r2 + *(input3 + c * iStep) * r3;
        }
        input0 += inStep;
        input1 += inStep;
        input2 += inStep;
        input3 += inStep;
        output += onStep;
    }
}

EE resize_bilinear_x86_fp32(
    TensorDesc inputDesc, F32 *input, TensorDesc outputDesc, F32 *tmp, F32 *output, ResizeParamSpec p)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    CHECK_REQUIREMENT(idf == DF_NCHWC8 || idf == DF_NCHW);
    EE ret = SUCCESS;

    U32 ocStep = oh * ow * 8;
    U32 onStep = oh * ow * oc;
    U32 icStep = ih * iw * 8;
    U32 inStep = ih * iw * ic;
    copy_func copy[2] = {copy_batch_nchw_fp32, copy_batch_nchwc8_fp32};
    compute_linear_func compute_linear[2] = {compute_linear_nchw_fp32, compute_linear_nchwc8_fp32};
    compute_bilinear_func compute_bilinear[2] = {
        compute_bilinear_nchw_fp32, compute_bilinear_nchwc8_fp32};
    U32 func_idx = (idf == DF_NCHWC8);
    U32 itile_size = (idf == DF_NCHWC8) ? 8 : 1;
    F32 *outArray = nullptr;
    if (odf == DF_NCHW) {
        outArray = output;
        output = tmp;
    }
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (U32 h = 0; h < oh; ++h) {
        for (U32 w = 0; w < ow; ++w) {
            F32 hC = infer_src(h, ih, oh, p.trans_mode);
            F32 wC = infer_src(w, iw, ow, p.trans_mode);
            U32 output_idx = h * ow * 8 + w * 8;
            if (h == 0 && w == 0) {
                copy[func_idx](input, output, icStep, ocStep, oc, inStep, onStep, on);
                continue;
            } else if (h == oh - 1 && w == ow - 1) {
                copy[func_idx](input + ((ih - 1) * iw + iw - 1) * itile_size, output + output_idx,
                    icStep, ocStep, oc, inStep, onStep, on);
                continue;
            } else if (h == 0 && w == ow - 1) {
                copy[func_idx](input + (iw - 1) * itile_size, output + output_idx, icStep, ocStep,
                    oc, inStep, onStep, on);
                continue;
            } else if (h == oh - 1 && w == 0) {
                copy[func_idx](input + (ih - 1) * iw * itile_size, output + output_idx, icStep,
                    ocStep, oc, inStep, onStep, on);
                continue;
            }

            // process edge pixel, linear
            hC = UNI_MIN(ih - 1, UNI_MAX(0, hC));
            wC = UNI_MIN(iw - 1, UNI_MAX(0, wC));

            I32 hT = floor(hC);
            I32 hB = ceil(hC);
            I32 wL = floor(wC);
            I32 wR = ceil(wC);

            if (hB == hT && wL == wR) {
                copy[func_idx](input + (hT * iw + wL) * itile_size, output + output_idx, icStep,
                    ocStep, oc, inStep, onStep, on);
            } else if (hB == hT) {
                compute_linear[func_idx](input + (hT * iw + wL) * itile_size,
                    input + (hT * iw + wR) * itile_size, output + output_idx, wR - wC, wC - wL,
                    icStep, ocStep, oc, inStep, onStep, on);
            } else if (wL == wR) {
                compute_linear[func_idx](input + (hT * iw + wL) * itile_size,
                    input + (hB * iw + wL) * itile_size, output + output_idx, hB - hC, hC - hT,
                    icStep, ocStep, oc, inStep, onStep, on);
            } else {
                compute_bilinear[func_idx](input + (hT * iw + wL) * itile_size,
                    input + (hT * iw + wR) * itile_size, input + (hB * iw + wL) * itile_size,
                    input + (hB * iw + wR) * itile_size, output + output_idx, (hB - hC) * (wR - wC),
                    (hB - hC) * (wC - wL), (hC - hT) * (wR - wC), (hC - hT) * (wC - wL), icStep,
                    ocStep, oc, inStep, onStep, on);
            }
        }
    }

    if (odf == DF_NCHW) {
        I32 ohow = oh * ow;
        __m256i v256index = _mm256_set_epi32(56, 48, 40, 32, 24, 16, 8, 0);
        for (U32 n = 0; n < on; ++n) {
            I32 c = 0;
            for (; c < (I32)oc - 7; c += 8) {
                I32 hw = 0;
                for (; hw < ohow - 7; hw += 8) {
                    F32 *src = output + n * oc * ohow + c * ohow + hw * 8;
                    F32 *dst = outArray + n * oc * ohow + c * ohow + hw;
                    _mm256_storeu_ps(dst, _mm256_i32gather_ps(src, v256index, 4));
                    _mm256_storeu_ps(dst + ohow, _mm256_i32gather_ps(src + 1, v256index, 4));
                    _mm256_storeu_ps(dst + ohow * 2, _mm256_i32gather_ps(src + 2, v256index, 4));
                    _mm256_storeu_ps(dst + ohow * 3, _mm256_i32gather_ps(src + 3, v256index, 4));
                    _mm256_storeu_ps(dst + ohow * 4, _mm256_i32gather_ps(src + 4, v256index, 4));
                    _mm256_storeu_ps(dst + ohow * 5, _mm256_i32gather_ps(src + 5, v256index, 4));
                    _mm256_storeu_ps(dst + ohow * 6, _mm256_i32gather_ps(src + 6, v256index, 4));
                    _mm256_storeu_ps(dst + ohow * 7, _mm256_i32gather_ps(src + 7, v256index, 4));
                }
                for (; hw < ohow; ++hw) {
                    for (I32 c8 = 0; c8 < 8; ++c8) {
                        outArray[n * oc * ohow + (c + c8) * ohow + hw] =
                            output[n * oc * ohow + c * ohow + hw * 8 + c8];
                    }
                }
            }
            I32 mainc = c;
            for (; c < (I32)oc - 3; c += 4) {
                for (I32 hw = 0; hw < ohow; ++hw) {
                    outArray[n * oc * ohow + c * ohow + hw] =
                        output[n * oc * ohow + mainc * ohow + hw * 4 + (c - mainc)];
                }
            }
            mainc = c;
            for (; c < (I32)oc; ++c) {
                for (I32 hw = 0; hw < ohow; ++hw) {
                    outArray[n * oc * ohow + c * ohow + hw] =
                        output[n * oc * ohow + mainc * ohow + hw * ((I32)oc - mainc) + (c - mainc)];
                }
            }
        }
    }

    return ret;
}

EE resize_bilinear_x86(
    TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *tmp, void *output, ResizeParamSpec p)
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
            ret = resize_bilinear_x86_fp32(inputDesc, (F32 *)input, outputDesc, (F32 *)tmp, (F32 *)output, p);
        default:
            break;
    }
    return ret;
}
