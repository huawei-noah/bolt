// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef CHEETAH_X86_TRANSFORM_FUNTIONS_FP32_H
#define CHEETAH_X86_TRANSFORM_FUNTIONS_FP32_H

#include "thread_affinity.h"

template <U32 C, U32 N>
inline void transformNCHWCxNx(U32 fc, U32 fh, U32 fw, U32 oc, const F32 *input, F32 *output)
{
    F32 *dest = nullptr;
    const F32 *src;
    U32 cSize = 0, cSizePadding = 0;
    U32 lstep = fc * fh * fw;
    __m256i vindex = _mm256_set_epi32(
        lstep * 7, lstep * 6, lstep * 5, lstep * 4, lstep * 3, lstep * 2, lstep, 0);
    for (U32 c = 0; c < fc; c += cSize) {
        cSize = UNI_MIN(fc - c, C);
        cSizePadding = UNI_MIN(oc - c, C);
        for (U32 hw = 0; hw < fh * fw; ++hw) {
            for (U32 c8 = 0; c8 < cSize; ++c8) {
                src = input + (c + c8) * fh * fw + hw;
                dest = output + c * fh * fw * N + hw * cSizePadding * N + c8 * N;
                if (N >= 8) {
                    _mm256_storeu_ps(dest, _mm256_i32gather_ps(src, vindex, 4));
                }
                if (N >= 16) {
                    _mm256_storeu_ps(dest + 8, _mm256_i32gather_ps(src + 8 * lstep, vindex, 4));
                }
                if (N >= 24) {
                    _mm256_storeu_ps(dest + 16, _mm256_i32gather_ps(src + 16 * lstep, vindex, 4));
                }
                if (N == 32) {
                    _mm256_storeu_ps(dest + 24, _mm256_i32gather_ps(src + 24 * lstep, vindex, 4));
                }
            }
            UNI_MEMSET(dest + N, 0, ((cSizePadding - cSize) * N * 4));
        }
    }
}

// N is 32/24
template <U32 C, U32 N>
inline EE transformNCHWToNCHWCxNx(
    TensorDesc inputDesc, const F32 *input, TensorDesc outputDesc, F32 *output)
{
    if (input == NULL || output == NULL) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt, odt;
    DataFormat fdf, odf;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    U32 remain = fn % N;
    fn -= remain;

    for (U32 n = 0; n < fn; n += N) {
        transformNCHWCxNx<C, N>(fc, fh, fw, oc, input, output);
        input += fc * fh * fw * N;
        output += oc * fh * fw * N;
    }
    if (remain >= 24) {
        transformNCHWCxNx<C, 24>(fc, fh, fw, oc, input, output);
        input += fc * fh * fw * 24;
        output += oc * fh * fw * 24;
        remain -= 24;
    }
    if (remain >= 16) {
        transformNCHWCxNx<C, 16>(fc, fh, fw, oc, input, output);
        input += fc * fh * fw * 16;
        output += oc * fh * fw * 16;
        remain -= 16;
    }
    if (remain >= 8) {
        transformNCHWCxNx<C, 8>(fc, fh, fw, oc, input, output);
        input += fc * fh * fw * 8;
        output += oc * fh * fw * 8;
        remain -= 8;
    }
    if (remain > 0) {
        F32 *dest = NULL;
        U32 cSize = 0, cSizePadding = 0;
        F32 m[8] = {0.0f};
        for (U32 i = 0; i < remain; ++i) {
            m[i] = -1.0f;
        }
        __m256 mask = _mm256_set_ps(m[7], m[6], m[5], m[4], m[3], m[2], m[1], m[0]);
        U32 lstep = fc * fh * fw;
        __m256i vindex = _mm256_set_epi32(
            lstep * 7, lstep * 6, lstep * 5, lstep * 4, lstep * 3, lstep * 2, lstep, 0);
        __m256 src256 = _mm256_setzero_ps();
        for (U32 c = 0; c < fc; c += cSize) {
            cSize = UNI_MIN(fc - c, C);
            cSizePadding = UNI_MIN(oc - c, C);
            for (U32 hw = 0; hw < fh * fw; ++hw) {
                for (U32 c8 = 0; c8 < cSize; ++c8) {
                    const F32 *src = input + (c + c8) * fh * fw + hw;
                    dest = output + c * fh * fw * 8 + hw * cSizePadding * 8 + c8 * 8;
                    _mm256_storeu_ps(dest, _mm256_mask_i32gather_ps(src256, src, vindex, mask, 4));
                }
                UNI_MEMSET(dest + 8, 0, ((cSizePadding - cSize) * 32));
            }
        }
        fn += remain;
    }
    return SUCCESS;
}

inline void PaddingNCHWC8(
    F32 *data, F32 *tmp, TensorDesc inputDesc, ConvolutionParamSpec convParamSpec)
{
    // NCHWC8
    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingB = convParamSpec.pad_bottom;
    U32 paddingL = convParamSpec.pad_left;
    U32 paddingR = convParamSpec.pad_right;

    U32 padih = paddingT + paddingB + ih;
    U32 padiw = paddingL + paddingR + iw;

    CHECK_REQUIREMENT((idf == DF_NCHWC8) && (ic % 8 == 0));
    ic /= 8;
#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
#endif
    {
#ifdef _USE_OPENMP
#pragma omp for schedule(static) nowait
#endif
        for (U32 c = 0; c < ic; ++c) {
            U32 coff = c * padih * padiw * 8;
            UNI_MEMSET(tmp + coff, 0, padiw * paddingT * 8 * bytesOf(idt));
            UNI_MEMSET(
                tmp + coff + (ih + paddingT) * padiw * 8, 0, padiw * paddingB * 8 * bytesOf(idt));
        }

#ifdef _USE_OPENMP
#pragma omp for schedule(static) nowait
#endif
        for (U32 hc = 0; hc < ih * ic; ++hc) {
            U32 c = hc / ih;
            U32 coff = c * padih * padiw * 8;
            U32 h = hc % ih;
            U32 hoff = (h + paddingT) * padiw;

            UNI_MEMSET(tmp + coff + hoff * 8, 0, paddingL * 8 * bytesOf(idt));
            UNI_MEMCPY(tmp + coff + (hoff + paddingL) * 8, data + c * ih * iw * 8 + h * iw * 8,
                iw * 8 * bytesOf(idt));
            UNI_MEMSET(tmp + coff + (hoff + (paddingL + iw)) * 8, 0, paddingR * 8 * bytesOf(idt));
        }
    }
}

inline void deconvOverlapAndCropF32(F32 *input,
    F32 *output,
    TensorDesc inputDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 fh = convParamSpec.kernel_h;
    U32 fw = convParamSpec.kernel_w;
    U32 fhfw = fh * fw;
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingL = convParamSpec.pad_left;
    __m256i vindex =
        _mm256_set_epi32(fhfw * 7, fhfw * 6, fhfw * 5, fhfw * 4, fhfw * 3, fhfw * 2, fhfw, 0);
    for (U32 kn = 0; kn < in; ++kn) {
        for (U32 kh = 0; kh < ih; ++kh) {
            for (U32 kw = 0; kw < iw; ++kw) {
                for (U32 kc = 0; kc < oc; kc += 8) {
                    for (U32 jh = 0; jh < fh; ++jh) {
                        for (U32 jw = 0; jw < fw; ++jw) {
                            U32 ohIdx = kh * strideH + jh;
                            U32 owIdx = kw * strideW + jw;
                            if ((ohIdx < paddingT) || (ohIdx >= oh + paddingT) ||
                                (owIdx < paddingL) || (owIdx >= ow + paddingL)) {
                                continue;
                            }
                            ohIdx -= paddingT;
                            owIdx -= paddingL;
                            U32 oidx = (kc * oh + ohIdx * 8) * ow + owIdx * 8;
                            U32 iidx = ((kh * iw + kw) * oc + kc) * fhfw + jh * fw + jw;
                            __m256 x = _mm256_i32gather_ps(input + iidx, vindex, 4);
                            x = _mm256_add_ps(x, _mm256_loadu_ps(output + oidx));
                            _mm256_storeu_ps(output + oidx, x);
                        }
                    }
                }
            }
        }
        input += oc * fh * fw * ih * iw;
        output += oc * oh * ow;
    }
}

inline void deconvOverlapAndCropI32(I32 *input,
    I32 *output,
    TensorDesc inputDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 fh = convParamSpec.kernel_h;
    U32 fw = convParamSpec.kernel_w;
    U32 fhfw = fh * fw;
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingL = convParamSpec.pad_left;
    __m256i vindex =
        _mm256_set_epi32(fhfw * 7, fhfw * 6, fhfw * 5, fhfw * 4, fhfw * 3, fhfw * 2, fhfw, 0);
    U32 cx = 8;
#ifdef _USE_AVX512_VNNI
    if (outputDesc.df == DF_NCHWC16) {
        cx = 16;
    }
    __m512i vindex512 =
        _mm512_set_epi32(fhfw * 15, fhfw * 14, fhfw * 13, fhfw * 12, fhfw * 11, fhfw * 10, fhfw * 9, fhfw * 8, 
                         fhfw * 7, fhfw * 6, fhfw * 5, fhfw * 4, fhfw * 3, fhfw * 2, fhfw, 0);
#endif
    for (U32 kn = 0; kn < in; ++kn) {
        for (U32 kh = 0; kh < ih; ++kh) {
            for (U32 kw = 0; kw < iw; ++kw) {
                for (U32 kc = 0; kc < oc; kc += cx) {
                    for (U32 jh = 0; jh < fh; ++jh) {
                        for (U32 jw = 0; jw < fw; ++jw) {
                            U32 ohIdx = kh * strideH + jh;
                            U32 owIdx = kw * strideW + jw;
                            if ((ohIdx < paddingT) || (ohIdx >= oh + paddingT) ||
                                (owIdx < paddingL) || (owIdx >= ow + paddingL)) {
                                continue;
                            }
                            ohIdx -= paddingT;
                            owIdx -= paddingL;
                            U32 oidx = (kc * oh + ohIdx * cx) * ow + owIdx * cx;
                            U32 iidx = ((kh * iw + kw) * oc + kc) * fhfw + jh * fw + jw;
                            if (cx == 8) {
                                __m256i x = _mm256_i32gather_epi32(input + iidx, vindex, 4);
                                x = _mm256_add_epi32(x, _mm256_loadu_si256((const __m256i *)(output + oidx)));
                                _mm256_storeu_si256((__m256i *)(output + oidx), x);
#ifdef _USE_AVX512_VNNI
                            } else if (cx == 16) {
                                __m512i x = _mm512_i32gather_epi32(vindex512, input + iidx, 4);
                                x = _mm512_add_epi32(x, _mm512_loadu_si512(output + oidx));
                                _mm512_storeu_si512(output + oidx, x);
#endif
                            }
                        }
                    }
                }
            }
        }
        input += oc * fh * fw * ih * iw;
        output += oc * oh * ow;
    }
}

inline void deconvOverlapAndCropNCHWC8F32(F32 *input,
    F32 *output,
    TensorDesc inputDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 fh = convParamSpec.kernel_h;
    U32 fw = convParamSpec.kernel_w;
    U32 fhfw = fh * fw;
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingL = convParamSpec.pad_left;
    U32 cx = 8;
#ifdef _USE_AVX512_VNNI
    if (outputDesc.df == DF_NCHWC16) {
        cx = 16;
    }
#endif
    for (U32 kn = 0; kn < in; ++kn) {
        for (U32 kh = 0; kh < ih; ++kh) {
            for (U32 kw = 0; kw < iw; ++kw) {
                for (U32 kc = 0; kc < oc; kc += cx) {
                    for (U32 jh = 0; jh < fh; ++jh) {
                        for (U32 jw = 0; jw < fw; ++jw) {
                            U32 ohIdx = kh * strideH + jh;
                            U32 owIdx = kw * strideW + jw;
                            if ((ohIdx < paddingT) || (ohIdx >= oh + paddingT) ||
                                (owIdx < paddingL) || (owIdx >= ow + paddingL)) {
                                continue;
                            }
                            ohIdx -= paddingT;
                            owIdx -= paddingL;
                            U32 oidx = (kc * oh + ohIdx * cx) * ow + owIdx * cx;
                            U32 iidx = ((jh * fw + jw) * oc + kc) * ih * iw + kh * iw * cx + kw * cx;
                            if (cx == 8) {
                                _mm256_storeu_ps(output + oidx,
                                    _mm256_add_ps(
                                        _mm256_loadu_ps(input + iidx), _mm256_loadu_ps(output + oidx)));
#ifdef _USE_AVX512_VNNI
                            } else if (cx == 16) {
                                _mm512_storeu_ps(output + oidx,
                                    _mm512_add_ps(
                                        _mm512_loadu_ps(input + iidx), _mm512_loadu_ps(output + oidx)));
#endif
                            }
                        }
                    }
                }
            }
        }
        input += oc * fh * fw * ih * iw;
        output += oc * oh * ow;
    }
}

inline void deconvOverlapAndCropNCHWC8I32(I32 *input,
    I32 *output,
    TensorDesc inputDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 fh = convParamSpec.kernel_h;
    U32 fw = convParamSpec.kernel_w;
    U32 fhfw = fh * fw;
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingL = convParamSpec.pad_left;
    U32 cx = 8;
#ifdef _USE_AVX512_VNNI
    if (outputDesc.df == DF_NCHWC16) {
        cx = 16;
    }
#endif
    for (U32 kn = 0; kn < in; ++kn) {
        for (U32 kh = 0; kh < ih; ++kh) {
            for (U32 kw = 0; kw < iw; ++kw) {
                for (U32 kc = 0; kc < oc; kc += cx) {
                    for (U32 jh = 0; jh < fh; ++jh) {
                        for (U32 jw = 0; jw < fw; ++jw) {
                            U32 ohIdx = kh * strideH + jh;
                            U32 owIdx = kw * strideW + jw;
                            if ((ohIdx < paddingT) || (ohIdx >= oh + paddingT) ||
                                (owIdx < paddingL) || (owIdx >= ow + paddingL)) {
                                continue;
                            }
                            ohIdx -= paddingT;
                            owIdx -= paddingL;
                            U32 oidx = (kc * oh + ohIdx * cx) * ow + owIdx * cx;
                            U32 iidx = ((jh * fw + jw) * oc + kc) * ih * iw + kh * iw * cx + kw * cx;
                            if (cx == 8) {
                                _mm256_storeu_si256((__m256i *)(output + oidx),
                                    _mm256_add_epi32(
                                        _mm256_loadu_si256((const __m256i *)(input + iidx)),
                                        _mm256_loadu_si256((const __m256i *)(output + oidx))));
#ifdef _USE_AVX512_VNNI
                            } else if (cx == 16) {
                                _mm512_storeu_si512(output + oidx,
                                    _mm512_add_epi32(
                                        _mm512_loadu_si512(input + iidx),
                                        _mm512_loadu_si512(output + oidx)));
#endif
                            }
                        }
                    }
                }
            }
        }
        input += oc * fh * fw * ih * iw;
        output += oc * oh * ow;
    }
}

template<typename T>
inline void deconvOverlapAndCropEqualNCHWC8(T *input,
    T *output,
    TensorDesc inputDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 fh = convParamSpec.kernel_h;
    U32 fw = convParamSpec.kernel_w;
    U32 fhfw = fh * fw;
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingL = convParamSpec.pad_left;
    U32 cx = 8;
    if (outputDesc.df == DF_NCHWC16) {
        cx = 16;
    }
    U32 tileSize = cx * bytesOf(odt);
#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS) if ((ih > OMP_NUM_THREADS) && (iw * fh * fw > 256))
#endif
    {
    for (U32 kn = 0; kn < in; ++kn) {
        for (U32 kc = 0; kc < oc; kc += cx) {
#ifdef _USE_OPENMP
#pragma omp for schedule(static)
#endif
            for (U32 jh = 0; jh < fh; ++jh) {
                for (U32 jw = 0; jw < fw; ++jw) {
                    for (U32 kh = 0; kh < ih; ++kh) {
                        for (U32 kw = 0; kw < iw; ++kw) {
                            U32 ohIdx = kh * strideH + jh;
                            U32 owIdx = kw * strideW + jw;
                            if ((ohIdx < paddingT) || (ohIdx >= oh + paddingT) ||
                                (owIdx < paddingL) || (owIdx >= ow + paddingL)) {
                                continue;
                            }
                            ohIdx -= paddingT;
                            owIdx -= paddingL;
                            U32 oidx = (kc * oh + ohIdx * cx) * ow + owIdx * cx;
                            U32 iidx = ((jh * fw + jw) * oc + kc) * ih * iw + kh * iw * cx + kw * cx;
                            UNI_MEMCPY(output + oidx, input + iidx, tileSize);
                        }
                    }
                }
            }
        }
        input += oc * fh * fw * ih * iw;
        output += oc * oh * ow;
    }
    }
}

#endif
