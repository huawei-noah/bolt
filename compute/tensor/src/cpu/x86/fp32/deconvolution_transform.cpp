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

template <U32 C, U32 N>
inline void transformCNHW2NCHWCxNxKernel(
    U32 fc, U32 fn, U32 fh, U32 fw, U32 fnPadding, const F32 *input, F32 *output)
{
    F32 *dest;
    const F32 *src;
    U32 cSize = 0, cSizePadding = 0;
    U32 lstep = fh * fw;
    U32 hwMax = fh * fw - 1;
    __m256i vindex = _mm256_set_epi32(
        lstep * 7, lstep * 6, lstep * 5, lstep * 4, lstep * 3, lstep * 2, lstep, 0);
    for (U32 n = 0; n < fn; n += cSize) {
        cSize = UNI_MIN(fn - n, C);
        cSizePadding = UNI_MIN(fnPadding - n, C);
        for (U32 hw = 0; hw < fh * fw; ++hw) {
            for (U32 c8 = 0; c8 < cSize; ++c8) {
                src = input + (n + c8) * fc * fh * fw + hwMax - hw;
                dest = output + n * fh * fw * N + hw * cSizePadding * N + c8 * N;
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
            memset(dest + N, 0, ((cSizePadding - cSize) * N * 4));
        }
    }
}

// N is 32/24
template <U32 C, U32 N>
inline EE transformCNHW2NCHWCxNx(
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

    U32 tail = fc % N;
    U32 remain = fc - tail;

    for (U32 c = 0; c < remain; c += N) {
        transformCNHW2NCHWCxNxKernel<C, N>(fc, fn, fh, fw, on, input, output);
        input += fh * fw * N;
        output += on * fh * fw * N;
    }
    if (tail >= 16) {
        transformCNHW2NCHWCxNxKernel<C, 16>(fc, fn, fh, fw, on, input, output);
        input += fh * fw * 16;
        output += on * fh * fw * 16;
        tail -= 16;
    }
    if (tail >= 8) {
        transformCNHW2NCHWCxNxKernel<C, 8>(fc, fn, fh, fw, on, input, output);
        input += fh * fw * 8;
        output += on * fh * fw * 8;
        tail -= 8;
    }
    if (tail > 0) {
        F32 *dest;
        const F32 *src;
        U32 cSize = 0, cSizePadding = 0;
        U32 hwMax = fh * fw - 1;
        F32 m[8] = {0.0f};
        for (U32 i = 0; i < tail; ++i) {
            m[i] = -1.0f;
        }
        __m256 mask = _mm256_set_ps(m[7], m[6], m[5], m[4], m[3], m[2], m[1], m[0]);
        U32 lstep = fh * fw;
        __m256i vindex = _mm256_set_epi32(
            lstep * 7, lstep * 6, lstep * 5, lstep * 4, lstep * 3, lstep * 2, lstep, 0);
        __m256 src256 = _mm256_setzero_ps();

        for (U32 n = 0; n < fn; n += cSize) {
            cSize = UNI_MIN(fn - n, C);
            cSizePadding = UNI_MIN(on - n, C);
            for (U32 hw = 0; hw < fh * fw; ++hw) {
                for (U32 c8 = 0; c8 < cSize; ++c8) {
                    src = input + (n + c8) * fc * fh * fw + hwMax - hw;
                    dest = output + n * fh * fw * 8 + hw * cSizePadding * 8 + c8 * 8;
                    _mm256_storeu_ps(dest, _mm256_mask_i32gather_ps(src256, src, vindex, mask, 4));
                }
                memset(dest + 8, 0, ((cSizePadding - cSize) * 32));
            }
        }
    }
    return SUCCESS;
}

inline EE deconvolution_transform_filter_kernel_fp32(TensorDesc filterDesc,
    const F32 *filterArray,
    TensorDesc *ftmDesc,
    F32 *ftmArray,
    DataFormat ftmDataFormat)
{
    if (nullptr == filterArray || nullptr == ftmDesc || nullptr == ftmArray) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    if (fdf == ftmDataFormat) {
        *ftmDesc = filterDesc;
        memcpy(ftmArray, filterArray, fn * fc * fh * fw * bytesOf(fdt));
        return SUCCESS;
    }
    if (fdf != DF_NCHW) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    EE ret = SUCCESS;
    switch (ftmDataFormat) {
        case DF_NCHWC24: {
            filterDesc = tensor4df(fdt, fdf, 1, fc, fh, fw);
            *ftmDesc = tensor4df(fdt, ftmDataFormat, 1, fc, fh, fw);
            transformCNHW2NCHWCxNx<1, 24>(filterDesc, filterArray, *ftmDesc, ftmArray);
            *ftmDesc = tensor4df(fdt, ftmDataFormat, fn, fc, fh, fw);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE deconvolution_transform_filter_fp32(TensorDesc filterDesc,
    const F32 *filter,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    F32 *filterTransformed)
{
    DataFormat ftmDataFormat;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_GROUP_DECONV:
            ftmDataFormat = DF_NCHWC24;
            break;
        default:
            return NOT_MATCH;
    }
    EE ret = deconvolution_transform_filter_kernel_fp32(
        filterDesc, filter, ftmDesc, filterTransformed, ftmDataFormat);
    CHECK_STATUS(ret);
    return ret;
}
