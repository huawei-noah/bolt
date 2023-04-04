// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cstring>
#include "cpu/x86/fp32/tensor_computing_fp32.h"
#include "cpu/x86/fp32/transform_functions_fp32.h"
#include "cpu/x86/fp32/convolution_functions.h"

EE convolution_winograd_transform_filter_fp32(TensorDesc filterDesc,
    const F32 *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    F32 *filterTransformed);

// N is 32/24
template <U32 N>
inline EE transformNCHWToNCHWCxNxWrapper(
    TensorDesc filterDesc, const F32 *filterArray, TensorDesc ftmDesc, F32 *ftmArray, U32 cx)
{
    EE ret = NOT_SUPPORTED;
    switch (cx) {
        case 128:
            ret = transformNCHWToNCHWCxNx<128, N>(filterDesc, filterArray, ftmDesc, ftmArray);
            break;
        case 8:
            ret = transformNCHWToNCHWCxNx<8, N>(filterDesc, filterArray, ftmDesc, ftmArray);
            break;
        case 1:
            ret = transformNCHWToNCHWCxNx<1, N>(filterDesc, filterArray, ftmDesc, ftmArray);
            break;
        default:
            break;
    }
    return ret;
}

inline EE convolution_transform_filter_kernel_fp32(TensorDesc filterDesc,
    const F32 *filterArray,
    TensorDesc *ftmDesc,
    F32 *ftmArray,
    DataFormat ftmDataFormat,
    U32 cx,
    bool paddingC)
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
        UNI_MEMCPY(ftmArray, filterArray, fn * fc * fh * fw * bytesOf(fdt));
        return SUCCESS;
    }
    if (fdf != DF_NCHW) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (paddingC) {
        fc = CeilDivide(fc, 8) * 8;
    }
    EE ret = SUCCESS;
    switch (ftmDataFormat) {
        case DF_NCHWCxN32: {
            /*
         *  NCHW => NCHWCxN32
         */
            *ftmDesc = tensor4df(fdt, ftmDataFormat, fn, fc, fh, fw);
            transformNCHWToNCHWCxNxWrapper<32>(filterDesc, filterArray, *ftmDesc, ftmArray, cx);
            break;
        }
        case DF_NCHWCxN24: {
            *ftmDesc = tensor4df(fdt, ftmDataFormat, fn, fc, fh, fw);
            transformNCHWToNCHWCxNxWrapper<24>(filterDesc, filterArray, *ftmDesc, ftmArray, cx);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE convolution_transform_filter_fp32(TensorDesc filterDesc,
    const F32 *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    F32 *filterTransformed)
{
    DataFormat ftmDataFormat;
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, ft, fh, fw;
    TensorDesc originalDesc = filterDesc;
    if (tensorIs4d(filterDesc)) {
        CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
        ft = 1;
    } else if (tensorIs5d(filterDesc)) {
        U32 pb = convParamSpec.pad_before;
        U32 pa = convParamSpec.pad_after;
        U32 st = convParamSpec.stride_t;
        U32 dt = convParamSpec.dilatedRate_t;
        CHECK_STATUS(tensor5dGet(filterDesc, &fdt, &fdf, &fn, &fc, &ft, &fh, &fw));
        if ((ft != 1 && fh != 1 && fw != 1) ||
            (pb != 0) ||
            (pa != 0) ||
            (st > 1) ||
            (dt > 1))
        {
            return NOT_SUPPORTED;
        }
        if (ft == 1) {
            fh *= ft;
        } else {
            fw *= fh;
            fh = ft;
        }
        filterDesc = tensor4df(fdt, fdf, fn, fc, fh, fw);
    } else {
        return NOT_SUPPORTED;
    }

    EE ret = NOT_SUPPORTED;
    bool paddingC = false;
    if (algorithm == CONVOLUTION_ALGORITHM_WINOGRAD) {
        ret = convolution_winograd_transform_filter_fp32(
            filterDesc, filter, convParamSpec, algorithm, ftmDesc, filterTransformed);
    } else {
        U32 cx = 0;
        U32 fnBlock = 0;
        fn = CeilDivide(fn, 8) * 8 / convParamSpec.group;
        switch (algorithm) {
            case CONVOLUTION_ALGORITHM_DIRECT: {
                fnBlock = InferConvDirectUnrollOc(fn);
                cx = 8;
                break;
            }
            case CONVOLUTION_ALGORITHM_POINTWISE: {
                fnBlock = InferConvPointwiseUnrollOc(fn);
                cx = 128;
                paddingC = true;
                break;
            }
            case CONVOLUTION_ALGORITHM_GEMM_ICNCHW: {
                fnBlock = InferConvPointwiseUnrollOc(fn);
                cx = 1;
                break;
            }
            default:
                return NOT_MATCH;
        }
        CHECK_STATUS(InferConvWeightFormat(ftmDataFormat, fnBlock));

        U32 channelAxis = filterDesc.nDims - 1;
        TensorDesc tmpFilterDesc = filterDesc;
        tmpFilterDesc.dims[channelAxis] /= convParamSpec.group;
        U32 fnPadding = CeilDivide(tmpFilterDesc.dims[channelAxis], 8) * 8;
        U32 originalTileSize = tensorNumElements(tmpFilterDesc);
        for (U32 g = 0; g < convParamSpec.group; g++) {
            CHECK_STATUS(convolution_transform_filter_kernel_fp32(
                tmpFilterDesc, filter, ftmDesc, filterTransformed, ftmDataFormat, cx, paddingC));
            U32 newTileSize = tensorNumElements(*ftmDesc) / tmpFilterDesc.dims[channelAxis] * fnPadding;
            filter += originalTileSize;
            filterTransformed += newTileSize;
        }
        ftmDesc->dims[channelAxis] = filterDesc.dims[channelAxis];
        ret = SUCCESS;
    }

    if (tensorIs5d(originalDesc)) {
        originalDesc.df = ftmDesc->df;
        *ftmDesc = originalDesc;
    }

    return ret;
}

void transformWeight4x4_3x3(
    const F32 *input, F32 *output, F32 *tmp, U32 blockIc, TensorDesc filterDesc)
{
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));

    __m256 v01666 = _mm256_set1_ps(0.1666666666666667f);
    __m256 minusV01666 = _mm256_set1_ps(-0.1666666666666667f);
    __m256 v00833 = _mm256_set1_ps(0.0833333333333333f);
    __m256 minusV00833 = _mm256_set1_ps(-0.0833333333333333f);
    __m256 v004166 = _mm256_set1_ps(0.0416666666666667f);
    __m256 v025 = _mm256_set1_ps(0.25f);

    // U32 fn32 = fn / 32;
    U32 fnBlocks[] = {8, 16, 24, 32};
    U32 lstep = fc * fh * fw;
    __m256i vindex = _mm256_set_epi32(
        lstep * 7, lstep * 6, lstep * 5, lstep * 4, lstep * 3, lstep * 2, lstep, 0);

    U32 cx = 0;
    for (U32 c = 0; c < fc; c += cx) {
        cx = UNI_MIN(blockIc, fc - c);
        U32 nSize = 0;
        for (U32 n = 0; n < fn; n += nSize) {
            nSize = UNI_MIN(32, fn - n);
            nSize = fnBlocks[(nSize >> 3) - 1];
            F32 *curO = output + (c * fn + n * cx) * 36;
            for (U32 cb = 0; cb < cx; ++cb) {
                for (U32 ni = 0; ni < (nSize / 8); ++ni) {
                    const F32 *curI = input + (n + ni * 8) * lstep + (c + cb) * fh * fw;
                    for (U32 i = 0; i < 3; ++i) {
                        __m256 xi0 = _mm256_i32gather_ps(curI + i, vindex, 4);
                        __m256 xi1 = _mm256_i32gather_ps(curI + 3 + i, vindex, 4);
                        __m256 xi2 = _mm256_i32gather_ps(curI + 3 * 2 + i, vindex, 4);

                        __m256 t0 = _mm256_mul_ps(v01666, xi2);
                        __m256 t1 = _mm256_sub_ps(_mm256_mul_ps(minusV01666, xi0), t0);
                        __m256 t2 = _mm256_fmadd_ps(v004166, xi0, t0);

                        __m256 o0 = _mm256_mul_ps(v025, xi0);
                        __m256 o1 = _mm256_fmadd_ps(xi1, minusV01666, t1);
                        __m256 o2 = _mm256_fmadd_ps(xi1, v01666, t1);
                        __m256 o3 = _mm256_fmadd_ps(xi1, v00833, t2);
                        __m256 o4 = _mm256_fmadd_ps(xi1, minusV00833, t2);

                        _mm256_storeu_ps(tmp + (i)*8, o0);
                        _mm256_storeu_ps(tmp + (3 + i) * 8, o1);
                        _mm256_storeu_ps(tmp + (3 * 2 + i) * 8, o2);
                        _mm256_storeu_ps(tmp + (3 * 3 + i) * 8, o3);
                        _mm256_storeu_ps(tmp + (3 * 4 + i) * 8, o4);
                        _mm256_storeu_ps(tmp + (3 * 5 + i) * 8, xi2);
                    }
                    for (U32 i = 0; i < 6; ++i) {
                        __m256 xi0 = _mm256_loadu_ps(tmp + (3 * i) * 8);
                        __m256 xi1 = _mm256_loadu_ps(tmp + (3 * i + 1) * 8);
                        __m256 xi2 = _mm256_loadu_ps(tmp + (3 * i + 2) * 8);

                        __m256 t0 = _mm256_mul_ps(v01666, xi2);
                        __m256 t1 = _mm256_sub_ps(_mm256_mul_ps(minusV01666, xi0), t0);
                        __m256 t2 = _mm256_fmadd_ps(v004166, xi0, t0);

                        __m256 o0 = _mm256_mul_ps(v025, xi0);
                        __m256 o1 = _mm256_fmadd_ps(xi1, minusV01666, t1);
                        __m256 o2 = _mm256_fmadd_ps(xi1, v01666, t1);
                        __m256 o3 = _mm256_fmadd_ps(xi1, v00833, t2);
                        __m256 o4 = _mm256_fmadd_ps(xi1, minusV00833, t2);

                        _mm256_storeu_ps(curO + (6 * i) * nSize * cx + cb * nSize + ni * 8, o0);
                        _mm256_storeu_ps(curO + (6 * i + 1) * nSize * cx + cb * nSize + ni * 8, o1);
                        _mm256_storeu_ps(curO + (6 * i + 2) * nSize * cx + cb * nSize + ni * 8, o2);
                        _mm256_storeu_ps(curO + (6 * i + 3) * nSize * cx + cb * nSize + ni * 8, o3);
                        _mm256_storeu_ps(curO + (6 * i + 4) * nSize * cx + cb * nSize + ni * 8, o4);
                        _mm256_storeu_ps(curO + (6 * i + 5) * nSize * cx + cb * nSize + ni * 8, xi2);
                    }
                }
            }
        }
    }
}

EE convolution_winograd_transform_filter_fp32(TensorDesc filterDesc,
    const F32 *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    F32 *filterTransformed)
{
    // F(4x4, 3x3)
    if (nullptr == filter || nullptr == ftmDesc || nullptr == filterTransformed) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    if (fdf != DF_NCHW) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    U32 blockIc = UNI_MIN(32, fc);
    F32 *tmp = filterTransformed + fn * fc * 36;
    transformWeight4x4_3x3(filter, filterTransformed, tmp, blockIc, filterDesc);
    *ftmDesc = tensor4df(fdt, DF_NCHWCxN32, fn, fc, fh, fw);

    return SUCCESS;
}
