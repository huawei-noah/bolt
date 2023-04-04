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
#include "uni.h"
#include "cpu/x86/int8/tensor_computing_int8.h"

inline EE transformNCHWToNCHWCxN48Cx(
    TensorDesc filterDesc, const INT8 *filterArray, TensorDesc ftmDesc, INT8 *ftmArray, U32 cx)
{
    if (filterArray == NULL || ftmArray == NULL) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    U32 fhfw = fh * fw;
    U32 count = 0;
    U32 nx = 48;
    if (fn % 32 == 0 && fn % nx != 0) {
        nx = 32;
    }
    U32 remain = fn % nx;

    I32 *offsetC = (I32 *)ftmArray;
    ftmArray += fn * bytesOf(DT_I32);
    for (U32 n = 0; n < fn; ++n) {
        I32 sum = 0;
        for (U32 i = 0; i < fc * fhfw; ++i) {
            sum += filterArray[i + n * fc * fhfw];
        }
        offsetC[n] = -128 * sum;
    }

    U32 fcPadding = UNI_ALIGN(fc, cx);

    U32 nxArray[4] = {8, 16, 32, 48};
    for (; count < fn; count += nx) {
        nx = UNI_MIN(fn - count, nx);
        nx = nxArray[nx >> 4];
        U32 c = 0;
        U32 realCx = cx;
        for (; c < fc / realCx; ++c) {
            for (U32 hw = 0; hw < fhfw; ++hw) {
                for (U32 c2 = 0; c2 < realCx; c2 += 4) {
                    for (U32 n = 0; n < nx; ++n) {
                        for (U32 c4 = 0; c4 < 4; ++c4) {
                            U32 iIdx = (n + count) * fc * fhfw + (c4 + c2 + c * realCx) * fhfw + hw;
                            U32 oIdx = count * fcPadding * fhfw + c * realCx * nx * fhfw +
                                hw * nx * realCx + c2 * nx + n * 4 + c4;
                            ftmArray[oIdx] = filterArray[iIdx];
                        }
                    }
                }
            }
        }
        c *= realCx;
        U32 resC = fc - c;
        if (resC > 0) {
            for (U32 hw = 0; hw < fhfw; ++hw) {
                U32 c2 = 0;
                for (; c2 < resC; c2 += 4) {
                    for (U32 n = 0; n < nx; ++n) {
                        for (U32 c4 = 0; c4 < 4; ++c4) {
                            U32 iIdx = (n + count) * fc * fhfw + (c4 + c2 + c) * fhfw + hw;
                            U32 oIdx = count * fcPadding * fhfw + c * nx * fhfw + hw * nx * realCx +
                                c2 * nx + n * 4 + c4;
                            if (c2 + c4 < resC) {
                                ftmArray[oIdx] = filterArray[iIdx];
                            } else {
                                ftmArray[oIdx] = 0;
                            }
                        }
                    }
                }
            }
        }
    }

    return SUCCESS;
}

inline EE transformNCHWToNCHWCxN24Cx(
    TensorDesc filterDesc, const INT8 *filterArray, TensorDesc ftmDesc, INT8 *ftmArray, U32 cx)
{
    if (filterArray == NULL || ftmArray == NULL) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    U32 fhfw = fh * fw;
    U32 count = 0;
    U32 nx = 24;
    // if (fn % 32 == 0 && fn % N != 0) {
    //     nx = 32;
    // }
    U32 remain = fn % nx;

    I32 *offsetC = (I32 *)ftmArray;
    ftmArray += fn * bytesOf(DT_I32);
    for (U32 n = 0; n < fn; ++n) {
        I32 sum = 0;
        for (U32 i = 0; i < fc * fhfw; ++i) {
            sum += filterArray[i + n * fc * fhfw];
        }
        offsetC[n] = -128 * sum;
    }

    U32 fcPadding = UNI_ALIGN(fc, cx);

    U32 nxArray[4] = {8, 16, 24, 32};
    for (; count < fn; count += nx) {
        nx = UNI_MIN(fn - count, nx);
        nx = nxArray[(nx >> 3) - 1];
        for (U32 c = 0; c < fc / cx; ++c) {
            for (U32 hw = 0; hw < fhfw; ++hw) {
                for (U32 c2 = 0; c2 < cx; c2 += 4) {
                    for (U32 n = 0; n < nx; ++n) {
                        for (U32 c4 = 0; c4 < 4; ++c4) {
                            U32 iIdx = (n + count) * fc * fhfw + (c4 + c2 + c * cx) * fhfw + hw;
                            U32 oIdx = count * fcPadding * fhfw + c * cx * nx * fhfw +
                                hw * nx * cx + c2 * nx + n * 4 + c4;
                            ftmArray[oIdx] = filterArray[iIdx];
                        }
                    }
                }
            }
        }
    }

    return SUCCESS;
}

inline EE convolution_transform_filter_kernel_int8(TensorDesc filterDesc,
    const INT8 *filterArray,
    TensorDesc *ftmDesc,
    INT8 *ftmArray,
    DataFormat ftmDataFormat,
    U32 cx,
    U32 fnBlock)
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
    EE ret = SUCCESS;
    U32 fcPadding = (fc + 3) / 4 * 4;
    switch (ftmDataFormat) {
        case DF_NCHWC2NxC4: {
            *ftmDesc = tensor4df(fdt, ftmDataFormat, fn, fcPadding, fh, fw);
            if (fnBlock == 48) {
                transformNCHWToNCHWCxN48Cx(filterDesc, filterArray, *ftmDesc, ftmArray, cx);
            } else if (fnBlock == 24) {
                transformNCHWToNCHWCxN24Cx(filterDesc, filterArray, *ftmDesc, ftmArray, cx);
            } else {
                ret = NOT_SUPPORTED;
            }
            *ftmDesc = tensor4df(fdt, ftmDataFormat, fn, UNI_ALIGN(fcPadding, cx), fh, fw);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE convolution_transform_filter_int8(TensorDesc filterDesc,
    const INT8 *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    INT8 *filterTransformed)
{
    DataFormat ftmDataFormat;
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    U32 cx = 0;
    U32 fnBlock = 0;
    fn = (fn + 7) / 8 * 8 / convParamSpec.group;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_POINTWISE:
        case CONVOLUTION_ALGORITHM_DIRECT: {
            ftmDataFormat = DF_NCHWC2NxC4;
#ifdef _USE_AVX512_VNNI
            fnBlock = 48;
            cx = 16;
#else
            fnBlock = 24;
            cx = 8;
#endif
            break;
        }
        default:
            return NOT_MATCH;
    }
    // CHECK_STATUS(InferConvWeightFormat(ftmDataFormat, fnBlock));
    U32 channelAxis = filterDesc.nDims - 1;
    TensorDesc tmpFilterDesc = filterDesc;
    tmpFilterDesc.dims[channelAxis] /= convParamSpec.group;
    U32 fnPadding = (tmpFilterDesc.dims[channelAxis] + 7) / 8 * 8;
    U32 originalTileSize = tensorNumElements(tmpFilterDesc);
    for (U32 g = 0; g < convParamSpec.group; g++) {
        CHECK_STATUS(convolution_transform_filter_kernel_int8(
            tmpFilterDesc, filter, ftmDesc, filterTransformed, ftmDataFormat, cx, fnBlock));
        U32 newTileSize = tensorNumElements(*ftmDesc) / tmpFilterDesc.dims[channelAxis] * fnPadding;
        filter += originalTileSize;
        filterTransformed += newTileSize;
    }
    ftmDesc->dims[channelAxis] = filterDesc.dims[channelAxis];
    return SUCCESS;
}
