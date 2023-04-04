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

template <U32 N>
inline EE transformCNHWToNCHWCxNxCx(
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
    U32 nx = N;
    U32 realFn = fc * fh * fw;
    if (realFn % 32 == 0 && realFn % N != 0) {
        nx = 32;
    }
    U32 fnPadding = UNI_ALIGN(fn, 16);
    I32 *offsetC = (I32 *)ftmArray;
    UNI_MEMSET(offsetC, 0, realFn * bytesOf(DT_I32));
    
    ftmArray += realFn * bytesOf(DT_I32);
    U32 nxArray[4] = {8, 16, 32, 48};
    U32 nSize = 0;
    U32 cSize = 4;
    for (U32 n = 0; n < realFn; n += nSize) {
        nSize = UNI_MIN(nx, fc * fh * fw - n);
        nSize = nxArray[nSize >> 4];
        for (U32 c = 0; c < fnPadding; c += cSize) {
            for (U32 nn = 0; nn < nSize; ++nn) {
                for (U32 cc = 0; cc < cSize; ++cc) {
                    U32 oidx = n * fnPadding + c * nSize + nn * cSize + cc;
                    if (c >= fn) {
                        ftmArray[oidx] = 0;
                    } else {
                        U32 iih = (n + nn) / (fw * fc);
                        U32 iiw = (n + nn) % (fw * fc) / fc;
                        U32 iic = (n + nn) % fc;
                        U32 iidx = (c + cc) * fc * fh * fw + iic * fh * fw + iih * fw + iiw;
                        ftmArray[oidx] = filterArray[iidx];
                        offsetC[n + nn] += filterArray[iidx];
                    }
                }
            }
        }
    }
    for (U32 n = 0; n < realFn; ++n) {
        offsetC[n] *= -128;
    }
    return SUCCESS;
}

inline EE deconvolution_transform_filter_kernel_int8(TensorDesc filterDesc,
    const INT8 *filterArray,
    TensorDesc *ftmDesc,
    INT8 *ftmArray,
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
        UNI_MEMCPY(ftmArray, filterArray, fn * fc * fh * fw * bytesOf(fdt));
        return SUCCESS;
    }
    if (fdf != DF_NCHW) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    EE ret = SUCCESS;
    U32 fcPadding = (fc * fh * fw + 3) / 4 * 4;
    switch (ftmDataFormat) {
        case DF_NCHWC2NxC4: {
            *ftmDesc = tensor4df(fdt, ftmDataFormat, fcPadding, fn, 1, 1);
            transformCNHWToNCHWCxNxCx<48>(filterDesc, filterArray, *ftmDesc, ftmArray, 16);
            *ftmDesc = tensor4df(fdt, ftmDataFormat, fcPadding, UNI_ALIGN(fn, 16), 1, 1);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE deconvolution_transform_filter_int8(TensorDesc filterDesc,
    const INT8 *filter,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    INT8 *filterTransformed)
{
    DataFormat ftmDataFormat;
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    fn = fc * fh * fw;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_POINTWISE: {
            ftmDataFormat = DF_NCHWC2NxC4;
            break;
        }
        default:
            return NOT_MATCH;
    }
    EE ret = deconvolution_transform_filter_kernel_int8(
        filterDesc, filter, ftmDesc, filterTransformed, ftmDataFormat);
    CHECK_STATUS(ret);
    return ret;
}
