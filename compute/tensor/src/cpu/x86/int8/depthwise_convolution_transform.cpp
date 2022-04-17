// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/int8/tensor_computing_int8.h"
#include "cpu/x86/int8/transform_functions_int8.h"

EE depthwise_convolution_transform_filter_int8(
    TensorDesc filterDesc, const INT8 *filter, TensorDesc *ftmDesc, INT8 *filterTransformed)
{
    DataFormat ftmDataFormat = DF_NCHWN8HW4;  // for flag, actually DF_NCHWN16HW4

    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    if (fdf == ftmDataFormat) {
        *ftmDesc = filterDesc;
        UNI_MEMCPY(filterTransformed, filter, fn * fc * fh * fw * bytesOf(fdt));
        return SUCCESS;
    }
    if (fdf != DF_NCHW) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    filterDesc = tensor4df(fdt, fdf, fc, 1, fh, fw);
    *ftmDesc = tensor4df(fdt, ftmDataFormat, fc, 1, fh, fw);

    U32 fhfw = fh * fw;
    U32 fhfwAligned = (fhfw + 3) / 4 * 4;

    I32 *offsetC = (I32 *)filterTransformed;
    filterTransformed += fc * bytesOf(DT_I32);
    for (U32 n = 0; n < fc; ++n) {
        I32 sum = 0;
        for (U32 i = 0; i < fh * fw; ++i) {
            sum += filter[i + n * fh * fw];
        }
        offsetC[n] = -128 * sum;
    }

    for (U32 n = 0; n < fn; ++n) {
        for (U32 c = 0; c < fc; c += 16) {
            for (U32 hw = 0; hw < fhfwAligned; hw += 4) {
                U32 c16;
                for (c16 = 0; (c16 < 16) && (c16 < (fc - c)); ++c16) {
                    U32 w4;
                    for (w4 = 0; (w4 < 4) && (w4 < (fhfw - hw)); ++w4) {
                        U32 iidx = n * c * fhfw + (c + c16) * fhfw + hw + w4;
                        U32 oidx = n * c * fhfwAligned + c * fhfwAligned + hw * 16 + 4 * c16 + w4;
                        filterTransformed[oidx] = filter[iidx];
                    }
                    for (; w4 < 4; ++w4) {
                        filterTransformed[n * c * fhfwAligned + c * fhfwAligned + hw * 16 +
                            4 * c16 + w4] = 0;
                    }
                }
                for (; c16 < 16; ++c16) {
                    UNI_MEMSET(
                        filterTransformed + n * c * fhfw + c * fhfw + hw * 16 + c16 * 4, 0, 4);
                }
            }
        }
    }

    return SUCCESS;
}

EE depthwise_pointwise_convolution_transform_filter_int8(TensorDesc dwFilterDesc,
    const INT8 *dwFilter,
    TensorDesc pwFilterDesc,
    const INT8 *pwFilter,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc *dwFtmDesc,
    INT8 *dwFilterTransformed,
    TensorDesc *pwFtmDesc,
    INT8 *pwFilterTransformed)
{
    EE ret = depthwise_convolution_transform_filter_int8(
        dwFilterDesc, dwFilter, dwFtmDesc, dwFilterTransformed);
    CHECK_STATUS(ret);
    if (pwFilter == nullptr) {
        return ret;
    }

    ConvolutionParamSpec p = createConvolutionParamSpec(1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
        1, pwFilterDesc.dims[pwFilterDesc.nDims - 1], CONVOLUTION_POINTWISE);
    ret = convolution_transform_filter_int8(
        pwFilterDesc, pwFilter, p, CONVOLUTION_ALGORITHM_POINTWISE, pwFtmDesc, pwFilterTransformed);
    return ret;
}
