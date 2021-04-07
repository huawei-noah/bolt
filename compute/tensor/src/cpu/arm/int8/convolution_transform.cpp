// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_INT8
#include "cpu/arm/int8/tensor_computing_int8.h"
#include "cpu/arm/fp16/convolution_winograd_transform.h"

#include <string.h>

#include "tensor_desc.h"
#include "error.h"
#include "tensor_computing.h"

inline EE convolution_transform_filter_kernel_int8(TensorDesc filterDesc,
    const void *filter,
    TensorDesc *ftmDesc,
    void *ftm,
    DataFormat ftmDataFormat)
{
    if (nullptr == filter || nullptr == ftmDesc || nullptr == ftm) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    if (fdf == ftmDataFormat) {
        *ftmDesc = filterDesc;
        memcpy(ftm, filter, fn * fc * fh * fw * bytesOf(fdt));
        return SUCCESS;
    }
    if (fdf != DF_NCHW) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    EE ret = SUCCESS;
    switch (ftmDataFormat) {
        case DF_NCHWN8C4: {
            INT8 *filterArray = (INT8 *)filter;
            INT8 *ftmArray = (INT8 *)ftm;
            U32 oc = fn / 8;
            U32 fc_quad = fc / 4;
            for (U32 o = 0; o < oc; o++) {
                for (U32 c = 0; c < fc_quad; c++) {
                    for (U32 hw = 0; hw < fh * fw; hw++) {
                        for (U32 o8 = 0; o8 < 8; o8++) {
                            for (U32 c4 = 0; c4 < 4; c4++) {
                                ftmArray[o * fh * fw * fc * 8 + c * fh * fw * 32 + hw * 32 +
                                    o8 * 4 + c4] = filterArray[(o * 8 + o8) * fc * fh * fw +
                                    (c * 4 + c4) * fh * fw + hw];
                            }
                        }
                    }
                }
            }
            break;
        }
        case DF_HWNCN8C4: {
            F16 *filterArray = (F16 *)filter;
            F16 *ftmArray = (F16 *)ftm;
            for (U32 o = 0; o < fn / 8; o++) {
                for (U32 c = 0; c < fc / 4; c++) {
                    // Each time deal with N2C4; 4 times we have N8C4
                    U32 f_off_0 = (o * 8) * fc * fh * fw + c * 4 * fh * fw;
                    U32 f_off_1 = (o * 8 + 2) * fc * fh * fw + c * 4 * fh * fw;
                    U32 f_off_2 = (o * 8 + 4) * fc * fh * fw + c * 4 * fh * fw;
                    U32 f_off_3 = (o * 8 + 6) * fc * fh * fw + c * 4 * fh * fw;

                    U32 ftm_off_0 = o * 36 * fc * 8 + c * 32;
                    U32 ftm_off_1 = o * 36 * fc * 8 + c * 32 + 8;
                    U32 ftm_off_2 = o * 36 * fc * 8 + c * 32 + 16;
                    U32 ftm_off_3 = o * 36 * fc * 8 + c * 32 + 24;

                    F16 F[9][8];  // N2C4 at a time
                    F16 *F_ptr[9];
                    F16 *Fw[36];

                    for (U32 hw = 0; hw < 9; hw++) {
                        for (U32 oo = 0; oo < 2; oo++) {
                            for (U32 cc = 0; cc < 4; cc++) {
                                F[hw][oo * 4 + cc] =
                                    filterArray[f_off_0 + hw + oo * fc * fh * fw + cc * fh * fw];
                            }
                        }
                        F_ptr[hw] = F[hw];
                    }
                    for (U32 hw = 0; hw < 36; hw++) {
                        Fw[hw] = ftmArray + ftm_off_0 + hw * fc * 8;  // Each hw fills N8*fc
                    }
                    trans_W_4x4_3x3(Fw, F_ptr);

                    for (U32 hw = 0; hw < 9; hw++) {
                        for (U32 oo = 0; oo < 2; oo++) {
                            for (U32 cc = 0; cc < 4; cc++) {
                                F[hw][oo * 4 + cc] =
                                    filterArray[f_off_1 + hw + oo * fc * fh * fw + cc * fh * fw];
                            }
                        }
                        F_ptr[hw] = F[hw];
                    }
                    for (U32 hw = 0; hw < 36; hw++) {
                        Fw[hw] = ftmArray + ftm_off_1 + hw * fc * 8;  // Each hw fills N8*fc
                    }
                    trans_W_4x4_3x3(Fw, F_ptr);

                    for (U32 hw = 0; hw < 9; hw++) {
                        for (U32 oo = 0; oo < 2; oo++) {
                            for (U32 cc = 0; cc < 4; cc++) {
                                F[hw][oo * 4 + cc] =
                                    filterArray[f_off_2 + hw + oo * fc * fh * fw + cc * fh * fw];
                            }
                        }
                        F_ptr[hw] = F[hw];
                    }
                    for (U32 hw = 0; hw < 36; hw++) {
                        Fw[hw] = ftmArray + ftm_off_2 + hw * fc * 8;  // Each hw fills N8*fc
                    }
                    trans_W_4x4_3x3(Fw, F_ptr);

                    for (U32 hw = 0; hw < 9; hw++) {
                        for (U32 oo = 0; oo < 2; oo++) {
                            for (U32 cc = 0; cc < 4; cc++) {
                                F[hw][oo * 4 + cc] =
                                    filterArray[f_off_3 + hw + oo * fc * fh * fw + cc * fh * fw];
                            }
                        }
                        F_ptr[hw] = F[hw];
                    }
                    for (U32 hw = 0; hw < 36; hw++) {
                        Fw[hw] = ftmArray + ftm_off_3 + hw * fc * 8;  // Each hw fills N8*fc
                    }
                    trans_W_4x4_3x3(Fw, F_ptr);
                }
            }
            fdt = DT_F16;
            fh = 6;
            fw = 6;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    *ftmDesc = tensor4df(fdt, ftmDataFormat, fn, fc, fh, fw);
    return ret;
}

EE convolution_transform_filter_int8(TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    void *filterTransformed)
{
    DataFormat ftmDataFormat;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_WINOGRAD:
            ftmDataFormat = DF_HWNCN8C4;
            break;
        case CONVOLUTION_ALGORITHM_GEMM:
            ftmDataFormat = DF_NCHWN8C4;
            break;
        default:
            return NOT_MATCH;
    }
    U32 channelAxis = filterDesc.nDims - 1;
    TensorDesc tmpFilterDesc = filterDesc;
    tmpFilterDesc.dims[channelAxis] /= convParamSpec.group;
    U32 fnPadding = tmpFilterDesc.dims[channelAxis];
    if (fnPadding % 8 != 0) {
        fnPadding = (fnPadding / 8 + 1) * 8;
    }
    U32 originalTileSize = tensorNumElements(tmpFilterDesc);
    for (U32 g = 0; g < convParamSpec.group; g++) {
        CHECK_STATUS(convolution_transform_filter_kernel_int8(
            tmpFilterDesc, filter, ftmDesc, filterTransformed, ftmDataFormat));
        U32 newTileSize = tensorNumElements(*ftmDesc) / tmpFilterDesc.dims[channelAxis] * fnPadding;
        filter = (const U8 *)filter + originalTileSize * bytesOf(filterDesc.dt);
        filterTransformed = (U8 *)filterTransformed + newTileSize * bytesOf(ftmDesc->dt);
    }
    ftmDesc->dims[channelAxis] = filterDesc.dims[channelAxis];
    return SUCCESS;
}
#endif
