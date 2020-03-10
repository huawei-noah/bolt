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
#include <string.h>

#include "tensor_computing_type.h"
#include "cpu/arm/int8/tensor_computing_int8.h"


inline EE depthwise_convolution_transform_filter_kernel_int8(TensorDesc filterDesc, const INT8* filterArray,
    TensorDesc *ftmDesc, INT8* ftmArray,
    DataFormat ftmDataFormat)
{
    if (nullptr == filterArray || nullptr == ftmDesc || nullptr == ftmArray)
        CHECK_STATUS(NULL_POINTER);
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));

    if (fdf == ftmDataFormat) {
        *ftmDesc = filterDesc;
        if (fdf == DF_NCHW || fdf == DF_NCHWC8) {
            memcpy(ftmArray, filterArray, fn*fc*fh*fw*bytesOf(fdt));
            return SUCCESS;
        }
        if (fdf == DF_CHW_NC || fdf == DF_CHWC8_NCN8C4) {
            memcpy(ftmArray, filterArray, (fc*fh*fw + fc*fn)*bytesOf(fdt));
            return SUCCESS;
        }
        return NOT_SUPPORTED;
    }

    switch (fdf) {
        case DF_NCHW: {
            if (ftmDataFormat == DF_NCHWC8) {
                U32 ic = fc / 8;
                for (U32 c = 0; c < ic; c++) {
                    for (U32 hw = 0; hw < fh*fw; hw++) {
                        for (U32 c8 = 0; c8 < 8; c8++) {
                            ftmArray[c*fh*fw*8 + hw*8 + c8] = filterArray[(c*8+c8)*fh*fw + hw];
                        }
                    }
                }
                *ftmDesc = tensor4df(fdt, DF_NCHWC8, fn, fc, fh, fw);
            } else {
                return NOT_SUPPORTED;
            }
            break;
        }
        case DF_CHW_NC: {
            if (ftmDataFormat == DF_CHWC8_NCN8C4) {
                /*
                 * CHW_NC => DF_CHWC8_NCN8C4
                 */
                const INT8 *pwFilterArray = filterArray + fc*fh*fw;
                INT8 *pwFtmArray = ftmArray + fc*fh*fw;
                
                U32 ic = fc / 8;
                for (U32 c = 0; c < ic; c++) {
                    for (U32 hw = 0; hw < fh*fw; hw++) {
                        for (U32 c8 = 0; c8 < 8; c8++) {
                            ftmArray[c*fh*fw*8 + hw*8 + c8] = filterArray[(c*8+c8)*fh*fw + hw];
                        }
                    }
                }
    
                U32 oc = fn / 8;
                ic *= 2;  // fc / 4
                for (U32 o = 0; o < oc; o++) {
                    for (U32 c = 0; c < ic; c++) {
                        for (U32 o8 = 0; o8 < 8; o8++) {
                            for (U32 c4 = 0; c4 < 4; c4++) {
                                pwFtmArray[o*fc*8 + c*32 + o8*4 + c4] = pwFilterArray[(o*8+o8)*fc + c*4 + c4];
                            }
                        }
                    }
                }
                *ftmDesc = tensor4df(fdt, DF_CHWC8_NCN8C4, fn, fc, fh, fw);
            } else {
                return NOT_SUPPORTED;
            }
            break;
        }
        default:
            return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE depthwise_convolution_transform_filter_int8(TensorDesc filterDesc, const INT8* filter,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, INT8* filterTransformed)
{
    DataFormat ftmDataFormat;
    switch (algorithm) {
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ftmDataFormat = DF_CHWC8_NCN8C4;
            break;
        default:
            return NOT_MATCH;
    }
    EE ret = depthwise_convolution_transform_filter_kernel_int8(filterDesc, filter, ftmDesc, filterTransformed, ftmDataFormat);
    CHECK_STATUS(ret);
    return ret;
}
#endif
