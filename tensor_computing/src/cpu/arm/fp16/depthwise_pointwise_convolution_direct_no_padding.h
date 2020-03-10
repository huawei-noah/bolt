// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_DEPTHWISE_POINTWISE_CONVOLUTION_DIRECT_NO_PADDING
#define _H_DEPTHWISE_POINTWISE_CONVOLUTION_DIRECT_NO_PADDING

#include <string.h>

#include "sys.h"
#include "type.h"
#include "error.h"
#include "tensor_desc.h"
#include "tensor_computing_type.h"


inline void calc_eight_channel_elements(I32 hw,
    I32 ih_base, I32 ih, I32 iw,
    I32 fh, I32 fw,
    I32 ow,
    F16 *inArray,
    I32 strideH, I32 strideW, I32 paddingT, I32 paddingL,
    const F16 *filterArray,
    float16x8_t bias,
    F16 *output)
{
    I32 h = hw / ow;
    I32 w = hw % ow;
    float16x8_t v0 = bias;
    I32 ih_start = h * strideH - paddingT;
    I32 iw_start = w * strideW - paddingL;
    I32 fh_start = 0;
    if (ih_start < 0)  {
        fh_start -= ih_start;
    }
    I32 fw_start = 0;
    if (iw_start < 0) {
        fw_start -= iw_start;
    }
    for (I32 fh_idx = fh_start; fh_idx < fh; fh_idx++) {
        I32 ih_idx = ih_start + fh_idx;
        if (ih_idx >= ih)
            break;
        I32 iw_base = ((ih_base + ih_idx) * iw);
        I32 filter_index = (fh_idx * fw + fw_start) * 8;
        for (I32 fw_idx = fw_start; fw_idx < fw; fw_idx++, filter_index+=8) {
            I32 iw_idx = iw_start + fw_idx;
            if (iw_idx >= iw)
                break;
            {
                U32 in_index = (iw_base + iw_idx) * 8;
                float16x8_t v1 = vld1q_f16(inArray + in_index);
                float16x8_t v2 = vld1q_f16(filterArray + filter_index);
                v0 = vfmaq_f16(v0, v1, v2);
            }
        }
    }
    vst1q_f16(output, v0);
}

EE depthwise_pointwise_convolution_direct_no_padding_A55(TensorDesc inputDesc, F16* inArray,
    TensorDesc filterDesc, const F16* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const F16* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F16* outArray,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode);

EE depthwise_pointwise_convolution_direct_no_padding_A76(TensorDesc inputDesc, F16* inArray,
    TensorDesc filterDesc, const F16* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const F16* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F16* outArray,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode);

inline EE depthwise_pointwise_convolution_direct_no_padding(TensorDesc inputDesc, F16* inArray,
    TensorDesc filterDesc, const F16* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const F16* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F16* outArray,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode,
    Arch arch)
{
    EE ret = SUCCESS;
    switch (arch) {
        case ARM_A55:
            ret = depthwise_pointwise_convolution_direct_no_padding_A55(inputDesc, inArray,
                                                                        filterDesc, filterArray,
                                                                        convDesc,
                                                                        biasDesc, biasArray,
                                                                        tmpBytes, tmp,
                                                                        outputDesc, outArray,
                                                                        depthwiseActivationMode,
                                                                        pointwiseActivationMode);
            break;
        case ARM_A76:
            ret = depthwise_pointwise_convolution_direct_no_padding_A76(inputDesc, inArray,
                                                                        filterDesc, filterArray,
                                                                        convDesc,
                                                                        biasDesc, biasArray,
                                                                        tmpBytes, tmp,
                                                                        outputDesc, outArray,
                                                                        depthwiseActivationMode,
                                                                        pointwiseActivationMode);
            break;
        default:
            return NOT_SUPPORTED;
    }
    return ret;
}
#endif
