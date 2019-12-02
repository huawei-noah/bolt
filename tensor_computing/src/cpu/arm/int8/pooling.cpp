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
#include "sys.h"
#include "tensor_desc.h"
#include "type.h"
#include "error.h"
#include "tensor_computing_type.h"

#include "cpu/arm/int8/pooling_int8.h"

EE pooling(const void* input, F16* inputScale, void* output, F16* outputScale,
           U32 in, U32 ic, U32 ih, U32 iw,
           U32 oh, U32 ow,
           U32 stride, U32 padding, U32 kernelSize,
           PoolingMode pm)
{
    INT8* inPtr = (INT8*)input;
    INT8* outPtr = (INT8*)output;
    ic /= 8;

    short khkw = kernelSize * kernelSize;
    short factor = 256 / khkw;

    switch (pm) {
        case Max: {
            *outputScale = *inputScale;
            break;
        }
        case Mean: {
            *outputScale = *inputScale * factor * khkw / 256;
            break;
        }
        default: {
            CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
        }
    }

    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 h = 0; h < oh; h++) {
                for (U32 w = 0; w < ow; w++) {
                    int hstart = (int)h * (int)stride - (int)padding;
                    int wstart = (int)w * (int)stride - (int)padding;
                    int hend = std::min(hstart + kernelSize, ih);
                    int wend = std::min(wstart + kernelSize, iw);
                    hstart = std::max(hstart, 0);
                    wstart = std::max(wstart, 0);

                    int8x8_t in1, out1;
                    int16x8_t out_mean = {0};
                    out1 = vdup_n_s8(-128);
                    short pool_size = (hend-hstart) * (wend-wstart);
                    for (int kernelH = hstart; kernelH < hend; kernelH++) {
                        for (int kernelW = wstart; kernelW < wend; kernelW++) {
                            const U32 index = (kernelH * iw + kernelW) * 8;
                            in1 = vld1_s8(inPtr + index);
                            switch (pm) {
                                case Max:
                                    out1 = vmax_s8(out1, in1);
                                    break;
                                case Mean:
                                    out_mean = vaddw_s8(out_mean, in1);
                                    break;
                                default:
                                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
                            }
                        }
                    }
                    if (pm == Max) {
                        vst1_s8(outPtr + (h * ow + w) * 8, out1);
                    } else {
                        short pool_factor = factor * khkw / pool_size;
                        if (pool_factor > 1) {
                            out_mean = vmulq_n_s16(out_mean, pool_factor);
                        }
                        in1 = vshrn_n_s16(out_mean, 8);
                        vst1_s8(outPtr + (h * ow + w) * 8, in1);
                    }
                }
            }
            inPtr += ih * iw * 8;
            outPtr += oh * ow * 8;
        }
    }
    return SUCCESS;
}

EE pooling_int8(TensorDesc inputDesc, const void* input, F16* inputScale, PoolingDesc poolingDesc, TensorDesc outputDesc, void* output, F16* outputScale)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in = 0, ic = 0, ih = 0, iw = 0,
        on = 0, oc = 0, oh = 0, ow = 0;
    CHECK_STATUS_WITH_RETURN(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (idt != odt || idt != DT_I8) {
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    }
    if (in != on || ic != oc) {
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    }
    if (idf != DF_NCHWC8 || odf != idf) {
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    }

    PoolingMode pm = poolingDesc.pm;
    U32 stride = poolingDesc.stride;
    U32 padding = poolingDesc.padding;
    U32 kernelSize = poolingDesc.kernelSize;
    if (padding >= kernelSize) {
        CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
    }

    if (kernelSize > 16 && pm == Mean) {
        CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);  
    }

    EE ret = pooling(input, inputScale, output, outputScale,
                        in, ic, ih, iw,
                        oh, ow,
                        stride, padding, kernelSize,
                        pm);
    return ret;
}
