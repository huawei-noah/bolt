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

EE pooling_int8(TensorDesc inputDesc, const INT8* input, F16* inputScale,
    PoolingDesc poolingDesc,
    TensorDesc outputDesc, INT8* output, F16* outputScale)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in = 0, ic = 0, ih = 0, iw = 0,
        on = 0, oc = 0, oh = 0, ow = 0;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (idt != odt || idt != DT_I8) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (in != on || ic != oc) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (idf != DF_NCHWC8 || odf != idf) {
        CHECK_STATUS(NOT_MATCH);
    }

    PoolingMode pm = poolingDesc.pm;
    U32 strideH = poolingDesc.stride_h;
    U32 strideW = poolingDesc.stride_w;
    U32 paddingT = poolingDesc.padding_top;
    U32 paddingL = poolingDesc.padding_left;
    U32 kernelSizeH = poolingDesc.kernelSize_h;
    U32 kernelSizeW = poolingDesc.kernelSize_w;
    if (paddingT >= kernelSizeH || paddingL >= kernelSizeW) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    if (kernelSizeH * kernelSizeW > 256 && pm == POOLING_MEAN) {
        CHECK_STATUS(NOT_SUPPORTED);  
    }

    ic /= 8;

    short khkw = kernelSizeH * kernelSizeW;
    short factor = 256 / khkw;

    switch (pm) {
        case POOLING_MAX: {
            *outputScale = *inputScale;
            break;
        }
        case POOLING_MEAN: {
            *outputScale = *inputScale * factor * khkw / 256;
            break;
        }
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }

    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 h = 0; h < oh; h++) {
                for (U32 w = 0; w < ow; w++) {
                    int hstart = (int)h * (int)strideH - (int)paddingT;
                    int wstart = (int)w * (int)strideW - (int)paddingL;
                    int hend = UNI_MIN(hstart + kernelSizeH, ih);
                    int wend = UNI_MIN(wstart + kernelSizeW, iw);
                    hstart = UNI_MAX(hstart, 0);
                    wstart = UNI_MAX(wstart, 0);

                    int8x8_t in1, out1;
                    int16x8_t out_mean = {0};
                    out1 = vdup_n_s8(-128);
                    short pool_size = (hend-hstart) * (wend-wstart);
                    for (int kernelH = hstart; kernelH < hend; kernelH++) {
                        for (int kernelW = wstart; kernelW < wend; kernelW++) {
                            const U32 index = (kernelH * iw + kernelW) * 8;
                            in1 = vld1_s8(input + index);
                            switch (pm) {
                                case POOLING_MAX:
                                    out1 = vmax_s8(out1, in1);
                                    break;
                                case POOLING_MEAN:
                                    out_mean = vaddw_s8(out_mean, in1);
                                    break;
                                default:
                                    CHECK_STATUS(NOT_SUPPORTED);
                            }
                        }
                    }
                    if (pm == POOLING_MAX) {
                        vst1_s8(output + (h * ow + w) * 8, out1);
                    } else {
                        short pool_factor = factor * khkw / pool_size;
                        if (pool_factor > 1) {
                            out_mean = vmulq_n_s16(out_mean, pool_factor);
                        }
                        in1 = vshrn_n_s16(out_mean, 8);
                        vst1_s8(output + (h * ow + w) * 8, in1);
                    }
                }
            }
            input += ih * iw * 8;
            output += oh * ow * 8;
        }
    }

    return SUCCESS;
}
#endif
