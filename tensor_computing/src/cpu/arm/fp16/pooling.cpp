// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "cpu/arm/fp16/tensor_computing_fp16.h"

EE pooling_fp16(TensorDesc inputDesc, const F16* input, PoolingDesc poolingDesc, TensorDesc outputDesc, F16* output)
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

    if (idt != odt || idt != DT_F16) {
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

    ic /= 8;
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
                    float16x8_t in1, out1;
                    float16x8_t poolSize = vdupq_n_f16(float16_t((hend - hstart)*(wend - wstart)));
                    out1 = vdupq_n_f16(float16_t((pm == POOLING_MAX) ? UNI_F16_MIN : 0));
                    for (int kernelH = hstart; kernelH < hend; kernelH++) {
                        for (int kernelW = wstart; kernelW < wend; kernelW++) {
                            const U32 index = (kernelH * iw + kernelW) * 8;
                            in1 = vld1q_f16(input + index);
                            switch (pm) {
                                case POOLING_MAX:
                                    out1 = vmaxq_f16(in1, out1);
                                    break;
                                case POOLING_MEAN:
                                    out1 = vaddq_f16(out1,  in1);
                                    break;
                                default:
                                    CHECK_STATUS(NOT_SUPPORTED);
                            }
                        }
                    }
                    vst1q_f16(output + (h * ow + w) * 8, ((pm == POOLING_MAX) ? out1 : vdivq_f16(out1, poolSize)));
                }
            }
            input += ih * iw * 8;
            output += oh * ow * 8;
        }
    }
    return SUCCESS;
}
