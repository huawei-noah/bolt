// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <float.h>
#include "cpu/arm/fp32/tensor_computing_fp32.h"

EE pooling_fp32(TensorDesc inputDesc, const F32* input, PoolingDesc poolingDesc, TensorDesc outputDesc, F32* output)
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

    if (idt != odt || idt != DT_F32) {
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
                    float32x4_t in0, in1, out0, out1;
                    float32x4_t poolSize = vdupq_n_f32((hend - hstart)*(wend - wstart));
                    out0 = vdupq_n_f32((pm == POOLING_MAX) ? FLT_MIN : 0);
                    out1 = out0;
                    for (int kernelH = hstart; kernelH < hend; kernelH++) {
                        for (int kernelW = wstart; kernelW < wend; kernelW++) {
                            const U32 index = (kernelH * iw + kernelW) * 8;
                            in0 = vld1q_f32(input + index);
                            in1 = vld1q_f32(input + index + 4);
                            switch (pm) {
                                case POOLING_MAX: {
                                    out0 = vmaxq_f32(in0, out0);
                                    out1 = vmaxq_f32(in1, out1);
                                    break;
                                }
                                case POOLING_MEAN: {
                                    out0 = vaddq_f32(out0, in0);
                                    out1 = vaddq_f32(out1, in1);
                                    break;
                                }
                                default:
                                    CHECK_STATUS(NOT_SUPPORTED);
                            }
                        }
                    }
                    vst1q_f32(output + (h * ow + w) * 8, ((pm == POOLING_MAX) ? out0 : vdivq_f32(out0, poolSize)));
                    vst1q_f32(output + (h * ow + w) * 8 + 4, ((pm == POOLING_MAX) ? out1 : vdivq_f32(out1, poolSize)));
                }
            }
            input += ih * iw * 8;
            output += oh * ow * 8;
        }
    }
    return SUCCESS;
}
