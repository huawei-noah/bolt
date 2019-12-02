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

#include "cpu/arm/fp16/pooling_fp16.h"


EE pooling_direct(const void* input, void* output,
                U32 in, U32 ic, U32 ih, U32 iw,
                U32 oh, U32 ow,
                U32 stride, U32 padding, U32 kernelSize,
                PoolingMode pm)
{
    F16* inPtr = (F16*)input;
    F16* outPtr = (F16*)output;
    ic /= 8;
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
                    float16x8_t in1, out1;
                    float16x8_t poolSize = vdupq_n_f16(float16_t((hend - hstart)*(wend - wstart)));
                    out1 = vdupq_n_f16(float16_t((pm == Max) ? (-65504) : 0));
                    for (int kernelH = hstart; kernelH < hend; kernelH++) {
                        for (int kernelW = wstart; kernelW < wend; kernelW++) {
                            const U32 index = (kernelH * iw + kernelW) * 8;
                            in1 = vld1q_f16(inPtr + index);
                            switch (pm) {
                                case Max:
                                    out1 = vmaxq_f16(in1, out1);
                                    break;
                                case Mean:
                                    out1 = vaddq_f16(out1,  in1);
                                    break;
                                default:
                                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
                            }
                        }
                    }
                    vst1q_f16(outPtr + (h * ow + w) * 8, ((pm == Max) ? out1 : vdivq_f16(out1, poolSize)));
                }
            }
            inPtr += ih * iw * 8;
            outPtr += oh * ow * 8;
        }
    }
    return SUCCESS;
}

EE pooling_fp16(TensorDesc inputDesc, const void* input, PoolingDesc poolingDesc, TensorDesc outputDesc, void* output)
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

    if (idt != odt || idt != DT_F16) {
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

    EE ret = pooling_direct(input, output,
                            in, ic, ih, iw,
                            oh, ow,
                            stride, padding, kernelSize,
                            pm);
    return ret;
}
