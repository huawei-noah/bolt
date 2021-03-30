// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/tensor_computing_arm.h"
#include "cpu/arm/fp16/tensor_computing_fp16.h"

EE prelu_fp16(TensorDesc inputDesc,
    F16 *input,
    F16 *weight,
    PReLUParamSpec preluDesc,
    TensorDesc outputDesc,
    F16 *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in = 0, ic = 0, ih = 0, iw = 0, on = 0, oc = 0, oh = 0, ow = 0;
    if (tensorIs4d(inputDesc) && tensorIs4d(outputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
        if (idf != DF_NCHWC8) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    } else {
        return NOT_SUPPORTED;
    }

    CHECK_REQUIREMENT(in == on && ic == oc && ih == oh && iw == ow);
    float16x8_t slope;
    uint16x8_t mask;
    float16x8_t in0, out0;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c += 8) {
            slope = preluDesc.propagate_down ? vdupq_n_f16(weight[0]) : vld1q_f16(weight + c);
            for (U32 hw = 0; hw < ih * iw; hw++) {
                in0 = vld1q_f16(input);
                mask = vcleq_f16(in0, vdupq_n_f16(0.f));
                float16x8_t tmp = vmulq_f16(in0, slope);
                out0 = vbslq_f16(mask, tmp, in0);
                vst1q_f16(output, out0);
                input += 8;
                output += 8;
            }
        }
    }
    return SUCCESS;
}
