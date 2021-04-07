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
#include "cpu/arm/fp32/tensor_computing_fp32.h"

EE prelu_fp32(TensorDesc inputDesc,
    F32 *input,
    F32 *weight,
    PReLUParamSpec preluDesc,
    TensorDesc outputDesc,
    F32 *output)
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
    float32x4_t slope0, slope1;
    uint32x4_t mask0, mask1;
    float32x4_t in0, in1, out0, out1;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c += 8) {
            if (preluDesc.propagate_down) {
                slope0 = slope1 = vdupq_n_f32(weight[0]);
            } else {
                slope0 = vld1q_f32(weight + c);
                slope1 = vld1q_f32(weight + c + 4);
            }
            for (U32 hw = 0; hw < ih * iw; hw++) {
                in0 = vld1q_f32(input);
                in1 = vld1q_f32(input + 4);
                mask0 = vcleq_f32(in0, vdupq_n_f32(0.f));
                mask1 = vcleq_f32(in1, vdupq_n_f32(0.f));
                float32x4_t tmp0 = vmulq_f32(in0, slope0);
                float32x4_t tmp1 = vmulq_f32(in1, slope1);
                out0 = vbslq_f32(mask0, tmp0, in0);
                out1 = vbslq_f32(mask1, tmp1, in1);
                vst1q_f32(output, out0);
                vst1q_f32(output + 4, out1);
                input += 8;
                output += 8;
            }
        }
    }
    return SUCCESS;
}
