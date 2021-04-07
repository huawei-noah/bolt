// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/general/tensor_computing_general.h"

template <typename T>
static EE prelu(
    T *input, T *output, T *weight, PReLUParamSpec preluDesc, U32 in, U32 ic, U32 ih, U32 iw)
{
    ic /= 8;
    T slope;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 hw = 0; hw < ih * iw; hw++) {
                for (U32 c8 = 0; c8 < 8; c8++) {
                    slope = preluDesc.propagate_down ? weight[0] : weight[c * 8 + c8];
                    U32 off = n * ic * ih * iw * 8 + c * ih * iw * 8 + hw * 8 + c8;
                    if (input[off] > 0) {
                        output[off] = input[off];
                    } else {
                        output[off] = input[off] * slope;
                    }
                }
            }
        }
    }
    return SUCCESS;
}

EE prelu_general(TensorDesc inputDesc,
    void *input,
    void *weight,
    PReLUParamSpec preluDesc,
    TensorDesc outputDesc,
    void *output)
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
    } else {
        return NOT_SUPPORTED;
    }
    if (idf != DF_NCHWC8) {
        return NOT_SUPPORTED;
    }
    CHECK_REQUIREMENT(in == on && ic == oc && ih == oh && iw == ow);
    EE ret = SUCCESS;
    switch (idt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = prelu((F32 *)input, (F32 *)output, (F32 *)weight, preluDesc, in, ic, ih, iw);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = prelu((F16 *)input, (F16 *)output, (F16 *)weight, preluDesc, in, ic, ih, iw);
            break;
        }
#endif
        default: {
            ret = NOT_SUPPORTED;
            break;
        }
    }
    return ret;
}
