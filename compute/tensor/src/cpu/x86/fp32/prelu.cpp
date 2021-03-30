// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/tensor_computing_x86.h"
#include "cpu/x86/fp32/tensor_computing_fp32.h"

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
    __m256 slope;
    __m256 mask0, mask1;
    __m256 data, out;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c += 8) {
            slope = preluDesc.propagate_down ? _mm256_set1_ps(weight[0])
                                             : _mm256_loadu_ps(weight + c);
            for (U32 hw = 0; hw < ih * iw; hw++) {
                data = _mm256_loadu_ps(input);
                mask0 = _mm256_max_ps(data, _mm256_setzero_ps());
                mask1 = _mm256_min_ps(data, _mm256_setzero_ps());
                out = _mm256_fmadd_ps(mask1, slope, mask0);
                _mm256_storeu_ps(output, out);
                input += 8;
                output += 8;
            }
        }
    }
    return SUCCESS;
}
