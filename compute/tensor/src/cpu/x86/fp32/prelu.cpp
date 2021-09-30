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
    } else if (tensorIs3d(inputDesc) && tensorIs3d(outputDesc)) {
        CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
        CHECK_STATUS(tensor3dGet(outputDesc, &odt, &odf, &on, &oc, &oh));
        iw = ow = 1;
    } else if (tensorIs2d(inputDesc) && tensorIs2d(outputDesc)) {
        CHECK_STATUS(tensor2dGet(inputDesc, &idt, &idf, &in, &ic));
        CHECK_STATUS(tensor2dGet(outputDesc, &odt, &odf, &on, &oc));
        ih = oh = iw = ow = 1;
    } else {
        return NOT_SUPPORTED;
    }

    CHECK_REQUIREMENT(in == on && ic == oc && ih == oh && iw == ow);
    I32 ihiw = ih * iw;
    if (idf == DF_NCHWC8) {
        for (U32 n = 0; n < in; n++) {
            for (U32 c = 0; c < ic; c += 8) {
                __m256 slope = preluDesc.propagate_down ? _mm256_set1_ps(weight[0])
                                                        : _mm256_loadu_ps(weight + c);
                for (I32 hw = 0; hw < ihiw; hw++) {
                    __m256 data = _mm256_loadu_ps(input);
                    __m256 mask0 = _mm256_max_ps(data, _mm256_setzero_ps());
                    __m256 mask1 = _mm256_min_ps(data, _mm256_setzero_ps());
                    __m256 out = _mm256_fmadd_ps(mask1, slope, mask0);
                    _mm256_storeu_ps(output, out);
                    input += 8;
                    output += 8;
                }
            }
        }
    } else {
        for (U32 n = 0; n < in; n++) {
            for (U32 c = 0; c < ic; c++) {
                F32 slope_s = preluDesc.propagate_down ? weight[0] : weight[c];
                __m256 slope = _mm256_set1_ps(slope_s);
                I32 hw;
                for (hw = 0; hw < ihiw - 7; hw += 8) {
                    __m256 data = _mm256_loadu_ps(input);
                    __m256 mask0 = _mm256_max_ps(data, _mm256_setzero_ps());
                    __m256 mask1 = _mm256_min_ps(data, _mm256_setzero_ps());
                    __m256 out = _mm256_fmadd_ps(mask1, slope, mask0);
                    _mm256_storeu_ps(output, out);
                    input += 8;
                    output += 8;
                }
                for (; hw < ihiw; hw++) {
                    if (input[0] >= 0) {
                        output[0] = input[0];
                    } else {
                        output[0] = input[0] * slope_s;
                    }
                    input++;
                    output++;
                }
            }
        }
    }
    return SUCCESS;
}
