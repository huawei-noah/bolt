// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/fp32/tensor_computing_fp32.h"
#include "cpu/x86/tensor_computing_x86.h"
#include "cpu/x86/fp32/x86_functions_fp32.h"

EE l2normalization_fp32(TensorDesc inputDesc, const F32 *input, TensorDesc outputDesc, F32 *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 ic = 0, ih = 0, iw = 0, oh = 0, ow = 0;
    if (tensorIs2d(inputDesc)) {
        CHECK_STATUS(tensor2dGet(inputDesc, &idt, &idf, &ih, &iw));
        ic = 1;
        CHECK_STATUS(tensor2dGet(outputDesc, &odt, &odf, &oh, &ow));
    } else if (tensorIs3d(inputDesc)) {
        U32 oc = 0;
        CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &ic, &ih, &iw));
        CHECK_STATUS(tensor3dGet(outputDesc, &odt, &odf, &oc, &oh, &ow));
        CHECK_REQUIREMENT(ic == oc);
    } else if (tensorIs4d(inputDesc)) {
        idt = inputDesc.dt;
        ic = inputDesc.dims[0];
        ih = inputDesc.dims[1];
        iw = inputDesc.dims[2];
    } else {
        CHECK_STATUS(NOT_MATCH);
    }

    // l2norm -> x / sqrt(sum(x^2))
    for (U32 c = 0; c < ic; c++) {
        for (U32 h = 0; h < ih; h++) {
            U32 index_off = (c * ih + h) * iw;
            F32 sum_row = array_var_f32(input + index_off, (I32)iw, 0.f) * static_cast<F32>(iw);
            F32 sqrt_sum_row = sqrt(sum_row);
            __m256 sqrt_sum_row_4 = _mm256_set1_ps(sqrt_sum_row);
            __m256 in, out;
            U32 w = 0;
            for (w = 0; w < iw - 7; w += 8) {
                in = _mm256_loadu_ps(input + index_off + w);
                out = _mm256_div_ps(in, sqrt_sum_row_4);
                _mm256_storeu_ps(output + index_off + w, out);
            }
            for (; w < iw; w++) {
                output[index_off + w] = input[index_off + w] / sqrt_sum_row;
            }
        }
    }
    return SUCCESS;
}
