// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/tensor_computing_cpu.h"
#include "cpu/cpu_functions.h"

EE l2normalization_cpu(
    TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output, Arch arch)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    ArrayVarFunction var_func = get_array_var_function(arch);
    ArrayScaleFunction scale_func = get_array_scale_function(arch);
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
            U32 index_off = (c * ih + h) * iw * bytesOf(idt);
            const U8 *input_ptr = (const U8 *)input + index_off;
            U8 *output_ptr = (U8 *)output + index_off;
            F32 sum_row = var_func(idt, input_ptr, (I32)iw, 0.f) * static_cast<F32>(iw);
            scale_func(idt, input_ptr, output_ptr, iw, 1.0 / sqrt(sum_row), 0);
        }
    }
    return SUCCESS;
}
