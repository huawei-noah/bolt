// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <string.h>
#include "cpu/tensor_computing_cpu.h"

EE reshape_infer_output_size_cpu(TensorDesc inputDesc, ReshapeParamSpec p, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        return NULL_POINTER;
    }
    I32 *shape = p.shape_dims;
    I32 shape_size = p.shape_size;
    int inputElementNum = tensorNumElements(inputDesc);
    int outputElementNum = 1;
    for (int i = 0; i < shape_size; i++) {
        outputElementNum *= shape[i];
    }
    int index_range = ((int)inputDesc.nDims > shape_size) ? shape_size : inputDesc.nDims;
    if (inputElementNum > 0 && outputElementNum > 0 && inputElementNum != outputElementNum) {
        for (int i = 0; i < index_range; i++) {
            if ((inputElementNum / (int)inputDesc.dims[inputDesc.nDims - 1 - i]) ==
                (outputElementNum / shape[i])) {
                shape[i] = inputDesc.dims[inputDesc.nDims - 1 - i];
                break;
            }
        }
    }

    *outputDesc = inputDesc;
    (*outputDesc).nDims = shape_size;
    if (shape_size == 2) {
        (*outputDesc).df = DF_NORMAL;
    }
    if (shape_size == 3) {
        (*outputDesc).df = DF_MTK;
    }
    if (shape_size >= 4) {
        (*outputDesc).df = DF_NCHW;
    }

    U32 factor = 1;
    I32 count = 0;
    for (I32 i = 0; i < shape_size; i++) {
        I32 value = shape[i];
        if (value == 0) {
            value = inputDesc.dims[inputDesc.nDims - 1 - i];
        }
        if (value == -1) {
            value = 0;
            count++;
        } else {
            factor *= value;
        }

        (*outputDesc).dims[shape_size - 1 - i] = value;
    }
    if (count > 1) {
        return NOT_SUPPORTED;
    }

    bool sameDim = ((*outputDesc).nDims == inputDesc.nDims);
    for (I32 i = 0; i < shape_size; i++) {
        if ((*outputDesc).dims[i] == 0) {
            (*outputDesc).dims[i] = tensorNumElements(inputDesc) / factor;
        }
        if ((*outputDesc).dims[i] != inputDesc.dims[i]) {
            sameDim = false;
        }
    }
    if (sameDim) {
        (*outputDesc).df = inputDesc.df;
    }

    return SUCCESS;
}

EE reshape_cpu(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }

    if (tensorNumElements(inputDesc) != tensorNumElements(outputDesc)) {
        // Only allow the removal of padded convolution channels
        CHECK_REQUIREMENT(DF_NCHWC8 == inputDesc.df);
        CHECK_REQUIREMENT(tensorNumElements(inputDesc) >= tensorNumElements(outputDesc));
        inputDesc.df = DF_NCHW;
    }

    bool sameDim = (outputDesc.nDims == inputDesc.nDims);
    for (U32 i = 0; i < inputDesc.nDims; i++) {
        if (outputDesc.dims[i] != inputDesc.dims[i]) {
            sameDim = false;
        }
    }

    if ((DF_NCHWC8 != inputDesc.df && DF_NCHWC16 != inputDesc.df) || sameDim) {
        if (output != input) {
            memcpy(output, input, tensorNumBytes(outputDesc));
        }
    } else {
        CHECK_REQUIREMENT(input != output);
        DataType idt;
        DataFormat idf;
        U32 in, ic, ih, iw;
        if (tensorIs4d(inputDesc)) {
            CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        } else if (tensorIs3d(inputDesc)) {
            CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
            iw = 1;
        } else {
            return NOT_SUPPORTED;
        }

        U32 cx = (DF_NCHWC8 == inputDesc.df) ? 8 : 16;
        U32 elementBytes = bytesOf(idt);
        ic /= cx;
        U8 *inPtr = (U8 *)input;
        U8 *outPtr = (U8 *)output;
        for (U32 n = 0; n < in; n++) {
            for (U32 c = 0; c < ic; c++) {
                for (U32 hw = 0; hw < ih * iw; hw++) {
                    for (U32 c8 = 0; c8 < cx; c8++) {
                        memcpy(outPtr +
                                elementBytes * (n * ic * cx * ih * iw + (c * cx + c8) * ih * iw + hw),
                            inPtr +
                                elementBytes *
                                    (n * ic * ih * iw * cx + c * ih * iw * cx + hw * cx + c8),
                            elementBytes);
                    }
                }
            }
        }
    }
    return SUCCESS;
}
