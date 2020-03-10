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

#include "cpu/arm/tensor_computing_arm.h"

EE transpose_arm(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output, U32 *dim) {
    if (nullptr == input || nullptr == output || nullptr == dim)
        CHECK_STATUS(NULL_POINTER);

    I32 inputDim = inputDesc.nDims;
    I32 outputDim = outputDesc.nDims;
    CHECK_REQUIREMENT(inputDim == outputDim);

    U32 size_inner = 1;
    I32 size_inner_index = 0;
    for (I32 i = inputDim-1; i >= 0; i--) {
        if ((I32)dim[i] == i) {
            size_inner *= inputDesc.dims[inputDim-1-i];
            size_inner_index++;
        } else {
            break;
        }
    }
    U32 inputSize = tensorNumElements(inputDesc) / size_inner;
    U32 outputSize = tensorNumElements(outputDesc) / size_inner;
    CHECK_REQUIREMENT(inputSize == outputSize);

    std::vector<U32> inputLocalIndex(inputDim);
    U8 *input_ptr = (U8 *)input;
    U8 *output_ptr = (U8 *)output;
    for (U32 i = 0; i < outputSize; i++) {
        U32 outputIndex = i;
        for (I32 j = size_inner_index; j < outputDim; j++) {
            U32 value = outputIndex % outputDesc.dims[j];
            outputIndex /= outputDesc.dims[j];
            inputLocalIndex[inputDim - 1 - dim[outputDim - 1 - j]] = value;
        }
        U32 inputIndex = 0;
        for (I32 j = inputDim-1; j > size_inner_index; j--) {
             inputIndex = (inputIndex + inputLocalIndex[j]) * inputDesc.dims[j-1];
        }
        inputIndex += inputLocalIndex[size_inner_index];
        memcpy(output_ptr+i*size_inner*bytesOf(outputDesc.dt),
               input_ptr+inputIndex*size_inner*bytesOf(inputDesc.dt),
               size_inner*bytesOf(inputDesc.dt));
    }

    return SUCCESS;
}

EE transpose_naive(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output, U32 *dim) {
    if (nullptr == input || nullptr == output || nullptr == dim)
        CHECK_STATUS(NULL_POINTER);

    U32 inputDim = inputDesc.nDims;
    U32 outputDim = outputDesc.nDims;
    CHECK_REQUIREMENT(inputDim == outputDim);

    U32 outputSize = tensorNumElements(outputDesc);
    CHECK_REQUIREMENT(inputDim == outputDim);
    std::vector<U32> inputLocalIndex(inputDim);
    U8 *input_ptr = (U8 *)input;
    U8 *output_ptr = (U8 *)output;
    for (U32 i = 0; i < outputSize; i++) {
        U32 outputIndex = i;
        for (U32 j = 0; j < outputDim; j++) {
            U32 value = outputIndex % outputDesc.dims[j];
            outputIndex /= outputDesc.dims[j];
            inputLocalIndex[inputDim - 1 - dim[outputDim - 1 - j]] = value;
        }
        U32 inputIndex = 0;
        for (U32 j = inputDim-1; j > 0; j--) {
             inputIndex = (inputIndex + inputLocalIndex[j]) * inputDesc.dims[j-1];
        }
        inputIndex += inputLocalIndex[0];
        memcpy(output_ptr+i*bytesOf(outputDesc.dt), input_ptr+inputIndex*bytesOf(inputDesc.dt), bytesOf(inputDesc.dt));
    }

    return SUCCESS;
}
