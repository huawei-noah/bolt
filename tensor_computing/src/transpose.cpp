// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "tensor_computing.h"
#include "cpu/general/tensor_computing_general.h"
#include "cpu/arm/tensor_computing_arm.h"

EE transpose(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output, U32 *dim, Arch arch) {
    EE ret = SUCCESS;
    switch (arch) {
        case CPU_GENERAL:
            ret = transpose_general(inputDesc, input, outputDesc, output, dim);
            break;
        case ARM_A55:
            ret = transpose_arm(inputDesc, input, outputDesc, output, dim);
            break;
        case ARM_A76:
            ret = transpose_arm(inputDesc, input, outputDesc, output, dim);
            break;
        case ARM_V8:
            ret = transpose_arm(inputDesc, input, outputDesc, output, dim);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE transpose_infer_output_size(TensorDesc inputDesc, TensorDesc *outputDesc, U32 *dim) {
    if (nullptr == outputDesc || nullptr == dim)
        CHECK_STATUS(NULL_POINTER);

    *outputDesc = inputDesc;
    U32 inputDim = inputDesc.nDims;
    U32 outputDim = (*outputDesc).nDims;
    for (U32 i = 0; i < inputDim; i++) {
        CHECK_REQUIREMENT(dim[i] < inputDim);
        // NOTE: TensorDesc.dims array is in [W H C N] order.
        // so if you want to transpose [N C H W] format data, we use (dims - 1 - *)
        // [5 6 7 8] + [0 3 2 1] = [5 8 7 6]
        // [8 7 6 5] + [0 3 2 1] = [6 7 8 5]
        (*outputDesc).dims[outputDim - 1 - i] = inputDesc.dims[inputDim - 1 - dim[i]];
    }
    return SUCCESS;
}
