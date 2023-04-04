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
#include "tensor_transpose.h"

EE transpose_cpu(
    TensorDesc inputDesc, U32 *inDim, const void *input, U32 *dim, TensorDesc outputDesc, U32 *outDim, void *output)
{
    if (nullptr == input && tensorNumElements(inputDesc) == 0) {
        return SUCCESS;
    }
    if (nullptr == input || nullptr == output || nullptr == dim) {
        return NULL_POINTER;
    }
    array_transpose(bytesOf(inputDesc.dt), inDim, input, outDim, output, dim,
        inputDesc.nDims, outputDesc.nDims);
    return SUCCESS;
}
