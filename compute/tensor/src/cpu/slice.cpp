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
#include <vector>
#include "cpu/tensor_computing_cpu.h"

EE slice_cpu(TensorDesc inputDesc,
    void *input,
    SliceParamSpec p,
    std::vector<TensorDesc> outputDesc,
    std::vector<void *> *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    U32 num = outputDesc.size();
    if (num < 1) {
        return NOT_MATCH;
    }

    int dim = inputDesc.nDims;
    int axis = (p.axis + dim) % dim;
    axis = dim - 1 - axis;
    U32 tileSize = bytesOf(inputDesc.dt);
    for (I32 i = 0; i < axis; i++) {
        tileSize *= inputDesc.dims[i];
    }
    U32 loops = 1;
    for (I32 i = axis + 1; i < dim; i++) {
        loops *= inputDesc.dims[i];
    }

    if (inputDesc.df == DF_NCHWC8) {
        if (axis < 2) {
            tileSize *= 8;
            loops /= 8;
        }
    }

    U8 *ptr = (U8 *)input;
    for (U32 i = 0; i < loops; i++) {
        for (U32 j = 0; j < num; j++) {
            U32 blockSize = outputDesc[j].dims[axis] * tileSize;
            if (blockSize > 0 && nullptr == (*output)[j]) {
                CHECK_STATUS(NULL_POINTER);
            }
            U8 *dstPtr = (U8 *)((*output)[j]) + i * blockSize;
            memcpy(dstPtr, ptr, blockSize);
            ptr += blockSize;
        }
    }
    return SUCCESS;
}
