// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <set>
#include <vector>
#include "cpu/tensor_computing_cpu.h"

EE slice_cpu(TensorDesc inputDesc,
    void *input,
    SliceParamSpec p,
    std::vector<TensorDesc> &outputDesc,
    std::vector<void *> &output)
{
    if (nullptr == input) {
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

    bool sameFormat = true;
    for (U32 j = 0; j < num; j++) {
        if (!isSameDataFormat(inputDesc.df, outputDesc[j].df)) {
            sameFormat = false;
            break;
        }
    }

    if (sameFormat && inputDesc.df == DF_NCHWC8) {
        if (axis < dim - 2) {
            tileSize *= 8;
            loops /= 8;
        }
    }

    if (sameFormat) {
        U8 *ptr = (U8 *)input;
        for (U32 i = 0; i < loops; i++) {
            for (U32 j = 0; j < num; j++) {
                U32 blockSize = outputDesc[j].dims[axis] * tileSize;
                if (blockSize > 0 && nullptr == output[j]) {
                    CHECK_STATUS(NULL_POINTER);
                }
                U8 *dstPtr = (U8 *)(output[j]) + i * blockSize;
                UNI_MEMCPY(dstPtr, ptr, blockSize);
                ptr += blockSize;
            }
        }
    } else {
        if (axis != dim - 2) {
            CHECK_STATUS(NOT_SUPPORTED);
            return NOT_SUPPORTED;
        }
        U8 *iPtr = (U8 *)input;
        U32 eleSize = bytesOf(inputDesc.dt);
        tileSize /= eleSize;
        U32 startDims = 0;
        U32 endDims = 0;

        for (U32 j = 0; j < num; j++) {
            endDims += outputDesc[j].dims[axis];
            U8 *oPtr = (U8 *)output[j];
            if (inputDesc.df == DF_NCHWC8 && outputDesc[j].df != DF_NCHWC8) {
                for (U32 i = 0; i < loops; i++) {
                    for (U32 d = startDims; d < endDims; ++d) {
                        U32 c8 = d % 8;
                        U32 c = d - c8;
                        for (U32 t = 0; t < tileSize; ++t) {
                            U32 oIdx = i * tileSize * (endDims - startDims) +
                                (d - startDims) * tileSize + t;
                            U32 iIdx =
                                i * tileSize * inputDesc.dims[axis] + c * tileSize + t * 8 + c8;
                            UNI_MEMCPY(oPtr + oIdx * eleSize, iPtr + iIdx * eleSize, eleSize);
                        }
                    }
                }
            } else {
                CHECK_STATUS(NOT_SUPPORTED);
                return NOT_SUPPORTED;
            }
            startDims = endDims;
        }
    }
    return SUCCESS;
}
