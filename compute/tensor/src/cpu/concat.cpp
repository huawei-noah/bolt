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
#if defined(_USE_NEON) && defined(_USE_INT8)
#include "cpu/arm/int8/tensor_computing_int8.h"
#endif

static EE concat(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    int axis,
    TensorDesc outputDesc,
    void *output,
    void *tmp)
{
    if (nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    U32 num = inputDesc.size();
    if (num < 1) {
        return NOT_MATCH;
    }

    int dim = outputDesc.nDims;
    axis = (axis + dim) % dim;
    axis = dim - 1 - axis;
    U32 tileSize = bytesOf(outputDesc.dt);
    for (I32 i = 0; i < axis; i++) {
        tileSize *= outputDesc.dims[i];
    }
    U32 loops = 1;
    for (I32 i = axis + 1; i < dim; i++) {
        loops *= outputDesc.dims[i];
    }

    if (outputDesc.df == DF_NCHWC8) {
        if (axis < 2) {
            tileSize *= 8;
            loops /= 8;
        }
    }

    bool isC8 = DF_NCHWC8 == outputDesc.df;

    U8 *ptr = (U8 *)output;
    U8 *tmpPtr = (U8 *)tmp;
    for (U32 i = 0; i < loops; i++) {
        for (U32 j = 0; j < num; j++) {
            U8 *inPtr = (U8 *)((input)[j]);
            if (nullptr == input[j] || tensorNumElements(inputDesc[j]) == 0) {
                continue;
            }

            if ((4 != inputDesc[j].nDims) || (1 != inputDesc[j].dims[1]) ||
                (1 != inputDesc[j].dims[0])) {
                if (isC8 && (DF_NCHW == inputDesc[j].df)) {
                    TensorDesc tmpDesc = inputDesc[j];
                    tmpDesc.df = DF_NCHWC8;
                    transformNCHWToNCHWC8(inputDesc[j], inPtr, tmpDesc, tmpPtr);
                    inPtr = tmpPtr;
                } else if (!isC8 && (DF_NCHWC8 == inputDesc[j].df)) {
                    TensorDesc tmpDesc = inputDesc[j];
                    tmpDesc.df = DF_NCHW;
                    transformToNCHW(inputDesc[j], inPtr, tmpDesc, tmpPtr);
                    inPtr = tmpPtr;
                }
            }
            U32 blockSize = inputDesc[j].dims[axis] * tileSize;
            U8 *srcPtr = inPtr + i * blockSize;
            memcpy(ptr, srcPtr, blockSize);
            ptr += blockSize;
            tmpPtr += tensorNumBytes(inputDesc[j]);
        }
    }
    return SUCCESS;
}

EE concat_cpu(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    void *inputScale,
    ConcatParamSpec p,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    void *outputScale)
{
    EE ret = NOT_SUPPORTED;
    if (outputDesc.dt == DT_I8) {
#if defined(_USE_NEON) && defined(_USE_INT8)
        ret = concat_int8(
            inputDesc, input, (F32 *)inputScale, p.axis, outputDesc, output, (F32 *)outputScale);
#endif
    } else {
        ret = concat(inputDesc, input, p.axis, outputDesc, output, tmp);
    }
    return ret;
}
