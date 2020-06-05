// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "cpu/arm/tensor_computing_arm.h"
#ifdef _USE_INT8
#include "cpu/arm/int8/tensor_computing_int8.h"
#endif
#include <cstring>

EE concat(std::vector<TensorDesc> inputDesc, std::vector<void*> input, TensorDesc outputDesc, void* output, int axis)
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

    // DF should either all be NCHWC8, or all be non-C8
    bool isC8 = DF_NCHWC8 == outputDesc.df;

    U8 *ptr = (U8 *)output;
    for (U32 i = 0; i < loops; i++) {
        for (U32 j = 0; j < num; j++) {
            if (nullptr == input[j]) {
                CHECK_STATUS(NULL_POINTER);
            }
            if (isC8) {
                if (DF_NCHWC8 != inputDesc[j].df) {
                    CHECK_REQUIREMENT(4 == inputDesc[j].nDims);
                    CHECK_REQUIREMENT(1 == inputDesc[j].dims[1] && 1 == inputDesc[j].dims[0]);
                }
            } else {
                if (DF_NCHWC8 == inputDesc[j].df) {
                    CHECK_REQUIREMENT(4 == inputDesc[j].nDims);
                    CHECK_REQUIREMENT(1 == inputDesc[j].dims[1] && 1 == inputDesc[j].dims[0]);
                }
            }
            U32 blockSize = inputDesc[j].dims[axis] * tileSize;
            U8* srcPtr = (U8*)((input)[j]) + i * blockSize;
            memcpy(ptr, srcPtr, blockSize);
            ptr += blockSize;
        }
    }
    return SUCCESS;
}

EE concat_arm(std::vector<TensorDesc> inputDesc, std::vector<void*> input, void* inputScale,
    TensorDesc outputDesc, void* output, void* outputScale, int axis)
{
    EE ret = SUCCESS;
    switch (outputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            UNUSED(inputScale);
            UNUSED(outputScale);
            ret = concat(inputDesc, input,
                         outputDesc, output,
                         axis);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            UNUSED(inputScale);
            UNUSED(outputScale);
            ret = concat(inputDesc, input,
                         outputDesc, output,
                         axis);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            ret = concat_int8(inputDesc, input, (F32*)inputScale,
                              outputDesc, output, (F32*)outputScale,
                              axis);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
