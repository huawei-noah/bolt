// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <vector>
#include "cpu/arm/tensor_computing_arm.h"
#ifdef _USE_FP32
#include "cpu/arm/fp32/tensor_computing_fp32.h"
#endif
#ifdef _USE_FP16
#include "cpu/arm/fp16/tensor_computing_fp16.h"
#endif

std::vector<int> calculateLocalIndex_arm(U32 index, TensorDesc desc) {
    std::vector<int> indexes(desc.nDims);
    for (U32 i = 0; i < desc.nDims; i++) {
        indexes[i] = index % desc.dims[i];
        index /= desc.dims[i];
    }
    return indexes;
}

U32 calculateGlobalIndex_arm(std::vector<int> indexes, TensorDesc desc) {
    U32 index = 0;
    for (int i = ((int)desc.nDims) - 1; i >= 0; i--) {
        index = index * desc.dims[i] + indexes[i];
    }
    return index;
    
}

std::vector<int> calculateRelativeLocalIndex_arm(std::vector<int> indexes, TensorDesc desc) {
    std::vector<int> relativeIndexes(desc.nDims);
    for (U32 i = 0; i < desc.nDims; i++) {
        relativeIndexes[i] = indexes[i] % desc.dims[i];
    }
    return relativeIndexes;
}

// [1, 10, 10] + [1, 10, 10] = [1, 10, 10]
// [1, 10, 1] + [1, 1, 10] = [1, 10, 10]
// [1, 20, 10] + [10] = [1. 20, 10] + [1, 1, 10] = [1, 20, 10]
EE eltwise_arm(std::vector<TensorDesc> inputDesc, std::vector<void*> input,
               TensorDesc outputDesc, void* output, EltwiseMode eltwiseMode) {
    U32 num = inputDesc.size();
    if(num <= 1 || outputDesc.nDims < 1) return NOT_MATCH;
    I32 oneCount = 0;
    for (int i = 0; i < ((int)outputDesc.nDims)-1; i++) {
        if(outputDesc.dims[i] == 1)
            oneCount ++;
        else
            break;
    }
    TensorDesc newOutputDesc = outputDesc;
    for (int i = 0; i < (int)outputDesc.nDims - oneCount; i++)
        newOutputDesc.dims[i] =  outputDesc.dims[oneCount+i];
    newOutputDesc.nDims = outputDesc.nDims - oneCount;

    std::vector<TensorDesc> newInputDesc(num);
    for (U32 i = 0; i < num; i++) {
        newInputDesc[i] = inputDesc[i];
        for (int j = 0; j < (int)inputDesc[i].nDims - oneCount; j++)
            newInputDesc[i].dims[j] =  inputDesc[i].dims[oneCount+j];
        newInputDesc[i].nDims = inputDesc[i].nDims - oneCount;
        for (U32 j = newInputDesc[i].nDims; j < newOutputDesc.nDims; j++) {
            newInputDesc[i].dims[j] = 1;
        }
        newInputDesc[i].nDims = newOutputDesc.nDims;
    }
    U32 size = tensorNumElements(newOutputDesc);
    U32 lastDimSize = newOutputDesc.dims[0];
    std::vector<int> lastDimSizes(num);
    for (U32 i = 0; i < num; i++)
        lastDimSizes[i] = newInputDesc[i].dims[0];
    for (U32 i = 1; i < newOutputDesc.nDims; i++) {
        bool sameDim = true;
        for (U32 j = 0; j < num; j++) {
            if (newInputDesc[j].dims[i] != newOutputDesc.dims[i]) {
                sameDim = false;
                break;
            }
        }
        if (sameDim) {
            lastDimSize *= newOutputDesc.dims[i];
            for (U32 j = 0; j < num; j++) {
                lastDimSizes[j] *= newInputDesc[j].dims[i];
            }
        } else {
            break;
        }
    }

    std::vector<void*> newInput(num);
    EE ret = SUCCESS;
    for (U32 i = 0; i < size; i+=lastDimSize) {
        std::vector<int> index = calculateLocalIndex_arm(i, newOutputDesc);
        for (U32 j = 0; j < num; j++) {
            std::vector<int> relativeIndex = calculateRelativeLocalIndex_arm(index, newInputDesc[j]);
            U32 globalIndex = calculateGlobalIndex_arm(relativeIndex, newInputDesc[j]);
            newInput[j] = (U8*)(input[j]) + globalIndex * bytesOf(newInputDesc[j].dt);
        }
        U8* newOutput = (U8*)output + i * bytesOf(newOutputDesc.dt);
        switch (newOutputDesc.dt) {
#ifdef _USE_FP32
            case DT_F32: {
                ret = eltwise_fp32(newInput, lastDimSizes, num, lastDimSize, newOutput, eltwiseMode);
                break;
            }
#endif
#ifdef _USE_FP16
            case DT_F16: {
                ret = eltwise_fp16(newInput, lastDimSizes, num, lastDimSize, newOutput, eltwiseMode);
                break;
            }
#endif
            default:
                ret = NOT_SUPPORTED;
                break;
        }

    }
    return ret;
}
