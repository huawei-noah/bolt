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
#include "cpu/tensor_computing_cpu.h"
#ifdef _USE_GENERAL
#include "cpu/general/tensor_computing_general.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif

static std::vector<U32> calculateRelativeLocalIndex_cpu(U32 *indexes, U32 *dims, U32 nDims)
{
    std::vector<U32> relativeIndexes(nDims);
    for (U32 i = 0; i < nDims; i++) {
        relativeIndexes[i] = indexes[i] % dims[i];
    }
    return relativeIndexes;
}

// [1, 10, 10] + [1, 10, 10] = [1, 10, 10]
// [1, 10, 1] + [1, 1, 10] = [1, 10, 10]
// [1, 20, 10] + [10] = [1. 20, 10] + [1, 1, 10] = [1, 20, 10]
EE eltwise_cpu(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input_,
    EltwiseParamSpec eltwiseDesc,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    Arch arch)
{
    U32 num = inputDesc.size();
    if (num <= 1 || outputDesc.nDims < 1) {
        return NOT_MATCH;
    }
    std::vector<void *> input = input_;
    U32 nchwc8Count = 0;
    U32 minDims = inputDesc[0].nDims;
    for (U32 i = 0; i < num; i++) {
        if (inputDesc[i].df == DF_NCHWC8) {
            nchwc8Count++;
        }
        if (inputDesc[i].nDims < minDims) {
            minDims = inputDesc[i].nDims;
        }
    }
    U8 *ptr = (U8 *)tmp;
    if (nchwc8Count > 0 && nchwc8Count != num) {
        for (U32 i = 0; i < num; i++) {
            if (inputDesc[i].df == DF_NCHWC8) {
                TensorDesc tmpDesc = inputDesc[i];
                tmpDesc.df = DF_NCHW;
                transformToNCHW(inputDesc[i], input[i], tmpDesc, ptr);
                inputDesc[i] = tmpDesc;
                input[i] = ptr;
                ptr += tensorNumBytes(inputDesc[i]);
                // Output from 1D-conv + 3D tensors
                if (inputDesc[i].dims[0] == 1 && minDims == 3) {
                    inputDesc[i] = tensor3df(inputDesc[i].dt, DF_NCHW,
                        inputDesc[i].dims[3], inputDesc[i].dims[2], inputDesc[i].dims[1]);
                }
            }
        }
    }

    I32 oneCount = 0;
    for (int i = 0; i < ((int)outputDesc.nDims) - 1; i++) {
        if (outputDesc.dims[i] == 1) {
            oneCount++;
        } else {
            break;
        }
    }
    TensorDesc newOutputDesc = outputDesc;
    for (int i = 0; i < (int)outputDesc.nDims - oneCount; i++) {
        newOutputDesc.dims[i] = outputDesc.dims[oneCount + i];
    }
    newOutputDesc.nDims = outputDesc.nDims - oneCount;

    std::vector<TensorDesc> newInputDesc(num);
    for (U32 i = 0; i < num; i++) {
        newInputDesc[i] = inputDesc[i];
        for (int j = 0; j < (int)inputDesc[i].nDims - oneCount; j++) {
            newInputDesc[i].dims[j] = inputDesc[i].dims[oneCount + j];
        }
        newInputDesc[i].nDims = inputDesc[i].nDims - oneCount;
        for (U32 j = newInputDesc[i].nDims; j < newOutputDesc.nDims; j++) {
            newInputDesc[i].dims[j] = 1;
        }
        newInputDesc[i].nDims = newOutputDesc.nDims;
    }
    U32 size = tensorNumElements(newOutputDesc);
    int lastDimSize = newOutputDesc.dims[0];
    std::vector<int> lastDimSizes(num);
    for (U32 i = 0; i < num; i++) {
        lastDimSizes[i] = newInputDesc[i].dims[0];
        if (lastDimSizes[i] != lastDimSize && newInputDesc[0].df == DF_NCHWC8) {
            UNI_ERROR_LOG("For NCHWc8, eltwise can only handle inputs with matching widths\n");
        }
    }
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

    std::vector<void *> newInput(num);
    EE ret = NOT_SUPPORTED;
    for (U32 i = 0; i < size; i += lastDimSize) {
        std::vector<U32> index = calculateLocalIndex(i, newOutputDesc.dims, newOutputDesc.nDims);
        for (U32 j = 0; j < num; j++) {
            std::vector<U32> relativeIndex = calculateRelativeLocalIndex_cpu(
                index.data(), newInputDesc[j].dims, newInputDesc[j].nDims);
            U32 globalIndex = calculateGlobalIndex(
                relativeIndex.data(), newInputDesc[j].dims, newInputDesc[j].nDims);
            newInput[j] = (U8 *)(input[j]) + globalIndex * bytesOf(newInputDesc[j].dt);
        }
        U8 *newOutput = (U8 *)output + i * bytesOf(newOutputDesc.dt);
        if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
            ret = eltwise_general(newOutputDesc.dt, newInput, lastDimSizes, num, lastDimSize,
                newOutput, eltwiseDesc.elt_mode);
#endif
#ifdef _USE_NEON
        } else if (IS_ARM(arch)) {
            ret = eltwise_arm(newOutputDesc.dt, newInput, lastDimSizes, num, lastDimSize, newOutput,
                eltwiseDesc.elt_mode);
#endif
#ifdef _USE_X86
        } else if (IS_X86_AVX2(arch)) {
            ret = eltwise_x86(newOutputDesc.dt, newInput, lastDimSizes, num, lastDimSize, newOutput,
                eltwiseDesc.elt_mode);
#endif
        }
    }
    if (ret == SUCCESS && eltwiseDesc.activation_type != ACTIVATION_NULL) {
        ActivationParamSpec p;
        p.mode = eltwiseDesc.activation_type;
        ret = activation_cpu(outputDesc, output, p, outputDesc, output, arch);
    }
    return ret;
}
