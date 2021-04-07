// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_ARRAY_TRANSPOSE
#define _H_ARRAY_TRANSPOSE

#include "string.h"

inline void array_transpose(unsigned int elementSize,
    unsigned int *inputDims,
    const void *input,
    unsigned int *outputDims,
    void *output,
    unsigned int *transposeDims,
    int inputDimsNum,
    int outputDimsNum)
{
    unsigned int inputSize = 1, outputSize = 1;
    for (int i = 0; i < inputDimsNum; i++) {
        inputSize *= inputDims[i];
    }
    for (int i = 0; i < outputDimsNum; i++) {
        outputSize *= outputDims[i];
    }
    CHECK_REQUIREMENT(inputSize == outputSize);

    unsigned int sizeInner = 1;
    int sizeInnerIndex = 0;
    for (int i = outputDimsNum - 1; i >= 0; i--) {
        if ((int)transposeDims[i] == i) {
            sizeInner *= inputDims[inputDimsNum - 1 - i];
            sizeInnerIndex++;
        } else {
            break;
        }
    }
    outputSize = outputSize / sizeInner;

    std::vector<unsigned int> inputLocalIndex(inputDimsNum, 0);
    const char *inputPtr = (const char *)input;
    char *outputPtr = (char *)output;
    unsigned int tileSize = sizeInner * elementSize;
    for (unsigned int i = 0; i < outputSize; i++) {
        unsigned int outputIndex = i;
        for (int j = sizeInnerIndex; j < outputDimsNum; j++) {
            unsigned int value = outputIndex % outputDims[j];
            outputIndex /= outputDims[j];
            inputLocalIndex[inputDimsNum - 1 - transposeDims[outputDimsNum - 1 - j]] = value;
        }
        unsigned int inputIndex = 0;
        for (int j = inputDimsNum - 1; j > sizeInnerIndex; j--) {
            inputIndex = (inputIndex + inputLocalIndex[j]) * inputDims[j - 1];
        }
        inputIndex += inputLocalIndex[sizeInnerIndex];
        memcpy(outputPtr + i * tileSize, inputPtr + inputIndex * tileSize, tileSize);
    }
}

inline void array_transpose_naive(unsigned int elementSize,
    unsigned int *inputDims,
    const void *input,
    unsigned int *outputDims,
    void *output,
    unsigned int *transposeDims,
    int dimsNum)
{
    if (dimsNum <= 1) {
        return;
    }
    unsigned int inputSize = 1, outputSize = 1;
    for (int i = 0; i < dimsNum; i++) {
        inputSize *= inputDims[i];
        outputSize *= outputDims[i];
    }
    std::vector<unsigned int> inputLocalIndex(dimsNum);
    const char *inputPtr = (const char *)input;
    char *outputPtr = (char *)output;
    for (unsigned int i = 0; i < outputSize; i++) {
        unsigned int outputIndex = i;
        for (int j = 0; j < dimsNum; j++) {
            unsigned int value = outputIndex % outputDims[j];
            outputIndex /= outputDims[j];
            inputLocalIndex[dimsNum - 1 - transposeDims[dimsNum - 1 - j]] = value;
        }
        unsigned int inputIndex = 0;
        for (int j = dimsNum - 1; j > 0; j--) {
            inputIndex = (inputIndex + inputLocalIndex[j]) * inputDims[j - 1];
        }
        inputIndex += inputLocalIndex[0];
        memcpy(outputPtr + i * elementSize, inputPtr + inputIndex * elementSize, elementSize);
    }
}
#endif
