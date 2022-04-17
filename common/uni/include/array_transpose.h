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

#include "secure_c_wrapper.h"
#include "affinity_policy.h"

template <int branch, typename T>
static inline void inner_transpose_template(unsigned int tileSize,
    unsigned int *inputDims,
    const T *input,
    unsigned int *outputDims,
    T *output,
    unsigned int *transposeDims,
    int inputDimsNum,
    int outputDimsNum,
    unsigned int outputSize,
    int sizeInnerIndex)
{
#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
#endif
    {
        std::vector<unsigned int> inputLocalIndex(inputDimsNum);
#ifdef _USE_OPENMP
#pragma omp for
#endif
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
            if (branch == 0) {
                *(output + i) = *(input + inputIndex);
            } else {
                UNI_MEMCPY(output + i * tileSize, input + inputIndex * tileSize, tileSize);
            }
        }
    }
}

inline void array_transpose(unsigned int elementSize,
    unsigned int *inputDims,
    const void *input,
    unsigned int *outputDims,
    void *output,
    unsigned int *transposeDims,
    int inputDimsNum,
    int outputDimsNum)
{
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
    int tileSize = elementSize * sizeInner;
    int in = inputDims[inputDimsNum - 1], ihiw = 0, ic = 0;
    if (outputDimsNum - sizeInnerIndex == 3 && transposeDims[0] == 0 && transposeDims[1] == 2 &&
        transposeDims[2] == 1) {
        ic = inputDims[inputDimsNum - 2];
        ihiw = inputDims[inputDimsNum - 3];
    }
    if (outputDimsNum - sizeInnerIndex == 4 && transposeDims[0] == 0 && transposeDims[1] == 2 &&
        transposeDims[2] == 3 && transposeDims[3] == 1) {
        ic = inputDims[inputDimsNum - 2];
        ihiw = inputDims[inputDimsNum - 3] * inputDims[inputDimsNum - 4];
    }
    if (ic > 0 && ihiw > 0 && input != output) {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (int o = 0; o < in * ihiw; o++) {
            int n = o / ihiw;
            int hw = o % ihiw;
            U8 *dst = (U8 *)output + o * ic * tileSize;
            for (int c = 0; c < ic; c++, dst += tileSize) {
                const U8 *src = (const U8 *)input + ((n * ic + c) * ihiw + hw) * tileSize;
                UNI_MEMCPY(dst, src, tileSize);
            }
        }
        return;
    }

    unsigned int inputSize = 1, outputSize = 1;
    for (int i = 0; i < inputDimsNum; i++) {
        inputSize *= inputDims[i];
    }
    for (int i = 0; i < outputDimsNum; i++) {
        outputSize *= outputDims[i];
    }
    CHECK_REQUIREMENT(inputSize == outputSize);
    outputSize = outputSize / sizeInner;

    const char *inputPtr = (const char *)input;
    char *outputPtr = (char *)output;
    if (sizeInner == 1 && elementSize == 4) {
        inner_transpose_template<0, int>(elementSize, inputDims, (const int *)input, outputDims,
            (int *)output, transposeDims, inputDimsNum, outputDimsNum, outputSize, sizeInnerIndex);
    } else if (sizeInner == 1 && elementSize == 2) {
        inner_transpose_template<0, short>(elementSize, inputDims, (const short *)input, outputDims,
            (short *)output, transposeDims, inputDimsNum, outputDimsNum, outputSize, sizeInnerIndex);
    } else {
        inner_transpose_template<1, char>(tileSize, inputDims, (const char *)input, outputDims,
            (char *)output, transposeDims, inputDimsNum, outputDimsNum, outputSize, sizeInnerIndex);
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
    const char *inputPtr = (const char *)input;
    char *outputPtr = (char *)output;
#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
#endif
    {
        std::vector<unsigned int> inputLocalIndex(dimsNum);
#ifdef _USE_OPENMP
#pragma omp for
#endif
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
            UNI_MEMCPY(
                outputPtr + i * elementSize, inputPtr + inputIndex * elementSize, elementSize);
        }
    }
}
#endif
