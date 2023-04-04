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
#include "affinity_policy.h"
#include "cpu/cpu_functions.h"

inline static void concat_v1(const std::vector<TensorDesc> &inputDesc,
    const std::vector<void *> &input,
    int axis,
    const TensorDesc &outputDesc,
    void *output,
    U32 loops,
    U32 num,
    U32 tileSize,
    const std::vector<bool> &jumpMemcpy)
{
    U32 stride = 0;
    for (U32 j = 0; j < num; j++) {
        stride += inputDesc[j].dims[axis] * tileSize;
    }
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (U32 i = 0; i < loops; i++) {
        U8 *dstPtr = (U8 *)output + i * stride;
        for (U32 j = 0; j < num; j++) {
            U32 blockSize = inputDesc[j].dims[axis] * tileSize;
            if (!jumpMemcpy[j]) {
                U8 *srcPtr = ((U8 *)input[j]) + i * blockSize;
                UNI_MEMCPY(dstPtr, srcPtr, blockSize);
            }
            dstPtr += blockSize;
        }
    }
}

inline static void concat_v2(const std::vector<TensorDesc> &inputDesc,
    const std::vector<void *> &input,
    int axis,
    const TensorDesc &outputDesc,
    void *output,
    U32 loops,
    U32 num,
    U32 tileSize,
    const std::vector<bool> &jumpMemcpy)
{
    U8 *dstPtr = (U8 *)output;
    for (U32 i = 0; i < loops; i++) {
        for (U32 j = 0; j < num; j++) {
            I32 blockSize = inputDesc[j].dims[axis] * tileSize;
            if (!jumpMemcpy[j]) {
                U8 *srcPtr = ((U8 *)input[j]) + i * blockSize;
#ifdef _USE_OPENMP
                I32 bblockNum = OMP_NUM_THREADS;
                I32 bblockSize = (blockSize + bblockNum - 1) / bblockNum;
                bblockSize = UNI_MIN(32, bblockSize);
                bblockNum = (blockSize + bblockSize - 1) / bblockSize;

#pragma omp parallel for num_threads(OMP_NUM_THREADS) if (bblockNum >= OMP_NUM_THREADS)
                for (I32 k = 0; k < bblockNum; ++k) {
                    I32 copyDst = k * bblockSize;
                    UNI_MEMCPY(dstPtr + copyDst, srcPtr + copyDst,
                        UNI_MIN(bblockSize, blockSize - copyDst));
                }
#else
                UNI_MEMCPY(dstPtr, srcPtr, blockSize);
#endif
            }
            dstPtr += blockSize;
        }
    }
}

static EE concat(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    int axis,
    TensorDesc outputDesc,
    void *output,
    void *tmp)
{
    if (nullptr == output) {
        return NULL_POINTER;
    }
    if (inputDesc.size() != input.size()) {
        return NOT_MATCH;
    }
    // remove null element
    auto descIter = inputDesc.begin();
    auto dataIter = input.begin();
    while (descIter != inputDesc.end()) {
        if (nullptr == *dataIter || tensorNumElements(*descIter) == 0) {
            inputDesc.erase(descIter);
            input.erase(dataIter);
        } else {
            descIter++;
            dataIter++;
        }
    }
    U32 num = inputDesc.size();
    if (num < 1) {
        return SUCCESS;
    }

    int dim = outputDesc.nDims;
    axis = (axis + dim) % dim;
    axis = dim - 1 - axis;
    U32 tileSize = bytesOf(outputDesc.dt);
    for (I32 i = 0; i < axis; i++) {
        tileSize *= outputDesc.dims[i];
    }

    I32 loops = 1;
    for (I32 i = axis + 1; i < dim; i++) {
        loops *= outputDesc.dims[i];
    }

    if (outputDesc.df == DF_NCHWC8) {
        if (axis < (int)outputDesc.nDims - 2) {
            tileSize *= 8;
            loops /= 8;
        }
    }

    // transform data
    bool isC8 = DF_NCHWC8 == outputDesc.df;
    std::vector<bool> jumpMemcpy(num, false);
    U8 *tmpPtr = (U8 *)tmp;
    U32 outputOff = 0;
    for (U32 j = 0; j < num; j++) {
        if (((4 == inputDesc[j].nDims) &&
                ((1 != inputDesc[j].dims[1]) || (1 != inputDesc[j].dims[0]))) ||
            ((3 == inputDesc[j].nDims) && (1 != inputDesc[j].dims[0])))
        {
            if ((inputDesc[j].df != outputDesc.df)) {
                TensorDesc desc = inputDesc[j];
                desc.df = outputDesc.df;
                transformFormat(inputDesc[j], input[j], desc, tmpPtr);
                input[j] = tmpPtr;
                tmpPtr += tensorNumBytes(inputDesc[j]);
            }
        }
        outputOff += inputDesc[j].dims[axis];
    }

    if (loops > OMP_NUM_THREADS) {
        concat_v1(inputDesc, input, axis, outputDesc, output, loops, num, tileSize, jumpMemcpy);
    } else {
        concat_v2(inputDesc, input, axis, outputDesc, output, loops, num, tileSize, jumpMemcpy);
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
    void *outputScale,
    Arch arch)
{
    if (outputDesc.dt == DT_I8 && inputScale != NULL && outputScale != NULL) {
#ifdef _USE_INT8
        F32 *scales = (F32 *)inputScale;
        F32 *oscale = (F32 *)outputScale;
        CHECK_STATUS(array_minmax_value_template<F32>(scales, inputDesc.size(), 1, oscale));
        for (U32 i = 0; i < inputDesc.size(); i++) {
            CHECK_STATUS(requantize_cpu(inputDesc[i], (INT8 *)input[i], scales[i], inputDesc[i],
                (INT8 *)input[i], *oscale, arch));
        }
#else
        return NOT_SUPPORTED;
#endif
    }
    return concat(inputDesc, input, p.axis, outputDesc, output, tmp);
}
