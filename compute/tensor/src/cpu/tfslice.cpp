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

EE tfslice_infer_output_size_cpu(TensorDesc inputDesc, TfSliceParamSpec p, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    int *begin = p.begin;
    int *end = p.end;
    int *strides = p.strides;
    char *beginMask = p.begin_mask;
    char *endMask = p.end_mask;
    U32 dimSize = inputDesc.nDims;

    CHECK_REQUIREMENT(dimSize == inputDesc.nDims);
    *outputDesc = inputDesc;
    for (U32 i = 0; i < dimSize; i++) {
        int axis = dimSize - 1 - i;
        int axisBegin = (beginMask[i] == 1) ? 0 : begin[i];
        int axisEnd = (endMask[i] == 1) ? inputDesc.dims[axis] : end[i];
        if (axisBegin < 0) {
            axisBegin = inputDesc.dims[axis] + axisBegin;
            axisBegin = UNI_MAX(axisBegin, 0);
        }
        if (axisEnd < 0) {
            axisEnd = inputDesc.dims[axis] + axisEnd;
        } else if (axisEnd > (int)(inputDesc.dims[axis])) {
            axisEnd = inputDesc.dims[axis];
        }
        CHECK_REQUIREMENT(axisBegin >= 0 && axisEnd >= 0);
        int num = (axisEnd - axisBegin) / strides[i];
        outputDesc->dims[axis] = num;
        begin[i] = axisBegin;
        end[i] = axisEnd;
    }
    if (inputDesc.df == DF_NCHWC8) {
        int channelAxis = 1;
        if (begin[channelAxis] % 8 != 0 || strides[channelAxis] != 1 ||
            (end[channelAxis] - begin[channelAxis]) / strides[channelAxis] % 8 != 0) {
            outputDesc->df = DF_NCHW;
        }
    }
    return SUCCESS;
}

#define TFSLICE_USE_RECURSIVE
#ifdef TFSLICE_USE_RECURSIVE
inline static void recursive_tfslice(U8 *src,
    U32 *srcDims,
    U32 srcNum,
    U8 *dst,
    U32 *dstDims,
    U32 dstNum,
    int *begin,
    int *end,
    int *strides,
    int i,
    int bound,
    int dimNum,
    U32 tileSize)
{
    if (i == bound) {
        memcpy(dst, src, tileSize);
        return;
    }
    U32 newSrcNum = srcNum / srcDims[dimNum - 1 - i];
    U32 newDstNum = dstNum / dstDims[dimNum - 1 - i];
    if (i + 1 == bound && strides[i] == 1) {
        memcpy(dst, src + begin[i] * newSrcNum, tileSize * (end[i] - begin[i]));
        return;
    }
    for (int j = begin[i]; j < end[i]; j += strides[i]) {
        U8 *newSrc = src + j * newSrcNum;
        recursive_tfslice(newSrc, srcDims, newSrcNum, dst, dstDims, newDstNum, begin, end, strides,
            i + 1, bound, dimNum, tileSize);
        dst += newDstNum;
    }
}
#endif

EE tfslice_cpu(
    TensorDesc inputDesc, void *input, TfSliceParamSpec p, TensorDesc outputDesc, void *output)
{
    int *begin = p.begin;
    int *end = p.end;
    int *strides = p.strides;
    char *beginMask = p.begin_mask;
    char *endMask = p.end_mask;
    U32 dimSize = inputDesc.nDims;
    for (U32 i = 0; i < dimSize; i++) {
        int axis = dimSize - 1 - i;
        int axisBegin = (beginMask[i] == 1) ? 0 : begin[i];
        int axisEnd = (endMask[i] == 1) ? inputDesc.dims[axis] : end[i];
        if (axisBegin < 0) {
            axisBegin = inputDesc.dims[axis] + axisBegin;
            axisBegin = UNI_MAX(axisBegin, 0);
        }
        if (axisEnd < 0) {
            axisEnd = inputDesc.dims[axis] + axisEnd;
        } else if (axisEnd > (int)(inputDesc.dims[axis])) {
            axisEnd = inputDesc.dims[axis];
        }
        CHECK_REQUIREMENT(axisBegin >= 0 && axisEnd >= 0);
        begin[i] = axisBegin;
        end[i] = axisEnd;
    }

    U32 num = tensorNumElements(outputDesc);
    U8 *dst = (U8 *)output;
    U32 elementSize = bytesOf(inputDesc.dt);
    int channelAxis = inputDesc.nDims - 2;
    if (inputDesc.df == outputDesc.df) {
        std::vector<U32> tmpInputDims(inputDesc.nDims), tmpOutputDims(outputDesc.nDims);
        memcpy(tmpInputDims.data(), inputDesc.dims, inputDesc.nDims * sizeof(U32));
        memcpy(tmpOutputDims.data(), outputDesc.dims, outputDesc.nDims * sizeof(U32));
        int startAxis = 0;
        int elementNum = 1;
        if (inputDesc.df == DF_NCHWC8) {
            elementNum *= 8;
            begin[1] /= 8;
            end[1] /= 8;
            tmpInputDims[channelAxis] /= 8;
            tmpOutputDims[channelAxis] /= 8;
            tmpInputDims.insert(tmpInputDims.begin(), 8);
            tmpOutputDims.insert(tmpOutputDims.begin(), 8);
#ifndef TFSLICE_USE_RECURSIVE
            startAxis = 1;
#endif
        }
        // aggregate dimension
        int i;
        for (i = dimSize - 1; i >= 0; i--) {
            int reverseAxis = dimSize - 1 - i;
            if (begin[i] == 0 && end[i] == (int)tmpInputDims[reverseAxis] && strides[i] == 1) {
                elementNum *= (end[i] - begin[i]);
            } else {
                break;
            }
        }
        U32 tileSize = elementSize * elementNum;
#ifdef TFSLICE_USE_RECURSIVE
        recursive_tfslice((U8 *)input, tmpInputDims.data(), tensorNumBytes(inputDesc), dst,
            tmpOutputDims.data(), tensorNumBytes(outputDesc), begin, end, strides, 0, i + 1,
            tmpInputDims.size(), tileSize);
#else
        for (U32 i = 0; i < num; i += elementNum, dst += tileSize) {
            std::vector<U32> localIndex =
                calculateLocalIndex(i, tmpOutputDims.data(), tmpOutputDims.size());
            for (U32 j = 0; j < dimSize; j++) {
                int reverseAxis = dimSize - 1 - j;
                localIndex[startAxis + j] =
                    localIndex[startAxis + j] * strides[reverseAxis] + begin[reverseAxis];
            }
            U32 srcIndex =
                calculateGlobalIndex(localIndex.data(), tmpInputDims.data(), tmpInputDims.size());
            U8 *src = (U8 *)input + srcIndex * elementSize;
            memcpy(dst, src, tileSize);
        }
#endif
        if (inputDesc.df == DF_NCHWC8) {
            begin[1] *= 8;
            end[1] *= 8;
        }
    } else {
        CHECK_REQUIREMENT(inputDesc.df == DF_NCHWC8);
        U32 tmpNDims = inputDesc.nDims + 1;
        std::vector<U32> tmpDims(tmpNDims);
        tmpDims[0] = 8;
        memcpy(&(tmpDims[1]), inputDesc.dims, inputDesc.nDims * sizeof(U32));
        for (U32 i = 0; i < num; i++, dst += elementSize) {
            std::vector<U32> localIndex = calculateLocalIndex(i, outputDesc.dims, outputDesc.nDims);
            for (U32 j = 0; j < dimSize; j++) {
                int reverseAxis = dimSize - 1 - j;
                localIndex[j] = localIndex[j] * strides[reverseAxis] + begin[reverseAxis];
            }
            int c8 = localIndex[channelAxis] % 8;
            localIndex[channelAxis] /= 8;
            localIndex.insert(localIndex.begin(), c8);
            U32 index = calculateGlobalIndex(localIndex.data(), tmpDims.data(), tmpNDims);
            U8 *src = (U8 *)input + index * elementSize;
            memcpy(dst, src, elementSize);
        }
    }
    return SUCCESS;
}
