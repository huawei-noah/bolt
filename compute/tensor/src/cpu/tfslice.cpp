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

bool isSingleCStepMode(TensorDesc inputDesc, TfSliceParamSpec p)
{
    int nDims = inputDesc.nDims;
    if (nDims < 3) {
        return false;
    }
    for (int i = 0; i < nDims; ++i) {
        int pi = nDims - 1 - i;
        if ((i != nDims - 2) &&
            (p.begin[pi] != 0 || p.end[pi] != int(inputDesc.dims[i]) || p.strides[pi] != 1)) {
            return false;
        }
    }
    return true;
}

EE tfslice_infer_output_size_cpu(TensorDesc inputDesc, TfSliceParamSpec p, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        return NULL_POINTER;
    }
    auto begin = p.begin;
    auto end = p.end;
    auto strides = p.strides;
    auto beginMask = p.begin_mask;
    auto endMask = p.end_mask;
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
            axisEnd = UNI_MAX(axisEnd, -1);
        } else if (axisEnd > (int)(inputDesc.dims[axis])) {
            axisEnd = inputDesc.dims[axis];
        }
        if (strides[i] > 0) {
            outputDesc->dims[axis] = (axisEnd - axisBegin + strides[i] - 1) / strides[i];
        } else {
            outputDesc->dims[axis] = (axisEnd - axisBegin + strides[i] + 1) / strides[i];
        }
        begin[i] = axisBegin;
        end[i] = axisEnd;
    }
    if (inputDesc.df == DF_NCHWC4 || inputDesc.df == DF_NCHWC8 || inputDesc.df == DF_NCHWC16) {
        int align = 1;
        if (inputDesc.df == DF_NCHWC4) {
            align = 4;
        }
        if (inputDesc.df == DF_NCHWC8) {
            align = 8;
        }
        if (inputDesc.df == DF_NCHWC16) {
            align = 16;
        }
        int channelAxis = 1;
        if (begin[channelAxis] % align != 0 || strides[channelAxis] != 1 ||
            (end[channelAxis] - begin[channelAxis]) / strides[channelAxis] % align != 0) {
            outputDesc->df = DF_NCHW;
        }
        if (isSingleCStepMode(inputDesc, p)) {
            if (inputDesc.df == DF_NCHWC4 && (outputDesc->dims[dimSize - 1 - channelAxis] % 4 == 0)) {
                outputDesc->df = DF_NCHWC4;
            }
            if (inputDesc.df == DF_NCHWC8 && (outputDesc->dims[dimSize - 1 - channelAxis] % 8 == 0)) {
                outputDesc->df = DF_NCHWC8;
            }
            if (inputDesc.df == DF_NCHWC16 &&
                (outputDesc->dims[dimSize - 1 - channelAxis] % 16 == 0)) {
                outputDesc->df = DF_NCHWC16;
            }
        }
    }
    return SUCCESS;
}

template <typename T>
void singleCStepMode(
    TensorDesc inputDesc, T *input, int start, int stride, TensorDesc outputDesc, T *output)
{
    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    if (tensorIs2d(inputDesc)) {
        CHECK_STATUS(tensor2dGet(inputDesc, &idt, &idf, &in, &ic));
        ih = iw = 1;
    } else if (tensorIs3d(inputDesc)) {
        CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
        iw = 1;
    } else if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    } else {
        UNI_ERROR_LOG("Slice currently on support 2/3/4 dim data.\n");
    }
    U32 ihiw = ih * iw;
    U32 oc = outputDesc.dims[outputDesc.nDims - 2];
    U32 icx = 1;
    if (idf == DF_NCHWC8) {
        icx = 8;
    }
    if (idf == DF_NCHWC16) {
        icx = 16;
    }
    U32 ocx = 1;
    if (outputDesc.df == DF_NCHWC8) {
        ocx = 8;
    }
    if (outputDesc.df == DF_NCHWC16) {
        ocx = 16;
    }
    int icc = ic / icx;
    int occ = oc / ocx;
    for (U32 n = 0, oidx = 0; n < in; ++n) {
        for (int c = 0; c < occ; c++) {
            for (U32 hw = 0; hw < ihiw; ++hw) {
                for (U32 c8 = 0; c8 < ocx; ++c8, oidx++) {
                    int ocidx = c * ocx + c8;
                    int icidx = (ocidx * stride + start + ic) % ic;
                    int ic0 = icidx / icx;
                    int ic1 = icidx % icx;
                    U32 iidx = ((n * icc + ic0) * ihiw + hw) * icx + ic1;
                    output[oidx] = input[iidx];
                }
            }
        }
    }
}

#define TFSLICE_USE_RECURSIVE
#ifdef TFSLICE_USE_RECURSIVE
inline static void recursive_tfslice(U8 *src,
    U32 *srcDims,
    U32 srcNum,
    U8 *dst,
    U32 *dstDims,
    U32 dstNum,
    I32 *begin,
    I32 *end,
    I32 *strides,
    int i,
    int bound,
    int dimNum,
    U32 tileSize)
{
    if (i == bound) {
        UNI_MEMCPY(dst, src, tileSize);
        return;
    }
    U32 newSrcNum = srcNum / srcDims[dimNum - 1 - i];
    U32 newDstNum = dstNum / dstDims[dimNum - 1 - i];
    if (i + 1 == bound) {
        if (strides[i] == 1) {
            UNI_MEMCPY(dst, src + begin[i] * newSrcNum, tileSize * (end[i] - begin[i]));
            return;
        }
    }
    if (strides[i] > 0) {
        for (int j = begin[i]; j < end[i]; j += strides[i]) {
            U8 *newSrc = src + j * newSrcNum;
            recursive_tfslice(newSrc, srcDims, newSrcNum, dst, dstDims, newDstNum, begin, end,
                strides, i + 1, bound, dimNum, tileSize);
            dst += newDstNum;
        }
    } else {
        for (int j = begin[i]; j > end[i]; j += strides[i]) {
            U8 *newSrc = src + j * newSrcNum;
            recursive_tfslice(newSrc, srcDims, newSrcNum, dst, dstDims, newDstNum, begin, end,
                strides, i + 1, bound, dimNum, tileSize);
            dst += newDstNum;
        }
    }
}
#endif

EE tfslice_cpu(
    TensorDesc inputDesc, void *input, TfSliceParamSpec p, TensorDesc outputDesc, void *output)
{
    if (tensorNumElements(outputDesc) == 0) {
        return SUCCESS;
    }
    auto begin = p.begin;
    auto end = p.end;
    auto strides = p.strides;
    auto beginMask = p.begin_mask;
    auto endMask = p.end_mask;
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
            axisEnd = UNI_MAX(axisEnd, -1);
        } else if (axisEnd > (int)(inputDesc.dims[axis])) {
            axisEnd = inputDesc.dims[axis];
        }
        begin[i] = axisBegin;
        end[i] = axisEnd;
    }

    if (isSingleCStepMode(inputDesc, p)) {
        EE ret = SUCCESS;
        switch (inputDesc.dt) {
            case DT_F32: {
                singleCStepMode<F32>(
                    inputDesc, (F32 *)input, p.begin[1], p.strides[1], outputDesc, (F32 *)output);
                break;
            }
#ifdef _USE_FP16
            case DT_F16: {
                singleCStepMode<F16>(
                    inputDesc, (F16 *)input, p.begin[1], p.strides[1], outputDesc, (F16 *)output);
                break;
            }
#endif
            case DT_U32:
            case DT_I32: {
                singleCStepMode<I32>(
                    inputDesc, (I32 *)input, p.begin[1], p.strides[1], outputDesc, (I32 *)output);
                break;
            }
            case DT_I8:
            case DT_U8:
            case DT_U8_Q: {
                singleCStepMode<INT8>(
                    inputDesc, (INT8 *)input, p.begin[1], p.strides[1], outputDesc, (INT8 *)output);
                break;
            }
            default:
                ret = NOT_SUPPORTED;
                break;
        }
        return ret;
    }

    U32 num = tensorNumElements(outputDesc);
    U8 *dst = (U8 *)output;
    U32 elementSize = bytesOf(inputDesc.dt);
    int channelAxis = inputDesc.nDims - 2;
    int cx = 1;
    if (inputDesc.df == DF_NCHWC8) {
        cx = 8;
    } else if (inputDesc.df == DF_NCHWC16) {
        cx = 16;
    }
    if (inputDesc.df == outputDesc.df) {
        std::vector<U32> tmpInputDims(inputDesc.nDims), tmpOutputDims(outputDesc.nDims);
        UNI_MEMCPY(tmpInputDims.data(), inputDesc.dims, inputDesc.nDims * sizeof(U32));
        UNI_MEMCPY(tmpOutputDims.data(), outputDesc.dims, outputDesc.nDims * sizeof(U32));
        int startAxis = 0;
        int elementNum = 1;
        if (cx > 1) {
            elementNum *= cx;
            begin[1] /= cx;
            end[1] /= cx;
            tmpInputDims[channelAxis] /= cx;
            tmpOutputDims[channelAxis] /= cx;
            tmpInputDims.insert(tmpInputDims.begin(), cx);
            tmpOutputDims.insert(tmpOutputDims.begin(), cx);
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
            UNI_MEMCPY(dst, src, tileSize);
        }
#endif
        if (cx > 1) {
            begin[1] *= cx;
            end[1] *= cx;
        }
    } else {
        CHECK_REQUIREMENT(cx > 1);
        U32 tmpNDims = inputDesc.nDims + 1;
        std::vector<U32> tmpDims(tmpNDims);
        tmpDims[0] = cx;
        UNI_MEMCPY(&(tmpDims[1]), inputDesc.dims, inputDesc.nDims * sizeof(U32));
        for (U32 i = 0; i < num; i++, dst += elementSize) {
            std::vector<U32> localIndex = calculateLocalIndex(i, outputDesc.dims, outputDesc.nDims);
            for (U32 j = 0; j < dimSize; j++) {
                int reverseAxis = dimSize - 1 - j;
                localIndex[j] = localIndex[j] * strides[reverseAxis] + begin[reverseAxis];
            }
            int c8 = localIndex[channelAxis] % cx;
            localIndex[channelAxis] /= cx;
            localIndex.insert(localIndex.begin(), c8);
            U32 index = calculateGlobalIndex(localIndex.data(), tmpDims.data(), tmpNDims);
            U8 *src = (U8 *)input + index * elementSize;
            UNI_MEMCPY(dst, src, elementSize);
        }
    }
    return SUCCESS;
}
