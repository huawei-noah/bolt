// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TENSOR_SHAPE_INFER
#define _H_TENSOR_SHAPE_INFER

#include "parameter_spec.h"

inline U32 reduceDims(U32 s, U32 e, TensorDesc desc)
{
    U32 dims = 1;
    for (U32 i = s; i < e; ++i) {
        dims *= desc.dims[i];
    }
    return dims;
}

inline bool isC8HasSameDim(TensorDesc inputDesc, TensorDesc outputDesc)
{
    int inDims = inputDesc.nDims;
    int onDims = outputDesc.nDims;
    int inCh = inDims - 2;
    int outCh = onDims - 2;
    if ((inCh < 0) || (outCh < 0)) {
        return false;
    }
    if ((reduceDims(inCh + 1, inDims, inputDesc) == reduceDims(outCh + 1, onDims, outputDesc)) &&
        (reduceDims(0, inCh, inputDesc) == reduceDims(0, outCh, outputDesc)) &&
        (inputDesc.dims[inCh] == outputDesc.dims[outCh])) {
        return true;
    }
    return false;
}

inline EE reshape_infer_output_size_cpu(
    TensorDesc inputDesc, ReshapeParamSpec p, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        return NULL_POINTER;
    }
    I32 *shape = p.shape;
    I32 shape_size = p.num_shape;
    int inputElementNum = tensorNumElements(inputDesc);
    int outputElementNum = 1;
    for (int i = 0; i < shape_size; i++) {
        outputElementNum *= shape[i];
    }
    int index_range = ((int)inputDesc.nDims > shape_size) ? shape_size : inputDesc.nDims;
    if (inputElementNum > 0 && outputElementNum > 0 && inputElementNum != outputElementNum) {
        for (int i = 0; i < index_range; i++) {
            if ((inputElementNum / (int)inputDesc.dims[inputDesc.nDims - 1 - i]) ==
                (outputElementNum / shape[i])) {
                shape[i] = inputDesc.dims[inputDesc.nDims - 1 - i];
                break;
            }
        }
    }

    *outputDesc = inputDesc;
    (*outputDesc).nDims = shape_size;
    if (shape_size == 2) {
        (*outputDesc).df = DF_NORMAL;
    }
    if (shape_size == 3) {
        (*outputDesc).df = DF_MTK;
    }
    if (shape_size >= 4) {
        (*outputDesc).df = DF_NCHW;
    }

    U32 factor = 1;
    I32 count = 0;
    for (I32 i = 0; i < shape_size; i++) {
        I32 value = shape[i];
        if (value == 0) {
            value = inputDesc.dims[inputDesc.nDims - 1 - i];
        }
        if (value == -1) {
            value = 0;
            count++;
        } else {
            factor *= value;
        }

        (*outputDesc).dims[shape_size - 1 - i] = value;
    }
    if (count > 1) {
        return NOT_SUPPORTED;
    }

    bool sameDim = ((*outputDesc).nDims == inputDesc.nDims);
    for (I32 i = 0; i < shape_size; i++) {
        if ((*outputDesc).dims[i] == 0) {
            (*outputDesc).dims[i] = tensorNumElements(inputDesc) / factor;
        }
        if ((*outputDesc).dims[i] != inputDesc.dims[i]) {
            sameDim = false;
        }
    }

    if (!sameDim && ((inputDesc.df == DF_NCHWC8) || (inputDesc.df == DF_NCHWC16))) {
        sameDim = isC8HasSameDim(inputDesc, *outputDesc);
    }
    if (sameDim) {
        (*outputDesc).df = inputDesc.df;
    }
    return SUCCESS;
}

inline EE transpose_infer_output_size_cpu(
    TensorDesc inputDesc, TransposeParamSpec p, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }

    U32 *dim = p.axes;
    *outputDesc = inputDesc;
    U32 num = inputDesc.nDims;
    U32 index = 0;
    for (U32 i = 0; i < p.num_axes; i++) {
        // use 5-dim array to transpose a NCHWC8 tensor. skip c8 axis
        if (dim[i] >= num) {
            continue;
        }
        // NOTE: TensorDesc.dims array is in [W H C N] order.
        // so if you want to transpose [N C H W] format data, we use (dims - 1 - *)
        // [5 6 7 8] + [0 3 2 1] = [5 8 7 6]
        // [8 7 6 5] + [0 3 2 1] = [6 7 8 5]
        outputDesc->dims[num - 1 - index] = inputDesc.dims[num - 1 - dim[i]];
        index++;
    }
    if ((inputDesc.df == DF_NCHWC8) && (p.axes[1] != 1)) {
        outputDesc->df = DF_NCHW;
    }
    //if (outputDesc->nDims == 4 && p.num_axes == 3 && outputDesc->dims[0] == 1) {
    //    (*outputDesc) = tensor3df(inputDesc.dt, DF_NCHW, outputDesc->dims[3],
    //        outputDesc->dims[2], outputDesc->dims[1]);
    //}
    if (p.df == DF_NCHWC8 && outputDesc->dims[num - 2] % 8 == 0) {
        outputDesc->df = DF_NCHWC8;
    }
    return SUCCESS;
}

inline EE squeeze_infer_output_size_cpu(
    TensorDesc inputDesc, SqueezeParamSpec p, TensorDesc *outputDesc)
{
    *outputDesc = inputDesc;
    if ((int)inputDesc.nDims == p.num_axes) {
        outputDesc->nDims = 1;
        outputDesc->df = DF_SCALAR;
        return SUCCESS;
    }
    for (int i = 0; i < p.num_axes; i++) {
        int axis = p.axes[i];
        if (axis < 0) {
            axis += inputDesc.nDims;
        }
        if (outputDesc->dims[inputDesc.nDims - 1 - axis] != 1) {
            UNI_ERROR_LOG(
                "try to squeeze non-one dimension in (%s).\n", tensorDesc2Str(inputDesc).c_str());
        }
        outputDesc->dims[inputDesc.nDims - 1 - axis] = INT_MAX;
    }
    U32 index = 0;
    for (U32 i = 0; i < inputDesc.nDims; i++) {
        if (outputDesc->dims[i] != INT_MAX) {
            outputDesc->dims[index++] = outputDesc->dims[i];
        }
    }
    CHECK_REQUIREMENT(index + p.num_axes == inputDesc.nDims);
    outputDesc->nDims = index;
    outputDesc->df = getTensorDefaultDataFormat(outputDesc->nDims);
    if (inputDesc.df == DF_NCHWC8 || inputDesc.df == DF_NCHWC16) {
        bool changeChannelAxis = false;
        for (int i = 0; i < p.num_axes; i++) {
            if (p.axes[i] < 1) {
                changeChannelAxis = true;
            }
        }
        if (!changeChannelAxis) {
            outputDesc->df = inputDesc.df;
        }
    }
    return SUCCESS;
}

inline EE unsqueeze_infer_output_size_cpu(
    TensorDesc inputDesc, UnsqueezeParamSpec p, TensorDesc *outputDesc)
{
    outputDesc->dt = inputDesc.dt;
    if (inputDesc.df == DF_SCALAR) {
        outputDesc->nDims = p.num_axes;
    } else {
        outputDesc->nDims = inputDesc.nDims + p.num_axes;
    }
    outputDesc->df = getTensorDefaultDataFormat(outputDesc->nDims);
    if (inputDesc.df == DF_NCHWC8 || inputDesc.df == DF_NCHWC16 || inputDesc.df == DF_NCHWC4) {
        bool changeChannelAxis = false;
        for (int i = 0; i < p.num_axes; i++) {
            if (p.axes[i] <= 1) {
                changeChannelAxis = true;
            }
        }
        if (!changeChannelAxis) {
            outputDesc->df = inputDesc.df;
        }
    }
    for (U32 i = 0; i < outputDesc->nDims; i++) {
        outputDesc->dims[i] = 0;
    }
    for (int i = 0; i < p.num_axes; i++) {
        int axis = p.axes[i];
        if (axis < 0) {
            axis += outputDesc->nDims;
        }
        outputDesc->dims[outputDesc->nDims - 1 - axis] = 1;
    }
    U32 index = 0;
    for (U32 i = 0; i < outputDesc->nDims; i++) {
        if (outputDesc->dims[i] == 0) {
            outputDesc->dims[i] = inputDesc.dims[index++];
        }
    }
    if (inputDesc.df != DF_SCALAR) {
        CHECK_REQUIREMENT(index == inputDesc.nDims);
    }
    return SUCCESS;
}
#endif
