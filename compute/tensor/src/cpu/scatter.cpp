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

template <typename T>
inline static void scatter_elements(const TensorDesc &dataDesc,
    const T *data,
    const TensorDesc &indexDesc,
    const int *index,
    const TensorDesc &updateDesc,
    const T *update,
    const ScatterParamSpec &p,
    const TensorDesc &outputDesc,
    T *output)
{
    int axis = (p.axis + dataDesc.nDims) % dataDesc.nDims;
    axis = dataDesc.nDims - 1 - axis;

    UNI_MEMCPY(output, data, tensorNumBytes(dataDesc));

    for (U32 i = 0; i < tensorNumElements(updateDesc); i++) {
        std::vector<U32> local = calculateLocalIndex(i, updateDesc.dims, updateDesc.nDims);
        local[axis] = index[i];
        U32 k = calculateGlobalIndex(local.data(), dataDesc.dims, dataDesc.nDims);
        output[k] = update[i];
    }
}

template <typename T>
inline static void scatterND(const TensorDesc &dataDesc,
    const T *data,
    TensorDesc indexDesc,
    const int *index,
    const TensorDesc &updateDesc,
    const T *update,
    const TensorDesc &outputDesc,
    T *output)
{
    UNI_MEMCPY(output, data, tensorNumBytes(dataDesc));

    int lastDim = indexDesc.dims[0];
    for (U32 i = 0; i < indexDesc.nDims - 1; i++) {
        indexDesc.dims[i] = indexDesc.dims[i + 1];
    }
    indexDesc.nDims--;

    U32 tileSize = 1;
    for (U32 j = 0; j < dataDesc.nDims - lastDim; j++) {
        tileSize *= dataDesc.dims[j];
    }
    TensorDesc local = dataDesc;
    for (U32 j = 0; j < local.nDims; j++) {
        local.dims[j] = 0;
    }
    for (U32 i = 0; i < tensorNumElements(indexDesc); i++) {
        for (int j = 0; j < lastDim; j++) {
            local.dims[dataDesc.nDims - 1 - j] = index[i * lastDim + j];
        }
        U32 k = calculateGlobalIndex(local.dims, dataDesc.dims, dataDesc.nDims);
        for (U32 j = 0; j < tileSize; j++) {
            output[k + j] = update[i * tileSize + j];
        }
    }
}

template <typename T>
inline static void scatter_kernel(const TensorDesc &dataDesc,
    const T *data,
    const TensorDesc &indexDesc,
    const int *index,
    const TensorDesc &updateDesc,
    const T *update,
    const ScatterParamSpec &p,
    const TensorDesc &outputDesc,
    T *output)
{
    if (p.axis == INT_MAX) {
        scatterND<T>(dataDesc, data, indexDesc, index, updateDesc, update, outputDesc, output);
    } else {
        scatter_elements<T>(
            dataDesc, data, indexDesc, index, updateDesc, update, p, outputDesc, output);
    }
}

EE scatter_cpu(TensorDesc dataDesc,
    const void *data,
    TensorDesc indexDesc,
    const void *index,
    TensorDesc updateDesc,
    const void *update,
    ScatterParamSpec p,
    void *tmp,
    TensorDesc outputDesc,
    void *output)
{
    if (dataDesc.df == DF_NCHWC8) {
        TensorDesc tmpTensorDesc = dataDesc;
        tmpTensorDesc.df = DF_NCHW;
        transformToNCHW(dataDesc, data, tmpTensorDesc, tmp);
        data = tmp;
        tmp = (U8 *)tmp + tensorNumBytes(dataDesc);
        dataDesc.df = DF_NCHW;
    }
    if (updateDesc.df == DF_NCHWC8) {
        TensorDesc tmpTensorDesc = updateDesc;
        tmpTensorDesc.df = DF_NCHW;
        transformToNCHW(updateDesc, update, tmpTensorDesc, tmp);
        update = tmp;
        tmp = (U8 *)tmp + tensorNumBytes(updateDesc);
        updateDesc.df = DF_NCHW;
    }
    EE ret = SUCCESS;
    switch (dataDesc.dt) {
#ifdef _USE_FP32
        case DT_F32:
            scatter_kernel<F32>(dataDesc, (const F32 *)data, indexDesc, (const int *)index,
                updateDesc, (const F32 *)update, p, outputDesc, (F32 *)output);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            scatter_kernel<F16>(dataDesc, (const F16 *)data, indexDesc, (const int *)index,
                updateDesc, (const F16 *)update, p, outputDesc, (F16 *)output);
            break;
#endif
        case DT_U32:
        case DT_I32:
            scatter_kernel<U32>(dataDesc, (const U32 *)data, indexDesc, (const int *)index,
                updateDesc, (const U32 *)update, p, outputDesc, (U32 *)output);
            break;
        case DT_U8:
        case DT_I8:
            scatter_kernel<INT8>(dataDesc, (const INT8 *)data, indexDesc, (const int *)index,
                updateDesc, (const INT8 *)update, p, outputDesc, (INT8 *)output);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
