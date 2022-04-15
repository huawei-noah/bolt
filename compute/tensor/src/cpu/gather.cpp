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
inline static void gather(const TensorDesc &dataDesc,
    const T *data,
    const TensorDesc &indexDesc,
    const int *index,
    const GatherParamSpec &p,
    const TensorDesc &outputDesc,
    T *output)
{
    int axis = (p.axis + dataDesc.nDims) % dataDesc.nDims;
    axis = dataDesc.nDims - 1 - axis;
    int outer_loop = 1, k = dataDesc.dims[axis], inner_loop = 1;
    for (int i = 0; i < axis; i++) {
        inner_loop *= dataDesc.dims[i];
    }
    for (U32 i = axis + 1; i < dataDesc.nDims; i++) {
        outer_loop *= dataDesc.dims[i];
    }
    int tile_size = inner_loop;
    for (int i = 0, dst_index = 0; i < outer_loop; i++) {
        for (U32 j = 0; j < tensorNumElements(indexDesc); j++, dst_index += tile_size) {
            int src_index = (i * k + index[j]) * tile_size;
            memcpy(output + dst_index, data + src_index, tile_size * sizeof(T));
        }
    }
}

template <typename T>
inline static void gather_elements(const TensorDesc &dataDesc,
    const T *data,
    const TensorDesc &indexDesc,
    const int *index,
    const GatherParamSpec &p,
    const TensorDesc &outputDesc,
    T *output)
{
    int axis = (p.axis + dataDesc.nDims) % dataDesc.nDims;
    axis = dataDesc.nDims - 1 - axis;

    for (U32 i = 0; i < tensorNumElements(dataDesc); i++) {
        std::vector<U32> local = calculateLocalIndex(i, dataDesc.dims, dataDesc.nDims);
        local[axis] = index[i];
        U32 k = calculateGlobalIndex(local.data(), dataDesc.dims, dataDesc.nDims);
        output[i] = data[k];
    }
}

template <typename T>
inline static void gatherND(const TensorDesc &dataDesc,
    const T *data,
    TensorDesc indexDesc,
    const int *index,
    const GatherParamSpec &p,
    const TensorDesc &outputDesc,
    T *output)
{
    int batch_dims_size = 1;
    for (int i = 0; i < p.batch_dims; i++) {
        int k = indexDesc.dims[indexDesc.nDims - 1 - i];
        batch_dims_size *= k;
    }
    int k = indexDesc.dims[indexDesc.nDims - 1];
    int t = tensorNumElements(indexDesc) / batch_dims_size / k;
    TensorDesc newDataDesc = dataDesc;
    int axis = (p.batch_dims + dataDesc.nDims) % dataDesc.nDims;
    axis = dataDesc.nDims - 1 - axis;
    newDataDesc.dims[axis + 1] = batch_dims_size;
    newDataDesc.nDims = axis + 1 + 1;

    U32 gather_index[16] = {0};
    int tile_dims = newDataDesc.nDims - (k + 1);
    gather_index[tile_dims + k] = p.batch_dims;
    U32 tile_size = 1;
    for (int i = 0; i < tile_dims; i++) {
        tile_size *= newDataDesc.dims[i];
    }
    for (int batch_dim = 0, i = 0, dst_index = 0; batch_dim < batch_dims_size; batch_dim++) {
        for (int outer_dim = 0; outer_dim < t; outer_dim++, i += k, dst_index += tile_size) {
            for (int j = 0; j < k; j++) {
                gather_index[tile_dims + k - 1 - j] = index[i + j];
            }
            U32 src_index = calculateGlobalIndex(gather_index, newDataDesc.dims, newDataDesc.nDims);
            memcpy(output + dst_index, data + src_index, tile_size * sizeof(T));
        }
    }
}

template <typename T>
inline static void gather_kernel(const TensorDesc &dataDesc,
    const T *data,
    const TensorDesc &indexDesc,
    const int *index,
    const GatherParamSpec &p,
    const TensorDesc &outputDesc,
    T *output)
{
    if (p.axis == INT_MAX) {
        gatherND<T>(dataDesc, data, indexDesc, index, p, outputDesc, output);
    } else if (p.element_level) {
        gather_elements<T>(dataDesc, data, indexDesc, index, p, outputDesc, output);
    } else {
        gather<T>(dataDesc, data, indexDesc, index, p, outputDesc, output);
    }
}

EE gather_cpu(TensorDesc dataDesc,
    const void *data,
    TensorDesc indexDesc,
    const void *index,
    GatherParamSpec p,
    void *tmp,
    TensorDesc outputDesc,
    void *output)
{
    if (dataDesc.df == DF_NCHWC8) {
        TensorDesc tmpTensorDesc = dataDesc;
        tmpTensorDesc.df = DF_NCHW;
        transformToNCHW(dataDesc, data, tmpTensorDesc, tmp);
        data = tmp;
        dataDesc.df = DF_NCHW;
    }
    EE ret = SUCCESS;
    switch (dataDesc.dt) {
        case DT_I32:
            gather_kernel<I32>(dataDesc, (const I32 *)data, indexDesc, (const int *)index, p,
                outputDesc, (I32 *)output);
            break;
#ifdef _USE_FP32
        case DT_F32:
            gather_kernel<F32>(dataDesc, (const F32 *)data, indexDesc, (const int *)index, p,
                outputDesc, (F32 *)output);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            gather_kernel<F16>(dataDesc, (const F16 *)data, indexDesc, (const int *)index, p,
                outputDesc, (F16 *)output);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
