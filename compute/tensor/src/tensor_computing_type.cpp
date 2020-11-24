// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "types.h"
#include "tensor_computing_type.h"

ConvolutionParamSpec createConvolutionParamSpec(U32 group,
    U32 kernelH,
    U32 kernelW,
    U32 strideH,
    U32 strideW,
    U32 paddingT,
    U32 paddingB,
    U32 paddingL,
    U32 paddingR,
    U32 dilateH,
    U32 dilateW,
    U32 num_outputs,
    ConvolutionMode convMode)
{
    ConvolutionParamSpec p;
    p.group = group;
    p.kernel_h = kernelH;
    p.kernel_w = kernelW;
    p.stride_h = strideH;
    p.stride_w = strideW;
    p.padding_top = paddingT;
    p.padding_bottom = paddingB;
    p.padding_left = paddingL;
    p.padding_right = paddingR;
    p.dilatedRate_h = dilateH;
    p.dilatedRate_w = dilateW;
    p.num_outputs = num_outputs;
    p.convolution_type = convMode;
    return p;
}

FullyConnectedParamSpec createFullyConnectedParamSpec(
    U32 num_outputs, U32 num_slices, I32 *slice_point)
{
    FullyConnectedParamSpec p;
    p.num_outputs = num_outputs;
    p.num_slices = num_slices;
    if (num_slices > 1 && slice_point != nullptr) {
        for (int i = 0; i < (int)num_slices; i++) {
            p.slice_point[i] = slice_point[i];
        }
    }
    return p;
}

PoolingParamSpec createPoolingParamSpec(PoolingMode pm,
    U32 ksH,
    U32 ksW,
    U32 strideH,
    U32 strideW,
    U32 paddingT,
    U32 paddingB,
    U32 paddingL,
    U32 paddingR,
    RoundMode rm)
{
    PoolingParamSpec p;
    p.mode = pm;
    p.kernel_h = ksH;
    p.kernel_w = ksW;
    p.stride_h = strideH;
    p.stride_w = strideW;
    p.padding_top = paddingT;
    p.padding_bottom = paddingB;
    p.padding_left = paddingL;
    p.padding_right = paddingR;
    p.rm = rm;
    return p;
}

ReshapeParamSpec createReshapeParamSpec(I32 *shape_dims, I32 shape_size, I32 axis, I32 num_axes)
{
    ReshapeParamSpec p;
    p.shape_size = shape_size;
    p.axis = axis;
    p.num_axes = num_axes;
    if (shape_dims != nullptr && shape_size != 0) {
        for (int i = 0; i < shape_size; i++) {
            p.shape_dims[i] = shape_dims[i];
        }
    }
    return p;
}

ClipParamSpec createClipParamSpec(float min, float max)
{
    ClipParamSpec p;
    p.min = min;
    p.max = max;
    return p;
}

SqueezeParamSpec createSqueezeParamSpec(int *axes, int axes_num)
{
    SqueezeParamSpec p;
    p.axes_num = axes_num;
    if (axes != nullptr && axes_num != 0) {
        for (int i = 0; i < axes_num; i++) {
            p.axes[i] = axes[i];
        }
    }
    return p;
}

std::vector<TensorDesc> get_desc_from_tensors(std::vector<Tensor> tensors)
{
    int size = tensors.size();
    std::vector<TensorDesc> result(size);
    for (int i = 0; i < size; i++) {
        result[i] = tensors[i].get_desc();
    }
    return result;
}

std::vector<TensorDesc> get_desc_from_tensor_ptrs(std::vector<Tensor *> tensors)
{
    int size = tensors.size();
    std::vector<TensorDesc> result(size);
    for (int i = 0; i < size; i++) {
        result[i] = tensors[i]->get_desc();
    }
    return result;
}

std::vector<F32> get_scale_from_tensors(std::vector<Tensor> tensors)
{
    int size = tensors.size();
    std::vector<F32> result(size);
    for (int i = 0; i < size; i++) {
        result[i] = tensors[i].get_scale();
    }
    return result;
}

template <typename T>
std::vector<T> get_data_from_tensors(std::vector<Tensor> tensors, Arch arch)
{
    int size = tensors.size();
    std::vector<T> result(size);
    for (int i = 0; i < size; i++) {
        result[i] = (T)get_ptr_from_tensor(tensors[i], arch);
    }
    return result;
}

template <typename T>
std::vector<T> get_data_from_tensor_ptrs(std::vector<Tensor *> tensors, Arch arch)
{
    int size = tensors.size();
    std::vector<T> result(size);
    for (int i = 0; i < size; i++) {
        result[i] = (T)get_ptr_from_tensor(*tensors[i], arch);
    }
    return result;
}

template std::vector<void *> get_data_from_tensors(std::vector<Tensor> tensors, Arch arch);
template std::vector<void *> get_data_from_tensor_ptrs(std::vector<Tensor *> tensors, Arch arch);
