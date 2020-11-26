// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TENSOR_COMPUTING_TYPE
#define _H_TENSOR_COMPUTING_TYPE

#include <vector>
#include "types.h"
#include "tensor.hpp"

#ifdef _USE_MALI
#include "gcl.h"
#include "ocl_desc_trans.h"
#define ALIGN(len, align_num) ((len + align_num - 1) / align_num * align_num)
#endif

ConvolutionParamSpec createConvolutionParamSpec(U32 group,
    U32 kernel_t,
    U32 kernel_h,
    U32 kernel_w,
    U32 stride_t,
    U32 stride_h,
    U32 stride_w,
    U32 padding_before,
    U32 padding_after,
    U32 padding_top,
    U32 padding_bottom,
    U32 padding_left,
    U32 padding_right,
    U32 dilateRate_t,
    U32 dilateRate_h,
    U32 dilateRate_w,
    U32 num_outputs,
    ConvolutionMode convMode);

PoolingParamSpec createPoolingParamSpec(PoolingMode pm,
    U32 kernel_t,
    U32 kernel_h,
    U32 kernel_w,
    U32 stride_t,
    U32 stride_h,
    U32 stride_w,
    U32 padding_before,
    U32 padding_after,
    U32 padding_top,
    U32 padding_bottom,
    U32 padding_left,
    U32 padding_right,
    RoundMode rm);

FullyConnectedParamSpec createFullyConnectedParamSpec(
    U32 num_outputs, U32 num_slices, I32 *slice_point);

ReshapeParamSpec createReshapeParamSpec(I32 *shape_dims, I32 shape_size, I32 axis, I32 num_axes);

ClipParamSpec createClipParamSpec(float min, float max);

SqueezeParamSpec createSqueezeParamSpec(int *axes, int axes_num);

std::vector<TensorDesc> get_desc_from_tensors(std::vector<Tensor> tensors);
std::vector<TensorDesc> get_desc_from_tensor_ptrs(std::vector<Tensor *> tensors);

std::vector<F32> get_scale_from_tensors(std::vector<Tensor> tensors);

template <typename T>
std::vector<T> get_data_from_tensors(std::vector<Tensor> tensors, Arch arch);
template <typename T>
std::vector<T> get_data_from_tensor_ptrs(std::vector<Tensor *> tensors, Arch arch);
#endif
