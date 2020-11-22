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
    ConvolutionMode convMode);

FullyConnectedParamSpec createFullyConnectedParamSpec(
    U32 num_outputs, U32 num_slices, I32 *slice_point);

PoolingParamSpec createPoolingParamSpec(PoolingMode pm,
    U32 ksH,
    U32 ksW,
    U32 strideH,
    U32 strideW,
    U32 paddingT,
    U32 paddingB,
    U32 paddingL,
    U32 paddingR,
    RoundMode rm);

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
