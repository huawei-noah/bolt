// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_IMAGE_CPU
#define _H_IMAGE_CPU

#include "tensor_desc.h"
#include "parameter_spec.h"
#include "uni.h"

EE resize_nearest_cpu(
    TensorDesc inputDesc, void *input, ResizeParamSpec p, TensorDesc outputDesc, void *output);

EE grid_sample_infer_output_size_cpu(
    TensorDesc inputDesc, TensorDesc gridDesc, TensorDesc *outputDesc);

EE grid_sample_cpu(TensorDesc inputDesc,
    void *input,
    TensorDesc gridDesc,
    void *grid,
    GridSampleParamSpec p,
    void *tmp,
    TensorDesc outputDesc,
    void *output);

EE convert_color_cpu(TensorDesc inputDesc,
    const void *input,
    ConvertColorParamSpec p,
    TensorDesc outputDesc,
    void *output);
#endif
