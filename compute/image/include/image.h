// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_IMAGE
#define _H_IMAGE

#include "tensor_auxiliary.h"
#include "parameter_spec.h"
#include "sys.h"

#ifdef _USE_GPU
#include "gcl.h"
#include "ocl_desc_trans.h"
#endif

EE resize_infer_output_size(
    Tensor *inputTensor, ResizeParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE resize_infer_forward_tmp_bytes(
    Tensor inputTensor, ResizeParamSpec p, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE resize(
    Tensor inputTensor, ResizeParamSpec p, Tensor tmpTensor, Tensor outputTensor, ArchInfo_t archInfo);

EE grid_sample_infer_output_size(
    Tensor *inputTensor, Tensor *gridTensor, Tensor *outputTensor, ArchInfo_t archInfo);

EE grid_sample_infer_forward_tmp_bytes(Tensor inputTensor,
    Tensor gridTensor,
    GridSampleParamSpec p,
    Tensor outputTensor,
    U32 *bytes,
    ArchInfo_t archInfo);

EE grid_sample(Tensor inputTensor,
    Tensor gridTensor,
    GridSampleParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE convert_color_infer_output_size(
    Tensor *inputTensor, ConvertColorParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE convert_color(
    Tensor inputTensor, ConvertColorParamSpec p, Tensor outputTensor, ArchInfo_t archInfo);

EE lut_preprocess_infer_output_size(Tensor *inputTensor, DataType dt, Tensor *outputTensor, ArchInfo_t archInfo);

EE lut_preprocess(Tensor inputTensor, Tensor outputTensor, ArchInfo_t archInfo);

EE lut_infer_output_size(
    Tensor *inputTensor, Tensor *lutTensor, LutParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE lut(
    Tensor inputTensor, Tensor lutTensor, LutParamSpec p, Tensor outputTensor, ArchInfo_t archInfo);
#endif
