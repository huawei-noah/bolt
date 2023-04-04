// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "sys.h"

#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/activation_mali_fp16.h"

static EE activation_checkpara_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ActivationParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (handle == nullptr || nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.df != outputDesc.df) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (p.mode != ACTIVATION_NULL && p.mode != ACTIVATION_RELU &&
        p.mode != ACTIVATION_RELU6 && p.mode != ACTIVATION_H_SIGMOID &&
        p.mode != ACTIVATION_H_SWISH && p.mode != ACTIVATION_GELU &&
        p.mode != ACTIVATION_TANH && p.mode != ACTIVATION_SIGMOID &&
        p.mode != ACTIVATION_ABS && p.mode != ACTIVATION_LOG &&
        p.mode != ACTIVATION_NEG && p.mode != ACTIVATION_EXP &&
        p.mode != ACTIVATION_SWISH && p.mode != ACTIVATION_FLOOR &&
        p.mode != ACTIVATION_ROUND && p.mode != ACTIVATION_CEIL) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (input->desc.memFormat != output->desc.memFormat) {
        CHECK_STATUS(NOT_MATCH)
    }
    return SUCCESS;
}

EE activation_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ActivationParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(
        activation_checkpara_mali(handle, inputDesc, input, p, outputDesc, output));
    return activation_mali_fp16(handle, inputDesc, input, p, outputDesc, output);
}
