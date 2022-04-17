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

inline EE activation_checkpara_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode activationMode)
{
    if (handle == nullptr || nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.df != outputDesc.df) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (activationMode != ACTIVATION_NULL && activationMode != ACTIVATION_RELU &&
        activationMode != ACTIVATION_RELU6 && activationMode != ACTIVATION_H_SIGMOID &&
        activationMode != ACTIVATION_H_SWISH && activationMode != ACTIVATION_GELU &&
        activationMode != ACTIVATION_TANH && activationMode != ACTIVATION_SIGMOID &&
        activationMode != ACTIVATION_ABS && activationMode != ACTIVATION_LOG &&
        activationMode != ACTIVATION_NEG && activationMode != ACTIVATION_EXP &&
        activationMode != ACTIVATION_SWISH) {
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
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode activationMode)
{
    EE ret = SUCCESS;
    CHECK_STATUS(
        activation_checkpara_mali(handle, inputDesc, input, outputDesc, output, activationMode));
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = activation_mali_fp16(handle, inputDesc, input, outputDesc, output, activationMode);
            break;
        }
        case DT_I8: {
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
