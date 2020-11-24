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
#include "error.h"
#include "types.h"
#include "gpu/mali/fp16/activation_mali_fp16.h"

inline EE activation_checkpara_mali_fp16(TensorDesc inputDesc)
{
    if (inputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE activation_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode activationMode)
{
    UNUSED(inputDesc);
    U32 ow, oh, oc, on;
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    cl_mem inbuf, outbuf;
    inbuf = input->mem;
    outbuf = output->mem;

    char modeName[16];
    switch (activationMode) {
        case ACTIVATION_NULL:
            return SUCCESS;
        case ACTIVATION_RELU:
            strcpy(modeName, "relu");
            break;
        case ACTIVATION_RELU6:
            strcpy(modeName, "relu6");
            break;
        case ACTIVATION_H_SIGMOID:
            strcpy(modeName, "hsigmoid");
            break;
        case ACTIVATION_H_SWISH:
            strcpy(modeName, "hswish");
            break;
        case ACTIVATION_GELU:
            strcpy(modeName, "gelu");
            break;
        case ACTIVATION_TANH:
            strcpy(modeName, "tanh");
            break;
        case ACTIVATION_SIGMOID:
            strcpy(modeName, "sigmoid");
            break;
        default:
            return NOT_SUPPORTED;
    }
    char kernelName[128];
    U32 H = 1;
    sprintf(kernelName, "activation_%s%d", modeName, H);
    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    U32 cd4 = (oc + 3) / 4;
    U32 ce4 = (oc & 3) == 0 ? 4 : (oc & 3);
    CHECK_STATUS(gcl_set_kernelArgs(kernel, oh, ow, cd4, ce4, ih_str, iw_str, ih_off, iw_off,
        oh_str, ow_str, oh_off, ow_off, inbuf, outbuf));
    U32 gs[3] = {oh, ow, (oc + 3) / 4};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

EE activation_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode activationMode)
{
    CHECK_STATUS(activation_checkpara_mali_fp16(inputDesc));
    if (input->mem != output->mem) {
        CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    }
    CHECK_STATUS(
        activation_core_mali_fp16(handle, inputDesc, input, outputDesc, output, activationMode));
    return SUCCESS;
}
