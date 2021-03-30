// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/embedding_mali_fp16.h"

inline EE embedding_checkpara_mali_fp16(TensorDesc weightDesc, TensorDesc outputDesc)
{
    if (weightDesc.dt != outputDesc.dt || weightDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE embedding_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc weightDesc,
    GCLMem_t weight,
    EmbedParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    U32 ow, oh, oc;
    U32 iw_str, iw_off, ih_off;
    U32 fw_str, fw_off, fh_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, NULL, NULL, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(weight->desc, &fw_str, NULL, NULL, &fw_off, &fh_off));
    CHECK_STATUS(gclmem_get_desc_dim(output->desc, NULL, NULL, NULL, &oc, &oh, &ow));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    cl_mem inbuf, weibuf, outbuf;
    inbuf = input->mem;
    weibuf = weight->mem;
    outbuf = output->mem;

    if (!p.transpose) {
        U32 gs[3] = {(ow + 3) / 4, oh, oc};
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        Kernel kernel;
        CHECK_STATUS(gcl_create_kernel(handle, "embedding", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, iw_off, ih_off, fw_str, fw_off, fh_off,
            ow_str, oh_str, ow_off, oh_off, ow, gs[0], gs[1], inbuf, weibuf, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, "embedding");
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "embedding"));
#endif
        return SUCCESS;
    } else {
        return NOT_SUPPORTED;
    }
}

EE embedding_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc weightDesc,
    GCLMem_t weight,
    EmbedParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(embedding_checkpara_mali_fp16(weightDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(embedding_core_mali_fp16(
        handle, inputDesc, input, weightDesc, weight, p, outputDesc, output));
    return SUCCESS;
}
