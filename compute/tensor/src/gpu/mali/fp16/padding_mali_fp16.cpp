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
#include "gpu/mali/fp16/padding_mali_fp16.h"

inline EE padding_checkpara_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    PadParamSpec padParamSpec,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (handle == nullptr || nullptr == input || nullptr == output) {
        return NULL_POINTER;
    }
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    if (inputDesc.df != outputDesc.df || inputDesc.df != DF_NCHW) {
        return NOT_SUPPORTED;
    }
    if (input->desc.memFormat != output->desc.memFormat || input->desc.memFormat != DF_NCWHC4) {
        return NOT_SUPPORTED;
    }
    if (padParamSpec.pad_mode == Pad_Reflect &&
        (padParamSpec.top >= inputDesc.dims[1] || padParamSpec.bottom >= inputDesc.dims[1])) {
        return NOT_SUPPORTED;
    }
    if (padParamSpec.pad_mode == Pad_Symmetric &&
        (padParamSpec.left > inputDesc.dims[0] || padParamSpec.right > inputDesc.dims[0])) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE padding_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    PadParamSpec padParamSpec,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);

    cl_mem inbuf, outbuf;
    inbuf = input->mem;
    outbuf = output->mem;

    U32 iw_str, ih_str, iw_off, ih_off;
    ih_str = input->desc.stride[0];
    iw_str = input->desc.stride[1];
    ih_off = input->desc.offset[0];
    iw_off = input->desc.offset[1];

    U32 ow_str, oh_str, ow_off, oh_off;
    oh_str = output->desc.stride[0];
    ow_str = output->desc.stride[1];
    oh_off = output->desc.offset[0];
    ow_off = output->desc.offset[1];

    U32 pw, ph, pr, pb;
    pw = padParamSpec.left;
    pr = padParamSpec.right;
    ph = padParamSpec.top;
    pb = padParamSpec.bottom;

    Kernel kernel;
    switch (padParamSpec.pad_mode) {
        case Pad_Constant: {
            CHECK_STATUS(gcl_create_kernel(handle, "padding_constant", &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ih_str, iw_str, ih_off, iw_off, oh, ow,
                oh_str, ow_str, oh_off, ow_off, ph, pb, pw, pr, inbuf, outbuf));

            U32 gs[3] = {oh, ow, (oc + 3) / 4 * on};
            U32 ls[3] = {0, 0, 0};
            U32 dim = 3;
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, "padding_constant");
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "padding_constant"));
            CHECK_STATUS(gcl_print_memory<F16>(handle, input, "padding_constant_input"));
            CHECK_STATUS(gcl_print_memory<F16>(handle, output, "padding_constant_output"));
#endif
            break;
        }
        case Pad_Reflect: {
            CHECK_STATUS(gcl_create_kernel(handle, "padding_reflect", &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ih_str, iw_str, ih_off, iw_off, oh, ow,
                oh_str, ow_str, oh_off, ow_off, ph, pb, pw, pr, inbuf, outbuf));

            U32 gs[3] = {oh, ow, (oc + 3) / 4 * on};
            U32 ls[3] = {0, 0, 0};
            U32 dim = 3;
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, "padding_reflect");
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "padding_reflect"));
            CHECK_STATUS(gcl_print_memory<F16>(handle, input, "padding_reflect_input"));
            CHECK_STATUS(gcl_print_memory<F16>(handle, output, "padding_reflect_output"));
#endif
            break;
        }
        case Pad_Edge: {
            CHECK_STATUS(gcl_create_kernel(handle, "padding_edge", &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ih_str, iw_str, ih_off, iw_off, oh, ow,
                oh_str, ow_str, oh_off, ow_off, ph, pb, pw, pr, inbuf, outbuf));

            U32 gs[3] = {oh, ow, (oc + 3) / 4 * on};
            U32 ls[3] = {0, 0, 0};
            U32 dim = 3;
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, "padding_edge");
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "padding_edge"));
            CHECK_STATUS(gcl_print_memory<F16>(handle, input, "padding_edge_input"));
            CHECK_STATUS(gcl_print_memory<F16>(handle, output, "padding_edge_output"));
#endif
            break;
        }
        case Pad_Symmetric: {
            CHECK_STATUS(gcl_create_kernel(handle, "padding_symmetric", &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ih_str, iw_str, ih_off, iw_off, oh, ow,
                oh_str, ow_str, oh_off, ow_off, ph, pb, pw, pr, inbuf, outbuf));

            U32 gs[3] = {oh, ow, (oc + 3) / 4 * on};
            U32 ls[3] = {0, 0, 0};
            U32 dim = 3;
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, "padding_symmetric");
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "padding_symmetric"));
            CHECK_STATUS(gcl_print_memory<F16>(handle, input, "padding_symmetric_input"));
            CHECK_STATUS(gcl_print_memory<F16>(handle, output, "padding_symmetric_output"));
#endif
            break;
        }
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    return SUCCESS;
}

EE padding_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    PadParamSpec padParamSpec,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(
        padding_checkpara_mali_fp16(handle, inputDesc, input, padParamSpec, outputDesc, output));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(padding_core_mali_fp16(handle, inputDesc, input, padParamSpec, outputDesc, output));
    return SUCCESS;
}
