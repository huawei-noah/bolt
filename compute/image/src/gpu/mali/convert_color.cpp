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
#include "gpu/mali/image_mali.h"
#include "gpu/mali/cl/kernel_option/convert_color_opt.h"

inline EE convert_color_checkpara_mali(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    if (handle == nullptr || input == nullptr || output == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.nDims != outputDesc.nDims) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (input->desc.memFormat != output->desc.memFormat) {
        CHECK_STATUS(NOT_MATCH);
    }
    return SUCCESS;
}

inline EE convert_color_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ConvertColorParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    if (inputDesc.df == DF_NHWC) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    } else {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ih, &iw, &ic));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oh, &ow, &oc));
    }
    U32 src_step, dst_step;
    U32 src_offset = 0, dst_offset = 0;
    U32 rows, cols;
    U32 height, width;
    if (p.src == YUV_NV21) {
        if (p.dst == RGB_0_255 || p.dst == RGB_0_1 || p.dst == BGR_0_255 || p.dst == BGR_0_1 ||
            p.dst == RGBA_0_255 || p.dst == RGBA_0_1 || p.dst == BGRA_0_255 || p.dst == BGRA_0_1) {
            height = oh;
            width = ow;
            src_step = width;
            dst_step = width * 3;
            rows = height;
            cols = width;
        } else {
            return NOT_SUPPORTED;
        }
    } else if (p.src == RGB_0_255 || p.src == RGB_0_1 || p.src == BGR_0_255 || p.src == BGR_0_1 ||
        p.src == RGBA_0_255 || p.src == RGBA_0_1 || p.src == BGRA_0_255 || p.src == BGRA_0_1) {
        if (p.dst == YUV_NV21) {
            height = ih;
            width = iw;
            src_step = width * 3;
            dst_step = width;
            rows = height / 2 * 3;
            cols = width;
        } else {
            return NOT_SUPPORTED;
        }
    } else {
        return NOT_SUPPORTED;
    }
    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    U32 dim = 3;
    U32 gs[3] = {width / 2, height / 2, in};
    U32 ls[3] = {0, 0, 0};
    CHECK_STATUS(set_convert_color_opt_mali(p, idt, odt, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(
        kernel, src_step, src_offset, dst_step, dst_offset, rows, cols, input->mem, output->mem));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
//    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

EE convert_color_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ConvertColorParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(convert_color_checkpara_mali(handle, inputDesc, input, outputDesc, output));
    //CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(convert_color_core_mali_fp16(handle, inputDesc, input, p, outputDesc, output));
    return SUCCESS;
}
