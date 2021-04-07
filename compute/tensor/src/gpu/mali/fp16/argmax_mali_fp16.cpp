// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/argmax_mali_fp16.h"

#define get_thread_num(len, maxThreadNum, threadNum)                               \
    {                                                                              \
        threadNum = ((len + 7) / 8 < maxThreadNum) ? (len + 7) / 8 : maxThreadNum; \
    }

inline EE argmax_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    if (outputDesc.dt != DT_U32) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE argmax_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    I32 axis,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    UNUSED(outputDesc);
    if (axis < 0) {
        axis += inputDesc.nDims;
    }
    axis = inputDesc.nDims - 1 - axis;
    if (axis == 0) {
        DataType dt = inputDesc.dt;
        U32 iw, ih, ic;
        U32 inDims = inputDesc.nDims;
        iw = inputDesc.dims[0];
        ih = (inDims > 1) ? inputDesc.dims[1] : 1;
        ic = (inDims > 2) ? inputDesc.dims[2] : 1;
        U32 iw_str, ih_str, iw_off, ih_off;
        U32 ow_str, oh_str, ow_off, oh_off;
        get_gclmem_dim(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off);
        U32 threadNum;
        Kernel kernel;
        U32 gs[3];
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        Mem inv1024 = input->mem;
        Mem ini1024 = input->mem;
        Mem inv128 = input->mem;
        Mem ini128 = input->mem;
        Mem inv1 = input->mem;
        Mem ini1 = input->mem;
        Mem outv1024, outi1024, outv128, outi128;
        char kernelName[128];
        char kernelNameIndex[128];
        sprintf(kernelName, "argmax_x");
        sprintf(kernelNameIndex, "argmax_x_index");
        bool use_index = false;
        U32 offset = 0;
        U32 len = iw;
        get_thread_num(len, 1024, threadNum);
        if (threadNum > 128) {
            U32 outNum = 1024 * ih * ic;
            U32 outvSize = outNum * bytesOf(dt);
            U32 outiSize = outNum * bytesOf(DT_U32);
            ow_str = threadNum;
            oh_str = ih;
            ow_off = 0;
            oh_off = 0;
            CHECK_STATUS(gcl_create_sub_buffer(outvSize, &offset, tmpbuf, &outv1024));
            CHECK_STATUS(gcl_create_sub_buffer(outiSize, &offset, tmpbuf, &outi1024));
            gs[0] = threadNum;
            gs[1] = ih;
            gs[2] = ic;
            CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str, oh_str,
                ow_off, oh_off, len, gs[0], gs[1], inv1024, ini1024, outv1024, outi1024));
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
            inv128 = outv1024;
            ini128 = outi1024;
            iw_str = ow_str;
            ih_str = oh_str;
            iw_off = ow_off;
            ih_off = oh_off;
            use_index = true;
            len = threadNum;
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
            CHECK_STATUS(gcl_print_buffer<F16>(handle, inv1024, input->desc.num, "argmax_input"));
            CHECK_STATUS(gcl_print_buffer<F16>(handle, outv1024, outNum, "argmax_output_value"));
            CHECK_STATUS(gcl_print_buffer<U32>(handle, outi1024, outNum, "argmax_output_value"));
#endif
        }

        get_thread_num(len, 128, threadNum);
        if (threadNum > 1) {
            U32 outNum = 128 * ih * ic;
            U32 outvSize = outNum * bytesOf(dt);
            U32 outiSize = outNum * bytesOf(DT_U32);
            ow_str = threadNum;
            oh_str = ih;
            ow_off = 0;
            oh_off = 0;
            CHECK_STATUS(gcl_create_sub_buffer(outvSize, &offset, tmpbuf, &outv128));
            CHECK_STATUS(gcl_create_sub_buffer(outiSize, &offset, tmpbuf, &outi128));
            gs[0] = threadNum;
            gs[1] = ih;
            gs[2] = ic;
            if (use_index) {
                CHECK_STATUS(gcl_create_kernel(handle, kernelNameIndex, &kernel));
            } else {
                CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
            }
            CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str, oh_str,
                ow_off, oh_off, len, gs[0], gs[1], inv128, ini128, outv128, outi128));
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
            inv1 = outv128;
            ini1 = outi128;
            iw_str = ow_str;
            ih_str = oh_str;
            iw_off = ow_off;
            ih_off = oh_off;
            use_index = true;
            len = threadNum;
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
            CHECK_STATUS(gcl_print_buffer<F16>(handle, outv128, outNum, "argmax_output_index"));
            CHECK_STATUS(gcl_print_buffer<U32>(handle, outi128, outNum, "argmax_output_value"));
#endif
        }

        gs[0] = 1;
        gs[1] = ih;
        gs[2] = ic;
        get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
        if (use_index) {
            CHECK_STATUS(gcl_create_kernel(handle, kernelNameIndex, &kernel));
        } else {
            CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        }
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str, oh_str,
            ow_off, oh_off, len, gs[0], gs[1], inv1, ini1, output->mem, output->mem));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        if (use_index) {
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelNameIndex));
        } else {
            CHECK_STATUS(gcl_print_memory<F16>(handle, input, "argmax_input"));
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
        }
        CHECK_STATUS(gcl_print_memory<U32>(handle, output, "argmax_output"));
#endif
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE argmax_infer_forward_tmp_bytes_mali_fp16(
    TensorDesc inputDesc, I32 axis, TensorDesc outputDesc, U32 *bytes)
{
    UNUSED(axis);
    UNUSED(outputDesc);
    DataType dt = inputDesc.dt;
    U32 iw, ih, ic;
    U32 inDims = inputDesc.nDims;
    iw = inputDesc.dims[0];
    ih = (inDims > 1) ? inputDesc.dims[1] : 1;
    ic = (inDims > 2) ? inputDesc.dims[2] : 1;
    U32 size = 1024 * ih * ic * bytesOf(dt);
    size += 1024 * ih * ic * bytesOf(DT_U32);
    size += (128 * ih * ic * bytesOf(dt) + 1023) / 1024 * 1024;
    size += (128 * ih * ic * bytesOf(DT_U32) + 1023) / 1024 * 1024;
    *bytes = size;
    return SUCCESS;
}

EE argmax_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    I32 axis,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(argmax_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(argmax_core_mali_fp16(handle, inputDesc, input, axis, tmpbuf, outputDesc, output));
    return SUCCESS;
}
