// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/topk_mali_fp16.h"

inline EE topk_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != DT_F16 && inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE topk_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TopKParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    TensorDesc outputIndicesDesc,
    GCLMem_t outputIndices)
{
    U32 iw_str, ih_str, ic_str, iw_off, ih_off;
    U32 ow_str, oh_str, oc_str, ow_off, oh_off;
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off));

    I32 axis = p.axis;
    if (axis < 0) {
        axis += inputDesc.nDims;
    }
    axis = inputDesc.nDims - 1 - axis;
    U32 len = inputDesc.dims[axis];
    I32 sorted = p.sorted;
    I32 top_k = p.topk;
    I32 largest = p.largest;
    char modeName[128];
    if (largest) {
        strcpy(modeName, "max");
    } else {
        strcpy(modeName, "min");
    }
    if (sorted) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    Mem outputId = outputIndices->mem;
    I32 need_out_id = 0;
    if (outputId) {
        need_out_id = 1;
    }

    if (len == tensorNumElements(inputDesc)) {
        if (iw_off != 0 || ih_off != 0 || ow_off != 0 || oh_off != 0) {
            CHECK_STATUS(NOT_SUPPORTED);
        }

        U32 sub_off = 0;
        Mem sub[4];
        Mem sub_id[4];
        U32 num = ALIGN(len, 16);
        U32 size = num * bytesOf(DT_F16);
        CHECK_STATUS(gcl_create_sub_buffer(size, &sub_off, tmpbuf, &sub[0]));
        CHECK_STATUS(gcl_create_sub_buffer(size, &sub_off, tmpbuf, &sub_id[0]));

        num = ((len + 15) / 16 + 1) / 2 * 16;
        size = num * bytesOf(DT_F16);
        CHECK_STATUS(gcl_create_sub_buffer(size, &sub_off, tmpbuf, &sub[1]));
        CHECK_STATUS(gcl_create_sub_buffer(size, &sub_off, tmpbuf, &sub_id[1]));

        num = (num / 16 + 1) / 2 * 16;
        size = num * bytesOf(DT_F16);
        CHECK_STATUS(gcl_create_sub_buffer(size, &sub_off, tmpbuf, &sub[2]));
        CHECK_STATUS(gcl_create_sub_buffer(size, &sub_off, tmpbuf, &sub_id[2]));

        num = (num / 16 + 1) / 2 * 16;
        size = num * bytesOf(DT_F16);
        CHECK_STATUS(gcl_create_sub_buffer(size, &sub_off, tmpbuf, &sub[3]));
        CHECK_STATUS(gcl_create_sub_buffer(size, &sub_off, tmpbuf, &sub_id[3]));

        Kernel kernel;
        char kernelName[1024];
        sprintf(kernelName, "topk_sort_%s", modeName);
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        U32 gs[3] = {0, 0, 0};
        U32 ls[3] = {0, 0, 0};
        U32 dim = 1;
        gs[0] = (len + 15) / 16;
        CHECK_STATUS(gcl_set_kernelArgs(kernel, len, gs[0], input->mem, sub[0], sub_id[0]));
        CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName));
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif

        U32 top_k_loop = (top_k + 15) / 16;
        for (U32 i = 0; i < top_k_loop; i++) {
            U32 mem_in_index = 0;
            U32 mem_out_index = 1;
            U32 out_off = 0;
            U32 out_val_num = 16;
            sprintf(kernelName, "topk_merge_%s", modeName);
            Mem merge_in, merge_out, merge_in_id, merge_out_id;
            gs[0] = (len + 15) / 16;
            ls[0] = 0;
            while (gs[0] > 1) {
                U32 total_group_num = gs[0];
                gs[0] = (gs[0] + 7) / 8;
                merge_in = sub[mem_in_index];
                merge_out = sub[mem_out_index];
                merge_in_id = sub_id[mem_in_index];
                merge_out_id = sub_id[mem_out_index];
                if (gs[0] == 1) {
                    merge_out = output->mem;
                    out_off = i * 16;
                    out_val_num = ((i * 16 + 16) <= (U32)top_k) ? 16 : (top_k % 16);
                }
                CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
                CHECK_STATUS(gcl_set_kernelArgs(kernel, total_group_num, out_val_num, out_off,
                    gs[0], merge_in, merge_in_id, merge_out, merge_out_id));
                CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName));
#ifdef _DEBUG
                CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
                if (gs[0] > 1) {
                    mem_in_index++;
                    mem_out_index++;
                    if (mem_in_index > 3) {
                        mem_in_index = 1;
                    }
                    if (mem_out_index > 3) {
                        mem_out_index = 1;
                    }
                }
            }

            if (i < top_k_loop - 1 || need_out_id) {
                sprintf(kernelName, "topk_update_%s", modeName);
                gs[0] = 16;
                ls[0] = 16;
                int out_id_off = out_off;
                int out_id_num = out_val_num;
                if (!need_out_id) {
                    outputId = sub_id[0];
                }
                CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
                CHECK_STATUS(gcl_set_kernelArgs(kernel, need_out_id, out_id_off, out_id_num, gs[0],
                    merge_out_id, sub[0], sub_id[0], outputId));
                CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName));
#ifdef _DEBUG
                CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
            }
        }
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE topk_infer_forward_tmp_bytes_mali_fp16(
    TensorDesc inputDesc, TopKParamSpec p, TensorDesc outputDesc, U32 *bytes)
{
    UNUSED(outputDesc);
    U32 totalNum = tensorNumElements(inputDesc);
    U32 axis = p.axis;
    if (axis < 0) {
        axis += inputDesc.nDims;
    }
    axis = inputDesc.nDims - 1 - axis;
    U32 len = inputDesc.dims[axis];

    U32 tmpBytes = 0;
    U32 num = ALIGN(len, 16);
    tmpBytes += ALIGN(num * bytesOf(DT_F16), 1024) * 2;
    num = ((len + 15) / 16 + 1) / 2 * 16;
    tmpBytes += ALIGN(num * bytesOf(DT_F16), 1024) * 2;
    num = (num / 16 + 1) / 2 * 16;
    tmpBytes += ALIGN(num * bytesOf(DT_F16), 1024) * 2;
    num = (num / 16 + 1) / 2 * 16;
    tmpBytes += ALIGN(num * bytesOf(DT_F16), 1024) * 2;

    tmpBytes *= (totalNum / len);
    *bytes = tmpBytes;
    return SUCCESS;
}

EE topk_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TopKParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    TensorDesc outputIndicesDesc,
    GCLMem_t outputIndices)
{
    CHECK_STATUS(topk_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(topk_core_mali_fp16(
        handle, inputDesc, input, p, tmpbuf, outputDesc, output, outputIndicesDesc, outputIndices));
    return SUCCESS;
}
