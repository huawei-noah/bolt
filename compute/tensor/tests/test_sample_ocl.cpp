// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ut_util_ocl.h"

void topk_max_cpu(F16 *val, U32 top_k, U32 len)
{
    for (U32 i = 0; i < top_k; i++) {
        U32 index = 0;
        F16 max_val = -65536;
        for (U32 j = 0; j < len; j++) {
            if (val[j] > max_val) {
                max_val = val[j];
                index = j;
            }
        }
        printf("%f %d\n", max_val, index);
        val[index] = -65536;
    }
}

int main()
{
    GCLHandle_t handle = OCLContext::getInstance().handle.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;

    U32 len = 3000;
    U32 top_k = 44;
    TensorDesc inputDesc = tensor1d(DT_F16, len);
    TensorDesc outputDesc = tensor1d(DT_F16, top_k);
    TensorDesc outputIdDesc = tensor1d(DT_I32, top_k);

    Tensor inputTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);
    Tensor outputIdTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);

    inputTensor.resize(inputDesc);
    outputTensor.resize(outputDesc);
    outputIdTensor.resize(outputIdDesc)

        U32 tmpBytes = 0;
    U32 num = (len + 15) / 16 * 16;
    tmpBytes += (num * bytesOf(DT_F16) + 1023) / 1024 * 1024 * 2;
    num = ((len + 15) / 16 + 1) / 2 * 16;
    tmpBytes += (num * bytesOf(DT_F16) + 1023) / 1024 * 1024 * 2;
    num = (num / 16 + 1) / 2 * 16;
    tmpBytes += (num * bytesOf(DT_F16) + 1023) / 1024 * 1024 * 2;
    num = (num / 16 + 1) / 2 * 16;
    tmpBytes += (num * bytesOf(DT_F16) + 1023) / 1024 * 1024 * 2;

    F16 *input_val = (F16 *)malloc(tensorNumBytes(inputDesc));
    for (U32 i = 0; i < len; i++)
        input_val[i] = (rand() % 1024) * 1.0;
    topk_max_cpu(input_val_cpu, top_k, len);

    U32 lenAlign = (len + 15) / 16 * 16;
    GCLMem_t input = alloc_padding(inputTensor, 0, lenAlign - len, 0, 0, input_val);
    GCLMem_t output = alloc(outputTensor);
    GCLMem_t outputId = alloc(outputIdTensor);
    GCLMem_t tmp = alloc_bytes(tmpTensor, tmpBytes);

    U32 off = 0;
    Mem sub[4];
    Mem sub_id[4];

    num = (len + 15) / 16 * 16;
    U32 size = num * bytesOf(DT_F16);
    CHECK_STATUS(gcl_create_sub_buffer(size, &off, tmp, &sub[0]));
    CHECK_STATUS(gcl_create_sub_buffer(size, &off, tmp, &sub_id[0]));

    num = ((len + 15) / 16 + 1) / 2 * 16;
    size = num * bytesOf(DT_F16);
    CHECK_STATUS(gcl_create_sub_buffer(size, &off, tmp, &sub[1]));
    CHECK_STATUS(gcl_create_sub_buffer(size, &off, tmp, &sub_id[1]));

    num = (num / 16 + 1) / 2 * 16;
    size = num * bytesOf(DT_F16);
    CHECK_STATUS(gcl_create_sub_buffer(size, &off, tmp, &sub[2]));
    CHECK_STATUS(gcl_create_sub_buffer(size, &off, tmp, &sub_id[2]));

    num = (num / 16 + 1) / 2 * 16;
    size = num * bytesOf(DT_F16);
    CHECK_STATUS(gcl_create_sub_buffer(size, &off, tmp, &sub[3]));
    CHECK_STATUS(gcl_create_sub_buffer(size, &off, tmp, &sub_id[3]));
    gcl_finish(handle);

    Kernel kernel;
    char kernelName[1024];
    sprintf(kernelName, "topk_sort_max");
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    U32 gs[3] = {0, 0, 0};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 1;
    gs[0] = (len + 15) / 16;
    CHECK_STATUS(gcl_set_kernelArgs(kernel, len, gs[0], input->mem, sub[0], sub_id[0]));
    CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName));
    CHECK_STATUS(gcl_run_kernel_select_ls(handle, &kernelVec[0]));

    U32 top_k_loop = (top_k + 15) / 16;
    for (U32 i = 0; i < top_k_loop; i++) {
        U32 mem_in_index = 0;
        U32 mem_out_index = 1;
        U32 out_off = 0;
        U32 out_id_off = 0;
        U32 out_val_num = 16;
        sprintf(kernelName, "topk_merge_max");
        Mem merge_in, merge_out, merge_in_id, merge_out_id;
        gs[0] = (len + 15) / 16;
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
                out_val_num = ((i * 16 + 16) <= top_k) ? 16 : (top_k % 16);
            }
            CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, total_group_num, out_val_num, out_off, gs[0],
                merge_in, merge_in_id, merge_out, merge_out_id));
            CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName));
            CHECK_STATUS(gcl_run_kernel_select_ls(handle, &kernelVec[kernelVec.size() - 1]));
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
            sprintf(kernelName, "topk_update_max");
            gs[0] = 16;
            ls[0] = 16;
            Mem outputIndex = sub_id[0];
            int out_id_off = out_off;
            int out_id_num = out_val_num;
            if (need_out_id) {
                outputIndex = outputId->mem;
            }
            CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, need_out_id, out_id_off, out_id_num, gs[0],
                merge_out_id, sub[0], sub_id[0], outputIndex));
            CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName));
            CHECK_STATUS(gcl_run_kernelVec_timing(handle, kernelVec.size() - 1, kernelVec.size()));
        }
    }
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernelVec_timing(handle, 0, handle->kernelVec->size()));
    CHECK_STATUS(gcl_check_buf<F16>(handle, output->mem, outputGclDesc.byteSize, false, "output"));
    CHECK_STATUS(
        gcl_check_buf<int>(handle, outputId->mem, outputIdGclDesc.byteSize, false, "outputId"));
#else
    CHECK_STATUS(gcl_run_kernelVec(handle));
#endif
    free(input_val);
    return 0;
}
