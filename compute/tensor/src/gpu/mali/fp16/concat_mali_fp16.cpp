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
#include "gpu/mali/fp16/concat_mali_fp16.h"

inline EE concat_checkpara_mali_fp16(std::vector<TensorDesc> inputDesc, TensorDesc outputDesc)
{
    for (auto it : inputDesc) {
        if (it.dt != outputDesc.dt) {
            return NOT_SUPPORTED;
        }
    }
    if (outputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE concat_core_mali_fp16(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    TensorDesc outputDesc,
    GCLMem_t output,
    GCLMem_t tmpbuf,
    I32 concatDim)
{
    U32 ow, oh, oc;
    tensorSelectGet(outputDesc, NULL, NULL, NULL, &oc, &oh, &ow);
    U32 ow_str, oh_str, oc_str, ow_off, oh_off;
    get_gclmem_dim(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off);
    U32 num = input.size();
    GCLMem_t inputMem[4];
    cl_mem inbuf[4];
    I32 dim = outputDesc.nDims;
    concatDim = (concatDim + dim) % dim;
    concatDim = dim - 1 - concatDim;
    char kernelName[128];
    char dimName[128];
    U32 axis;
    if (inputDesc[0].df == DF_NCHW) {
        switch (concatDim) {
            case 0:
                strcpy(dimName, "w");
                axis = 1;
                break;
            case 1:
                strcpy(dimName, "h");
                axis = 0;
                break;
            case 2:
                strcpy(dimName, "c");
                axis = 2;
                break;
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    if (inputDesc[0].df == DF_MKT) {
        concatDim = 1 - concatDim;
    }
    if (inputDesc[0].df == DF_MKT || inputDesc[0].df == DF_MTK) {
        switch (concatDim) {
            case 0:
                strcpy(dimName, "c");
                axis = 2;
                break;
            case 1:
                strcpy(dimName, "h");
                axis = 0;
                break;
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    bool concatDimCAlign = true;
    if (axis == 2) {
        for (auto p : inputDesc) {
            U32 tc;
            tensorSelectGet(p, NULL, NULL, NULL, &tc, NULL, NULL);
            if (tc % 4 != 0) {
                concatDimCAlign = false;
                break;
            }
        }
    }
    U32 ic[4];
    U32 axis_len[4];
    U32 bn = (num + 3) / 4;
    U32 en, nmax, axis_max;
    U32 out_size = 0;
    U32 ih_str[4];
    U32 iw_str[4];
    U32 ih_off[4];
    U32 iw_off[4];
    U32 oh_val = oh_str;
    U32 ohw_val = oh_str * ow_str;
    U32 oh_off_val = oh_off;
    U32 ow_off_val = ow_off;
    cl_mem outbuf = output->mem;
    if (!concatDimCAlign) {
        oh_val = oh;
        ohw_val = oh * ow;
        oh_off_val = 0;
        ow_off_val = 0;
        outbuf = tmpbuf->mem;
    }
    for (U32 i = 0; i < bn; i++) {
        en = (i * 4 + 4 <= num) ? 4 : (num & 3);
        axis_max = 0;
        nmax = en - 1;
        for (U32 j = 0; j < en; ++j) {
            inputMem[j] = (GCLMem_t)input[i * 4 + j];
            inbuf[j] = inputMem[j]->mem;
            get_gclmem_dim(inputMem[j]->desc, &iw_str[j], &ih_str[j], NULL, &iw_off[j], &ih_off[j]);
        }
        for (U32 j = 0; j < en; ++j) {
            axis_len[j] = inputDesc[i * 4 + j].dims[concatDim];
            ic[j] = 0;
            if (axis == 2) {
                ic[j] = axis_len[j];
                axis_len[j] = (axis_len[j] + 3) / 4;
            }
            axis_max += axis_len[j];
        }
        U32 gs[3] = {oh, ow, (oc + 3) / 4};
        gs[axis] = axis_max;
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        axis_max -= axis_len[nmax];
        if (!concatDimCAlign) {
            sprintf(kernelName, "concat_nonalign_c_p1_%d", en);
        } else {
            sprintf(kernelName, "concat_%s%d", dimName, en);
        }
        Kernel kernel;
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        switch (en) {
            case 1:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, oh_val, ohw_val, oh_off_val, ow_off_val,
                    axis_max, nmax, out_size, gs[0], gs[1], ih_str[0], iw_str[0], ih_off[0],
                    iw_off[0], ic[0], inbuf[0], outbuf));
                break;
            case 2:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, oh_val, ohw_val, oh_off_val, ow_off_val,
                    axis_max, nmax, out_size, gs[0], gs[1], ih_str[0], iw_str[0], ih_off[0],
                    iw_off[0], ic[0], inbuf[0], ih_str[1], iw_str[1], ih_off[1], iw_off[1], ic[1],
                    axis_len[0], inbuf[1], outbuf));
                break;
            case 3:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, oh_val, ohw_val, oh_off_val, ow_off_val,
                    axis_max, nmax, out_size, gs[0], gs[1], ih_str[0], iw_str[0], ih_off[0],
                    iw_off[0], ic[0], inbuf[0], ih_str[1], iw_str[1], ih_off[1], iw_off[1], ic[1],
                    axis_len[0], inbuf[1], ih_str[2], iw_str[2], ih_off[2], iw_off[2], ic[2],
                    axis_len[1], inbuf[2], outbuf));
                break;
            case 4:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, oh_val, ohw_val, oh_off_val, ow_off_val,
                    axis_max, nmax, out_size, gs[0], gs[1], ih_str[0], iw_str[0], ih_off[0],
                    iw_off[0], ic[0], inbuf[0], ih_str[1], iw_str[1], ih_off[1], iw_off[1], ic[1],
                    axis_len[0], inbuf[1], ih_str[2], iw_str[2], ih_off[2], iw_off[2], ic[2],
                    axis_len[1], inbuf[2], ih_str[3], iw_str[3], ih_off[3], iw_off[3], ic[3],
                    axis_len[2], inbuf[3], outbuf));
                break;
            default:
                return NOT_SUPPORTED;
        }
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
        if (!concatDimCAlign) {
            out_size += oh * ow * (ic[0] + ic[1] + ic[2] + ic[3]);
        } else {
            if (axis == 0) {
                out_size += gs[0] * 4;
            }
            if (axis == 1) {
                out_size += oh_str * gs[1] * 4;
            }
            if (axis == 2) {
                out_size += oh_str * ow_str * gs[2] * 4;
            }
        }
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    }
    if (!concatDimCAlign) {
        U32 gs[3] = {(oh + 3) / 4, ow, (oc + 3) / 4};
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        Kernel kernel;
        CHECK_STATUS(gcl_create_kernel(handle, "mem_trans_nchw_to_ncwhc4_input_tran", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, ow, oh, 0, 0, ow_str, oh_str, ow_off, oh_off, ow,
            oh, oc, ow, oh, oc, 0, 0, tmpbuf->mem, output->mem));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, "mem_trans_nchw_to_ncwhc4_input_tran");
#ifdef _DEBUG
        CHECK_STATUS(
            gcl_run_kernel(handle, kernel, dim, gs, ls, "mem_trans_nchw_to_ncwhc4_input_tran"));
#endif
    }
    return SUCCESS;
}

EE concat_infer_forward_tmp_bytes_mali_fp16(std::vector<TensorDesc> inputDesc, U32 *bytes)
{
    *bytes = 0;
    bool concatDimCAlign = true;
    for (auto p : inputDesc) {
        U32 tc;
        tensorSelectGet(p, NULL, NULL, NULL, &tc, NULL, NULL);
        if (tc % 4 != 0) {
            concatDimCAlign = false;
            break;
        }
    }
    if (!concatDimCAlign) {
        for (auto p : inputDesc) {
            *bytes += tensorNumBytes(p);
        }
    }
    return SUCCESS;
}

EE concat_mali_fp16(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    TensorDesc outputDesc,
    GCLMem_t output,
    GCLMem_t tmpbuf,
    I32 concatDim)
{
    CHECK_STATUS(concat_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(
        concat_core_mali_fp16(handle, inputDesc, input, outputDesc, output, tmpbuf, concatDim));
    return SUCCESS;
}
