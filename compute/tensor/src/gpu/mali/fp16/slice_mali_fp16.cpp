// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/slice_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/slice_opt.h"

bool slice_axis_c_align4(U32 target_axis, std::vector<TensorDesc> outputDesc)
{
    bool align4 = true;
    if (target_axis == 2) {
        if (outputDesc[0].nDims != 4) {
            CHECK_STATUS(NOT_MATCH);
        }
        for (U32 i = 0; i < outputDesc.size(); ++i) {
            if (outputDesc[i].dims[2] % 4 != 0) {
                align4 = false;
                break;
            }
        }
    }
    return align4;
}

inline EE slice_checkpara_mali_fp16(TensorDesc inputDesc, std::vector<TensorDesc> outputDesc)
{
    if (inputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    for (auto p : outputDesc) {
        if (p.dt != DT_F16) {
            return NOT_SUPPORTED;
        }
    }
    return SUCCESS;
}

inline EE slice_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    SliceParamSpec p,
    GCLMem_t tmpbuf,
    std::vector<TensorDesc> outputDesc,
    std::vector<void *> *output)
{
    U32 in, ic, ih, iw;
    U32 iw_str, ih_str, iw_off, ih_off;
    CHECK_STATUS(gclmem_get_desc_dim(input->desc, NULL, NULL, &in, &ic, &ih, &iw));
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off));
    Mem inbuf = input->mem;
    I32 axis = p.axis;
    axis = (axis + inputDesc.nDims) % inputDesc.nDims;
    U32 target_axis = inputDesc.nDims - 1 - axis;
    U32 num = outputDesc.size();
    if (target_axis > 2) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    bool useNchwFormat = (input->desc.memFormat == DF_NCHW) ? true : false;
    bool needTrans = false;
    if (!useNchwFormat) {
        bool align4 = slice_axis_c_align4(target_axis, outputDesc);
        if (!align4) {
            useNchwFormat = true;
            needTrans = true;
        }
    }
    if (needTrans) {
        GCLMem tMem;
        GCLMemDesc desc = input->desc;
        iw_str = iw;
        ih_str = ih;
        iw_off = 0;
        ih_off = 0;
        inbuf = tmpbuf->mem;
        desc.stride[0] = iw_str;
        desc.stride[1] = ih_str;
        desc.stride[2] = ic * in;
        desc.offset[0] = 0;
        desc.offset[1] = 0;
        desc.offset[2] = 0;
        desc.memFormat = DF_NCHW;
        tMem.mem = inbuf;
        tMem.desc = desc;
        CHECK_STATUS(ocl_data_trans_form(handle, input, &tMem, 0, 0, NCWHC4_TO_NCHW));
    }

    char kernelName[128];
    Kernel kernel;
    KernelOpt kernelOpt;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    GCLMem_t outMem[4];
    Mem outbuf[4];
    U32 ow[4];
    U32 ow_str[4], oh_str[4], ow_off[4], oh_off[4];
    U32 axis_len[4];
    U32 in_size = 0;
    U32 slice_len = 0;
    for (U32 i = 0; i < num; i += 4) {
        U32 slice_num = ((i + 4) <= num) ? 4 : (num & 3);
        U32 nmax = slice_num;
        U32 axis_max = 0;
        U32 axis_total = 0;
        for (U32 j = 0; j < slice_num; j++) {
            outMem[j] = (GCLMem_t)((*output)[j]);
            ow[j] = outMem[j]->desc.dims[0];
            CHECK_STATUS(gclmem_get_desc_padding(
                input->desc, ow_str + j, oh_str + j, NULL, ow_off + j, oh_off + j));
            outbuf[j] = outMem[j]->mem;

            axis_len[j] = outMem[j]->desc.dims[target_axis];
            slice_len += axis_len[j];
            if ((useNchwFormat && target_axis == 0) || (!useNchwFormat && target_axis == 2)) {
                axis_len[j] = (axis_len[j] + 3) / 4;
            }
            axis_total += axis_len[j];
        }
        axis_max = axis_total - axis_len[slice_num - 1];

        if (useNchwFormat) {
            gs[0] = (iw + 3) / 4;
            ;
            gs[1] = ih;
            gs[2] = ic;
        } else {
            gs[0] = ih;
            gs[1] = iw;
            gs[2] = (ic + 3) / 4;
        }
        gs[target_axis] = axis_total;

        CHECK_STATUS(set_scale_opt_mali(
            useNchwFormat, target_axis, slice_num, DT_F16, kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        switch (slice_num) {
            case 1:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, axis_max,
                    nmax, in_size, gs[0], gs[1], inbuf, ow_str[0], oh_str[0], ow_off[0], oh_off[0],
                    ow[0], outbuf[0]));
                break;
            case 2:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, axis_max,
                    nmax, in_size, gs[0], gs[1], inbuf, ow_str[0], oh_str[0], ow_off[0], oh_off[0],
                    ow[0], outbuf[0], ow_str[1], oh_str[1], ow_off[1], oh_off[1], ow[1],
                    axis_len[0], outbuf[1]));
                break;
            case 3:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, axis_max,
                    nmax, in_size, gs[0], gs[1], inbuf, ow_str[0], oh_str[0], ow_off[0], oh_off[0],
                    ow[0], outbuf[0], ow_str[1], oh_str[1], ow_off[1], oh_off[1], ow[1],
                    axis_len[0], outbuf[1], ow_str[2], oh_str[2], ow_off[2], oh_off[2], ow[2],
                    axis_len[1], outbuf[2]));
                break;
            case 4:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, axis_max,
                    nmax, in_size, gs[0], gs[1], inbuf, ow_str[0], oh_str[0], ow_off[0], oh_off[0],
                    ow[0], outbuf[0], ow_str[1], oh_str[1], ow_off[1], oh_off[1], ow[1],
                    axis_len[0], outbuf[1], ow_str[2], oh_str[2], ow_off[2], oh_off[2], ow[2],
                    axis_len[1], outbuf[2], ow_str[3], oh_str[3], ow_off[3], oh_off[3], ow[3],
                    axis_len[2], outbuf[3]));
                break;
            default:
                CHECK_STATUS(NOT_MATCH);
        }
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
        if (useNchwFormat) {
            if (target_axis == 0) {
                in_size += slice_len;
            } else if (target_axis == 1) {
                in_size += slice_len * iw_str;
            } else if (target_axis == 2) {
                in_size += slice_len * iw_str * ih_str;
            }
        } else {
            if (target_axis == 1) {
                in_size += slice_len * 4;
            } else if (target_axis == 0) {
                in_size += slice_len * ih_str * 4;
            } else if (target_axis == 2) {
                in_size += slice_len * iw_str * ih_str;
            }
        }
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    }
    return SUCCESS;
}

EE slice_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    GCLMemDesc gclmemInputDesc,
    SliceParamSpec p,
    std::vector<TensorDesc> outputDesc,
    U32 *bytes)
{
    I32 axis = p.axis;
    axis = (axis + inputDesc.nDims) % inputDesc.nDims;
    U32 target_axis = inputDesc.nDims - 1 - axis;
    U32 size = 0;
    if (gclmemInputDesc.memFormat == DF_NCWHC4) {
        bool align4 = slice_axis_c_align4(target_axis, outputDesc);
        if (!align4) {
            size = tensorNumBytes(inputDesc);
        }
    }
    *bytes = size;
    return SUCCESS;
}

EE slice_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    SliceParamSpec p,
    GCLMem_t tmpbuf,
    std::vector<TensorDesc> outputDesc,
    std::vector<void *> *output)
{
    std::vector<void *> outputArray = *output;
    CHECK_STATUS(slice_checkpara_mali_fp16(inputDesc, outputDesc));
    for (U32 i = 0; i < outputArray.size(); i++) {
        CHECK_STATUS(fill_output_zero(handle, (GCLMem_t)(outputArray[i]), outputDesc[i]));
    }
    CHECK_STATUS(slice_core_mali_fp16(handle, inputDesc, input, p, tmpbuf, outputDesc, output));
    return SUCCESS;
}
