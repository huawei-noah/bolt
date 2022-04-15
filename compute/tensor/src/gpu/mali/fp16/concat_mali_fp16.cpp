// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/concat_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/concat_opt.h"

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

inline EE concat_core_mali_nchw_fp16(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    TensorDesc outputDesc,
    GCLMem_t output,
    GCLMem_t tmpbuf,
    I32 concatDim)
{
    U32 ow, oh, oc, on;
    U32 ow_str, oh_str, oc_str, ow_off, oh_off, ohw_str, o_off;
    GCLMemType outputMemType = output->desc.memType;
    CHECK_STATUS(gclmem_get_desc_dim(output->desc, NULL, NULL, &on, &oc, &oh, &ow));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off));
    ohw_str = ow_str * oh_str;
    o_off = oh_off * ow_str + ow_off;

    U32 num = input.size();
    GCLMem_t inputMem[4];
    cl_mem inbuf[4];
    I32 dim = outputDesc.nDims;
    concatDim = (concatDim + dim) % dim;
    concatDim = dim - 1 - concatDim;
    bool concatDimWAlign = true;
    if (concatDim == 0) {
        for (auto p : inputDesc) {
            U32 tw;
            CHECK_STATUS(tensorSelectGet(p, NULL, NULL, NULL, NULL, NULL, &tw));
            if (tw % 4 != 0) {
                concatDimWAlign = false;
                break;
            }
        }
    }
    if (!concatDimWAlign) {
        if (outputMemType != GCL_MEM_BUF) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    char kernelName[128];
    KernelOpt kernelOpt;

    U32 bn = (num + 3) / 4;
    U32 iw[4];
    U32 axis_len[4];
    U32 en, nmax, axis_max;
    U32 out_size = 0;
    U32 ih_str[4];
    U32 iw_str[4];
    U32 ih_off[4];
    U32 iw_off[4];
    U32 i_off[4];
    GCLMemType inputMemType[4];
    cl_mem outbuf = output->mem;

    for (U32 i = 0; i < bn; i++) {
        en = (i * 4 + 4 <= num) ? 4 : (num & 3);
        axis_max = 0;
        nmax = en - 1;
        for (U32 j = 0; j < en; ++j) {
            inputMem[j] = (GCLMem_t)input[i * 4 + j];
            inbuf[j] = inputMem[j]->mem;
            CHECK_STATUS(gclmem_get_desc_padding(
                inputMem[j]->desc, &iw_str[j], &ih_str[j], NULL, &iw_off[j], &ih_off[j]));
            i_off[j] = ih_off[j] * iw_str[j] + iw_off[j];
            inputMemType[j] = inputMem[j]->desc.memType;

            iw[j] = inputDesc[i * 4 + j].dims[0];
            axis_len[j] = inputDesc[i * 4 + j].dims[concatDim];
            if (concatDim == 0) {
                axis_len[j] = (axis_len[j] + 3) / 4;
            }
            axis_max += axis_len[j];
        }

        U32 gs[3] = {(ow + 3) / 4, oh, oc * on};
        gs[concatDim] = axis_max;
        if (concatDim == 2) {
            gs[2] *= on;
        }
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        CHECK_STATUS(set_concat_opt_mali(concatDim, en, true, concatDimWAlign, DT_F16, inputMemType,
            outputMemType, kernelName, &kernelOpt));
        Kernel kernel;
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        switch (en) {
            case 1:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ow_str, ohw_str, o_off, axis_max, nmax,
                    out_size, oc, gs[0], gs[1], iw_str[0], ih_str[0], i_off[0], iw[0], axis_len[0],
                    inbuf[0], outbuf));
                break;
            case 2:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ow_str, ohw_str, o_off, axis_max, nmax,
                    out_size, oc, gs[0], gs[1], iw_str[0], ih_str[0], i_off[0], iw[0], axis_len[0],
                    inbuf[0], iw_str[1], ih_str[1], i_off[1], iw[1], axis_len[1], inbuf[1], outbuf));
                break;
            case 3:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ow_str, ohw_str, o_off, axis_max, nmax,
                    out_size, oc, gs[0], gs[1], iw_str[0], ih_str[0], i_off[0], iw[0], axis_len[0],
                    inbuf[0], iw_str[1], ih_str[1], i_off[1], iw[1], axis_len[1], inbuf[1],
                    iw_str[2], ih_str[2], i_off[2], iw[2], axis_len[2], inbuf[2], outbuf));
                break;
            case 4:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ow_str, ohw_str, o_off, axis_max, nmax,
                    out_size, oc, gs[0], gs[1], iw_str[0], ih_str[0], i_off[0], iw[0], axis_len[0],
                    inbuf[0], iw_str[1], ih_str[1], i_off[1], iw[1], axis_len[1], inbuf[1],
                    iw_str[2], ih_str[2], i_off[2], iw[2], axis_len[2], inbuf[2], iw_str[3],
                    ih_str[3], i_off[3], iw[3], axis_len[3], inbuf[3], outbuf));
                break;
            default:
                return NOT_SUPPORTED;
        }
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
        if (outputMemType == GCL_MEM_BUF) {
            if (concatDim == 0) {
                out_size += iw[0] + iw[1] + iw[2] + iw[3];
            } else if (concatDim == 1) {
                out_size += ow_str * gs[1];
            } else if (concatDim == 2) {
                out_size += oh_str * ow_str * gs[2] / on;
            }
        } else {
            if (concatDim == 2) {
                out_size += gs[concatDim] / on;
            } else {
                out_size += gs[concatDim];
            }
        }
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
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
    U32 ow, oh, oc, on;
    U32 ow_str, oh_str, oc_str, ow_off, oh_off, o_off;
    GCLMemType outputMemType;
    CHECK_STATUS(gclmem_get_desc_dim(output->desc, NULL, NULL, &on, &oc, &oh, &ow));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off));
    o_off = oh_off * ow_str + ow_off;
    outputMemType = output->desc.memType;

    U32 num = input.size();
    GCLMem_t inputMem[4];
    cl_mem inbuf[4];
    I32 dim = outputDesc.nDims;
    concatDim = (concatDim + dim) % dim;
    concatDim = dim - 1 - concatDim;
    char kernelName[128];
    KernelOpt kernelOpt;
    bool concatDimCAlign = true;
    U32 bn = (num + 3) / 4;
    if (concatDim == 2) {
        for (auto p : inputDesc) {
            U32 tc;
            CHECK_STATUS(tensorSelectGet(p, NULL, NULL, NULL, &tc, NULL, NULL));
            if (tc % 4 != 0) {
                concatDimCAlign = false;
                break;
            }
        }
    }
    U32 ic[4];
    U32 axis_len[4];
    U32 en, nmax, axis_max;
    U32 out_size = 0;
    U32 ih_str[4];
    U32 iw_str[4];
    U32 ih_off[4];
    U32 iw_off[4];
    U32 i_off[4];
    GCLMemType inputMemType[4];

    U32 ow_val = ow_str;
    U32 ohw_val = oh_str * ow_str;
    U32 o_off_val = o_off;
    GCLMemType outputMemTypeVal = outputMemType;
    cl_mem outbuf = output->mem;
    U32 offset = 0;
    if (!concatDimCAlign) {
        Mem subOut;
        U32 size = tensorNumBytes(outputDesc);
        ow_val = ow;
        ohw_val = oh * ow;
        o_off_val = 0;
        outputMemTypeVal = GCL_MEM_BUF;
        CHECK_STATUS(gcl_create_sub_buffer(size, &offset, tmpbuf, &subOut));
        outbuf = subOut;
    }

    for (U32 i = 0; i < bn; i++) {
        en = (i * 4 + 4 <= num) ? 4 : (num & 3);
        axis_max = 0;
        nmax = en - 1;
        for (U32 j = 0; j < en; ++j) {
            GCLMem subInMem;
            inputMem[j] = (GCLMem_t)input[i * 4 + j];
            if (inputMem[j]->desc.memFormat == DF_NCHW) {
                GCLMemDesc desc = inputMem[j]->desc;
                Mem subIn;
                U32 size = tensorNumBytes(inputDesc[i * 4 + j]);
                CHECK_STATUS(gcl_create_sub_buffer(size, &offset, tmpbuf, &subIn));
                subInMem.mem = subIn;
                U32 iw, ih, ic, in;
                CHECK_STATUS(tensorSelectGet(inputDesc[i * 4 + j], NULL, NULL, &in, &ic, &ih, &iw));
                desc.stride[0] = iw;
                desc.stride[1] = ih;
                desc.stride[2] = (ic + 3) / 4 * in;
                desc.offset[0] = 0;
                desc.offset[1] = 0;
                desc.offset[2] = 0;
                desc.memFormat = DF_NCHWC4;
                desc.memType = GCL_MEM_BUF;
                subInMem.mem = subIn;
                subInMem.desc = desc;
                CHECK_STATUS(
                    ocl_data_trans_form(handle, inputMem[j], &subInMem, 0, 0, NCHW_TO_NCHWC4));
                inputMem[j] = &subInMem;
            }
            inbuf[j] = inputMem[j]->mem;
            get_gclmem_dim(inputMem[j]->desc, &iw_str[j], &ih_str[j], NULL, &iw_off[j], &ih_off[j]);
            i_off[j] = ih_off[j] * iw_str[j] + iw_off[j];
            inputMemType[j] = inputMem[j]->desc.memType;
        }
        for (U32 j = 0; j < en; ++j) {
            axis_len[j] = inputDesc[i * 4 + j].dims[concatDim];
            ic[j] = 0;
            if (concatDim == 2) {
                ic[j] = axis_len[j];
                axis_len[j] = (axis_len[j] + 3) / 4;
            }
            axis_max += axis_len[j];
        }
        U32 gs[3] = {ow, oh, (oc + 3) / 4 * on};
        gs[concatDim] = axis_max;
        if (concatDim == 2) {
            gs[2] *= on;
        }
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        CHECK_STATUS(set_concat_opt_mali(concatDim, en, false, concatDimCAlign, DT_F16,
            inputMemType, outputMemTypeVal, kernelName, &kernelOpt));
        Kernel kernel;
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        switch (en) {
            case 1:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ow_val, ohw_val, o_off_val, axis_max, nmax,
                    out_size, oc, gs[0], gs[1], iw_str[0], ih_str[0], i_off[0], ic[0], axis_len[0],
                    inbuf[0], outbuf));
                break;
            case 2:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ow_val, ohw_val, o_off_val, axis_max, nmax,
                    out_size, oc, gs[0], gs[1], iw_str[0], ih_str[0], i_off[0], ic[0], axis_len[0],
                    inbuf[0], iw_str[1], ih_str[1], i_off[1], ic[1], axis_len[1], inbuf[1], outbuf));
                break;
            case 3:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ow_val, ohw_val, o_off_val, axis_max, nmax,
                    out_size, oc, gs[0], gs[1], iw_str[0], ih_str[0], i_off[0], ic[0], axis_len[0],
                    inbuf[0], iw_str[1], ih_str[1], i_off[1], ic[1], axis_len[1], inbuf[1],
                    iw_str[2], ih_str[2], i_off[2], ic[2], axis_len[2], inbuf[2], outbuf));
                break;
            case 4:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ow_val, ohw_val, o_off_val, axis_max, nmax,
                    out_size, oc, gs[0], gs[1], iw_str[0], ih_str[0], i_off[0], ic[0], axis_len[0],
                    inbuf[0], iw_str[1], ih_str[1], i_off[1], ic[1], axis_len[1], inbuf[1],
                    iw_str[2], ih_str[2], i_off[2], ic[2], axis_len[2], inbuf[2], iw_str[3],
                    ih_str[3], i_off[3], ic[3], axis_len[3], inbuf[3], outbuf));
                break;
            default:
                return NOT_SUPPORTED;
        }
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
        if (!concatDimCAlign) {
            out_size += oh * ow * (ic[0] + ic[1] + ic[2] + ic[3]);
        } else {
            if (outputMemTypeVal == GCL_MEM_BUF) {
                if (concatDim == 0) {
                    out_size += gs[0] * 4;
                }
                if (concatDim == 1) {
                    out_size += ow_str * gs[1] * 4;
                }
                if (concatDim == 2) {
                    out_size += (oh_str * ow_str * gs[2] * 4 / on);
                }
            } else {
                if (concatDim == 2) {
                    out_size += gs[concatDim] / on;
                } else {
                    out_size += gs[concatDim];
                }
            }
        }
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    }
    if (!concatDimCAlign) {
        GCLMem tMem;
        GCLMemDesc desc = output->desc;
        U32 str[3] = {ow, oh, oc * on};
        U32 off[3] = {0, 0, 0};
        MemFlags flag = CL_MEM_READ_WRITE;
        CHECK_STATUS(gclmem_set_desc_padding(&desc, str, off, DT_F16, DF_NCHW, GCL_MEM_BUF, flag));
        tMem.desc = desc;
        tMem.mem = tmpbuf->mem;
        CHECK_STATUS(ocl_data_trans_form(handle, &tMem, output, 0, 0, NCHW_TO_NCHWC4));
    }
    return SUCCESS;
}

EE concat_infer_forward_tmp_bytes_mali_fp16(
    std::vector<TensorDesc> inputDesc, std::vector<GCLMemDesc> gclmemInputDesc, U32 *bytes)
{
    U32 size = 0;
    bool useNchw = true;
    for (auto p : gclmemInputDesc) {
        if (p.memFormat == DF_NCHWC4) {
            useNchw = false;
            break;
        }
    }
    if (!useNchw) {
        bool concatDimCAlign = true;
        for (auto p : inputDesc) {
            U32 tc;
            CHECK_STATUS(tensorSelectGet(p, NULL, NULL, NULL, &tc, NULL, NULL));
            if (tc % 4 != 0) {
                concatDimCAlign = false;
                break;
            }
        }
        if (!concatDimCAlign) {
            for (auto p : inputDesc) {
                size += tensorNumBytes(p);
            }
            size = ALIGN(size, BUFFER_ALIGN_BASE);
        }
        for (U32 i = 0; i < gclmemInputDesc.size(); i++) {
            if (gclmemInputDesc[i].memFormat == DF_NCHW) {
                DataType dt;
                U32 iw, ih, ic, in;
                CHECK_STATUS(tensorSelectGet(inputDesc[i], &dt, NULL, &in, &ic, &ih, &iw));
                U32 sizeAlignC4 = iw * ih * ALIGN(ic, 4) * in * bytesOf(dt);
                size += ALIGN(sizeAlignC4, BUFFER_ALIGN_BASE);
            }
        }
    }
    *bytes = size;
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

    DataFormat mf = ((GCLMem_t)output)->desc.memFormat;
    if (mf == DF_NCHWC4) {
        CHECK_STATUS(
            concat_core_mali_fp16(handle, inputDesc, input, outputDesc, output, tmpbuf, concatDim));
    } else if (mf == DF_NCHW) {
        CHECK_STATUS(concat_core_mali_nchw_fp16(
            handle, inputDesc, input, outputDesc, output, tmpbuf, concatDim));
    } else {
        CHECK_STATUS(NOT_SUPPORTED)
    }
    return SUCCESS;
}
