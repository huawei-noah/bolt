// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include <algorithm>
#include <unordered_set>

#include "gpu/mali/fp16/eltwise_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/eltwise_opt.h"

bool eltwise_same_desc(std::vector<TensorDesc> inputDesc, U32 *arrayDimMax)
{
    U32 size = inputDesc.size();
    U32 dimMax = 0;
    for (U32 i = 1; i < size; i++) {
        if (inputDesc[i].nDims > inputDesc[dimMax].nDims) {
            dimMax = i;
        } else if (inputDesc[i].nDims == inputDesc[dimMax].nDims) {
            U32 nDims = inputDesc[dimMax].nDims;
            U32 sign[8];
            if (nDims > 8) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
            for (U32 j = 0; j < nDims; j++) {
                if (inputDesc[i].dims[j] > inputDesc[dimMax].dims[j]) {
                    sign[j] = 2;
                } else if (inputDesc[i].dims[j] == inputDesc[dimMax].dims[j]) {
                    sign[j] = 1;
                } else {
                    sign[j] = 0;
                }
            }
            if (*std::max_element(sign, sign + nDims) == 2 &&
                *std::min_element(sign, sign + nDims) == 1) {
                dimMax = i;
            }
            if (*std::max_element(sign, sign + nDims) == 2 &&
                *std::min_element(sign, sign + nDims) == 0) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
    }

    bool sameDesc = true;
    DataFormat idf;
    U32 in, ic, ih, iw, it;
    tensorSelectGet(inputDesc[0], NULL, &idf, &in, &ic, &ih, &iw, &it);
    for (U32 i = 1; i < size; i++) {
        DataFormat tdf;
        U32 tn, tc, th, tw, tt;
        tensorSelectGet(inputDesc[i], NULL, &tdf, &tn, &tc, &th, &tw, &tt);
        if (tdf != idf || in != tn || ic != tc || ih != th || iw != tw || it != tt) {
            sameDesc = false;
            break;
        }
    }
    *arrayDimMax = dimMax;
    return sameDesc;
}

inline bool needTransInput(GCLMemDesc gclmemInputDesc, GCLMemDesc gclmemBroadDesc)
{
    DataFormat imf = gclmemInputDesc.memFormat;
    DataFormat bmf = gclmemBroadDesc.memFormat;
    U32 bw_str, bh_str, bw_off, bh_off;
    CHECK_STATUS(gclmem_get_desc_padding(gclmemBroadDesc, &bw_str, &bh_str, NULL, NULL, NULL));
    bool needTrans = true;
    if (imf == bmf) {
        needTrans = false;
    }
    if (bw_str == 1 && bh_str == 1) {
        needTrans = false;
    }
    return needTrans;
}

inline EE eltwise_checkpara_mali_fp16(
    std::vector<TensorDesc> inputDesc, std::vector<void *> input, TensorDesc outputDesc)
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

inline EE eltwise_core_mali_fp16(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    EltwiseParamSpec eltwiseDesc)
{
    UNUSED(outputDesc);
    U32 iw, ih, ic, in, it;
    U32 arrayDimMax;
    bool sameDesc = eltwise_same_desc(inputDesc, &arrayDimMax);
    tensorSelectGet(inputDesc[arrayDimMax], NULL, NULL, &in, &ic, &ih, &iw, &it);

    U32 num = input.size();
    std::vector<GCLMem_t> inputMem;
    for (U32 i = 0; i < num; ++i) {
        inputMem.push_back((GCLMem_t)input[i]);
    }
    cl_mem outbuf;
    outbuf = output->mem;

    U32 ow_str, oh_str, oc_str, ow_off, oh_off;
    std::vector<U32> iw_str;
    std::vector<U32> ih_str;
    std::vector<U32> iw_off;
    std::vector<U32> ih_off;
    for (U32 i = 0; i < num; ++i) {
        U32 w_str, h_str, w_off, h_off;
        CHECK_STATUS(
            gclmem_get_desc_padding(inputMem[i]->desc, &w_str, &h_str, NULL, &w_off, &h_off));
        iw_str.push_back(w_str);
        ih_str.push_back(h_str);
        iw_off.push_back(w_off);
        ih_off.push_back(h_off);
    }
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off));

    Kernel kernel;
    KernelOpt kernelOpt;
    char kernelName[128];
    bool useNchwFormat = (inputMem[arrayDimMax]->desc.memFormat == DF_NCHW) ? true : false;
    EltwiseMode eltwiseMode = eltwiseDesc.elt_mode;
    ActivationMode activeMode = eltwiseDesc.activation_type;
    U32 gs[3] = {ih, iw, (ic + 3) / 4 * in * it};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    if (useNchwFormat) {
        gs[0] = (iw + 3) / 4;
        gs[1] = ih;
        gs[2] = ic;
    }

    if (sameDesc) {
        CHECK_STATUS(set_eltwise_opt_mali(
            num, useNchwFormat, eltwiseMode, activeMode, DT_F16, kernelName, &kernelOpt));
        ic = ic * in * it;
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        switch (num) {
            case 1:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ic, oh_str, ow_str, oh_off, ow_off,
                    gs[0], gs[1], ih_str[0], iw_str[0], ih_off[0], iw_off[0], inputMem[0]->mem,
                    outbuf));
                break;
            case 2:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ic, oh_str, ow_str, oh_off, ow_off,
                    gs[0], gs[1], ih_str[0], iw_str[0], ih_off[0], iw_off[0], inputMem[0]->mem,
                    ih_str[1], iw_str[1], ih_off[1], iw_off[1], inputMem[1]->mem, outbuf));
                break;
            case 3:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ic, oh_str, ow_str, oh_off, ow_off,
                    gs[0], gs[1], ih_str[0], iw_str[0], ih_off[0], iw_off[0], inputMem[0]->mem,
                    ih_str[1], iw_str[1], ih_off[1], iw_off[1], inputMem[1]->mem, ih_str[2],
                    iw_str[2], ih_off[2], iw_off[2], inputMem[2]->mem, outbuf));
                break;
            case 4:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ic, oh_str, ow_str, oh_off, ow_off,
                    gs[0], gs[1], ih_str[0], iw_str[0], ih_off[0], iw_off[0], inputMem[0]->mem,
                    ih_str[1], iw_str[1], ih_off[1], iw_off[1], inputMem[1]->mem, ih_str[2],
                    iw_str[2], ih_off[2], iw_off[2], inputMem[2]->mem, ih_str[3], iw_str[3],
                    ih_off[3], iw_off[3], inputMem[3]->mem, outbuf));
                break;
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
        handle->t_total += handle->t_execute;
#endif
    } else {
        if (num > 2) {
            CHECK_STATUS(NOT_SUPPORTED)
        }
        GCLMemDesc gclmemImaxDesc = inputMem[arrayDimMax]->desc;
        GCLMemDesc gclmemBroadDesc = inputMem[1 - arrayDimMax]->desc;
        bool needTrans = needTransInput(gclmemImaxDesc, gclmemBroadDesc);
        Mem iMaxMem = inputMem[arrayDimMax]->mem;
        Mem broadMem = inputMem[1 - arrayDimMax]->mem;
        Mem tmp = tmpbuf->mem;
        U32 bn, bc, bh, bw;
        U32 mw_str, mh_str, mw_off, mh_off;
        U32 bw_str, bh_str, bw_off, bh_off;
        tensorSelectGet(inputDesc[1 - arrayDimMax], NULL, NULL, &bn, &bc, &bh, &bw);
        CHECK_STATUS(gclmem_get_desc_padding(
            inputMem[arrayDimMax]->desc, &mw_str, &mh_str, NULL, &mw_off, &mh_off));
        CHECK_STATUS(gclmem_get_desc_padding(
            inputMem[1 - arrayDimMax]->desc, &bw_str, &bh_str, NULL, &bw_off, &bh_off));

        if (needTrans) {
            GCLMem tMem;
            GCLMemDesc desc = gclmemBroadDesc;
            desc.offset[0] = 0;
            desc.offset[1] = 0;
            desc.offset[2] = 0;
            bw_str = bw;
            bh_str = bh;
            bw_off = 0;
            bh_off = 0;
            if (desc.memFormat == DF_NCWHC4) {
                desc.stride[0] = bw;
                desc.stride[1] = bh;
                desc.stride[2] = bc * bn;
                desc.memFormat = DF_NCHW;
                tMem.desc = desc;
                tMem.mem = tmp;
                CHECK_STATUS(ocl_data_trans_form(
                    handle, inputMem[1 - arrayDimMax], &tMem, 0, 0, NCWHC4_TO_NCHW));
                broadMem = tmp;
            } else if (desc.memFormat == DF_NCHW) {
                desc.stride[0] = bh;
                desc.stride[1] = bw;
                desc.stride[2] = (bc + 3) / 4 * bn;
                desc.memFormat = DF_NCWHC4;
                tMem.desc = desc;
                tMem.mem = tmp;
                CHECK_STATUS(ocl_data_trans_form(
                    handle, inputMem[1 - arrayDimMax], &tMem, 0, 0, NCHW_TO_NCWHC4));
                broadMem = tmp;
            }
        }
        bool axisSpeMode = false;
        if (gclmemImaxDesc.memFormat == DF_NCHW && bw == 1) {
            axisSpeMode = true;
        }
        if (gclmemImaxDesc.memFormat == DF_NCWHC4 && bc == 1) {
            axisSpeMode = true;
        }
        CHECK_STATUS(set_eltwise_broadcast_opt_mali(
            useNchwFormat, axisSpeMode, eltwiseMode, activeMode, DT_F16, kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, mh_str, mw_str, mh_off, mw_off, bh_str, bw_str,
            bh_off, bw_off, oh_str, ow_str, oh_off, ow_off, iw, bh, bw, bc, gs[0], gs[1], iMaxMem,
            broadMem, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
        handle->t_total += handle->t_execute;
#endif
    }
    return SUCCESS;
}

EE eltwise_infer_forward_tmp_bytes_mali_fp16(
    std::vector<TensorDesc> inputDesc, std::vector<GCLMemDesc> gclmemInputDesc, U32 *bytes)
{
    U32 size = 0;
    U32 arrayDimMax;
    bool sameDesc = eltwise_same_desc(inputDesc, &arrayDimMax);
    if (!sameDesc) {
        bool needTrans =
            needTransInput(gclmemInputDesc[arrayDimMax], gclmemInputDesc[1 - arrayDimMax]);
        if (needTrans) {
            size = tensorNumBytes(inputDesc[1 - arrayDimMax]);
        }
    }
    *bytes = size;
    return SUCCESS;
}

EE eltwise_mali_fp16(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    EltwiseParamSpec eltwiseDesc)
{
    CHECK_STATUS(eltwise_checkpara_mali_fp16(inputDesc, input, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(
        eltwise_core_mali_fp16(handle, inputDesc, input, tmpbuf, outputDesc, output, eltwiseDesc));
    return SUCCESS;
}
