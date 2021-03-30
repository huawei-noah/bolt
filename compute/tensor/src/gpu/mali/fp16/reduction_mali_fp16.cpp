// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "tensor_desc.h"

#include "gpu/mali/fp16/reduction_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/copy_opt.h"

inline EE reduction_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != DT_F16) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return SUCCESS;
}

inline EE reduction_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc maskDesc,
    GCLMem_t mask,
    ReductionParamSpec p,
    GCLMem_t tmp,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    int axisTran[6];
    int axis;
    for (int i = 0; i < p.axes_num; i++) {
        axis = p.axes[i];
        if (axis < 0) {
            axis = inputDesc.nDims + axis;
        }
        axis = inputDesc.nDims - 1 - axis;
        axisTran[i] = axis;
    }

    DataFormat imf, omf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    U32 iw_str, ih_str, ic_str, iw_off, ih_off;
    U32 ow_str, oh_str, oc_str, ow_off, oh_off;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off);
    axis = axisTran[0];
    U32 od = outputDesc.nDims;
    imf = input->desc.memFormat;
    omf = output->desc.memFormat;
    Mem inbuf = input->mem;
    Mem outbuf = output->mem;
    Mem tmpbuf = tmp->mem;
    int keep_dim = (p.keep_dim) ? 1 : 0;
    char kernelName[128];
    char modeName[16];
    KernelOpt kernelOpt;
    Kernel kernel;
    U32 gs[3] = {0, 0, 0};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    switch (p.reduction_mode) {
        case REDUCTION_SUM:
            strcpy(modeName, "sum");
            break;
        case REDUCTION_MEAN:
            strcpy(modeName, "mean");
            break;
        case REDUCTION_STD_DEVIATION:
            strcpy(modeName, "std_deviation");
            break;
        case REDUCTION_SCALAR_PRODUCT:
            strcpy(modeName, "scalar_product");
            break;
        default:
            return NOT_SUPPORTED;
    }

    if (imf == DF_NCWHC4 && omf == DF_NCWHC4 && keep_dim) {
        sprintf(kernelName, "reduction_oc4_%s%d", modeName, axis);
        gs[0] = oh;
        gs[1] = ow;
        gs[2] = (oc + 3) / 4 * on;
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, iw_str, ih_off, iw_off, oh_str, ow_str,
            oh_off, ow_off, ih, iw, ic, keep_dim, od, gs[0], gs[1], inbuf, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
        return SUCCESS;
    } else {
        bool needTran = (omf == DF_NCWHC4) ? true : false;
        U32 th_str = oh;
        U32 tw_str = ow;
        U32 th_off = 0;
        U32 tw_off = 0;
        if (!needTran) {
            tmpbuf = outbuf;
            th_str = oh_str;
            tw_str = ow_str;
            th_off = oh_off;
            th_off = ow_off;
        }
        if (imf == DF_NCWHC4) {
            sprintf(kernelName, "reduction_%s%d", modeName, axis);
            gs[0] = ih;
            gs[1] = iw;
            gs[2] = ic * in;
            if (axis == 2) {
                gs[2] = 1;
            } else {
                gs[1 - axis] = 1;
            }
            CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, iw_str, ih_off, iw_off, th_str, tw_str,
                th_off, tw_off, ih, iw, ic * in, keep_dim, od, gs[0], gs[1], inbuf, tmpbuf));
        } else if (imf == DF_NCHW) {
            gs[0] = (iw + 3) >> 2;
            gs[1] = ih;
            gs[2] = ic * in;
            gs[axis] = 1;
            sprintf(kernelName, "reduction_nchw_%s%d", modeName, axis);
            CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, iw_str, ih_off, iw_off, th_str, tw_str,
                th_off, tw_off, ih, iw, ic * in, ow, oh, keep_dim, od, gs[0], gs[1], inbuf, tmpbuf));
        } else {
            return NOT_SUPPORTED;
        }
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
        if (needTran) {
            U32 max_str = (ow_str > oh_str) ? ow_str : oh_str;
            max_str = (max_str > oc_str) ? max_str : oc_str;
            char kernelname[128];
            if (ow_off == 0 && oh_off == 0 && max_str == ow_str * oh_str * oc_str) {
                set_copy_opt_mali(false, DT_F16, kernelName, &kernelOpt);
                U32 copy_len = max_str * 4;
                gs[0] = max_str;
                dim = 1;
                CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel, &kernelOpt));
                CHECK_STATUS(
                    gcl_set_kernelArgs(kernel, copy_len, copy_len, 0, 0, gs[0], inbuf, outbuf));
            } else {
                sprintf(kernelname, "mem_trans_nchw_to_ncwhc4");
                gs[0] = (ow + 3) >> 2;
                gs[1] = oh;
                gs[2] = (oc + 3) / 4 * on;
                CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ow, oh, 0, 0, ow_str, oh_str, ow_off,
                    oh_off, ow, oh, oc, ow, oh, oc, 0, 0, tmpbuf, outbuf));
            }
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
#endif
        }
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE reduction_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    ReductionParamSpec p,
    TensorDesc outputDesc,
    GCLMemDesc gclmemInputDesc,
    GCLMemDesc gclmemOutputDesc,
    U32 *bytes)
{
    UNUSED(inputDesc);
    UNUSED(outputDesc);
    U32 size = 0;
    if (gclmemOutputDesc.memFormat == DF_NCWHC4) {
        if (gclmemInputDesc.memFormat == DF_NCHW || !p.keep_dim) {
            size = gclmemOutputDesc.byteSize;
        }
    }
    *bytes = size;
    return SUCCESS;
}

EE reduction_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc maskDesc,
    GCLMem_t mask,
    ReductionParamSpec p,
    GCLMem_t tmp,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(reduction_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(reduction_core_mali_fp16(
        handle, inputDesc, input, maskDesc, mask, p, tmp, outputDesc, output));
    return ret;
}
