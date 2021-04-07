// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/reshape_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/copy_opt.h"

inline EE reshape_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    if (outputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE reshape_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output,
    GCLMem_t tmpbuf)
{
    DataFormat idf, odf;
    U32 iw, ih, ic, in, it;
    U32 ow, oh, oc, on, ot;
    U32 inDims;
    tensorSelectGet(inputDesc, NULL, &idf, &in, &ic, &ih, &iw, &it);
    tensorSelectGet(outputDesc, NULL, &odf, &on, &oc, &oh, &ow, &ot);
    inDims = inputDesc.nDims;
    U32 iw_str, ih_str, ic_str, iw_off, ih_off;
    U32 ow_str, oh_str, oc_str, ow_off, oh_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off);
    DataFormat imf = input->desc.memFormat;
    DataFormat omf = output->desc.memFormat;
    cl_mem inbuf = input->mem;
    cl_mem outbuf = output->mem;
    cl_mem tmp = tmpbuf->mem;
    bool dataCopy = false;
    U32 copy_len_in = iw * ih * ic * in * it;
    U32 copy_len_out = ow * oh * oc * on * ot;

    if ((iw_str == 1 && ih_str == 1 && omf == DF_NCHW && ow_off == 0 && oh_off == 0) ||
        (ow_str == 1 && oh_str == 1 && imf == DF_NCHW && iw_off == 0 && ih_off == 0)) {
        if (inbuf == outbuf) {
            return SUCCESS;
        } else {
            dataCopy = true;
            goto DATACOPY;
        }
    }

    if (imf == omf) {
        if (imf == DF_NCHW) {
            if ((iw_off == 0 && ih_off == 0 && ow_off == 0 && oh_off == 0) ||
                (iw_str == ow_str && ih_str == oh_str && iw_off == ow_off && ih_off == oh_off &&
                    iw == ow && ih == oh)) {
                if (inbuf == outbuf) {
                    return SUCCESS;
                } else {
                    dataCopy = true;
                    goto DATACOPY;
                }
            }
        }

        if (imf == DF_NCWHC4) {
            if (iw_str == ow_str && ih_str == oh_str && iw_off == ow_off && ih_off == oh_off &&
                iw == ow && ih == oh) {
                if (it == ot) {
                    if (inbuf == outbuf) {
                        return SUCCESS;
                    } else {
                        dataCopy = true;
                        goto DATACOPY;
                    }
                } else {
                    goto DATACOPY;
                }
            }
        }

        if (iw == ow && ih == oh) {
            if (inbuf == outbuf) {
                outbuf = tmp;
                dataCopy = true;
                copy_len_in = copy_len_out;
            }
            char kernelName[128];
            U32 gs[3];
            U32 ls[3] = {0, 0, 0};
            U32 dim = 3;
            if (imf == DF_NCHW) {
                sprintf(kernelName, "mem_trans_nchw_to_nchw");
                gs[0] = (ow + 3) / 4;
                gs[1] = oh;
                gs[2] = oc * ot * on;
            } else {
                if (it != ot) {
                    dataCopy = false;
                    goto DATACOPY;
                }
                sprintf(kernelName, "mem_trans_ncwhc4_to_ncwhc4");
                gs[0] = oh;
                gs[1] = ow;
                gs[2] = (oc + 3) / 4 * ot * on;
                ic = ALIGN(ic, 4);
            }
            Kernel kernel;
            CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str, oh_str,
                ow_off, oh_off, iw, ih, ic * it * in, ow, oh, oc * ot * on, 0, 0, inbuf, outbuf));
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
            if (dataCopy) {
                inbuf = tmp;
                goto DATACOPY;
            } else {
                return SUCCESS;
            }
        }
    }

    if (imf != omf && it == 1 && ot == 1) {
        if ((imf == DF_NCWHC4 && ih == ow && iw == 1) || (omf == DF_NCWHC4 && iw == oh && ow == 1)) {
            if (inbuf == outbuf) {
                outbuf = tmp;
                dataCopy = true;
                copy_len_in = copy_len_out;
            }
            char kernelName[128];
            U32 gs[3];
            U32 ls[3] = {0, 0, 0};
            U32 dim = 3;
            U32 h_val, c_val;
            if (imf == DF_NCWHC4) {
                sprintf(kernelName, "mem_trans_ncwhc4_to_nchw_ih_equal_ow");
                gs[0] = ih;
                gs[1] = iw;
                gs[2] = (ic + 3) / 4;
                h_val = oh;
                c_val = oc;
            } else {
                sprintf(kernelName, "mem_trans_nchw_to_ncwhc4_iw_equal_oh");
                gs[0] = oh;
                gs[1] = ow;
                gs[2] = (oc + 3) / 4;
                h_val = ih;
                c_val = ic;
            }
            Kernel kernel;
            CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str, oh_str,
                ow_off, oh_off, h_val, c_val, gs[0], gs[1], inbuf, outbuf));
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
            if (dataCopy) {
                inbuf = tmp;
            } else {
                return SUCCESS;
            }
        }
    }

DATACOPY:
    if (dataCopy) {
        U32 gs = (copy_len_out + 3) / 4;
        U32 ls = 0;
        U32 dim = 1;
        char kernelName[128];
        Kernel kernel;
        KernelOpt kernelOpt;
        set_copy_opt_mali(false, DT_F16, kernelName, &kernelOpt);
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, copy_len_in, copy_len_out, 0, 0, gs, inbuf, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, &gs, &ls, kernelName);
        inbuf = tmp;
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs, &ls, "copy_f16"));
#endif
        return SUCCESS;
    }

    bool noNeedOutTrans = false;
    if (ow_str == 1 && oh_str == 1) {
        noNeedOutTrans = true;
        tmp = outbuf;
    }

    if (imf == DF_NCHW && (iw_off > 0 || ih_off > 0)) {
        U32 gs[3] = {(iw + 3) / 4, ih, ic * it};
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        Kernel kernel;
        CHECK_STATUS(gcl_create_kernel(handle, "mem_trans_nchw_to_nchw", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, iw, ih, 0, 0, iw,
            ih, ic * it, iw, ih, ic * it, 0, 0, inbuf, tmp));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, "mem_trans_nchw_to_nchw");
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "mem_trans_nchw_to_nchw"));
#endif
        if (noNeedOutTrans) {
            return SUCCESS;
        } else {
            inbuf = tmp;
        }
    }

    if (imf == DF_NCWHC4) {
        U32 gs[3] = {ih, (iw + 3) / 4, (ic + 3) / 4 * it};
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        Kernel kernel;
        char kernelName[128];
        if (inDims == 5) {
            sprintf(kernelName, "mem_trans_3d_ncwhc4_to_nchw");
            CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, iw, ih, 0, 0,
                iw, ih, ic, it, iw, ih, ic, it, 0, 0, inbuf, tmp));
        } else {
            sprintf(kernelName, "mem_trans_ncwhc4_to_nchw");
            CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, iw, ih, 0, 0,
                iw, ih, ic, iw, ih, ic, 0, 0, inbuf, tmp));
        }
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
        if (noNeedOutTrans) {
            return SUCCESS;
        } else {
            inbuf = tmp;
        }
    }

    if (omf == DF_NCHW) {
        U32 gs[3] = {(ow + 3) / 4, oh, oc * ot};
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        Kernel kernel;
        CHECK_STATUS(gcl_create_kernel(handle, "mem_trans_nchw_to_nchw", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, ow, oh, 0, 0, ow_str, oh_str, ow_off, oh_off, ow,
            oh, oc * ot, ow, oh, oc * ot, 0, 0, inbuf, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, "mem_trans_nchw_to_nchw");
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "mem_trans_nchw_to_nchw"));
#endif
        return SUCCESS;
    }

    if (omf == DF_NCWHC4) {
        U32 gs[3] = {(ow + 3) / 4, oh, (oc + 3) / 4 * on};
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        Kernel kernel;
        CHECK_STATUS(gcl_create_kernel(handle, "mem_trans_nchw_to_ncwhc4", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, ow, oh, 0, 0, ow_str, oh_str, ow_off, oh_off, ow,
            oh, oc, ow, oh, oc, 0, 0, inbuf, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, "mem_trans_nchw_to_ncwhc4");
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "mem_trans_nchw_to_ncwhc4"));
#endif
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE reshape_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc,
    U32 *bytes)
{
    U32 maxSize = tensorNumBytes(inputDesc);
    U32 tmpSize = tensorNumBytes(outputDesc);
    maxSize = (maxSize > tmpSize) ? maxSize : tmpSize;
    tmpSize = gclmemInputDesc->byteSize;
    maxSize = (maxSize > tmpSize) ? maxSize : tmpSize;
    tmpSize = gclmemOutputDesc->byteSize;
    maxSize = (maxSize > tmpSize) ? maxSize : tmpSize;
    *bytes = maxSize;
    return SUCCESS;
}

EE reshape_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output,
    GCLMem_t tmpbuf)
{
    CHECK_STATUS(reshape_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(reshape_core_mali_fp16(handle, inputDesc, input, outputDesc, output, tmpbuf));
    return SUCCESS;
}
