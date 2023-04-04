// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/tfslice_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/tfslice_opt.h"

inline EE tfslice_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE tfslice_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TfSliceParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    DataType dt;
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    U32 iw_str, ih_str, ic_str, iw_off, ih_off, i_off;
    U32 ow_str, oh_str, oc_str, ow_off, oh_off, o_off;
    CHECK_STATUS(gclmem_get_desc_dim(input->desc, &dt, NULL, &in, &ic, &ih, &iw));
    CHECK_STATUS(gclmem_get_desc_dim(output->desc, NULL, NULL, &on, &oc, &oh, &ow));
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off));
    U32 be[8], stride[8];
    U32 nDims = inputDesc.nDims;
    for (U32 i = 0; i < 8; i++) {
        be[i] = 0;
        stride[i] = 1;
    }
    for (U32 i = 0; i < nDims; i++) {
        U32 j = nDims - 1 - i;
        be[j] = (p.begin_mask[i]) ? 0 : p.begin[i];
        stride[j] = p.strides[i];
        if (j > 3) {
            if (inputDesc.dims[j] != outputDesc.dims[j]) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
    }
    DataFormat imf = input->desc.memFormat;
    DataFormat omf = output->desc.memFormat;
    GCLMemType inputMemType = input->desc.memType;
    GCLMemType outputMemType = output->desc.memType;
    Mem inMem = input->mem;
    Mem outMem = output->mem;

    U32 dim = 3;
    U32 gs[3] = {0, 0, 0};
    U32 ls[3] = {0, 0, 0};

    bool useNchw = true;
    if (imf == DF_NCHWC4 && omf == DF_NCHWC4) {
        useNchw = false;
        ic /= 4;
        oc /= 4;
        be[nDims - 2] /= 4;
    } else {
        if (imf == DF_NCHWC4) {
            GCLMem tMem;
            GCLMemDesc desc = input->desc;
            U32 str[3] = {iw, ih, ic * in};
            U32 off[3] = {0, 0, 0};
            MemFlags flag = CL_MEM_READ_WRITE;
            CHECK_STATUS(
                gclmem_set_desc_padding(&desc, str, off, dt, DF_NCHW, GCL_MEM_BUF, flag));
            tMem.desc = desc;
            tMem.mem = tmpbuf->mem;
            CHECK_STATUS(ocl_data_trans_form(handle, input, &tMem, 0, 0, NCHWC4_TO_NCHW));
            iw_str = iw;
            ih_str = ih;
            iw_off = 0;
            ih_off = 0;
            inMem = tmpbuf->mem;
        }
    }
    i_off = ih_off * iw_str + iw_off;
    o_off = oh_off * ow_str + ow_off;
    gs[0] = ow;
    gs[1] = oh;
    gs[2] = oc * on;
    Kernel kernel;
    KernelOpt kernelOpt;
    char kernelName[128];
    CHECK_STATUS(
        set_tfslice_opt_mali(useNchw, dt, inputMemType, outputMemType, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, ow_str, oh_str, i_off, o_off, ic, oc,
        be[0], be[1], be[2], be[3], stride[0], stride[1], stride[2], stride[3], gs[0], gs[1], inMem,
        outMem));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

EE tfslice_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc outputDesc,
    GCLMemDesc gclmemInputDesc,
    GCLMemDesc gclmemOutputDesc,
    U32 *bytes)
{
    U32 tmpBytes = 0;
    if (gclmemInputDesc.memFormat != gclmemOutputDesc.memFormat) {
        tmpBytes += tensorNumBytes(inputDesc);
    }
    *bytes = tmpBytes;
    return SUCCESS;
}

EE tfslice_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TfSliceParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(tfslice_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(tfslice_core_mali_fp16(handle, inputDesc, input, p, tmpbuf, outputDesc, output));
    return SUCCESS;
}
