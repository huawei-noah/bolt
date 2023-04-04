// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/roialign_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/roialign_opt.h"
inline EE roialign_checkpara_mali_fp16(std::vector<TensorDesc> inputDescs, TensorDesc outputDesc)
{
    if (outputDesc.dt != inputDescs[0].dt || outputDesc.dt != inputDescs[1].dt) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return SUCCESS;
}

inline bool need_process_input(TensorDesc inputDesc, GCLMemDesc gclmemInputDesc)
{
    if (inputDesc.df == DF_NCHW && gclmemInputDesc.memType != GCL_MEM_BUF) {
        return true;
    }
    return false;
}

inline EE process_input(GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, Mem inMem)
{
    DataType dt;
    DataFormat df;
    U32 iw, ih, ic, in;
    CHECK_STATUS(tensorSelectGet(inputDesc, &dt, &df, &in, &ic, &ih, &iw));
    MemTransFormType type = NCHW_TO_NCHW;
    GCLMem InTmp;
    InTmp.desc = input->desc;
    InTmp.mem = inMem;
    U32 str[3] = {iw, ih, ic * in};
    U32 off[3] = {0, 0, 0};
    MemFlags flag = CL_MEM_READ_WRITE;
    CHECK_STATUS(
        gclmem_set_desc_padding(&(InTmp.desc), str, off, dt, DF_NCHW, GCL_MEM_BUF, flag));
    CHECK_STATUS(ocl_data_trans_form(handle, input, &InTmp, 0, 0, type));
    return SUCCESS;
}

inline bool need_process_output(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.df == DF_NCHWC4 && outputDesc.df == DF_NCHW) {
        return true;
    }
    return false;
}

inline EE process_output(GCLHandle_t handle, TensorDesc outputDesc, Mem outMem, GCLMem_t output)
{
    DataType dt;
    DataFormat df;
    U32 ow, oh, oc, on;
    CHECK_STATUS(tensorSelectGet(outputDesc, &dt, &df, &on, &oc, &oh, &ow));
    MemTransFormType type = NCHWC4_TO_NCHW;
    GCLMem outTmp;
    outTmp.desc = output->desc;
    outTmp.desc.df = DF_NCHWC4;
    outTmp.mem = outMem;
    U32 str[3] = {ow, oh, (oc + 3) / 4 * on};
    U32 off[3] = {0, 0, 0};
    MemFlags flag = CL_MEM_READ_WRITE;
    CHECK_STATUS(
        gclmem_set_desc_padding(&(outTmp.desc), str, off, dt, DF_NCHWC4, GCL_MEM_BUF, flag));
    CHECK_STATUS(ocl_data_trans_form(handle, &outTmp, output, 0, 0, type));
    return SUCCESS;
}

inline EE roialign_core_mali_fp16(GCLHandle_t handle,
    std::vector<TensorDesc> inputDescs,
    std::vector<void *> inputs,
    RoIAlignParamSpec roiAlignParamSpec,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    DataType dt;
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    U32 rw_str, rh_str, rw_off, rh_off;
    TensorDesc inputDesc = inputDescs[0];
    GCLMem_t input = (GCLMem_t)inputs[0];
    GCLMem_t roi = (GCLMem_t)inputs[1];
    CHECK_STATUS(tensorSelectGet(inputDesc, &dt, NULL, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow));
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(roi->desc, &rw_str, &rh_str, NULL, &rw_off, &rh_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    U32 i_off = ih_off * iw_str + iw_off;
    U32 o_off = oh_off * ow_str + ow_off;
    U32 r_off = rh_off * rw_str + rw_off;

    Mem inMem = input->mem;
    Mem outMem = output->mem;
    GCLMemType imt = input->desc.memType;
    GCLMemType omt = output->desc.memType;
    U32 tmpOff = 0;
    if (need_process_input(inputDesc, input->desc)) {
        U32 size = tensorNumBytes(inputDesc);
        CHECK_STATUS(gcl_create_sub_buffer(size, &tmpOff, tmpbuf, &inMem));
        CHECK_STATUS(process_input(handle, inputDesc, input, inMem));
        iw_str = iw;
        ih_str = ih;
        i_off = 0;
        imt = GCL_MEM_BUF;
    }

    if (need_process_output(inputDesc, outputDesc)) {
        U32 size = tensorNumBytes(outputDesc);
        CHECK_STATUS(gcl_create_sub_buffer(size, &tmpOff, tmpbuf, &outMem));
        ow_str = ow;
        oh_str = oh;
        o_off = 0;
        omt = GCL_MEM_BUF;
    }

    bool useNchwFormat = (inputDesc.df != DF_NCHWC4) ? true : false;
    char kernelName[128];
    KernelOpt kernelOpt;
    CHECK_STATUS(set_roialign_opt_mali(
        useNchwFormat, roiAlignParamSpec.mode, dt, imt, omt, kernelName, &kernelOpt));
    Kernel kernel;
    U32 gs[3] = {1, 1, 1};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    if (useNchwFormat) {
        gs[0] = ow;
        gs[1] = oh;
        gs[2] = oc * on;
    } else {
        oc = (oc + 3) / 4;
        gs[0] = ow;
        gs[1] = oh;
        gs[2] = oc * on;
    }
    int sampling_ratio = roiAlignParamSpec.sampling_ratio;
    float spatial_scale = roiAlignParamSpec.spatial_scale;
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, ow_str, oh_str, i_off, o_off, r_off, iw,
        ih, ow, oh, oc, gs[0], gs[1], sampling_ratio, spatial_scale, roi->mem, inMem, outMem));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
    if (need_process_output(inputDesc, outputDesc)) {
        CHECK_STATUS(process_output(handle, outputDesc, outMem, output));
    }
    return SUCCESS;
}

EE roialign_infer_forward_tmp_bytes_mali_fp16(
    TensorDesc inputDesc, GCLMemDesc gclmemInputDesc, TensorDesc outputDesc, U32 *bytes)
{
    *bytes = 0;
    if (need_process_input(inputDesc, gclmemInputDesc)) {
        *bytes = UNI_ALIGN(tensorNumBytes(inputDesc), BUFFER_ALIGN_BASE);
    }
    if (need_process_output(inputDesc, outputDesc)) {
        *bytes += tensorNumBytes(outputDesc);
    }
    return SUCCESS;
}

EE roialign_mali_fp16(GCLHandle_t handle,
    std::vector<TensorDesc> inputDescs,
    std::vector<void *> inputs,
    RoIAlignParamSpec roiAlignParamSpec,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(roialign_checkpara_mali_fp16(inputDescs, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(roialign_core_mali_fp16(
        handle, inputDescs, inputs, roiAlignParamSpec, tmpbuf, outputDesc, output));
    return SUCCESS;
}
