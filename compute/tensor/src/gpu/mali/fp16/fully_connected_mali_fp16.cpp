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
#include "gpu/mali/fp16/fully_connected_mali_fp16.h"

inline EE fully_connected_checkpara_mali_fp16(
    TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != filterDesc.dt || inputDesc.dt != DT_F16) {
        return NOT_MATCH;
    }
    return SUCCESS;
}

inline EE fully_connected_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    std::vector<GCLMem_t> filter,
    TensorDesc biasDesc,
    std::vector<GCLMem_t> bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    std::vector<GCLMem_t> output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    UNUSED(outputDesc);

    U32 ih_str, iw_str, ih_off, iw_off, ihw_str;
    U32 oh_str, ow_str, oh_off, ow_off;
    U32 fw, fh, fc, fn;
    cl_mem inbuf, fltbuf, biasmem, outbuf, tmp;
    inbuf = input->mem;
    fltbuf = filter[0]->mem;
    biasmem = bias[0]->mem;
    outbuf = output[0]->mem;
    tmp = tmpBuf->mem;

    tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw);
    ih_str = input->desc.stride[0];
    iw_str = input->desc.stride[1];
    ih_off = input->desc.offset[0];
    iw_off = input->desc.offset[1];
    oh_str = output[0]->desc.stride[0];
    ow_str = output[0]->desc.stride[1];
    oh_off = output[0]->desc.offset[0];
    ow_off = output[0]->desc.offset[1];
    ihw_str = ih_str * iw_str;
    char kernelname[128];
    Kernel kernel;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    U32 item_w = forwardRunInfo->best_w[0];
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];

    if (fw == 1 && fh == 1) {
        if (inputDesc.df == DF_NCHW || inputDesc.df == DF_NORMAL) {
            U32 ic_str;
            ic_str = filter[0]->desc.stride[1];
            if (ih_str > 1 || iw_str > 1) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
            sprintf(kernelname, "conv_direct_spe_fwhs1_%d", item_c);
            gs[0] = fn;
            gs[1] = 1;
            gs[2] = 1;
            dim = 1;
            CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, ihw_str, ic_str, ih_off, iw_off, oh_str,
                ow_str, oh_off, ow_off, fn, gs[0], gs[1], inbuf, fltbuf, biasmem, outbuf));
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
            CHECK_STATUS(gcl_print_memory<F16>(handle, input, "fc_wh1_input"));
            CHECK_STATUS(gcl_print_memory<F16>(handle, filter[0], "fc_wh1_filter"));
            CHECK_STATUS(gcl_print_memory<F16>(handle, bias[0], "fc_wh1_bias"));
            CHECK_STATUS(gcl_print_memory<F16>(handle, output[0], "fc_wh1_output"));
            handle->t_total += handle->t_execute;
#endif
        }
        if (inputDesc.df == DF_MKT) {
            item_k = item_k >> 2;
            U32 ic_str = input->desc.stride[2];
            U32 ohw_str;
            U32 step = inputDesc.dims[0];
            sprintf(kernelname, "conv_direct_s%d_%d%d%d", 1, 1, item_w, item_k);
            for (U32 i = 0; i < filter.size(); ++i) {
                fltbuf = filter[i]->mem;
                biasmem = bias[i]->mem;
                outbuf = output[i]->mem;
                iw_str = input->desc.stride[0];
                ih_str = input->desc.stride[1];
                iw_off = input->desc.offset[0];
                ih_off = input->desc.offset[1];
                ow_str = output[i]->desc.stride[0];
                oh_str = output[i]->desc.stride[1];
                ow_off = output[i]->desc.offset[0];
                oh_off = output[i]->desc.offset[1];
                ohw_str = oh_str * ow_str;
                if (ih_str != 1 || ih_off != 0) {
                    CHECK_STATUS(NOT_SUPPORTED);
                }
                gs[0] = 1;
                gs[1] = (step + item_w - 1) / item_w;
                gs[2] = output[i]->desc.stride[2] / item_k;
                CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, ihw_str, ic_str, ih_off, iw_off,
                    oh_str, ohw_str, oh_off, ow_off, step, 1, gs[0], gs[1], inbuf, fltbuf, biasmem,
                    outbuf));
                gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
#ifdef _DEBUG
                CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
                CHECK_STATUS(gcl_print_memory<F16>(handle, input, "conv_direct_input"));
                CHECK_STATUS(gcl_print_memory<F16>(handle, filter[i], "conv_direct_filter"));
                CHECK_STATUS(gcl_print_memory<F16>(handle, bias[i], "conv_direct_bias"));
                CHECK_STATUS(gcl_print_memory<F16>(handle, output[i], "conv_direct_output"));
                handle->t_total += handle->t_execute;
#endif
            }
        }
    } else {
        U32 ihy_str, fhy_str, fhw_str, fwc_str;
        ihy_str = ih_str * item_w;
        fc = (fc + item_c - 1) / item_c;
        fn = (fn + item_k - 1) / item_k;
        fhy_str = fh * item_w;
        fhw_str = fh * fw;
        fwc_str = fw * fc;
        CHECK_STATUS(gcl_create_kernel(handle, "fc_p1", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, item_w, ih_str, iw_str, ih_off, iw_off, ihy_str,
            ihw_str, fh, fw, fc, fn, fhy_str, fhw_str, fwc_str, fltbuf, inbuf, tmp));
        gs[0] = fh;
        gs[1] = item_w;
        gs[2] = fn;
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, "fc_p1");
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "fc_p1"));
        CHECK_STATUS(gcl_print_memory<F16>(handle, input, "fc_p1_input"));
        CHECK_STATUS(gcl_print_memory<F16>(handle, filter[0], "fc_p1_filter"));
        CHECK_STATUS(gcl_print_buffer<F16>(handle, tmp, fh * item_w * fn * item_k, "fc_p1_output"));
        handle->t_total += handle->t_execute;
#endif
        CHECK_STATUS(gcl_create_kernel(handle, "fc_p2", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(
            kernel, fh * item_w, fn, oh_str, ow_str, oh_off, ow_off, tmp, biasmem, outbuf));
        U32 gs2 = fn;
        U32 ls2 = 0;
        dim = 1;
        gcl_set_kernelVec(handle, kernel, dim, &gs2, &ls2, "fc_p2");
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs2, &ls2, "fc_p2"));
        CHECK_STATUS(gcl_print_memory<F16>(handle, bias[0], "fc_p2_bias"));
        CHECK_STATUS(gcl_print_memory<F16>(handle, output[0], "fc_p2_output"));
        handle->t_total += handle->t_execute;
#endif
    }
    return SUCCESS;
}

EE fully_connected_transform_filter_bytes_mali_fp16(TensorDesc filterDesc,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw);
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    U32 s0 = 0;
    U32 s1 = 0;
    U32 s2 = 0;
    U32 num = 0;
    U32 byteSize;

    if (item_k == 0) {
        s0 = fn;
        s1 = (fc + item_c - 1) / item_c;
        s2 = 1;
        DataFormat df = DF_CHWNC4;
        if (item_c == 8) {
            df = DF_CHWNC8;
        }
        if (item_c == 16) {
            df = DF_CHWNC16;
        }
        gclmemFilterDesc->memFormat = df;
        num = s0 * s1 * s2 * item_c;
    } else if (fw == 1 && fh == 1) {
        s0 = item_k >> 2;
        s1 = (fc + item_c - 1) / item_c;
        s2 = (fn + item_k - 1) / item_k;
        gclmemFilterDesc->memFormat = DF_NCHWN4C4;
        num = s0 * s1 * s2 * item_c * item_k / (item_k >> 2);
    } else {
        s0 = fh;
        s1 = fw;
        s2 = ((fc + item_c - 1) / item_c) * ((fn + item_k - 1) / item_k);
        num = s0 * s1 * s2 * item_c * item_k;
        gclmemFilterDesc->memFormat = DF_NCWHN4C4;
    }
    byteSize = num * bytesOf(DT_F16);
    gclmemFilterDesc->stride[0] = s0;
    gclmemFilterDesc->stride[1] = s1;
    gclmemFilterDesc->stride[2] = s2;
    gclmemFilterDesc->offset[0] = 0;
    gclmemFilterDesc->offset[1] = 0;
    gclmemFilterDesc->offset[2] = 0;
    gclmemFilterDesc->num = num;
    gclmemFilterDesc->byteSize = byteSize;
    gclmemFilterDesc->memType = GCL_MEM_BUF;
    gclmemFilterDesc->flags = CL_MEM_READ_WRITE;
    gclmemFilterDesc->host_ptr = NULL;
    *bytes = 0;
    return SUCCESS;
}

EE fully_connected_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc *fltmemDesc,
    std::vector<GCLMem_t> fltmem,
    ForwardRunInfoMali_t forwardRunInfo)
{
    DataType fdt;
    DataFormat fdf;
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw);
    char kernelname[128];
    Kernel kernel;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    U32 fwh = fw * fh;
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    if (fw == 1 && fh == 1) {
        if (item_k == 0) {
            sprintf(kernelname, "conv_direct_trans_fltbuf_%d%d", item_c, item_k);
            CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, fwh, fc, fn, filter->mem, fltmem[0]->mem));
            gs[0] = fwh;
            gs[1] = (fc + item_c - 1) / item_c;
            gs[2] = fn;
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
        } else {
            sprintf(kernelname, "conv_direct_trans_fltbuf_%d%d", item_c, item_k);
            CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
            if (fltmem.size() == 1) {
                CHECK_STATUS(gcl_set_kernelArgs(kernel, fwh, fc, fn, filter->mem, fltmem[0]->mem));
                gs[0] = fwh;
                gs[1] = (fc + item_c - 1) / item_c;
                gs[2] = (fn + item_k - 1) / item_k * item_k;
                CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
            } else {
                GCLMem_t tmp = gcl_create_gclmem();
                tmp->desc.byteSize = 0;
                for (U32 i = 0; i < fltmem.size(); ++i) {
                    tmp->desc.byteSize += fltmem[i]->desc.byteSize;
                }
                tmp->desc.memType = GCL_MEM_BUF;
                tmp->desc.flags = CL_MEM_READ_WRITE;
                CHECK_STATUS(gcl_create_memory(handle, tmp));
                CHECK_STATUS(gcl_set_kernelArgs(kernel, fwh, fc, fn, filter->mem, tmp->mem));
                gs[0] = fwh;
                gs[1] = (fc + item_c - 1) / item_c;
                gs[2] = (fn + item_k - 1) / item_k * item_k;
                CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
                U32 offset[2] = {0, 0};
                for (U32 i = 0; i < fltmem.size(); i++) {
                    U32 size = fltmem[i]->desc.byteSize;
                    CHECK_STATUS(gcl_trans_memory(
                        handle, tmp, fltmem[i], &size, DEVICE_BUF_TO_BUF, CL_TRUE, offset));
                    offset[0] += size;
                }
                gcl_destroy_gclmem(tmp);
            }
        }
    } else {
        sprintf(kernelname, "fc_trans_fltbuf_%d%d", item_c, item_k);
        CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, fw, fh, fwh, fc, fn, filter->mem, fltmem[0]->mem));
        gs[0] = fw;
        gs[1] = fh;
        gs[2] = (fc + item_c - 1) / item_c * ((fn + item_k - 1) / item_k) * item_k;
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    }
    *fltmemDesc = tensor4df(fdt, fdf, fn, fc, fh, fw);
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, filter, "fc_filter_org"));
    for (U32 i = 0; i < fltmem.size(); ++i) {
        CHECK_STATUS(gcl_print_memory<F16>(handle, fltmem[i], "fc_filter_tran"));
    }
#endif
    return SUCCESS;
}

EE fully_connected_infer_forward_tmp_bytes_mali_fp16(
    TensorDesc inputDesc, TensorDesc filterDesc, U32 *bytes, ForwardRunInfoMali_t forwardRunInfo)
{
    U32 fn, fw, fh;
    tensorSelectGet(filterDesc, NULL, NULL, &fn, NULL, &fh, &fw);
    if (fh == 1 && fw == 1) {
        *bytes = 0;
    } else {
        DataType dt;
        U32 ic, ih, iw;
        tensorSelectGet(inputDesc, &dt, NULL, NULL, &ic, &ih, &iw);
        U32 item_w = forwardRunInfo->best_w[0];
        U32 item_k = forwardRunInfo->best_k[0];
        *bytes = ih * item_w * ((fn + item_k - 1) / item_k * item_k) * bytesOf(dt);
    }
    return SUCCESS;
}

EE fully_connected_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    std::vector<GCLMem_t> filter,
    TensorDesc biasDesc,
    std::vector<GCLMem_t> bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    std::vector<GCLMem_t> output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    CHECK_STATUS(fully_connected_checkpara_mali_fp16(inputDesc, filterDesc, outputDesc));
    for (U32 i = 0; i < output.size(); i++) {
        CHECK_STATUS(fill_output_zero(handle, output[i], outputDesc));
    }
    CHECK_STATUS(fully_connected_core_mali_fp16(handle, inputDesc, input, filterDesc, filter,
        biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, forwardRunInfo));
    return SUCCESS;
}
