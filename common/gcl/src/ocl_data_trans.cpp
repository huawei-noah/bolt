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
#include "tensor_desc.h"
#include "error.h"
#include "gcl_common.h"
#include "gcl_func.h"
#include "gclmem_desc_infer.h"
#include "ocl_data_trans.h"

EE ocl_data_trans_form(GCLHandle_t handle,
    GCLMem_t input,
    GCLMem_t output,
    U32 in_off,
    U32 out_off,
    MemTransFormType type,
    bool setKernelVec)
{
    U32 iw, ih, ic, in;
    U32 iw_str, ih_str, iw_off, ih_off, i_off;
    U32 ow, oh, oc, on;
    U32 ow_str, oh_str, ow_off, oh_off, o_off;
    U32 iDims = input->desc.nDims;
    U32 oDims = output->desc.nDims;
    iw = (iDims > 0) ? input->desc.dims[0] : 1;
    ih = (iDims > 1) ? input->desc.dims[1] : 1;
    ic = (iDims > 2) ? input->desc.dims[2] : 1;
    in = (iDims > 3) ? input->desc.dims[3] : 1;
    for (U32 i = 4; i < iDims; i++) {
        in *= input->desc.dims[i];
    }
    ow = (oDims > 0) ? output->desc.dims[0] : 1;
    oh = (oDims > 1) ? output->desc.dims[1] : 1;
    oc = (oDims > 2) ? output->desc.dims[2] : 1;
    on = (oDims > 3) ? output->desc.dims[3] : 1;
    for (U32 i = 4; i < oDims; i++) {
        on *= output->desc.dims[i];
    }
    if (iw != ow || ih != oh || ic != oc || in != on) {
        CHECK_STATUS(NOT_MATCH);
    }
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    i_off = ih_off * iw_str + iw_off;
    o_off = oh_off * ow_str + ow_off;
    Mem inbuf = input->mem;
    Mem outbuf = output->mem;
    Kernel kernel;
    KernelOpt kernelOpt;
    char kernelName[128];
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    switch (type) {
        case NCHW_TO_NCHW:
            gs[0] = (ow + 3) / 4;
            gs[1] = oh;
            gs[2] = oc * on;
            break;
        case NCHW_TO_NCHWC4:
            gs[0] = (input->desc.memType == GCL_MEM_BUF) ? ow : (ow + 3) / 4;
            gs[1] = oh;
            gs[2] = (oc + 3) / 4 * on;
            break;
        case NCHWC4_TO_NCHW:
            gs[0] = (ow + 3) / 4;
            gs[1] = oh;
            gs[2] = (oc + 3) / 4 * on;
            break;
        case NCHWC4_TO_NCHWC4:
            gs[0] = ow;
            gs[1] = oh;
            gs[2] = (oc + 3) / 4 * on;
            break;
        default:
            CHECK_STATUS(NOT_SUPPORTED);
    }

    CHECK_STATUS(set_mem_trans_opt_mali(
        type, false, DT_F16, input->desc.memType, output->desc.memType, kernelName, &kernelOpt));
    if (setKernelVec) {
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    } else {
        CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel, &kernelOpt));
    }
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, ow_str, oh_str, i_off, o_off, ow, oh,
        oc, in_off, out_off, inbuf, outbuf));
    if (setKernelVec) {
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    } else {
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    }
    return SUCCESS;
}

EE ocl_data_trans_form_3d(GCLHandle_t handle,
    GCLMem_t input,
    GCLMem_t output,
    U32 in_off,
    U32 out_off,
    MemTransFormType type,
    bool setKernelVec)
{
    U32 iw, ih, ic, in, it;
    U32 iw_str, ih_str, iw_off, ih_off, i_off;
    U32 ow, oh, oc, on, ot;
    U32 ow_str, oh_str, ow_off, oh_off, o_off;
    U32 iDims = input->desc.nDims;
    U32 oDims = output->desc.nDims;
    CHECK_STATUS(gclmem_get_desc_dim_5d(input->desc, NULL, NULL, &in, &ic, &it, &ih, &iw));
    CHECK_STATUS(gclmem_get_desc_dim_5d(output->desc, NULL, NULL, &on, &oc, &ot, &oh, &ow));
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    i_off = ih_off * iw_str + iw_off;
    o_off = oh_off * ow_str + ow_off;
    if (iw != ow || ih != oh || ic != oc || in != on || it != ot) {
        CHECK_STATUS(NOT_MATCH);
    }
    Mem inbuf = input->mem;
    Mem outbuf = output->mem;
    Kernel kernel;
    KernelOpt kernelOpt;
    char kernelName[128];
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    if (in > 1 || on > 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    switch (type) {
        case NCHW_TO_NCHW:
            gs[0] = (ow + 3) / 4;
            gs[1] = oh;
            gs[2] = oc * ot;
            break;
        case NCHW_TO_NCHWC4:
            gs[0] = (input->desc.memType == GCL_MEM_BUF) ? ow : (ow + 3) / 4;
            gs[1] = oh;
            gs[2] = (oc + 3) / 4 * ot;
            break;
        case NCHWC4_TO_NCHW:
            gs[0] = (ow + 3) / 4;
            gs[1] = oh;
            gs[2] = (oc + 3) / 4 * ot;
            break;
        case NCHWC4_TO_NCHWC4:
            gs[0] = ow;
            gs[1] = oh;
            gs[2] = (oc + 3) / 4 * ot;
            break;
        default:
            CHECK_STATUS(NOT_SUPPORTED);
    }
    CHECK_STATUS(set_mem_trans_opt_mali(
        type, true, DT_F16, input->desc.memType, output->desc.memType, kernelName, &kernelOpt));
    if (setKernelVec) {
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    } else {
        CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel, &kernelOpt));
    }
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, ow_str, oh_str, i_off, o_off, ow, oh,
        oc, ot, in_off, out_off, inbuf, outbuf));
    if (setKernelVec) {
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    } else {
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    }
    return SUCCESS;
}

EE ocl_data_trans_c(GCLHandle_t handle,
    GCLMem_t input,
    GCLMem_t output,
    U32 in_off,
    U32 out_off,
    MemTransCType type,
    bool setKernelVec)
{
    U32 x, y, c, n;
    U32 ix_str, iy_str, ix_off, iy_off;
    U32 ox_str, oy_str, ox_off, oy_off;
    U32 iDims = input->desc.nDims;
    U32 oDims = output->desc.nDims;
    if (iDims < 4 || oDims < 4) {
        CHECK_STATUS(NOT_MATCH);
    }
    switch (type) {
        case C1_TO_C4:
            x = output->desc.dims[1];
            y = output->desc.dims[0];
            c = output->desc.dims[2];
            n = output->desc.dims[3];
            for (U32 i = 4; i < oDims; i++) {
                n = n * output->desc.dims[i];
            }
            break;
        case C4_TO_C1:
            x = input->desc.dims[1];
            y = input->desc.dims[0];
            c = input->desc.dims[2];
            n = input->desc.dims[3];
            for (U32 i = 4; i < iDims; i++) {
                n = n * input->desc.dims[i];
            }
            break;
        default:
            CHECK_STATUS(NOT_SUPPORTED);
    }
    ix_str = input->desc.stride[0];
    iy_str = input->desc.stride[1];
    ix_off = input->desc.offset[0];
    iy_off = input->desc.offset[1];
    ox_str = output->desc.stride[0];
    oy_str = output->desc.stride[1];
    ox_off = output->desc.offset[0];
    oy_off = output->desc.offset[1];
    Mem inbuf = input->mem;
    Mem outbuf = output->mem;
    Kernel kernel;
    KernelOpt kernelOpt;
    char kernelName[128];
    U32 gs[3] = {(x + 3) / 4, y, (c + 3) / 4 * n};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    CHECK_STATUS(set_mem_trans_c_opt_mali(
        type, DT_F16, input->desc.memType, output->desc.memType, kernelName, &kernelOpt));
    if (setKernelVec) {
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    } else {
        CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel, &kernelOpt));
    }
    CHECK_STATUS(gcl_set_kernelArgs(kernel, ix_str, iy_str, ix_off, iy_off, ox_str, oy_str, oy_off,
        oy_off, x, y, c, in_off, out_off, inbuf, outbuf));
    if (setKernelVec) {
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    } else {
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    }
    return SUCCESS;
}

EE ocl_set_input(GCLHandle_t handle,
    GCLMem_t dst,
    TensorDesc hostDesc,
    const U8 *hostPtr,
    GCLMem_t tmpBuf,
    bool blocking)
{
    DataType hdt;
    DataFormat hdf;
    U32 hw, hh, hc, hn, ht;
    tensorSelectGet(hostDesc, &hdt, &hdf, &hn, &hc, &hh, &hw, &ht);
    U32 w, h, c, pw, ph;
    get_gclmem_dim(dst->desc, &w, &h, &c, &pw, &ph);
    bool needTrans = false;
    if ((pw != 0 || ph != 0) && dst->desc.memFormat == DF_NCHW && dst->desc.memType != GCL_MEM_BUF) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (dst->desc.memFormat == DF_NCHWC4 || w != hw || h != hh) {
        needTrans = true;
    }
    if (dst->desc.memType != GCL_MEM_BUF && (w & 3) != 0) {
        needTrans = true;
    }
    GCLMem tMem;
    U32 size[3] = {1, 1, 1};
    MemFlags flag = CL_MEM_READ_WRITE;
    GCLMemTransType hostToDevType = HOST_TO_DEVICE_BUF;
    if (needTrans || dst->desc.memType == GCL_MEM_BUF) {
        GCLMemDesc desc = dst->desc;
        U32 stride[3] = {hw, hh, hn * hc * ht};
        U32 offset[3] = {0, 0, 0};
        CHECK_STATUS(
            gclmem_set_desc_padding(&desc, stride, offset, DT_F16, DF_NCHW, GCL_MEM_BUF, flag));
        tMem.desc = desc;
        tMem.mem = (needTrans) ? tmpBuf->mem : dst->mem;
        size[0] = tensorNumBytes(hostDesc);
    } else {
        tMem = *dst;
        size[0] = dst->desc.stride[0];
        size[1] = dst->desc.stride[1];
        size[2] = dst->desc.stride[2];
        hostToDevType = HOST_TO_DEVICE_IMG;
    }
    CHECK_STATUS(
        gcl_trans_memory(handle, (void *)hostPtr, (void *)(&tMem), size, hostToDevType, CL_TRUE));
    if (needTrans) {
        if (dst->desc.dt != DT_F16) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        MemTransFormType type = (dst->desc.memFormat == DF_NCHWC4) ? NCHW_TO_NCHWC4 : NCHW_TO_NCHW;
        if (ht > 1) {
            CHECK_STATUS(ocl_data_trans_form_3d(handle, &tMem, dst, 0, 0, type, false));
        } else {
            CHECK_STATUS(ocl_data_trans_form(handle, &tMem, dst, 0, 0, type, false));
        }
    }
    return SUCCESS;
}

EE ocl_get_output(GCLHandle_t handle,
    const GCLMem_t src,
    TensorDesc hostDesc,
    const U8 *hostPtr,
    GCLMem_t tmpBuf,
    bool blocking)
{
    DataType hdt;
    DataFormat hdf;
    U32 hw, hh, hc, hn, ht;
    tensorSelectGet(hostDesc, &hdt, &hdf, &hn, &hc, &hh, &hw, &ht);
    U32 w, h, c, pw, ph;
    get_gclmem_dim(src->desc, &w, &h, &c, &pw, &ph);
    bool needTrans = false;
    if (src->desc.memFormat == DF_NCHWC4 || w != hw || h != hh) {
        needTrans = true;
    }
    if (src->desc.memType != GCL_MEM_BUF && (w & 3) != 0) {
        needTrans = true;
    }
    GCLMem tMem;
    U32 size[3] = {1, 1, 1};
    MemFlags flag = CL_MEM_READ_WRITE;
    GCLMemTransType devToHostType = DEVICE_BUF_TO_HOST;
    if (needTrans || src->desc.memType == GCL_MEM_BUF) {
        GCLMemDesc desc = src->desc;
        U32 stride[3] = {hw, hh, hn * hc * ht};
        U32 offset[3] = {0, 0, 0};
        CHECK_STATUS(
            gclmem_set_desc_padding(&desc, stride, offset, DT_F16, DF_NCHW, GCL_MEM_BUF, flag));
        tMem.desc = desc;
        tMem.mem = (needTrans) ? tmpBuf->mem : src->mem;
        size[0] = tensorNumBytes(hostDesc);
    } else {
        tMem = *src;
        size[0] = src->desc.stride[0];
        size[1] = src->desc.stride[1];
        size[2] = src->desc.stride[2];
        devToHostType = HOST_TO_DEVICE_IMG;
    }
    if (needTrans) {
        MemTransFormType type = (src->desc.memFormat == DF_NCHWC4) ? NCHWC4_TO_NCHW : NCHW_TO_NCHW;
        if (ht > 1) {
            CHECK_STATUS(ocl_data_trans_form_3d(handle, src, &tMem, 0, 0, type, false));
        } else {
            CHECK_STATUS(ocl_data_trans_form(handle, src, &tMem, 0, 0, type, false));
        }
    }
    CHECK_STATUS(
        gcl_trans_memory(handle, (void *)(&tMem), (void *)(hostPtr), size, devToHostType, CL_TRUE));
    return SUCCESS;
}

EE ocl_trans_mem(
    GCLHandle_t handle, GCLMem_t src, GCLMemDesc srcDesc, GCLMem_t dst, GCLMemDesc dstDesc)
{
    if (srcDesc.memType == dstDesc.memType && srcDesc.memType == GCL_MEM_BUF) {
        U32 sw_str, sh_str, sc_str, sw_off, sh_off;
        U32 dw_str, dh_str, dc_str, dw_off, dh_off;
        DataFormat sf, df;
        sf = srcDesc.memFormat;
        df = dstDesc.memFormat;
        get_gclmem_dim(srcDesc, &sw_str, &sh_str, &sc_str, &sw_off, &sh_off);
        get_gclmem_dim(dstDesc, &dw_str, &dh_str, &dc_str, &dw_off, &dh_off);
        U32 gs[3] = {0, 0, 0};
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        Mem srcMem = src->mem;
        Mem dstMem = dst->mem;
        Kernel kernel;
        KernelOpt kernelOpt;
        char kernelName[128];
        if (sf == df) {
            if (sw_str == dw_str && sh_str == dh_str && sc_str == dc_str && sw_off == dw_off &&
                sh_off == dh_off) {
                U32 len = srcDesc.num;
                gs[0] = (len + 3) / 4;
                ls[0] = 0;
                dim = 1;
                CHECK_STATUS(gcl_create_kernel(handle, "copy_f16", &kernel));
                CHECK_STATUS(gcl_set_kernelArgs(kernel, len, len, 0, 0, gs[0], srcMem, dstMem));
                gcl_set_kernelVec(handle, kernel, dim, gs, ls, "copy_fp16");
            } else if (sf == DF_NCHW && sw_off == 0 && sh_off == 0 && sc_str == dc_str) {
                gs[0] = (dw_str + 3) / 4;
                gs[1] = dh_str;
                gs[2] = dc_str;
                dim = 3;
                U32 pl = dw_off;
                U32 pr = dw_str - sw_str - pl;
                U32 pt = dh_off;
                U32 pb = dh_str - sh_str - pt;
                if (pr < 0 || pb < 0) {
                    CHECK_STATUS(NOT_MATCH);
                }
                CHECK_STATUS(set_padding_opt_mali(
                    true, PAD_CONSTANT, DT_F16, GCL_MEM_BUF, GCL_MEM_BUF, kernelName, &kernelOpt));
                CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
                CHECK_STATUS(gcl_set_kernelArgs(kernel, sw_str, sh_str, dw_str, dh_str, 0, 0,
                    sw_str, sh_str, dw_str, dh_str, pl, pr, pt, pb, gs[0], gs[1], srcMem, dstMem));
                gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
            } else {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        } else if (sf == DF_NCHW && df == DF_NCHWC4) {
            GCLMem sMem = *src;
            sMem.desc = srcDesc;
            GCLMem dMem = *dst;
            dMem.desc = dstDesc;
            CHECK_STATUS(ocl_data_trans_form(handle, &sMem, &dMem, 0, 0, NCHW_TO_NCHWC4));
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return SUCCESS;
}

EE ocl_map_mem_write(
    GCLHandle_t handle, GCLMem_t gclMem, GCLMemDesc desc, TensorDesc hostDesc, U8 *host_ptr)
{
    DataType hdt;
    DataFormat hdf;
    U32 n, c, t, h, w;
    tensorSelectGet(hostDesc, &hdt, &hdf, &n, &c, &h, &w, &t);
    DataFormat mf = desc.memFormat;
    U32 w_str, h_str, c_str, w_off, h_off;
    get_gclmem_dim(desc, &w_str, &h_str, &c_str, &w_off, &h_off);
    U32 size = tensorNumBytes(hostDesc);
    if ((mf != DF_NCHWC4 && w == w_str && h == h_str) ||
        (hdf == DF_NCHW && mf == DF_NCHWC4 && w * h * t * n * w_str * h_str == 1)) {
        if (size > desc.byteSize) {
            size = desc.byteSize;
        }
        CHECK_STATUS(gcl_trans_memory(handle, host_ptr, gclMem, &size, HOST_TO_DEVICE_BUF, CL_TRUE));
    } else {
        U32 offset = desc.byteSize / 2;
        Kernel kernel;
        CHECK_STATUS(
            gcl_trans_memory(handle, host_ptr, gclMem, &size, HOST_TO_DEVICE_BUF, CL_TRUE, &offset));
        offset = offset / bytesOf(DT_F16);
        if (hdt != DT_F16) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        if (mf == DF_NCHWC4) {
            if (t > 1) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
            if (w != w_str || h != h_str) {
                gclMem->desc.byteSize = desc.byteSize / 2;
                gcl_fill_memory_zero(handle, gclMem);
            }
            GCLMem sMem = *gclMem;
            GCLMem dMem = *gclMem;
            sMem.desc = desc;
            dMem.desc = desc;
            U32 stride[3] = {w, h, n * c * t};
            U32 offset[3] = {0, 0, 0};
            MemFlags flag = CL_MEM_READ_WRITE;
            CHECK_STATUS(gclmem_set_desc_padding(
                &(sMem.desc), stride, offset, DT_F16, DF_NCHW, GCL_MEM_BUF, flag));
            CHECK_STATUS(ocl_data_trans_form(handle, &sMem, &dMem, 0, 0, NCHW_TO_NCHWC4, false));
        } else {
            KernelOpt kernelOpt;
            char kernelName[128];
            U32 gs[3] = {(w_str + 3) / 4, h_str, c * t * n};
            U32 ls[3] = {0, 0, 0};
            U32 dim = 3;
            U32 pl = w_off;
            U32 pr = w_str - w - pl;
            U32 pt = h_off;
            U32 pb = h_str - h - pt;
            if (pr < 0 || pb < 0) {
                CHECK_STATUS(NOT_MATCH);
            }
            CHECK_STATUS(set_padding_opt_mali(
                true, PAD_CONSTANT, DT_F16, GCL_MEM_BUF, GCL_MEM_BUF, kernelName, &kernelOpt));
            CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel, &kernelOpt));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, w, h, w_str, h_str, offset, 0, w, h, w_str,
                h_str, pl, pr, pt, pb, gs[0], gs[1], gclMem->mem, gclMem->mem));
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
        }
        /*        
        U32 size = desc.byteSize / 2;
        U32 offset = 0;
        U32 mapPtrIndex = gclMem->mapPtrArray.size();
        CHECK_STATUS(gcl_map_memory(handle, gclMem, &offset, &size, CL_MAP_WRITE, CL_TRUE));
        F16* gpuMapPtr = (F16*) gclMem->mapPtrArray[mapPtrIndex];
        F16* host_val = (F16*)host_ptr;
        U32 in_whtc = w * h * t * c;
        U32 in_wht = w * h * t;
        U32 in_wh = w * h;
        if (mf == DF_NCHWC4) {
            U32 out_whtc = w_str * h_str * t * 4 * (c + 3) / 4;
            U32 out_wht = w_str * h_str * t;//j is 4x
            U32 out_wh = w_str * h_str * 4;
            U32 gpu_base = w_off * h_str * 4 + h_off * 4;

            for (U32 i = 0; i < n; i++) {
                U32 host_off_i = i * in_whtc;
                U32 gpu_off_i = i * out_whtc;
                for (U32 j = 0; j < c; j += 4) {
                    U32 host_off_j = j * in_wht;
                    U32 gpu_off_j = j * out_wht;
                    for (U32 k = 0; k < t; k++) {
                        U32 host_off_k = k * in_wh;
                        U32 gpu_off_k = k * out_wh;
                        for (U32 ii = 0; ii < w; ii++) {
                            for (U32 jj = 0; jj < h; jj++) {
                                for (U32 kk = 0; kk < 4; kk++) {
                                    F16 val = (j + kk < c) ? host_val[host_off_i + host_off_j + host_off_k + ii + jj * w + in_wht * kk] : 0;
                                    gpuMapPtr[gpu_base + gpu_off_i + gpu_off_j + gpu_off_k + ii * h_str * 4 + jj * 4 + kk] = val;
                                }
                            }
                        }
                    }
                }
            }
            
        } else {
            U32 out_whtc = w_str * h_str * t * c;
            U32 out_wht = w_str * h_str * t;
            U32 out_wh = w_str * h_str;
            U32 gpu_base = h_off * w_str + w_off;

            for (U32 i = 0; i < n; i++) {
                U32 host_off_i = i * in_whtc;
                U32 gpu_off_i = i * out_whtc;
                for (U32 j = 0; j < c; j ++) {
                    U32 host_off_j = j * in_wht;
                    U32 gpu_off_j = j * out_wht;
                    for (U32 k = 0; k < t; k++) {
                        U32 host_off_k = k * in_wh;
                        U32 gpu_off_k = k * out_wh;
                        for (U32 ii = 0; ii < h; ii++) {
                            U32 host_off_ii = ii * w;
                            U32 gpu_off_ii = ii * w_str;
                            for (U32 jj = 0; jj < w; jj++) {
                                F16 val = host_val[host_off_i + host_off_j + host_off_k + host_off_ii + jj];
                                gpuMapPtr[gpu_base + gpu_off_i + gpu_off_j + gpu_off_k + gpu_off_ii + jj] = val;
                            }
                        }
                    }
                }
            }
        }
        gcl_unmap_memory(handle, gclMem);*/
    }
    return SUCCESS;
}

EE ocl_map_mem_read(GCLHandle_t handle, GCLMem_t gclMem, GCLMemDesc desc)
{
    DataType dt;
    DataFormat df;
    U32 n, c, h, w;
    CHECK_STATUS(gclmem_get_desc_dim(desc, &dt, &df, &n, &c, &h, &w));
    DataFormat mf = desc.memFormat;
    U32 w_str, h_str, c_str, w_off, h_off;
    get_gclmem_dim(desc, &w_str, &h_str, &c_str, &w_off, &h_off);
    bool needTrans = true;
    U32 offset = 0;
    U32 size = n * c * h * w * bytesOf(dt);
    if (mf != DF_NCHWC4) {
        if (w_str == w && h_str == h) {
            needTrans = false;
        }
        if (w_off == 0 && h_off == 0) {
            U32 totalNum = w_str * h_str * c_str;
            if (totalNum == w_str || totalNum == h_str || totalNum == c_str) {
                needTrans = false;
            }
        }
    } else if (w_str == 1 && h_str == 1) {
        needTrans = false;
    }
    if (needTrans) {
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        Kernel kernel;
        Mem buf = gclMem->mem;
        offset = desc.num;
        GCLMem sMem = *gclMem;
        GCLMem dMem = *gclMem;
        sMem.desc = desc;
        dMem.desc = desc;
        U32 str[3] = {w, h, n * c};
        U32 off[3] = {0, 0, 0};
        MemFlags flag = CL_MEM_READ_WRITE;
        CHECK_STATUS(
            gclmem_set_desc_padding(&(dMem.desc), str, off, DT_F16, DF_NCHW, GCL_MEM_BUF, flag));
        if (mf == DF_NCHWC4) {
            CHECK_STATUS(
                ocl_data_trans_form(handle, &sMem, &dMem, 0, offset, NCHWC4_TO_NCHW, false));
        } else if (mf == DF_NCHW) {
            CHECK_STATUS(ocl_data_trans_form(handle, &sMem, &dMem, 0, offset, NCHW_TO_NCHW, false));
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        offset = desc.num * bytesOf(dt);
    }
    CHECK_STATUS(gcl_map_memory(handle, gclMem, &offset, &size, CL_MAP_READ, CL_TRUE));
    return SUCCESS;
}
