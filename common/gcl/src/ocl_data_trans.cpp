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
#include "types.h"
#include "tensor_desc.h"
#include "error.h"
#include "gcl_common.h"
#include "gcl_func.h"
#include "gclmem_desc_infer.h"
#include "ocl_data_trans.h"

EE ocl_set_input(GCLHandle_t handle,
    GCLMem_t input,
    TensorDesc hostDesc,
    const U8 *hostPtr,
    GCLMem_t tmpBuf,
    bool blocking)
{
    GCLMemDesc desc = input->desc;
    if (desc.memType == GCL_MEM_BUF) {
        U32 size = tensorNumBytes(hostDesc);
        Kernel kernel;
        U32 iw, ih, ic, in;
        DataType hdt;
        DataFormat hdf;
        if (hostDesc.df == DF_NCHW || hostDesc.df == DF_NHWC) {
            tensorSelectGet(hostDesc, &hdt, &hdf, &in, &ic, &ih, &iw);
        } else if (hostDesc.df == DF_NORMAL) {
            tensor2dGet(hostDesc, &hdt, &hdf, &ih, &iw);
            ic = 1;
            in = 1;
            hdf = DF_NORMAL;
        } else {
            return NOT_SUPPORTED;
        }
        if (hdf == DF_NCHW) {
            U32 ow, oh, pw, ph;
            ow = input->desc.stride[0];
            oh = input->desc.stride[1];
            pw = input->desc.offset[0];
            ph = input->desc.offset[1];
            if (desc.memFormat == DF_NCHW || (ow == 1 && oh == 1 && pw == 0 && ph == 0)) {
                GCLMem_t dst = (iw == ow && ih == oh) ? input : tmpBuf;
                CHECK_STATUS(gcl_trans_memory(
                    handle, (void *)hostPtr, (void *)dst, &size, HOST_TO_DEVICE_BUF, CL_TRUE));
                if (iw != ow || ih != oh) {
                    CHECK_STATUS(gcl_get_kernel_from_map(handle, "padding_input_gclmem", &kernel));
                    CHECK_STATUS(
                        gcl_set_kernelArgs(kernel, iw, ih, pw, ph, ow, oh, tmpBuf->mem, input->mem));
                    U32 gs[3] = {(ow + 3) / 4 * 4, (oh + 3) / 4 * 4, ic};
                    U32 ls[3] = {0, 0, 0};
                    U32 dim = 3;
                    CHECK_STATUS(
                        gcl_run_kernel(handle, kernel, dim, gs, ls, "padding_input_gclmem"));
                }
                return SUCCESS;
            }

            if (desc.memFormat == DF_NCWHC4) {
                if (hdt != DT_F16) {
                    return NOT_SUPPORTED;
                }
                oh = input->desc.stride[0];
                ow = input->desc.stride[1];
                ph = input->desc.offset[0];
                pw = input->desc.offset[1];
                gcl_trans_memory(
                    handle, (void *)hostPtr, (void *)tmpBuf, &size, HOST_TO_DEVICE_BUF, blocking);
                CHECK_STATUS(gcl_get_kernel_from_map(handle, "mem_trans_nchw_to_ncwhc4", &kernel));
                CHECK_STATUS(gcl_set_kernelArgs(kernel, iw, ih, 0, 0, ow, oh, pw, ph, iw, ih, ic,
                    iw, ih, ic, 0, 0, tmpBuf->mem, input->mem));
                U32 gs[3] = {(iw + 3) / 4, ih, (ic + 3) / 4 * in};
                U32 ls[3] = {0, 0, 0};
                U32 dim = 3;
                CHECK_STATUS(
                    gcl_run_kernel(handle, kernel, dim, gs, ls, "mem_trans_nchw_to_ncwhc4"));
                return SUCCESS;
            }
            return NOT_SUPPORTED;
        }

        if (hdf == DF_NHWC) {
            U32 oc, ow, pc, pw;
            oc = input->desc.stride[0];
            ow = input->desc.stride[1];
            pc = input->desc.offset[0];
            pw = input->desc.offset[1];
            if (desc.memFormat == DF_NHWC) {
                if (ic == oc && iw == ow && pc == 0 && pw == 0) {
                    gcl_trans_memory(handle, (void *)hostPtr, (void *)input, &size,
                        HOST_TO_DEVICE_BUF, blocking);
                    return SUCCESS;
                }
            }
            return NOT_SUPPORTED;
        }

        if (hdf == DF_NORMAL) {
            U32 oh, ow, ph, pw;
            ow = input->desc.stride[0];
            oh = input->desc.stride[1];
            pw = input->desc.offset[0];
            ph = input->desc.offset[1];
            if (desc.memFormat == DF_NCHW) {
                if (iw == ow && ih == oh && pw == 0 && ph == 0) {
                    gcl_trans_memory(handle, (void *)hostPtr, (void *)input, &size,
                        HOST_TO_DEVICE_BUF, blocking);
                    return SUCCESS;
                }
            }
            return NOT_SUPPORTED;
        }
    }
    return NOT_SUPPORTED;
}

EE ocl_get_output(GCLHandle_t handle, const GCLMem_t input, TensorDesc hostDesc, bool blocking)
{
    GCLMemDesc desc = input->desc;
    Kernel kernel;
    DataType host_dt;
    DataFormat host_df, device_df;
    U32 ow, oh, oc, on;
    U32 iw, ih, ic, pw, ph;
    tensorSelectGet(hostDesc, &host_dt, &host_df, &on, &oc, &oh, &ow);
    U32 size = tensorNumBytes(hostDesc);
    U32 offset = 0;
    get_gclmem_dim(desc, &iw, &ih, &ic, &pw, &ph);
    device_df = desc.memFormat;
    if (desc.byteSize < size) {
        CHECK_STATUS(NOT_MATCH);
    }

    if (device_df == DF_NCWHC4 && (host_df == DF_NCHW || host_df == DF_NORMAL) &&
        host_dt == DT_F16 && (ih != 1 || iw != 1)) {
        if (desc.byteSize < size * 2) {
            CHECK_STATUS(NOT_MATCH);
        }
        offset = iw * ih * ic * 4;
        CHECK_STATUS(gcl_get_kernel_from_map(handle, "mem_trans_ncwhc4_to_nchw", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw, ih, pw, ph, ow, oh, 0, 0, ow, oh, oc, ow, oh,
            oc, 0, offset, input->mem, input->mem));
        U32 gs[3] = {oh, (ow + 3) >> 2, (oc + 3) / 4 * on};
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "mem_trans_ncwhc4_to_nchw"));
        offset = offset * bytesOf(host_dt);
    }

    if (device_df == DF_NCWHC4 && host_df == DF_MKT) {
        if (desc.byteSize < size * 2) {
            CHECK_STATUS(NOT_MATCH);
        }
        offset = iw * ih * ic * 4;
        U32 gs[2] = {oh, (oc + 3) / 4};
        U32 ls[2] = {0, 0};
        U32 dim = 2;
        CHECK_STATUS(gcl_get_kernel_from_map(handle, "mem_trans_ncwhc4_to_mtk", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(
            kernel, ih, iw, ph, pw, oc, offset, gs[0], gs[1], input->mem, input->mem));
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "mem_trans_ncwhc4_to_mtk"));
        offset = offset * bytesOf(host_dt);
    }
    CHECK_STATUS(gcl_map_memory(handle, input, &offset, &size, CL_MAP_READ, blocking));
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
                gs[0] = (dw_str + 3) / 4 * 4;
                gs[1] = (dh_str + 3) / 4 * 4;
                gs[2] = dc_str;
                dim = 3;
                CHECK_STATUS(gcl_create_kernel(handle, "padding_input_gclmem", &kernel));
                CHECK_STATUS(gcl_set_kernelArgs(
                    kernel, sw_str, sh_str, dw_off, dh_off, dw_str, dh_str, srcMem, dstMem));
                gcl_set_kernelVec(handle, kernel, dim, gs, ls, "padding_input_gclmem");
            } else {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        } else if (sf == DF_NCHW && df == DF_NCWHC4) {
            U32 iw, ih, ic;
            TensorDesc cpuDesc = tensor4df(srcDesc.dt, srcDesc.df, srcDesc.dims[3], srcDesc.dims[2],
                srcDesc.dims[1], srcDesc.dims[0]);
            tensorSelectGet(cpuDesc, NULL, NULL, NULL, &ic, &ih, &iw);
            gs[0] = (iw + 3) / 4;
            gs[1] = ih;
            gs[2] = (ic + 3) / 4;
            U32 ls[3] = {0, 0, 0};
            U32 dim = 3;
            CHECK_STATUS(gcl_create_kernel(handle, "mem_trans_nchw_to_ncwhc4", &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, sw_str, sh_str, sw_off, sh_off, dw_str, dh_str,
                dw_off, dh_off, iw, ih, ic, iw, ih, ic, 0, 0, srcMem, dstMem));
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, "mem_trans_nchw_to_ncwhc4");
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return SUCCESS;
}

EE ocl_map_mem(GCLHandle_t handle, GCLMem_t gclMem, GCLMemDesc desc)
{
    DataType dt;
    DataFormat df;
    U32 n, c, h, w;
    CHECK_STATUS(gclmem_get_desc_non_padding(desc, &dt, &df, &n, &c, &h, &w));

    DataFormat mf = desc.memFormat;
    U32 w_str, h_str, c_str, w_off, h_off;
    get_gclmem_dim(desc, &w_str, &h_str, &c_str, &w_off, &h_off);
    bool needTrans = true;
    U32 offset = 0;
    U32 size = n * c * h * w * bytesOf(dt);
    if (w_str == w && h_str == h && c_str == c && mf != DF_NCWHC4) {
        needTrans = false;
    }
    if (w_str == 1 && h_str == 1 && mf == DF_NCWHC4) {
        needTrans = false;
    }
    if (needTrans) {
        if (mf == DF_NCWHC4) {
            U32 gs[3] = {h, (w + 3) >> 2, (c + 3) / 4 * n};
            U32 ls[3] = {0, 0, 0};
            U32 dim = 3;
            Kernel kernel;
            Mem buf = gclMem->mem;
            offset = desc.num;
            CHECK_STATUS(gcl_get_kernel_from_map(handle, "mem_trans_ncwhc4_to_nchw", &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, w_str, h_str, w_off, h_off, w, h, 0, 0, w, h, c,
                w, h, c, 0, offset, buf, buf));
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "mem_trans_ncwhc4_to_nchw"));
            offset = desc.num * bytesOf(dt);
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    CHECK_STATUS(gcl_map_memory(handle, gclMem, &offset, &size, CL_MAP_READ, CL_TRUE));
    return SUCCESS;
}
