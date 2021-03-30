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
    DataTransFormType type,
    bool setKernelVec)
{
    U32 iw, ih, ic, in;
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow, oh, oc, on;
    U32 ow_str, oh_str, ow_off, oh_off;
    CHECK_STATUS(gclmem_get_desc_dim(input->desc, NULL, NULL, &in, &ic, &ih, &iw));
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_dim(output->desc, NULL, NULL, &on, &oc, &oh, &ow));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    Mem inbuf = input->mem;
    Mem outbuf = output->mem;
    U32 nDims = input->desc.nDims;
    if (on > 1 && oc % 4 != 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (type == NCWHC4_TO_NCHW) {
        if (nDims != 4) {
            CHECK_STATUS(NOT_MATCH)
        }
    }
    Kernel kernel;
    char kernelName[128];
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    switch (type) {
        case NCWHC4_TO_NCHW:
            sprintf(kernelName, "mem_trans_ncwhc4_to_nchw");
            gs[0] = oh;
            gs[1] = (ow + 3) / 4;
            gs[2] = (oc + 3) / 4 * on;
            break;
        case NCHW_TO_NCWHC4:
            sprintf(kernelName, "mem_trans_nchw_to_ncwhc4");
            gs[0] = (ow + 3) / 4;
            gs[1] = oh;
            gs[2] = (oc + 3) / 4 * on;
            break;
        case NCHW_TO_NCHW:
            sprintf(kernelName, "mem_trans_nchw_to_nchw");
            gs[0] = (ow + 3) / 4;
            gs[1] = oh;
            gs[2] = oc * on;
            break;
        default:
            CHECK_STATUS(NOT_SUPPORTED);
    }
    if (setKernelVec) {
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    } else {
        CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel));
    }
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str, oh_str, ow_off,
        oh_off, iw, ih, ic, ow, oh, oc, in_off, out_off, inbuf, outbuf));
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
        U32 iw, ih, ic, in, it;
        DataType hdt;
        DataFormat hdf;
        tensorSelectGet(hostDesc, &hdt, &hdf, &in, &ic, &ih, &iw, &it);
        U32 ow, oh, oc, pw, ph;
        get_gclmem_dim(input->desc, &ow, &oh, &oc, &pw, &ph);
        if (hdf == DF_NCHW || hdf == DF_MTK) {
            if (desc.memFormat == DF_NCHW || (ow == 1 && oh == 1 && pw == 0 && ph == 0)) {
                GCLMem_t dst = (iw == ow && ih == oh) ? input : tmpBuf;
                CHECK_STATUS(gcl_trans_memory(
                    handle, (void *)hostPtr, (void *)dst, &size, HOST_TO_DEVICE_BUF, CL_TRUE));
                if (iw != ow || ih != oh) {
                    CHECK_STATUS(gcl_get_kernel_from_map(handle, "padding_input_gclmem", &kernel));
                    CHECK_STATUS(gcl_set_kernelArgs(
                        kernel, iw, ih, pw, ph, ow, oh, 0, 0, tmpBuf->mem, input->mem));
                    U32 gs[3] = {(ow + 3) / 4, oh, ic * it * in};
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
            if (desc.memFormat == DF_NHWC) {
                if (ic == oc && iw == ow && pw == 0 && ph == 0) {
                    gcl_trans_memory(handle, (void *)hostPtr, (void *)input, &size,
                        HOST_TO_DEVICE_BUF, blocking);
                    return SUCCESS;
                }
            }
            return NOT_SUPPORTED;
        }

        if (hdf == DF_NORMAL) {
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
    U32 ow, oh, oc, on, ot;
    U32 iw, ih, ic, pw, ph;
    tensorSelectGet(hostDesc, &host_dt, &host_df, &on, &oc, &oh, &ow, &ot);
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
        if ((on * oc & 3) != (oc & 3)) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        CHECK_STATUS(gcl_get_kernel_from_map(handle, "mem_trans_3d_ncwhc4_to_nchw", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw, ih, pw, ph, ow, oh, 0, 0, ow, oh, oc * on, ot,
            ow, oh, oc * on, ot, 0, offset, input->mem, input->mem));
        U32 gs[3] = {oh, (ow + 3) >> 2, (oc + 3) / 4 * on * ot};
        U32 ls[3] = {0, 0, 0};
        U32 dim = 3;
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "mem_trans_3d_ncwhc4_to_nchw"));
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
                gs[0] = (dw_str + 3) / 4;
                gs[1] = dh_str;
                gs[2] = dc_str;
                dim = 3;
                CHECK_STATUS(gcl_create_kernel(handle, "padding_input_gclmem", &kernel));
                CHECK_STATUS(gcl_set_kernelArgs(
                    kernel, sw_str, sh_str, dw_off, dh_off, dw_str, dh_str, 0, 0, srcMem, dstMem));
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
    if ((mf == DF_NCHW && w == w_str && h == h_str) ||
        (hdf == DF_NCHW && mf == DF_NCWHC4 && w * h * t * n * w_str * h_str == 1)) {
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
        if (mf == DF_NCWHC4) {
            if (t > 1) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
            if (w != w_str || h != h_str) {
                gclMem->desc.byteSize = desc.byteSize / 2;
                gcl_fill_memory_zero(handle, gclMem);
            }
            CHECK_STATUS(gcl_get_kernel_from_map(handle, "mem_trans_nchw_to_ncwhc4", &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, w, h, 0, 0, w_str, h_str, w_off, h_off, w, h, c,
                w, h, c, offset, 0, gclMem->mem, gclMem->mem));
            U32 gs[3] = {(w + 3) / 4, h, (c + 3) / 4 * n};
            U32 ls[3] = {0, 0, 0};
            U32 dim = 3;
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "mem_trans_nchw_to_ncwhc4"));
        } else {
            CHECK_STATUS(gcl_get_kernel_from_map(handle, "padding_input_gclmem", &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(
                kernel, w, h, w_off, h_off, w_str, h_str, offset, 0, gclMem->mem, gclMem->mem));
            U32 gs[3] = {(w_str + 3) / 4, h_str, c * t * n};
            U32 ls[3] = {0, 0, 0};
            U32 dim = 3;
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "padding_input_gclmem"));
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
        if (mf == DF_NCWHC4) {
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
    if (mf != DF_NCWHC4) {
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
        if (mf == DF_NCWHC4) {
            if (n > 1) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
            U32 gs[3] = {h, (w + 3) >> 2, (c + 3) / 4};
            CHECK_STATUS(gcl_get_kernel_from_map(handle, "mem_trans_ncwhc4_to_nchw", &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, w_str, h_str, w_off, h_off, w, h, 0, 0, w, h, c,
                w, h, c, 0, offset, buf, buf));
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "mem_trans_ncwhc4_to_nchw"));
        } else if (mf == DF_NCHW) {
            U32 gs[3] = {(w + 3) >> 2, h, c * n};
            CHECK_STATUS(gcl_get_kernel_from_map(handle, "mem_trans_nchw_to_nchw", &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, w_str, h_str, w_off, h_off, w, h, 0, 0, w, h,
                c * n, w, h, c * n, 0, offset, buf, buf));
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "mem_trans_nchw_to_nchw"));
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        offset = desc.num * bytesOf(dt);
    }
    CHECK_STATUS(gcl_map_memory(handle, gclMem, &offset, &size, CL_MAP_READ, CL_TRUE));
    return SUCCESS;
}
