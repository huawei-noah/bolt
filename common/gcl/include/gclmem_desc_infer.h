// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _GCLMEM_DESC_INFER
#define _GCLMEM_DESC_INFER
#include <stdio.h>
#include "tensor_desc.h"
#include "gcl_func.h"

inline EE trans_gclmem_desc_nchw_ncwhc4(
    U32 iw, U32 ih, U32 ic, U32 pw, U32 ph, DataType dt, GCLMemDesc_t gclmemDesc, bool need_pad = false)
{
    U32 s0, s1, s2;
    U32 num, byteSize;
    U32 pw_org, ph_org;
    U32 s0_org, s1_org, s2_org;
    U32 byteSize_org;

    if (gclmemDesc) {
        if (gclmemDesc->memFormat != DF_NCHW) {
            return NOT_SUPPORTED;
        }
        s0_org = gclmemDesc->stride[1];
        s1_org = gclmemDesc->stride[0];
        s2_org = gclmemDesc->stride[2];
        ph_org = gclmemDesc->offset[1];
        pw_org = gclmemDesc->offset[0];
        if (pw_org == 0 && ph_org == 0) {
            if (s2_org == 1 && (s0_org == 1 || s1_org == 1)) {
                s2_org = (s0_org == 1) ? s1_org : s0_org;
                s0_org = 1;
                s1_org = 1;
            }
        }
        s2_org = (s2_org + 3) / 4;
        byteSize_org = gclmemDesc->byteSize;
        bool need_pad_org = gclmemDesc->need_pad;
        if (pw == 0 && ph == 0) {
            if (ic == 1 && (iw == 1 || ih == 1)) {
                ic = (iw == 1) ? ih : iw;
                iw = 1;
                ih = 1;
            }
        }
        ph = (ph > ph_org) ? ph : ph_org;
        pw = (pw > pw_org) ? pw : pw_org;

        s0 = ih + (ph << 1);
        s1 = iw + (pw << 1);
        s2 = (ic + 3) / 4;
        s0 = (s0 > s0_org) ? s0 : s0_org;
        s1 = (s1 > s1_org) ? s1 : s1_org;
        s2 = (s2 > s2_org) ? s2 : s2_org;
        num = s0 * s1 * s2 * 4;
        byteSize = num * bytesOf(dt);
        byteSize = (byteSize > byteSize_org) ? byteSize : byteSize_org;

        gclmemDesc->stride[0] = s0;
        gclmemDesc->stride[1] = s1;
        gclmemDesc->stride[2] = s2;
        gclmemDesc->offset[0] = ph;
        gclmemDesc->offset[1] = pw;
        gclmemDesc->offset[2] = 0;
        gclmemDesc->num = num;
        gclmemDesc->byteSize = byteSize;
        gclmemDesc->memType = GCL_MEM_BUF;
        gclmemDesc->memFormat = DF_NCWHC4;
        gclmemDesc->flags = CL_MEM_READ_WRITE;
        gclmemDesc->host_ptr = NULL;
        gclmemDesc->need_pad = need_pad | need_pad_org;
    }
    return SUCCESS;
}

inline EE infer_gclmem_desc_ncwhc4(U32 iw,
    U32 ih,
    U32 ic,
    U32 pw,
    U32 ph,
    U32 ow,
    U32 oh,
    U32 oc,
    DataType idt,
    DataType odt,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc,
    bool need_pad = false)
{
    U32 s0, s1, s2;
    U32 num, byteSize;
    U32 pw_org, ph_org;
    U32 s0_org, s1_org, s2_org;
    U32 byteSize_org;
    if (gclmemOutputDesc) {
        s0 = oh;
        s1 = ow;
        s2 = (oc + 3) / 4;
        num = s0 * s1 * s2 * 4;
        byteSize = num * bytesOf(odt);
        gclmemOutputDesc->stride[0] = s0;
        gclmemOutputDesc->stride[1] = s1;
        gclmemOutputDesc->stride[2] = s2;
        gclmemOutputDesc->offset[0] = 0;
        gclmemOutputDesc->offset[1] = 0;
        gclmemOutputDesc->offset[2] = 0;
        gclmemOutputDesc->num = num;
        gclmemOutputDesc->byteSize = byteSize;
        gclmemOutputDesc->memType = GCL_MEM_BUF;
        gclmemOutputDesc->memFormat = DF_NCWHC4;
        gclmemOutputDesc->flags = CL_MEM_READ_WRITE;
        gclmemOutputDesc->host_ptr = NULL;
    }

    if (gclmemInputDesc) {
        byteSize_org = gclmemInputDesc->byteSize;
        if (byteSize_org != 0 && gclmemInputDesc->memFormat != DF_NCWHC4) {
            return trans_gclmem_desc_nchw_ncwhc4(iw, ih, ic, pw, ph, idt, gclmemInputDesc, need_pad);
        }
        s0_org = gclmemInputDesc->stride[0];
        s1_org = gclmemInputDesc->stride[1];
        s2_org = gclmemInputDesc->stride[2];
        ph_org = gclmemInputDesc->offset[0];
        pw_org = gclmemInputDesc->offset[1];
        if (pw_org == 0 && ph_org == 0) {
            if (s2_org == 1 && (s0_org == 1 || s1_org == 1)) {
                s2_org = (s0_org == 1) ? s1_org : s0_org;
                s0_org = 1;
                s1_org = 1;
            }
        }
        bool need_pad_org = gclmemInputDesc->need_pad;
        if (pw == 0 && ph == 0) {
            if (ic == 1 && (iw == 1 || ih == 1)) {
                ic = (iw == 1) ? ih : iw;
                iw = 1;
                ih = 1;
            }
        }

        ph = (ph > ph_org) ? ph : ph_org;
        pw = (pw > pw_org) ? pw : pw_org;

        s0 = ih + (ph << 1);
        s1 = iw + (pw << 1);
        s2 = (ic + 3) / 4;
        s0 = (s0 > s0_org) ? s0 : s0_org;
        s1 = (s1 > s1_org) ? s1 : s1_org;
        s2 = (s2 > s2_org) ? s2 : s2_org;
        num = s0 * s1 * s2 * 4;
        byteSize = num * bytesOf(idt);
        byteSize = (byteSize > byteSize_org) ? byteSize : byteSize_org;

        gclmemInputDesc->stride[0] = s0;
        gclmemInputDesc->stride[1] = s1;
        gclmemInputDesc->stride[2] = s2;
        gclmemInputDesc->offset[0] = ph;
        gclmemInputDesc->offset[1] = pw;
        gclmemInputDesc->offset[2] = 0;
        gclmemInputDesc->num = num;
        gclmemInputDesc->byteSize = byteSize;
        gclmemInputDesc->memType = GCL_MEM_BUF;
        gclmemInputDesc->memFormat = DF_NCWHC4;
        gclmemInputDesc->flags = CL_MEM_READ_WRITE;
        gclmemInputDesc->host_ptr = NULL;
        gclmemInputDesc->need_pad = need_pad | need_pad_org;
    }
    return SUCCESS;
}

inline EE infer_gclmem_desc_ncwhc4_3d(U32 iw,
    U32 ih,
    U32 ic,
    U32 it,
    U32 in,
    U32 pw,
    U32 ph,
    U32 ow,
    U32 oh,
    U32 oc,
    U32 ot,
    U32 on,
    DataType idt,
    DataType odt,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc,
    bool need_pad = false)
{
    ic = (ic + 3) / 4 * 4 * it * in;
    oc = (oc + 3) / 4 * 4 * ot * on;
    return infer_gclmem_desc_ncwhc4(
        iw, ih, ic, pw, ph, ow, oh, oc, idt, odt, gclmemInputDesc, gclmemOutputDesc, need_pad);
}

inline EE infer_gclmem_desc_nhwc(U32 iw,
    U32 ih,
    U32 ic,
    U32 pc,
    U32 pw,
    U32 ow,
    U32 oh,
    U32 oc,
    DataType idt,
    DataType odt,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc,
    bool need_pad = false)
{
    U32 s0, s1, s2;
    U32 num, byteSize;
    U32 pc_org, pw_org;
    U32 s0_org, s1_org, s2_org;
    U32 byteSize_org;

    if (gclmemOutputDesc) {
        s0 = oc;
        s1 = ow;
        s2 = oh;
        num = s0 * s1 * s2;
        byteSize = num * bytesOf(odt);
        gclmemOutputDesc->stride[0] = s0;
        gclmemOutputDesc->stride[1] = s1;
        gclmemOutputDesc->stride[2] = s2;
        gclmemOutputDesc->offset[0] = 0;
        gclmemOutputDesc->offset[1] = 0;
        gclmemOutputDesc->offset[2] = 0;
        gclmemOutputDesc->num = num;
        gclmemOutputDesc->byteSize = byteSize;
        gclmemOutputDesc->memType = GCL_MEM_BUF;
        gclmemOutputDesc->memFormat = DF_NHWC;
        gclmemOutputDesc->flags = CL_MEM_READ_WRITE;
        gclmemOutputDesc->host_ptr = NULL;
    }

    if (gclmemInputDesc) {
        s0_org = gclmemInputDesc->stride[0];
        s1_org = gclmemInputDesc->stride[1];
        s2_org = gclmemInputDesc->stride[2];
        pc_org = gclmemInputDesc->offset[0];
        pw_org = gclmemInputDesc->offset[1];
        byteSize_org = gclmemInputDesc->byteSize;
        bool need_pad_org = gclmemInputDesc->need_pad;
        if (byteSize_org != 0 && gclmemInputDesc->memFormat != DF_NHWC) {
            return NOT_SUPPORTED;
        }

        pc = (pc > pc_org) ? pc : pc_org;
        pw = (pw > pw_org) ? pw : pw_org;
        s0 = ic + (pc << 1);
        s1 = iw + (pw << 1);
        s2 = ih;
        s0 = (s0 > s0_org) ? s0 : s0_org;
        s1 = (s1 > s1_org) ? s1 : s1_org;
        s2 = (s2 > s2_org) ? s2 : s2_org;

        num = s0 * s1 * s2;
        byteSize = num * bytesOf(idt);
        byteSize = (byteSize > byteSize_org) ? byteSize : byteSize_org;
        gclmemInputDesc->stride[0] = s0;
        gclmemInputDesc->stride[1] = s1;
        gclmemInputDesc->stride[2] = s2;
        gclmemInputDesc->offset[0] = pc;
        gclmemInputDesc->offset[1] = pw;
        gclmemInputDesc->offset[2] = 0;
        gclmemInputDesc->num = num;
        gclmemInputDesc->byteSize = byteSize;
        gclmemInputDesc->memType = GCL_MEM_BUF;
        gclmemInputDesc->memFormat = DF_NHWC;
        gclmemInputDesc->flags = CL_MEM_READ_WRITE;
        gclmemInputDesc->host_ptr = NULL;
        gclmemInputDesc->need_pad = need_pad | need_pad_org;
    }
    return SUCCESS;
}

inline EE infer_gclmem_desc_nchw(U32 iw,
    U32 ih,
    U32 ic,
    U32 pw,
    U32 ph,
    U32 ow,
    U32 oh,
    U32 oc,
    DataType idt,
    DataType odt,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc,
    bool need_pad = false)
{
    U32 s0, s1, s2;
    U32 num, byteSize;
    U32 pw_org, ph_org;
    U32 s0_org, s1_org, s2_org;
    U32 byteSize_org;

    if (gclmemOutputDesc) {
        s0 = ow;
        s1 = oh;
        s2 = oc;
        num = s0 * s1 * s2;
        byteSize = num * bytesOf(odt);
        gclmemOutputDesc->stride[0] = s0;
        gclmemOutputDesc->stride[1] = s1;
        gclmemOutputDesc->stride[2] = s2;
        gclmemOutputDesc->offset[0] = 0;
        gclmemOutputDesc->offset[1] = 0;
        gclmemOutputDesc->offset[2] = 0;
        gclmemOutputDesc->num = num;
        gclmemOutputDesc->byteSize = byteSize;
        gclmemOutputDesc->memType = GCL_MEM_BUF;
        gclmemOutputDesc->memFormat = DF_NCHW;
        gclmemOutputDesc->flags = CL_MEM_READ_WRITE;
        gclmemOutputDesc->host_ptr = NULL;
    }

    if (gclmemInputDesc) {
        s0_org = gclmemInputDesc->stride[0];
        s1_org = gclmemInputDesc->stride[1];
        s2_org = gclmemInputDesc->stride[2];
        pw_org = gclmemInputDesc->offset[0];
        ph_org = gclmemInputDesc->offset[1];
        byteSize_org = gclmemInputDesc->byteSize;
        bool need_pad_org = gclmemInputDesc->need_pad;
        if (byteSize_org != 0 && gclmemInputDesc->memFormat != DF_NCHW) {
            return NOT_SUPPORTED;
        }

        pw = (pw > pw_org) ? pw : pw_org;
        ph = (ph > ph_org) ? ph : ph_org;
        s0 = iw + (pw << 1);
        s1 = ih + (ph << 1);
        s2 = ic;
        s0 = (s0 > s0_org) ? s0 : s0_org;
        s1 = (s1 > s1_org) ? s1 : s1_org;
        s2 = (s2 > s2_org) ? s2 : s2_org;

        num = s0 * s1 * s2;
        byteSize = num * bytesOf(idt);
        byteSize = (byteSize > byteSize_org) ? byteSize : byteSize_org;
        gclmemInputDesc->stride[0] = s0;
        gclmemInputDesc->stride[1] = s1;
        gclmemInputDesc->stride[2] = s2;
        gclmemInputDesc->offset[0] = pw;
        gclmemInputDesc->offset[1] = ph;
        gclmemInputDesc->offset[2] = 0;
        gclmemInputDesc->num = num;
        gclmemInputDesc->byteSize = byteSize;
        gclmemInputDesc->memType = GCL_MEM_BUF;
        gclmemInputDesc->memFormat = DF_NCHW;
        gclmemInputDesc->flags = CL_MEM_READ_WRITE;
        gclmemInputDesc->host_ptr = NULL;
        gclmemInputDesc->need_pad = need_pad | need_pad_org;
    }
    return SUCCESS;
}

inline EE infer_gclmem_desc_nchw_3d(U32 iw,
    U32 ih,
    U32 ic,
    U32 it,
    U32 in,
    U32 pw,
    U32 ph,
    U32 ow,
    U32 oh,
    U32 oc,
    U32 ot,
    U32 on,
    DataType idt,
    DataType odt,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc,
    bool need_pad = false)
{
    ic = ic * it * in;
    oc = oc * ot * on;
    return infer_gclmem_desc_nchw(
        iw, ih, ic, pw, ph, ow, oh, oc, idt, odt, gclmemInputDesc, gclmemOutputDesc);
}

inline void get_nlp_mkt_val(TensorDesc desc, DataType *dt, U32 *m, U32 *k, U32 *t)
{
    if (dt) {
        *dt = desc.dt;
    }
    if (desc.df == DF_MTK) {
        if (m) {
            *m = desc.dims[2];
        }
        if (t) {
            *t = desc.dims[1];
        }
        if (k) {
            *k = desc.dims[0];
        }
    } else if (desc.df == DF_MKT) {
        if (m) {
            *m = desc.dims[2];
        }
        if (k) {
            *k = desc.dims[1];
        }
        if (t) {
            *t = desc.dims[0];
        }
    } else {
        CHECK_STATUS(NOT_MATCH);
    }
}

inline void map_nlp_mkt_to_ncwhc4(U32 m, U32 k, U32 t, U32 *gw, U32 *gh, U32 *gc)
{
    if (gw) {
        *gw = 1;
    }
    if (gh) {
        *gh = t;
    }
    if (gc) {
        *gc = (k + 3) / 4 * m;
    }
}

inline void get_gclmem_dim(
    GCLMemDesc desc, U32 *w_str, U32 *h_str, U32 *c_str, U32 *w_off, U32 *h_off)
{
    if (desc.memFormat == DF_NCHW) {
        if (w_str) {
            *w_str = desc.stride[0];
        }
        if (h_str) {
            *h_str = desc.stride[1];
        }
        if (c_str) {
            *c_str = desc.stride[2];
        }
        if (w_off) {
            *w_off = desc.offset[0];
        }
        if (h_off) {
            *h_off = desc.offset[1];
        }
    } else if (desc.memFormat == DF_NCWHC4) {
        if (w_str) {
            *w_str = desc.stride[1];
        }
        if (h_str) {
            *h_str = desc.stride[0];
        }
        if (c_str) {
            *c_str = desc.stride[2];
        }
        if (w_off) {
            *w_off = desc.offset[1];
        }
        if (h_off) {
            *h_off = desc.offset[0];
        }
    } else if (desc.memFormat == DF_NHWC) {
        if (w_str) {
            *w_str = desc.stride[1];
        }
        if (h_str) {
            *h_str = desc.stride[2];
        }
        if (c_str) {
            *c_str = desc.stride[0];
        }
        if (w_off) {
            *w_off = desc.offset[1];
        }
        if (h_off) {
            *h_off = desc.offset[0];
        }
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
}

inline EE fill_output_zero(GCLHandle_t handle, GCLMem_t output, TensorDesc outputDesc)
{
    GCLMemDesc outGCLDesc = output->desc;
    if (!outGCLDesc.need_pad) {
        return SUCCESS;
    }
    DataType dt;
    U32 ow_str, oh_str, oc_str;
    get_gclmem_dim(outGCLDesc, &ow_str, &oh_str, &oc_str, NULL, NULL);
    char kernelname[128];
    U32 gs = ow_str * oh_str * oc_str;
    U32 ls = 0;
    U32 dim = 1;
    Kernel kernel;
    U32 ow, oh;
    if (outGCLDesc.memFormat == DF_NCWHC4) {
        if (outputDesc.df == DF_NCHW) {
            tensorSelectGet(outputDesc, &dt, NULL, NULL, NULL, &oh, &ow);
            if (ow_str != ow || oh_str != oh) {
                if (dt == DT_F16) {
                    sprintf(kernelname, "fill_memory_zero_vec4_f16");
                } else if (dt == DT_I32 || dt == DT_U32) {
                    sprintf(kernelname, "fill_memory_zero_vec4_i32");
                } else {
                    return NOT_SUPPORTED;
                }
                CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
                CHECK_STATUS(gcl_set_kernelArgs(kernel, gs * 4, 0, gs, output->mem));
                gcl_set_kernelVec(handle, kernel, dim, &gs, &ls, kernelname);
#ifdef _DEBUG
                CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs, &ls, kernelname));
#endif
            }
            return SUCCESS;
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    } else if (outGCLDesc.memFormat == DF_NCHW || outGCLDesc.memFormat == DF_NHWC) {
        if (outputDesc.df == DF_NCHW || outputDesc.df == DF_NORMAL || outputDesc.df == DF_NHWC ||
            outputDesc.df == DF_MTK) {
            tensorSelectGet(outputDesc, &dt, NULL, NULL, NULL, &oh, &ow);
            if (ow_str != ow || oh_str != oh) {
                if (dt == DT_F16) {
                    sprintf(kernelname, "fill_memory_zero_vec4_f16");
                } else if (dt == DT_I32 || dt == DT_U32) {
                    sprintf(kernelname, "fill_memory_zero_vec4_i32");
                } else {
                    return NOT_SUPPORTED;
                }
                U32 len = gs;
                gs = (gs + 3) / 4;
                CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
                CHECK_STATUS(gcl_set_kernelArgs(kernel, len, 0, gs, output->mem));
                gcl_set_kernelVec(handle, kernel, dim, &gs, &ls, kernelname);
#ifdef _DEBUG
                CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs, &ls, kernelname));
#endif
            }
            return SUCCESS;

        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    return NOT_SUPPORTED;
}

inline GCLMemDesc gclmem_build_desc()
{
    GCLMemDesc desc;
    for (U32 i = 0; i < 6; i++) {
        desc.dims[i] = 0;
    }
    for (U32 i = 0; i < 3; i++) {
        desc.stride[i] = 0;
        desc.offset[i] = 0;
    }
    desc.nDims = 4;
    desc.dt = DT_U8;
    desc.df = DF_NCHW;
    desc.memFormat = DF_NCWHC4;
    desc.memType = GCL_MEM_BUF;
    desc.byteSize = 0;
    desc.num = 0;
    desc.flags = CL_MEM_READ_WRITE;
    desc.imgFormat.image_channel_order = CL_RGBA;
    desc.imgFormat.image_channel_data_type = CL_HALF_FLOAT;
    desc.host_ptr = NULL;
    desc.need_pad = false;
    return desc;
}

inline EE gclmem_set_desc_padding(GCLMemDesc *desc,
    U32 *stride,
    U32 *offset,
    DataType dt,
    DataFormat mf,
    GCLMemType mt,
    MemFlags flags,
    void *host_ptr = NULL)
{
    if (desc == NULL) {
        return NULL_POINTER;
    }
    desc->stride[0] = stride[0];
    desc->stride[1] = stride[1];
    desc->stride[2] = stride[2];
    desc->offset[0] = offset[0];
    desc->offset[1] = offset[1];
    desc->offset[2] = offset[2];
    desc->memFormat = mf;
    desc->memType = mt;
    desc->flags = flags;
    desc->host_ptr = host_ptr;
    U32 num = 0;
    U32 bytes = 0;
    if (mf == DF_NHWC || mf == DF_NCHW || mt != GCL_MEM_BUF) {
        num = stride[0] * stride[1] * stride[2];
    } else if (mf == DF_NCWHC4) {
        num = stride[0] * stride[1] * stride[2] * 4;
    } else {
        return NOT_SUPPORTED;
    }
    bytes = num * bytesOf(dt);
    if (mt != GCL_MEM_BUF) {
        bytes = bytes * 4;
    }
    desc->num = num;
    desc->byteSize = bytes;
    return SUCCESS;
}

inline EE gclmem_get_desc_dim(
    GCLMemDesc desc, DataType *dt, DataFormat *df, U32 *num, U32 *numChannels, U32 *height, U32 *width)
{
    TensorDesc cpuDesc;
    cpuDesc.dt = desc.dt;
    cpuDesc.nDims = desc.nDims;
    for (U32 i = 0; i < desc.nDims; ++i) {
        cpuDesc.dims[i] = desc.dims[i];
    }
    tensorSelectGet(cpuDesc, dt, df, num, numChannels, height, width);
    return SUCCESS;
}

inline EE gclmem_get_desc_dim_5d(GCLMemDesc desc,
    DataType *dt,
    DataFormat *df,
    U32 *num,
    U32 *numChannels,
    U32 *time,
    U32 *height,
    U32 *width)
{
    U32 ndims = desc.nDims;
    if (time) {
        if (ndims == 5) {
            *time = desc.dims[2];
        } else {
            *time = 1;
        }
    }
    return gclmem_get_desc_dim(desc, dt, df, num, numChannels, height, width);
}

inline EE gclmem_get_desc_padding(
    GCLMemDesc desc, U32 *w_str, U32 *h_str, U32 *c_str, U32 *w_off, U32 *h_off)
{
    get_gclmem_dim(desc, w_str, h_str, c_str, w_off, h_off);
    return SUCCESS;
}

#endif
