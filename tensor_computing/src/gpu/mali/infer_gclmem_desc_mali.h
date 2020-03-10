// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _INFER_GCLMEM_DESC_MALI_F16
#define _INFER_GCLMEM_DESC_MALI_F16

inline EE infer_gclmem_desc_nchw_to_ncwhc4(U32 iw, U32 ih, U32 ic, U32 pw, U32 ph, U32 ow, U32 oh, U32 oc,
                                      GCLMemDesc_t gclmemInputDesc, GCLMemDesc_t gclmemOutputDesc){
    if(gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) return NULL_POINTER;
    U32 s0, s1, s2;
    s0 = ow;
    s1 = oh;
    s2 = (oc + 3) / 4;

    U32 num, byteSize;
    num      = s0 * s1 * s2 * 4;
    byteSize = num * bytesOf(DT_F16);
    gclmemOutputDesc->stride[0] = s0;
    gclmemOutputDesc->stride[1] = s1;
    gclmemOutputDesc->stride[2] = s2;
    gclmemOutputDesc->offset[0] = 0;
    gclmemOutputDesc->offset[1] = 0;
    gclmemOutputDesc->offset[2] = 0;
    gclmemOutputDesc->num       = num;
    gclmemOutputDesc->byteSize  = byteSize;

    U32 pw_org, ph_org;
    U32 s0_org, s1_org, s2_org;
    U32 byteSize_org;

    s0_org = gclmemInputDesc->stride[0];
    s1_org = gclmemInputDesc->stride[1];
    s2_org = gclmemInputDesc->stride[2];
    pw_org = gclmemInputDesc->offset[0];
    ph_org = gclmemInputDesc->offset[1];
    byteSize_org = gclmemInputDesc->byteSize;
    if(byteSize_org != 0 && gclmemInputDesc->memFormat != DF_NCWHC4) return NOT_SUPPORTED;

    pw = (pw > pw_org) ? pw : pw_org;
    ph = (ph > ph_org) ? ph : ph_org;

    s0 = iw + (pw << 1);
    s1 = ih + (ph << 1);
    s2 = (ic + 3) >> 2;
    s0 = (s0 > s0_org) ? s0 : s0_org;
    s1 = (s1 > s1_org) ? s1 : s1_org;
    s2 = (s2 > s2_org) ? s2 : s2_org;

    num      = s0 * s1 * s2 * 4;
    byteSize = num * bytesOf(DT_F16);
    byteSize = (byteSize > byteSize_org) ? byteSize : byteSize_org;

    gclmemInputDesc->stride[0] = s0;
    gclmemInputDesc->stride[1] = s1;
    gclmemInputDesc->stride[2] = s2;
    gclmemInputDesc->offset[0] = pw;
    gclmemInputDesc->offset[1] = ph;
    gclmemInputDesc->offset[2] = 0;
    gclmemInputDesc->num       = num;
    gclmemInputDesc->byteSize  = byteSize;

    gclmemInputDesc->memType    = GCL_MEM_BUF;
    gclmemInputDesc->memFormat  = DF_NCWHC4;
    gclmemInputDesc->flags      = CL_MEM_READ_WRITE;
    gclmemInputDesc->host_ptr   = NULL;
    gclmemOutputDesc->memType   = GCL_MEM_BUF;
    gclmemOutputDesc->memFormat = DF_NCWHC4;
    gclmemOutputDesc->flags     = CL_MEM_READ_WRITE;
    gclmemOutputDesc->host_ptr  = NULL;
    return SUCCESS;
}

inline EE infer_gclmem_desc_ncwhc4_to_ncwhc4(U32 iw, U32 ih, U32 ic, U32 pw, U32 ph, U32 ow, U32 oh, U32 oc,
                                             GCLMemDesc_t gclmemInputDesc, GCLMemDesc_t gclmemOutputDesc){
    if(gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) return NULL_POINTER;
    U32 s0, s1, s2;
    s0 = ow;
    s1 = oh;
    s2 = (oc + 3) / 4;

    U32 num, byteSize;
    num      = s0 * s1 * s2 * 4;
    byteSize = num * bytesOf(DT_F16);
    gclmemOutputDesc->stride[0] = s0;
    gclmemOutputDesc->stride[1] = s1;
    gclmemOutputDesc->stride[2] = s2;
    gclmemOutputDesc->offset[0] = 0;
    gclmemOutputDesc->offset[1] = 0;
    gclmemOutputDesc->offset[2] = 0;
    gclmemOutputDesc->num       = num;
    gclmemOutputDesc->byteSize  = byteSize;

    U32 pw_org, ph_org;
    U32 s0_org, s1_org, s2_org;
    U32 byteSize_org;

    s0_org = gclmemInputDesc->stride[0];
    s1_org = gclmemInputDesc->stride[1];
    s2_org = gclmemInputDesc->stride[2];
    pw_org = gclmemInputDesc->offset[0];
    ph_org = gclmemInputDesc->offset[1];
    byteSize_org = gclmemInputDesc->byteSize;
    if(byteSize_org != 0 && gclmemInputDesc->memFormat != DF_NCWHC4) return NOT_SUPPORTED;

    pw = (pw > pw_org) ? pw : pw_org;
    ph = (ph > ph_org) ? ph : ph_org;

    s0 = iw + (pw << 1);
    s1 = ih + (ph << 1);
    s2 = (ic + 3) >> 2;
    s0 = (s0 > s0_org) ? s0 : s0_org;
    s1 = (s1 > s1_org) ? s1 : s1_org;
    s2 = (s2 > s2_org) ? s2 : s2_org;

    num      = s0 * s1 * s2 * 4;
    byteSize = num * bytesOf(DT_F16);
    byteSize = (byteSize > byteSize_org) ? byteSize : byteSize_org;

    gclmemInputDesc->stride[0] = s0;
    gclmemInputDesc->stride[1] = s1;
    gclmemInputDesc->stride[2] = s2;
    gclmemInputDesc->offset[0] = pw;
    gclmemInputDesc->offset[1] = ph;
    gclmemInputDesc->offset[2] = 0;
    gclmemInputDesc->num       = num;
    gclmemInputDesc->byteSize  = byteSize;

    gclmemInputDesc->memType    = GCL_MEM_BUF;
    gclmemInputDesc->memFormat  = DF_NCWHC4;
    gclmemInputDesc->flags      = CL_MEM_READ_WRITE;
    gclmemInputDesc->host_ptr   = NULL;
    gclmemOutputDesc->memType   = GCL_MEM_BUF;
    gclmemOutputDesc->memFormat = DF_NCWHC4;
    gclmemOutputDesc->flags     = CL_MEM_READ_WRITE;
    gclmemOutputDesc->host_ptr  = NULL;
    return SUCCESS;
}
inline EE infer_gclmem_desc_nchwc3_to_nchw(U32 iw, U32 ih, U32 ic, U32 pw, U32 ph, U32 ow, U32 oh, U32 oc,
                                         GCLMemDesc_t gclmemInputDesc, GCLMemDesc_t gclmemOutputDesc){
    if(gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) return NULL_POINTER;
    U32 s0, s1, s2;
    s0 = ow;
    s1 = oh;
    s2 = oc;

    U32 num, byteSize;
    num      = s0 * s1 * s2;
    byteSize = num * bytesOf(DT_F16);
    gclmemOutputDesc->stride[0] = s0;
    gclmemOutputDesc->stride[1] = s1;
    gclmemOutputDesc->stride[2] = s2;
    gclmemOutputDesc->offset[0] = 0;
    gclmemOutputDesc->offset[1] = 0;
    gclmemOutputDesc->offset[2] = 0;
    gclmemOutputDesc->num       = num;
    gclmemOutputDesc->byteSize  = byteSize;

    U32 pw_org, ph_org;
    U32 s0_org, s1_org, s2_org;
    U32 byteSize_org;

    s0_org = gclmemInputDesc->stride[0];
    s1_org = gclmemInputDesc->stride[1];
    s2_org = gclmemInputDesc->stride[2];
    pw_org = gclmemInputDesc->offset[0];
    ph_org = gclmemInputDesc->offset[1];
    byteSize_org = gclmemInputDesc->byteSize;
    if(byteSize_org != 0 && gclmemInputDesc->memFormat != DF_NCHWC3) return NOT_SUPPORTED;

    pw = (pw > pw_org) ? pw : pw_org;
    ph = (ph > ph_org) ? ph : ph_org;

    s0 = iw + (pw << 1);
    s1 = ih + (ph << 1);
    s2 = (ic + 2) / 3;
    s0 = (s0 > s0_org) ? s0 : s0_org;
    s1 = (s1 > s1_org) ? s1 : s1_org;
    s2 = (s2 > s2_org) ? s2 : s2_org;

    num      = s0 * s1 * s2 * 3;
    byteSize = num * bytesOf(DT_F16);
    byteSize = (byteSize > byteSize_org) ? byteSize : byteSize_org;

    gclmemInputDesc->stride[0] = s0;
    gclmemInputDesc->stride[1] = s1;
    gclmemInputDesc->stride[2] = s2;
    gclmemInputDesc->offset[0] = pw;
    gclmemInputDesc->offset[1] = ph;
    gclmemInputDesc->offset[2] = 0;
    gclmemInputDesc->num       = num;
    gclmemInputDesc->byteSize  = byteSize;

    gclmemInputDesc->memType    = GCL_MEM_BUF;
    gclmemInputDesc->memFormat  = DF_NCHWC3;
    gclmemInputDesc->flags      = CL_MEM_READ_WRITE;
    gclmemInputDesc->host_ptr   = NULL;
    gclmemOutputDesc->memType   = GCL_MEM_BUF;
    gclmemOutputDesc->memFormat = DF_NCHW;
    gclmemOutputDesc->flags     = CL_MEM_READ_WRITE;
    gclmemOutputDesc->host_ptr  = NULL;
    return SUCCESS;
}

inline EE infer_gclmem_desc_ncwhc4(U32 iw, U32 ih, U32 ic, U32 pw, U32 ph, U32 ow, U32 oh, U32 oc, DataType idt, DataType odt,
                                   GCLMemDesc_t gclmemInputDesc, GCLMemDesc_t gclmemOutputDesc){
    if(gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) return NULL_POINTER;
    U32 s0, s1, s2;
    s0 = oh;
    s1 = ow;
    s2 = (oc + 3) / 4;

    U32 num, byteSize;
    num      = s0 * s1 * s2 * 4;
    byteSize = num * bytesOf(odt);
    gclmemOutputDesc->stride[0] = s0;
    gclmemOutputDesc->stride[1] = s1;
    gclmemOutputDesc->stride[2] = s2;
    gclmemOutputDesc->offset[0] = 0;
    gclmemOutputDesc->offset[1] = 0;
    gclmemOutputDesc->offset[2] = 0;
    gclmemOutputDesc->num       = num;
    gclmemOutputDesc->byteSize  = byteSize;

    U32 pw_org, ph_org;
    U32 s0_org, s1_org, s2_org;
    U32 byteSize_org;

    s0_org = gclmemInputDesc->stride[0];
    s1_org = gclmemInputDesc->stride[1];
    s2_org = gclmemInputDesc->stride[2];
    ph_org = gclmemInputDesc->offset[0];
    pw_org = gclmemInputDesc->offset[1];
    byteSize_org = gclmemInputDesc->byteSize;
    if(byteSize_org != 0 && gclmemInputDesc->memFormat != DF_NCWHC4) return NOT_SUPPORTED;

    ph = (ph > ph_org) ? ph : ph_org;
    pw = (pw > pw_org) ? pw : pw_org;

    s0 = ih + (ph << 1);
    s1 = iw + (pw << 1);
    s2 = (ic + 3) / 4;
    s0 = (s0 > s0_org) ? s0 : s0_org;
    s1 = (s1 > s1_org) ? s1 : s1_org;
    s2 = (s2 > s2_org) ? s2 : s2_org;

    num      = s0 * s1 * s2 * 4;
    byteSize = num * bytesOf(idt);
    byteSize = (byteSize > byteSize_org) ? byteSize : byteSize_org;

    gclmemInputDesc->stride[0] = s0;
    gclmemInputDesc->stride[1] = s1;
    gclmemInputDesc->stride[2] = s2;
    gclmemInputDesc->offset[0] = ph;
    gclmemInputDesc->offset[1] = pw;
    gclmemInputDesc->offset[2] = 0;
    gclmemInputDesc->num       = num;
    gclmemInputDesc->byteSize  = byteSize;

    gclmemInputDesc->memType    = GCL_MEM_BUF;
    gclmemInputDesc->memFormat  = DF_NCWHC4;
    gclmemInputDesc->flags      = CL_MEM_READ_WRITE;
    gclmemInputDesc->host_ptr   = NULL;
    gclmemOutputDesc->memType   = GCL_MEM_BUF;
    gclmemOutputDesc->memFormat = DF_NCWHC4;
    gclmemOutputDesc->flags     = CL_MEM_READ_WRITE;
    gclmemOutputDesc->host_ptr  = NULL;
    return SUCCESS;
}

inline EE infer_gclmem_desc_nhwc(U32 iw, U32 ih, U32 ic, U32 pc, U32 pw, U32 ow, U32 oh, U32 oc, DataType idt, DataType odt,
    GCLMemDesc_t gclmemInputDesc, GCLMemDesc_t gclmemOutputDesc){
    if(gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) return NULL_POINTER;
    U32 s0, s1, s2;
    s0 = oc;
    s1 = ow;
    s2 = oh;

    U32 num, byteSize;
    num      = s0 * s1 * s2;
    byteSize = num * bytesOf(odt);
    gclmemOutputDesc->stride[0] = s0;
    gclmemOutputDesc->stride[1] = s1;
    gclmemOutputDesc->stride[2] = s2;
    gclmemOutputDesc->offset[0] = 0;
    gclmemOutputDesc->offset[1] = 0;
    gclmemOutputDesc->offset[2] = 0;
    gclmemOutputDesc->num       = num;
    gclmemOutputDesc->byteSize  = byteSize;

    U32 pc_org, pw_org;
    U32 s0_org, s1_org, s2_org;
    U32 byteSize_org;

    s0_org = gclmemInputDesc->stride[0];
    s1_org = gclmemInputDesc->stride[1];
    s2_org = gclmemInputDesc->stride[2];
    pc_org = gclmemInputDesc->offset[0];
    pw_org = gclmemInputDesc->offset[1];
    byteSize_org = gclmemInputDesc->byteSize;
    if(byteSize_org != 0 && gclmemInputDesc->memFormat != DF_NHWC) return NOT_SUPPORTED;

    pc = (pc > pc_org) ? pc : pc_org;
    pw = (pw > pw_org) ? pw : pw_org;

    s0 = ic + (pc << 1);
    s1 = iw + (pw << 1);
    s2 = ih;
    s0 = (s0 > s0_org) ? s0 : s0_org;
    s1 = (s1 > s1_org) ? s1 : s1_org;
    s2 = (s2 > s2_org) ? s2 : s2_org;

    num      = s0 * s1 * s2;
    byteSize = num * bytesOf(idt);
    byteSize = (byteSize > byteSize_org) ? byteSize : byteSize_org;

    gclmemInputDesc->stride[0] = s0;
    gclmemInputDesc->stride[1] = s1;
    gclmemInputDesc->stride[2] = s2;
    gclmemInputDesc->offset[0] = pc;
    gclmemInputDesc->offset[1] = pw;
    gclmemInputDesc->offset[2] = 0;
    gclmemInputDesc->num       = num;
    gclmemInputDesc->byteSize  = byteSize;

    gclmemInputDesc->memType    = GCL_MEM_BUF;
    gclmemInputDesc->memFormat  = DF_NHWC;
    gclmemInputDesc->flags      = CL_MEM_READ_WRITE;
    gclmemInputDesc->host_ptr   = NULL;
    gclmemOutputDesc->memType   = GCL_MEM_BUF;
    gclmemOutputDesc->memFormat = DF_NHWC;
    gclmemOutputDesc->flags     = CL_MEM_READ_WRITE;
    gclmemOutputDesc->host_ptr  = NULL;
    return SUCCESS;
}
#endif
