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

#include "tensor_desc.h"
#include "gcl_func.h"
#include "fill_memory_zero_vec4_opt.h"

inline void get_gclmem_dim(
    GCLMemDesc desc, U32 *w_str, U32 *h_str, U32 *c_str, U32 *w_off, U32 *h_off)
{
    if (desc.memFormat == DF_NCHW || desc.memFormat == DF_NCHWC4) {
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

inline cl_channel_type get_channel_type(DataType dt)
{
    cl_channel_type ret;
    switch (dt) {
        case DT_F16: {
            ret = CL_HALF_FLOAT;
            break;
        }
        case DT_F32: {
            ret = CL_FLOAT;
            break;
        }
        //case DT_U8: {
        //    ret = CL_UNSIGNED_INT8;
        //    break;
        //}
        //case DT_I8: {
        //    ret = CL_SIGNED_INT8;
        //    break;
        //}
        //case DT_U32: {
        //    ret = CL_UNSIGNED_INT32;
        //    break;
        //}
        //case DT_I32: {
        //    ret = CL_SIGNED_INT32;
        //    break;
        //}
        default: {
            UNI_ERROR_LOG("not support to create %s type image.\n", DataTypeName()[dt]);
        }
    }
    return ret;
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
    desc.memFormat = DF_NCHW;
    desc.memType = GCL_MEM_BUF;
    desc.byteSize = 0;
    desc.num = 0;
    desc.flags = CL_MEM_READ_WRITE;
    desc.imgFormat.image_channel_order = CL_RGBA;
#ifdef _USE_FP16
    desc.imgFormat.image_channel_data_type = CL_HALF_FLOAT;
#else
    desc.imgFormat.image_channel_data_type = CL_FLOAT;
#endif
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
    desc->dt = dt;
    U32 num = 0;
    U32 bytes = 0;
    if (mf == DF_NHWC || mf == DF_NCHW || mt != GCL_MEM_BUF) {
        num = stride[0] * stride[1] * stride[2];
    } else if (mf == DF_NCHWC4) {
        num = stride[0] * stride[1] * stride[2] * 4;
    } else {
        return NOT_SUPPORTED;
    }
    bytes = num * bytesOf(dt);
    if (mt != GCL_MEM_BUF) {
        bytes = bytes * 4;
    }
    desc->imgFormat.image_channel_data_type = get_channel_type(dt);
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

inline EE ocl_fill_memory_zero(GCLHandle_t handle, GCLMem_t gclMem, U32 offset)
{
    U32 w_str, h_str, c_str;
    CHECK_STATUS(gclmem_get_desc_padding(gclMem->desc, &w_str, &h_str, &c_str, NULL, NULL));
    char kernelName[128];
    Kernel kernel;
    KernelOpt kernelOpt;
    U32 gs[3] = {0, 0, 0};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 1;
    U32 len = w_str * h_str * c_str;
    GCLMemType mType = gclMem->desc.memType;
    CHECK_STATUS(set_fill_memory_zero_vec4_opt_mali(gclMem->desc.dt, mType, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    if (mType == GCL_MEM_BUF) {
        if (gclMem->desc.memFormat == DF_NCHWC4) {
            gs[0] = len;
            len *= 4;
        } else {
            gs[0] = (len + 3) / 4;
        }
    } else {
        gs[0] = w_str;
        gs[1] = h_str;
        gs[2] = c_str;
        dim = 3;
    }
    CHECK_STATUS(gcl_set_kernelArgs(kernel, len, offset, gs[0], gs[1], gclMem->mem));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
//    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

inline EE fill_output_zero(GCLHandle_t handle, GCLMem_t output, TensorDesc outputDesc)
{
    GCLMem mem = *output;
    GCLMemDesc desc = output->desc;
    bool need_pad = desc.need_pad;
    if (desc.memType != GCL_MEM_BUF) {
        U32 w_str, h_str, c_str;
        CHECK_STATUS(gclmem_get_desc_padding(desc, &w_str, &h_str, &c_str, NULL, NULL));
        U32 width, height, depth;
        CHECK_STATUS(gcl_get_image_size(output, &width, &height, &depth));
        if (width == 0) {
            width = 1;
        }
        if (height == 0) {
            height = 1;
        }
        if (depth == 0) {
            depth = 1;
        }
        if (width == w_str && height == h_str && depth == c_str) {
            need_pad = false;
        }
        if (need_pad) {
            desc.stride[0] += desc.offset[3];
            desc.stride[1] += desc.offset[4];
            desc.stride[2] += desc.offset[5];
        }
        mem.desc = desc;
    }
    if (need_pad) {
        CHECK_STATUS(ocl_fill_memory_zero(handle, &mem, 0));
    }
    return SUCCESS;
}

#endif
