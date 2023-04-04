// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _MEMORY_OCL_IMG_H
#define _MEMORY_OCL_IMG_H

#include "memory_ocl.hpp"

class OclMemoryImg : public OclMemory {
public:
    OclMemoryImg() : OclMemory()
    {
        this->desc.imgFormat.image_channel_order = CL_RGBA;
#ifdef _USE_FP16
        this->desc.imgFormat.image_channel_data_type = CL_HALF_FLOAT;
#else
        this->desc.imgFormat.image_channel_data_type = CL_FLOAT;
#endif
        this->desc.memType = GCL_MEM_IMG_3D;
    }

    OclMemoryImg(MemoryType type) : OclMemory()
    {
        new (this)OclMemoryImg();
        if (type == OCLMemImg) {
            this->desc.memType = GCL_MEM_IMG_3D;
        } else if (type == OCLMemImg2D) {
            this->desc.memType = GCL_MEM_IMG_2D;
        } else if (type == OCLMemImg1D) {
            this->desc.memType = GCL_MEM_IMG_1D;
        } else {
            UNI_ERROR_LOG("Unsupported GPU Memory image type\n");
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }

    MemoryType get_mem_type() override
    {
        MemoryType type = OCLMemImg;
        if (this->desc.memType == GCL_MEM_IMG_2D) {
            type = OCLMemImg2D;
        } else if (this->desc.memType == GCL_MEM_IMG_1D) {
            type = OCLMemImg1D;
        }
        return type;
    }

    void resize(TensorDesc desc) override
    {
        this->desc.nDims = desc.nDims;
        for (U32 i = 0; i < desc.nDims; i++) {
            this->desc.dims[i] = desc.dims[i];
        }
        this->desc.dt = desc.dt;
        this->desc.df = desc.df;
        set_image_shape(desc);
    }

    EE padding(U32 pl, U32 pr, U32 pt, U32 pb, U32 pf, U32 pa) override
    {
        if (this->desc.memFormat != DF_NCHWC4) {
            pr = (pr + 3) / 4;
        }
        if (pr > this->desc.offset[3]) {
            this->desc.offset[3] = pr;
        }
        if (pb > this->desc.offset[4]) {
            this->desc.offset[4] = pb;
        }
        if (pa > this->desc.offset[5]) {
            this->desc.offset[5] = pa;
        }
        if (pr > 0 || pb > 0 || pa > 0) {
            this->desc.need_pad = true;
        }
        return SUCCESS;
    }

    void alloc() override
    {
        if (!this->allocated) {
            if (this->desc.stride[0] > this->capacitySize[0] ||
                this->desc.stride[1] > this->capacitySize[1] ||
                this->desc.stride[2] > this->capacitySize[2]) {
                GCLMem_t mem = ocl_alloc_gclmem(this->desc);
                this->val = std::shared_ptr<GCLMem>(mem, ocl_release_gclmem);
                this->allocated = true;
                this->capacitySize[0] = this->desc.stride[0];
                this->capacitySize[1] = this->desc.stride[1];
                this->capacitySize[2] = this->desc.stride[2];
            }
        }
    }

    void alloc(U32 width, U32 height, U32 depth)
    {
        if (width > this->desc.stride[0] || height > this->desc.stride[1] ||
            depth > this->desc.stride[2]) {
            this->allocated = false;
        }
        this->desc.stride[0] = width;
        this->desc.stride[1] = height;
        this->desc.stride[2] = depth;
        this->desc.offset[0] = 0;
        this->desc.offset[1] = 0;
        this->desc.offset[2] = 0;
        this->desc.num = this->desc.stride[0] * this->desc.stride[1] * this->desc.stride[2] * 4;
        this->desc.byteSize = desc.num * bytesOf(this->desc.dt);
        this->alloc();
    }

    EE copy_from(Memory *other) override
    {
        EE ret = SUCCESS;
        MemoryType srcType = other->get_mem_type();
        if (srcType == CPUMem) {
            TensorDesc cpuDesc = ((CpuMemory *)other)->get_desc();
            U8 *host_ptr = (U8 *)((CpuMemory *)other)->get_ptr();
            U32 size = ((CpuMemory *)other)->bytes();
            U8 *tmp = nullptr;
            if (size < this->desc.byteSize) {
                if (this->get_mem_type() == OCLMemImg1D) {
                    tmp = (U8 *)UNI_OPERATOR_NEW(this->bytes());
                    UNI_MEMSET(tmp, 0, this->bytes());
                    UNI_MEMCPY(tmp, host_ptr, size);
                    host_ptr = tmp;
                } else {
                    CHECK_STATUS(NOT_MATCH);
                }
            }

            if (!allocated) {
                OclMemory::alloc(host_ptr);
            } else {
                if (this->desc.memFormat != DF_NCHWC4) {
                    CHECK_STATUS(gcl_trans_memory(OCLContext::getInstance().handle.get(), host_ptr,
                        this->val.get(), this->desc.stride, HOST_TO_DEVICE_IMG, CL_TRUE));
                } else {
                    CHECK_STATUS(NOT_SUPPORTED);
                }
            }
            if (tmp != nullptr) {
                UNI_OPERATOR_DELETE(tmp);
            }
        } else {
            if (!allocated) {
                this->alloc();
            }
            void *srcPtr = ((OclMemory *)other)->get_ptr();
            void *dstPtr = this->val.get();
            U32 srcBytes = ((OclMemory *)other)->bytes();
            if (srcType == OCLMem) {
                if (this->bytes() > srcBytes) {
                    CHECK_STATUS(NOT_MATCH);
                }
                U32 region[3] = {this->desc.stride[0], this->desc.stride[1], this->desc.stride[2]};
                CHECK_STATUS(gcl_trans_memory(OCLContext::getInstance().handle.get(), srcPtr,
                    dstPtr, region, DEVICE_BUF_TO_IMG, CL_TRUE))
            } else {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        return ret;
    }

    void set_shared_ptr(std::shared_ptr<GCLMem> val)
    {
        this->val = val;
        this->allocated = true;
        this->capacitySize[0] = this->desc.stride[0];
        this->capacitySize[1] = this->desc.stride[1];
        this->capacitySize[2] = this->desc.stride[2];
    }

    EE reuse(Memory *other) override
    {
        MemoryType type = other->get_mem_type();
        if (type == CPUMem) {
            CHECK_STATUS(this->copy_from(other));
        } else if (type == this->get_mem_type()) {
            U32 size[3];
            other->capacity(size);
            if (size[0] >= this->desc.stride[0] && size[1] >= this->desc.stride[1] &&
                size[2] >= this->desc.stride[2]) {
                this->val = ((OclMemory *)other)->get_shared_ptr();
                this->allocated = true;
                other->capacity(this->capacitySize);
            } else {
                UNI_ERROR_LOG("small OCL image can not meet big OCL image demand\n");
                CHECK_STATUS(NOT_SUPPORTED);
            }
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        return SUCCESS;
    }

    void capacity(U32 *size) override
    {
        size[0] = this->capacitySize[0];
        size[1] = this->capacitySize[1];
        size[2] = this->capacitySize[2];
    }

private:
    void set_image_shape(TensorDesc cpuDesc)
    {
        DataType dt;
        U32 n, c, t, h, w;
        CHECK_STATUS(tensorSelectGet(cpuDesc, &dt, NULL, &n, &c, &h, &w, &t));
        if (cpuDesc.nDims > 5) {
            for (U32 i = 4; i < cpuDesc.nDims; i++) {
                n = n * cpuDesc.dims[i];
            }
        }
        if (cpuDesc.df == DF_NCHWC4 || cpuDesc.df == DF_NHWC) {
            this->desc.memFormat = cpuDesc.df;
        } else {
            this->desc.memFormat = DF_NCHW;
        }
        if (this->desc.memFormat == DF_NCHWC4) {
            this->desc.stride[0] = w;
            this->desc.stride[1] = h;
            this->desc.stride[2] = ((c + 3) / 4) * t * n;
        } else if (this->desc.memFormat == DF_NHWC) {
            this->desc.stride[0] = ((c + 3) / 4) * t * n;
            this->desc.stride[1] = w;
            this->desc.stride[2] = h;
        } else {
            this->desc.stride[0] = (w + 3) / 4;
            this->desc.stride[1] = h;
            this->desc.stride[2] = c * t * n;
        }
        for (U32 i = 0; i < 6; i++) {
            this->desc.offset[i] = 0;
        }
        this->desc.num = this->desc.stride[0] * this->desc.stride[1] * this->desc.stride[2] * 4;
        this->desc.byteSize = this->desc.num * bytesOf(this->desc.dt);
        for (U32 i = 0; i < 3; i++) {
            if (this->desc.stride[i] > this->capacitySize[i]) {
                this->allocated = false;
                break;
            }
        }
        this->desc.imgFormat.image_channel_data_type = get_channel_type(dt);
    }
};
#endif
