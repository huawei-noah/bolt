// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _MEMORY_OCL_H
#define _MEMORY_OCL_H

#include "memory.hpp"
#include "gcl.h"
#include "ocl_data_alloc.h"
#include "ocl_data_trans.h"

class OclMemory : public Memory {
public:
    OclMemory()
    {
        UNI_MEMSET(&(this->desc), 0, sizeof(GCLMemDesc));
        this->desc.memFormat = DF_NCHW;
        this->desc.memType = GCL_MEM_BUF;
        this->desc.flags = CL_MEM_READ_WRITE;
        this->allocated = false;
        this->mapped = false;
        this->capacitySize[0] = 0;
        this->capacitySize[1] = 0;
        this->capacitySize[2] = 0;
    }

    ~OclMemory() = default;

    MemoryType get_mem_type() override
    {
        return OCLMem;
    }

    std::shared_ptr<Memory> clone(bool allocate) override
    {
        OclMemory *mem = new OclMemory();
        mem->desc = this->desc;
        if (allocate) {
            mem->alloc();
        }
        return std::shared_ptr<Memory>(mem);
    }

    void resize(TensorDesc desc) override
    {
        this->desc.nDims = desc.nDims;
        for (U32 i = 0; i < desc.nDims; i++) {
            this->desc.dims[i] = desc.dims[i];
        }
        this->desc.dt = desc.dt;
        this->desc.df = desc.df;
        set_shape(desc);
    }

    virtual EE padding(U32 pl, U32 pr, U32 pt, U32 pb, U32 pf, U32 pa)
    {
        if (this->desc.byteSize == 0) {
            return NOT_MATCH;
        }
        if (pl == 0 && pr == 0 && pt == 0 && pb == 0 && pf == 0 && pa == 0) {
            return SUCCESS;
        }
        U32 w = this->desc.dims[0];
        U32 h = (this->desc.nDims < 2) ? 1 : this->desc.dims[1];
        U32 num;
        DataFormat mf = this->desc.memFormat;
        if (pf > 0 || pa > 0) {
            U32 c = 1;
            if (mf == DF_NCHWC4) {
                if (this->desc.nDims == 5) {
                    c = (this->desc.dims[3] + 3) / 4 * this->desc.dims[2] * this->desc.dims[4];
                } else if (this->desc.nDims == 4) {
                    c = (this->desc.dims[2] + 3) / 4 * this->desc.dims[3];
                }
            } else {
                for (U32 i = 2; i < this->desc.nDims; i++) {
                    c *= this->desc.dims[i];
                }
            }
            if (pf > this->desc.offset[2]) {
                this->desc.offset[2] = pf;
            } else {
                pf = this->desc.offset[2];
            }
            c += pf + pa;
            if (c > this->desc.stride[2]) {
                this->desc.stride[2] = c;
                this->desc.need_pad = true;
            }
        }

        if (pl > this->desc.offset[0]) {
            this->desc.offset[0] = pl;
        } else {
            pl = this->desc.offset[0];
        }
        if (pt > this->desc.offset[1]) {
            this->desc.offset[1] = pt;
        } else {
            pt = this->desc.offset[1];
        }
        w += pl + pr;
        h += pt + pb;
        if (w > this->desc.stride[0]) {
            this->desc.stride[0] = w;
        }
        if (h > this->desc.stride[1]) {
            this->desc.stride[1] = h;
        }

        if (mf == DF_NCHWC4) {
            num = this->desc.stride[0] * this->desc.stride[1] * this->desc.stride[2] * 4;
        } else if (mf == DF_NCHW) {
            num = this->desc.stride[0] * this->desc.stride[1] * this->desc.stride[2];
        } else {
            return NOT_SUPPORTED;
        }
        if (num > this->desc.num) {
            this->desc.num = num;
            this->desc.byteSize = num * bytesOf(this->desc.dt);
        }
        if (w != this->desc.dims[0] || h != this->desc.dims[1]) {
            this->desc.need_pad = true;
        }
        if (this->desc.byteSize > this->capacitySize[0]) {
            this->allocated = false;
        }
        return SUCCESS;
    }

    EE padding(U32 pl, U32 pr, U32 pt, U32 pb)
    {
        return padding(pl, pr, pt, pb, 0, 0);
    }

    void alloc() override
    {
        U32 size = this->desc.byteSize;
        if (!this->allocated && size > this->capacitySize[0]) {
            GCLMem_t mem = ocl_alloc_gclmem(this->desc);
            this->val = std::shared_ptr<GCLMem>(mem, ocl_release_gclmem);
            this->allocated = true;
            this->capacitySize[0] = size;
        }
    }

    void alloc(U8 *cpuPtr)
    {
        this->desc.host_ptr = cpuPtr;
        this->desc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
        this->alloc();
    }

    TensorDesc get_dims() override  //replace get_desc after delete GCLMemDesc
    {
        TensorDesc desc;
        desc.nDims = this->desc.nDims;
        desc.df = this->desc.df;
        desc.dt = this->desc.dt;
        for (U32 i = 0; i < desc.nDims; i++) {
            desc.dims[i] = this->desc.dims[i];
        }
        return desc;
    }

    GCLMemDesc get_desc()
    {
        return this->desc;
    }

    void set_desc(GCLMemDesc _desc)
    {
        this->desc = _desc;
        if (this->val) {
            this->val->desc = _desc;
        }
    }

    void stride(U32 *stride)
    {
        stride[0] = this->desc.stride[0];
        stride[1] = this->desc.stride[1];
        stride[2] = this->desc.stride[2];
    }

    void offset(U32 *offset)
    {
        offset[0] = this->desc.offset[0];
        offset[1] = this->desc.offset[1];
        offset[2] = this->desc.offset[2];
    }

    EE copy_from(Memory *other) override
    {
        EE ret = SUCCESS;
        MemoryType srcType = other->get_mem_type();
        if (srcType == CPUMem) {
            U32 size = ((CpuMemory *)other)->bytes();
            U8 *host_ptr = (U8 *)((CpuMemory *)other)->get_ptr();
            if (!allocated) {
                U8 *tmp = nullptr;
                if (size < this->desc.byteSize) {
                    tmp = (U8 *)UNI_OPERATOR_NEW(this->desc.byteSize);
                    UNI_MEMCPY(tmp, host_ptr, size);
                    UNI_MEMSET(tmp + size, 0, this->desc.byteSize - size);
                    host_ptr = tmp;
                }
                this->alloc(host_ptr);
                if (tmp != nullptr) {
                    UNI_OPERATOR_DELETE(tmp);
                }
            } else {
                this->val->desc = this->desc;  //TODO DELETE AFTER SPLITE DESC FROM GCLMEM
                TensorDesc hostDesc = ((CpuMemory *)other)->get_desc();
                if (this->mapped) {
                    CHECK_STATUS(ocl_map_mem_write(OCLContext::getInstance().handle.get(),
                        this->val.get(), this->desc, hostDesc, (U8 *)host_ptr));
                } else {
                    if (size > this->desc.byteSize) {
                        size = this->desc.byteSize;
                    }
                    CHECK_STATUS(gcl_trans_memory(OCLContext::getInstance().handle.get(), host_ptr,
                        this->val.get(), &size, HOST_TO_DEVICE_BUF, CL_TRUE));
                }
            }
        } else {
            if (!allocated) {
                this->alloc();
            }
            void *srcPtr = ((OclMemory *)other)->get_ptr();
            void *dstPtr = this->val.get();
            U32 srcBytes = ((OclMemory *)other)->bytes();
            if (srcBytes > this->bytes()) {
                CHECK_STATUS(NOT_MATCH);
            }
            if (srcType == OCLMem) {
                CHECK_STATUS(gcl_trans_memory(OCLContext::getInstance().handle.get(), srcPtr,
                    dstPtr, &srcBytes, DEVICE_BUF_TO_BUF, CL_TRUE));
            } else if (srcType == OCLMemImg || srcType == OCLMemImg1D || srcType == OCLMemImg2D) {
                GCLMemDesc srcDesc = ((OclMemory *)other)->get_desc();
                U32 region[3] = {srcDesc.stride[0], srcDesc.stride[1], srcDesc.stride[2]};
                CHECK_STATUS(gcl_trans_memory(OCLContext::getInstance().handle.get(), srcPtr,
                    dstPtr, region, DEVICE_IMG_TO_BUF, CL_TRUE));
            } else {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        return ret;
    }

    void *get_ptr()
    {
        if (this->capacitySize[0]) {
            this->val->desc = this->desc;  //TODO DELETE AFTER SPLITE DESC FROM GCLMEM
        } else {
            UNI_DETAIL_LOG("Get memory val without allocated, the capacitySize is %d\n",
                this->capacitySize[0]);
        }
        return this->val.get();
    }

    void set_shared_ptr(std::shared_ptr<GCLMem> val)
    {
        this->val = val;
        this->allocated = true;
        this->capacitySize[0] = this->bytes();
    }

    std::shared_ptr<GCLMem> get_shared_ptr()
    {
        if (this->capacitySize[0]) {
            this->val->desc = this->desc;  //TODO DELETE AFTER SPLITE DESC FROM GCLMEM
        } else {
            UNI_DETAIL_LOG("Get memory val without allocated, the capacitySize is %d\n",
                this->capacitySize[0]);
        }
        return this->val;
    }

    void mapped_alloc()
    {
        if (!this->mapped) {
            this->desc.flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
            this->desc.byteSize *= 2;
            this->allocated = this->mapped;
            this->mapped = true;
            this->alloc();
        }
    }

    void *get_mapped_ptr()
    {
        if (!mapped) {
            CHECK_STATUS(NOT_MATCH);
        }
        ocl_map_mem_read(OCLContext::getInstance().handle.get(), this->val.get(), this->desc);
        if (desc.df == DF_NCHWC4) {
            desc.df = DF_NCHW;
        }
        return this->val->mapPtrArray.back();
    }

    EE reuse(Memory *other) override
    {
        MemoryType type = other->get_mem_type();
        if (type == CPUMem) {
            CHECK_STATUS(this->copy_from(other));
        } else if (type == OCLMem) {
            U32 size;
            other->capacity(&size);
            if (size >= this->bytes()) {
                this->val = ((OclMemory *)other)->get_shared_ptr();
                this->allocated = true;
                other->capacity(this->capacitySize);
            } else {
                UNI_ERROR_LOG("small OCL memory can not meet big OCL memory demand\n");
                CHECK_STATUS(NOT_SUPPORTED);
            }
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        return SUCCESS;
    }

    U32 length() override
    {
        U32 num = 1;
        for (U32 i = 0; i < this->desc.nDims; i++) {
            num *= this->desc.dims[i];
        }
        return num;
    }

    U32 bytes() override
    {
        return this->desc.byteSize;
    }

    void capacity(U32 *size) override
    {
        *size = this->capacitySize[0];
    }

    std::string string(U32 num, F32 factor) override
    {
        std::string line = "desc:" + gclMemDesc2Str(this->desc);
        if (num > 0) {
            line += " data:";
#ifdef _DEBUG
            line +=
                gcl_string(OCLContext::getInstance().handle.get(), this->desc, get_ptr(), num, 0);
#else
            if (mapped) {
                for (U32 i = 0; i < num; i++) {
                    line += std::to_string(this->element(i) / factor) + " ";
                }
                double sum = 0;
                for (U32 i = 0; i < this->length(); i++) {
                    sum += this->element(i) / factor;
                }
                line += " sum:" + std::to_string(sum);
            }
#endif
        }
        return line;
    }

    F32 element(U32 index) override
    {
        F32 result = 0;
        if (this->mapped) {
            U8 *ptr = this->val->mapPtrArray.back();
            transformToFloat(desc.dt, ptr + index * bytesOf(desc.dt), &result, 1);
        } else {
            UNI_ERROR_LOG("Currently not support to get element on OCL memory\n");
        }
        return result;
    }

    void set_mapped(bool _mapped)
    {
        this->mapped = _mapped;
    }

    bool get_mapped()
    {
        return this->mapped;
    }

    GCLMemType gclMemType()
    {
        return this->desc.memType;
    }

private:
    void set_shape(TensorDesc cpuDesc)
    {
        U32 n, c, t, h, w;
        tensorSelectGet(cpuDesc, NULL, NULL, &n, &c, &h, &w, &t);
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
            this->desc.num = this->desc.stride[0] * this->desc.stride[1] * this->desc.stride[2] * 4;
        } else if (this->desc.memFormat == DF_NHWC) {
            this->desc.stride[0] = c * t * n;
            this->desc.stride[1] = w;
            this->desc.stride[2] = h;
            this->desc.num = this->desc.stride[0] * this->desc.stride[1] * this->desc.stride[2];
        } else {
            this->desc.stride[0] = w;
            this->desc.stride[1] = h;
            this->desc.stride[2] = c * t * n;
            this->desc.num = this->desc.stride[0] * this->desc.stride[1] * this->desc.stride[2];
        }
        for (U32 i = 0; i < 3; i++) {
            this->desc.offset[i] = 0;
        }
        this->desc.byteSize = this->desc.num * bytesOf(this->desc.dt);
        if (this->desc.byteSize > this->capacitySize[0]) {
            this->allocated = false;
        }
    }

protected:
    GCLMemDesc desc;
    std::shared_ptr<GCLMem> val;
    U32 capacitySize[3];
    bool allocated;
    bool mapped;
};
#endif
