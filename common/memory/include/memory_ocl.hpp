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
        memset(&(this->desc), 0, sizeof(GCLMemDesc));
        this->desc.memFormat = DF_NCHW;
        this->allocated = false;
        this->mapped = false;
        this->capacitySize = 0;
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
        if (this->desc.byteSize == 0) {
            this->desc.memType = GCL_MEM_BUF;
            this->desc.flags = CL_MEM_READ_WRITE;
        }
        if (tensorNumBytes(desc) > this->capacity()) {
            this->allocated = false;
        }
    }

    void padding(GCLMemDesc desc)
    {
        if (desc.byteSize > this->capacity()) {
            this->allocated = false;
        }
        for (U32 i = 0; i < 3; i++) {
            this->desc.stride[i] = desc.stride[i];
            this->desc.offset[i] = desc.offset[i];
        }
        this->desc.memType = desc.memType;
        this->desc.memFormat = desc.memFormat;
        this->desc.byteSize = desc.byteSize;
        this->desc.num = desc.num;
        this->desc.flags = desc.flags;
        this->desc.imgFormat = desc.imgFormat;
        this->desc.host_ptr = desc.host_ptr;
        this->desc.need_pad = desc.need_pad;
    }

    void alloc() override
    {
        if (this->desc.byteSize == 0) {
            U32 num = (this->desc.nDims == 0) ? 0 : 1;
            for (U32 i = 0; i < this->desc.nDims; i++) {
                num *= this->desc.dims[i];
            }
            this->desc.byteSize = num * bytesOf(this->desc.dt);
        }
        U32 size = this->desc.byteSize;
        if (!this->allocated && size > this->capacity()) {
            GCLMem_t mem = ocl_alloc_gclmem(this->desc);
            this->val = std::shared_ptr<GCLMem>(mem, ocl_release_gclmem);
            this->allocated = true;
            this->capacitySize = size;
        }
    }

    GCLMemDesc get_desc()
    {
        return this->desc;
    }

    EE copy_from(Memory *other) override
    {
        EE ret = SUCCESS;
        if (other->get_mem_type() == CPUMem) {
            U32 size = ((CpuMemory *)other)->bytes();
            void *host_ptr = ((CpuMemory *)other)->get_ptr();
            if (!allocated) {
                U8 *tmp = nullptr;
                if (size < this->desc.byteSize) {
                    U8 *tmp = (U8 *)operator new(this->desc.byteSize);
                    memset(tmp, 0, this->desc.byteSize);
                    memcpy(tmp, host_ptr, size);
                    host_ptr = tmp;
                }
                this->desc.host_ptr = host_ptr;
                this->desc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
                this->alloc();
                if (tmp) {
                    delete tmp;
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
        } else if (other->get_mem_type() == OCLMem) {
            if (!allocated) {
                this->alloc();
            } else {
                GCLMemDesc srcDesc = ((OclMemory *)other)->get_desc();
                GCLMemType srcMt = srcDesc.memType;
                GCLMemType dstMt = this->desc.memType;
                void *srcPtr = ((OclMemory *)other)->get_ptr();
                void *dstPtr = this->val.get();
                if (srcMt == GCL_MEM_BUF && dstMt == GCL_MEM_BUF) {
                    if (srcDesc.byteSize > this->desc.byteSize) {
                        CHECK_STATUS(NOT_MATCH);
                    }
                    U32 size = srcDesc.byteSize;
                    CHECK_STATUS(gcl_trans_memory(OCLContext::getInstance().handle.get(), srcPtr,
                        dstPtr, &size, DEVICE_BUF_TO_BUF, CL_TRUE));
                } else if (srcMt != GCL_MEM_BUF && dstMt == GCL_MEM_BUF) {
                    if (srcDesc.byteSize > this->desc.byteSize) {
                        CHECK_STATUS(NOT_MATCH);
                    }
                    U32 region[3] = {srcDesc.stride[0], srcDesc.stride[1], srcDesc.stride[2]};
                    CHECK_STATUS(gcl_trans_memory(OCLContext::getInstance().handle.get(), srcPtr,
                        dstPtr, region, DEVICE_IMG_TO_BUF, CL_TRUE));
                } else if (srcMt == GCL_MEM_BUF && dstMt != GCL_MEM_BUF) {
                    if (this->desc.byteSize > srcDesc.byteSize) {
                        CHECK_STATUS(NOT_MATCH);
                    }
                    U32 region[3] = {
                        this->desc.stride[0], this->desc.stride[1], this->desc.stride[2]};
                    CHECK_STATUS(gcl_trans_memory(OCLContext::getInstance().handle.get(), srcPtr,
                        dstPtr, region, DEVICE_BUF_TO_IMG, CL_TRUE));
                } else {
                    CHECK_STATUS(NOT_SUPPORTED);
                }
            }
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        return ret;
    }

    void *get_ptr()
    {
        if (allocated) {
            this->val->desc = this->desc;  //TODO DELETE AFTER SPLITE DESC FROM GCLMEM
        }
        return this->val.get();
    }

    void set_shared_ptr(std::shared_ptr<GCLMem> val)
    {
        this->val = val;
        this->allocated = true;
        this->capacitySize = this->bytes();
    }

    std::shared_ptr<GCLMem> get_shared_ptr()
    {
        if (allocated) {
            this->val->desc = this->desc;  //TODO DELETE AFTER SPLITE DESC FROM GCLMEM
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
        return this->val->mapPtrArray.back();
    }

    EE reuse(Memory *other) override
    {
        EE ret;
        if (other->get_mem_type() != OCLMem) {
            ret = this->copy_from(other);
        } else {
            U32 size = other->capacity();
            if (size >= this->bytes()) {
                this->val = ((OclMemory *)other)->get_shared_ptr();
                this->allocated = true;
                this->capacitySize = other->capacity();
                ret = SUCCESS;
            } else {
                UNI_ERROR_LOG("small OCL memory can not meet big OCL memory demand\n");
                ret = NOT_SUPPORTED;
            }
        }
        return ret;
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

    U32 capacity() override
    {
        return this->capacitySize;
    }

    std::string string(U32 num, F32 factor) override
    {
        std::string line = "desc: " + gclMemDesc2Str(this->desc) + "data: \n";
#ifdef _DEBUG
        DataType dt = (this->desc.dt == DT_U8) ? DT_F16 : this->desc.dt;
        if (dt == DT_U32) {
            dt = DT_I32;
        }
        switch (dt) {
            case DT_F16:
                line += gcl_check_data<F16>(
                    OCLContext::getInstance().handle.get(), this->desc, get_ptr(), num, 0, false);
                break;
            case DT_I32:
                line += gcl_check_data<I32>(
                    OCLContext::getInstance().handle.get(), this->desc, get_ptr(), num, 0, false);
                break;
            default:
                UNI_ERROR_LOG("Currently not support to get %d type OCL Memory\n", this->desc.dt);
                break;
        }
#else
        if (mapped) {
            for (U32 i = 0; i < num; i++) {
                line += std::to_string(this->element(i) * factor) + " ";
            }
        }
#endif
        return line;
    }

    F32 element(U32 index) override
    {
        F32 result = 0;
        if (this->mapped) {
            if (desc.dt == DT_F16) {
                F16 *res = (F16 *)this->val->mapPtrArray.back();
                result = res[index];
            } else if (desc.dt == DT_F32) {
                F32 *res = (F32 *)this->val->mapPtrArray.back();
                result = res[index];
            } else if (desc.dt == DT_I32 || desc.dt == DT_U32) {
                I32 *res = (I32 *)this->val->mapPtrArray.back();
                result = res[index];
            } else {
                UNI_ERROR_LOG("Get mapped ptr data type not support\n");
            }
        } else {
            UNI_ERROR_LOG("Currently not support to get element on OCL memory\n");
        }
        return result;
    }

    bool check_mapped()
    {
        return this->mapped;
    }

private:
    GCLMemDesc desc;
    std::shared_ptr<GCLMem> val;
    U32 capacitySize;
    bool allocated;
    bool mapped;
};
#endif
