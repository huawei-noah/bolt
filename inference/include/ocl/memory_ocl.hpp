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
#include "tensor_computing.h"

class OclMemory : public Memory_
{
    public:
    OclMemory(std::shared_ptr<GCLHandle> handle){
        this->handle = handle;
        GCLMem_t mem = gcl_create_gclmem();
        std::shared_ptr<GCLMem> tmp(mem);
        val = tmp;
        type = OCLMem;
    };    

    virtual ~OclMemory() override {
        if(val->desc.use_map) CHECK_STATUS(gcl_unmap_memory(handle.get(), val.get()));
        CHECK_STATUS(gcl_release_memory(val.get()));
    };

    virtual void alloc(TensorDesc desc) override {
        UNUSED(desc);
        if(val->desc.byteSize) {
            CHECK_STATUS(gcl_create_memory(handle.get(), val.get()));
            if(val->desc.host_ptr == NULL) CHECK_STATUS(gcl_fill_memory_zero(handle.get(), val.get()));
        }
    }

    virtual void set_val(PtrCaster val) override {
        std::shared_ptr<GCLMem> tmp((GCLMem_t)val);
        this->val = tmp;
    };

    virtual void* get_val() override {
        return this->val.get();
    };

    void set_val_from_hostptr(TensorDesc hostDesc, U8* hostPtr, bool blocking){
        ExtInfo extInfo;
        extInfo.maliInfo.handle = handle.get();
        CHECK_STATUS(tensor_computing_set_input((void*)val.get(), hostDesc, (const void*)hostPtr, (void*)tmpBuf, blocking, MALI, &extInfo));
        
    }

    void get_val_to_hostptr(TensorDesc hostDesc, U8** hostPtr, bool blocking){
        ExtInfo extInfo;
        extInfo.maliInfo.handle = handle.get();
        CHECK_STATUS(tensor_computing_get_output((const void*)val.get(), hostDesc, (void**)hostPtr, (void*)tmpBuf, blocking, MALI, &extInfo));
    }

    void set_tmpBuf(GCLMem_t tmpBuf){
        this->tmpBuf = tmpBuf;
    }

    void set_mem_desc(GCLMemDesc desc){val.get()->desc = desc;}

    GCLMemDesc get_mem_desc(){return val.get()->desc;}

    MemoryType get_mem_type() override {
        return type;
    }
   
    virtual void set_shared_ptr(PtrCasterShared val) override {
        this->val = val;
    }

    virtual std::shared_ptr<void> get_shared_ptr() override {
        return val;
    }

    private:
    std::shared_ptr<GCLHandle> handle;
    std::shared_ptr<GCLMem> val;
    GCLMem_t    tmpBuf;
    MemoryType  type;
};
#endif
