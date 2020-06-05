// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _TENSOR_H
#define _TENSOR_H

#include <memory>
#include <vector>
#include <set>
#include <map>
#include <iostream>
#include "memory.hpp"
#include "cpu/memory_cpu.hpp"
#ifdef _USE_MALI
#include "ocl/memory_ocl.hpp"
#endif
#include <string.h>
#include "type.h"
#include "tensor_desc.h"

#define HashMap std::map
#define Vec std::vector

class Tensor {
public:
    Tensor()
    {
        this->val = std::shared_ptr<Memory_>(new CpuMemory());
        this->scalePtr = std::shared_ptr<F32>((F32*)operator new(bytesOf(DT_F32)));
    }

#ifdef _USE_MALI
    Tensor(std::shared_ptr<GCLHandle> handle)
    {
        this->val = std::shared_ptr<Memory_>(new OclMemory(handle));
        this->scalePtr = std::shared_ptr<F32>((F32*)operator new(bytesOf(DT_F32)));
    }
#endif

    void alloc()
    {
        this->val->alloc(desc);
    }

    void set_desc(TensorDesc d)
    {
        this->desc = d;
    }

    TensorDesc get_desc()
    {
        return this->desc;
    };

    void set_scale(F32 s)
    {
        if (nullptr == this->scalePtr.get()) {
            this->scalePtr = std::shared_ptr<F32>((F32*)operator new(bytesOf(DT_F32)));
        }
        *(this->scalePtr.get()) = s;
    }

    F32 get_scale()
    {
        if (nullptr != this->scalePtr.get()) {
            return *(this->scalePtr.get());
        } else {
            return 1.0;
        }
    }

    void set_val_by_copy(TensorDesc desc, U8* ptr) {
        this->val->set_val_by_copy(desc, ptr);
    }

    void set_shared_ptr(std::shared_ptr<void> val)
    {
        this->val->set_shared_ptr_caster(val);
    }

    PtrCaster get_val() 
    {
        return this->val->get_val_caster();
    }

    PtrCasterShared get_shared_ptr()
    {
        return this->val->get_shared_ptr_caster();
    }

    void set_memory(std::shared_ptr<Memory_> mem)
    {
        this->val = mem;
    }

    Memory_* get_memory()
    {
        return this->val.get();
    }

    bool isInvalid()
    {
        U32 num = tensorNumElements(this->desc);
        for (U32 i = 0; i < num; i++) {
            if (UNI_ISNAN(getElement(i)) || UNI_ISINF(getElement(i))) {
                return true;
            }
        }
        return false;
    }

    void print()
    {
        U32 num = tensorNumElements(this->desc);
        std::cout << num << std::endl;
        num = (num > 64) ? 64 : num;
        for(U32 i = 0; i < num; i++) {
            std::cout << getElement(i) << " ";
        }
        std::cout << std::endl;
    }

    F32 getElement(U32 index)
    {
        F32 value = 0;
        U8* res = NULL;
#ifdef _USE_MALI
        if(val->get_mem_type() == OCLMem) {
            std::shared_ptr<GCLMem> oclMem = this->val->get_shared_ptr_caster();
            if(!oclMem->desc.use_map) {
                std::cerr << "[ERROR] Not support check unmapped gcl memory value" << std::endl;
                exit(1);
            }
            res = oclMem->desc.map_ptr;
        } else {
#endif
            res = (U8*)this->val->get_val();
#ifdef _USE_MALI
        }
#endif
        switch (this->desc.dt) {
            case DT_F32: {
                 F32* data = (F32*)res;
                 value = data[index];
                 break;
            }
#ifdef __aarch64__
            case DT_F16: {
                 F16* data = (F16*)res;
                 value = data[index];
                 break;
            }
#endif
            case DT_U32: {
                 U32* data = (U32*)res;
                 value = data[index];
                 break;
            }
            case DT_I32: {
                 I32* data = (I32*)res;
                 value = data[index];
                 break;
            }
            case DT_I8: {
                 INT8* data = (INT8*)res;
                 value = data[index] / this->get_scale();
                 break;
            }
            case DT_U8: {
                 U8* data = (U8*)res;
                 value = data[index];
                 break;
            }
            default:
                 CHECK_STATUS(NOT_SUPPORTED);
        }
        return value;
    }

private:
    TensorDesc desc;
    std::shared_ptr<Memory_> val;
    std::shared_ptr<F32> scalePtr;
};
#endif //_TENSOR_H
