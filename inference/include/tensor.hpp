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
#define Set std::set

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

        void set_val(void* v)
        {
            this->val->set_val_caster(v);
        }

        PtrCaster get_val() 
        {
            return this->val->get_val_caster();
        }

        void set_shared_ptr(std::shared_ptr<void> val)
        {
            this->val->set_shared_ptr_caster(val);
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
#ifdef _USE_MALI
            if (val->get_mem_type() == OCLMem) {
                return false;
            } else {
#endif
                U32 num = tensorNumElements(this->desc);
                for (U32 i = 0; i < num; i++) {
                    if (UNI_ISNAN(getElement(i)) || UNI_ISINF(getElement(i))) {
                        return true;
                    }
                }
                return false;
#ifdef _USE_MALI
            }
#endif
        }

        void print()
        {
#ifdef _USE_MALI
            if (val->get_mem_type() == OCLMem) {
            } else {
#endif
                U32 num = tensorNumElements(this->desc);
                std::cout << num << std::endl;
                if(num > 64) num = 64;
                for(U32 i = 0; i < num; i++) {
                    std::cout << getElement(i) << " ";
                }
                std::cout << std::endl;
#ifdef _USE_MALI
            }
#endif
        }

       F32 getElement(U32 index)
       {
           F32 value = 0;
           switch (this->desc.dt) {
               case DT_F32: {
                    F32* data = (F32*) this->val->get_val();
                    value = data[index];
                    break;
               }
#ifdef _USE_FP16
               case DT_F16: {
                    F16* data = (F16*) this->val->get_val();
                    value = data[index];
                    break;
               }
#endif
               case DT_U32: {
                    U32* data = (U32*) this->val->get_val();
                    value = data[index];
                    break;
               }
               case DT_I32: {
                    I32* data = (I32*) this->val->get_val();
                    value = data[index];
                    break;
               }
               default:
                    CHECK_STATUS(NOT_SUPPORTED);
           }
           return value;
       }

    public:
        TensorDesc desc;
        std::shared_ptr<Memory_> val;
        std::shared_ptr<F32> scalePtr;
};
#endif //_TENSOR_H
