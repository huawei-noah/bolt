// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _MEMORY_H
#define _MEMORY_H

#include <type.h>
#include <tensor_desc.h>
#include "point_cast.hpp"

typedef enum{
    OCLMem = 0,
    CPUMem = 1
} MemoryType;

class Memory_ {
    public:
    Memory_(){}
    virtual ~Memory_(){}
    virtual void  alloc(TensorDesc desc) = 0;
    virtual void  set_val(PtrCaster val) = 0;
    virtual void* get_val() = 0;
    inline  void  set_val_caster(void* val){set_val(PtrCaster(val));}
    inline PtrCaster get_val_caster(){return PtrCaster(this->get_val());}
    virtual MemoryType get_mem_type() = 0;

    virtual void  set_shared_ptr(PtrCasterShared val) = 0;
    virtual std::shared_ptr<void> get_shared_ptr() = 0;
    inline  void  set_shared_ptr_caster(std::shared_ptr<void> val) {set_shared_ptr(PtrCasterShared(val));}
    inline PtrCasterShared get_shared_ptr_caster() {return PtrCasterShared(this->get_shared_ptr());}
};
#endif
