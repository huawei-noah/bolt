// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _MEMORY_CPU_H
#define _MEMORY_CPU_H
#include <math.h>
#include <memory>
#include "memory.hpp"

class CpuMemory : public Memory_
{
    public:
    CpuMemory(){type = CPUMem;}   
    virtual ~CpuMemory() = default;

    virtual void alloc(TensorDesc desc) override{
	    U32 size = tensorNumElements(desc);
	    this->val = std::shared_ptr<U8>((U8*)operator new(size*bytesOf(desc.dt)));
    }

    virtual void  set_val(PtrCaster val) override{
//        std::shared_ptr<U8> tmp((U8*)val);
        U8* tmp = (U8*)val;
        this->val.reset(tmp);
    };

    virtual void* get_val() override{
        return this->val.get();
    };
    
    virtual MemoryType get_mem_type() override{
        return type;
    }

    virtual void set_shared_ptr(PtrCasterShared val) override{
        this->val = val;
    }

    virtual std::shared_ptr<void> get_shared_ptr() override{
        return val;
    }

    private:
    std::shared_ptr<U8> val;
    MemoryType type;
};
#endif
