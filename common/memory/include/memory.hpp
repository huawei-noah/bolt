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

#include <string>
#include "tensor_desc.h"
#include "uni.h"

typedef enum { OCLMem = 0, CPUMem = 1 } MemoryType;

class Memory {
public:
    Memory()
    {}

    virtual ~Memory() = default;

    virtual MemoryType get_mem_type() = 0;

    virtual std::shared_ptr<Memory> clone(bool allocate = true) = 0;

    virtual void resize(TensorDesc desc) = 0;

    virtual void alloc() = 0;

    virtual EE reuse(Memory *other) = 0;

    virtual EE copy_from(Memory *other) = 0;

    virtual EE copy_to(Memory *other)
    {
        return other->copy_from(this);
    }

    virtual U32 length() = 0;
    virtual U32 bytes() = 0;
    virtual U32 capacity() = 0;
    virtual std::string string(U32 num, F32 factor) = 0;
    virtual F32 element(U32 index) = 0;
};
#endif
