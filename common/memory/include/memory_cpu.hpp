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

#include "memory.hpp"

class CpuMemory : public Memory {
public:
    CpuMemory()
    {
        this->capacitySize = 0;
        this->allocated = false;
    }

    ~CpuMemory()
    {}

    std::shared_ptr<Memory> clone(bool allocate) override
    {
        CpuMemory *mem = new CpuMemory();
        mem->desc = this->desc;
        if (allocate) {
            mem->alloc();
        }
        return std::shared_ptr<Memory>(mem);
    }

    MemoryType get_mem_type() override
    {
        return CPUMem;
    }

    void resize(TensorDesc desc) override
    {
        this->desc = desc;
        if (tensorNumBytes(desc) > this->capacitySize) {
            this->allocated = false;
        }
    }

    void alloc() override
    {
        U32 size = this->bytes();
        if (!this->allocated && size > this->capacitySize) {
            this->capacitySize = size;
#ifndef _USE_X86
            this->val = std::shared_ptr<U8>((U8 *)UNI_OPERATOR_NEW(size), UNI_OPERATOR_DELETE);
#else
            this->val = std::shared_ptr<U8>((U8 *)UNI_ALIGNED_MALLOC(64, size), UNI_ALIGNED_FREE);
#endif
        }
        this->allocated = true;
    }

    TensorDesc get_desc()
    {
        return this->desc;
    }

    TensorDesc get_dims() override
    {
        return this->desc;
    }

    void set_ptr(U8 *val)
    {
        this->set_shared_ptr(std::shared_ptr<U8>(val));
    }

    void *get_ptr()
    {
        return this->val.get();
    }

    void set_shared_ptr(std::shared_ptr<U8> val)
    {
        if (val != this->val) {
            this->val = val;
            this->allocated = true;
            this->capacitySize = this->bytes();
        }
    }

    std::shared_ptr<U8> get_shared_ptr()
    {
        return this->val;
    }

    U32 length() override
    {
        return tensorNumElements(this->desc);
    }

    U32 bytes() override
    {
        return tensorNumBytes(this->desc);
    }

    void capacity(U32 *size) override
    {
        *size = this->capacitySize;
    }

    EE reuse(Memory *other) override
    {
        EE ret;
        if (other->get_mem_type() != CPUMem) {
            ret = this->copy_from(other);
        } else {
            U32 other_size;
            other->capacity(&other_size);
            if (other_size >= this->bytes()) {
                this->set_shared_ptr(((CpuMemory *)other)->get_shared_ptr());
                other->capacity(&this->capacitySize);
                ret = SUCCESS;
            } else {
                UNI_ERROR_LOG("Small CPU memory can not meet big CPU memory demand\n");
                ret = NOT_SUPPORTED;
            }
        }
        return ret;
    }

    EE copy_from(Memory *other) override
    {
        if (!this->allocated) {
            this->alloc();
        }
        EE ret = SUCCESS;
        if (CPUMem == other->get_mem_type()) {
            auto *src = ((CpuMemory *)other)->val.get();
            auto *dst = this->val.get();
            auto dst_size = this->bytes();
            auto src_size = other->bytes();
            U32 min_size = UNI_MIN(src_size, dst_size);
            U32 max_size = UNI_MAX(src_size, dst_size);
            if (min_size <= 0) {
                min_size = max_size;
            }
            UNI_MEMCPY(dst, src, min_size);
        } else {
            //todo
            ret = NOT_SUPPORTED;
        }
        return ret;
    }

    std::string string(U32 num, F32 factor) override
    {
        std::string line = "desc:" + tensorDesc2Str(this->desc);
        if (num > 0) {
            line += " data:";
            U32 capacityNum = this->capacitySize / bytesOf(this->desc.dt);
            U32 length = UNI_MIN(tensorNumElements(this->desc), capacityNum);
            std::vector<F32> tmp(length);
            transformToFloat(this->desc.dt, this->get_ptr(), tmp.data(), length, factor);
            F32 sum = 0;
            for (U32 i = 0; i < length; i++) {
                sum += tmp[i];
                if (i < num) {
                    line += std::to_string(tmp[i]) + " ";
                }
            }
            line += " sum:" + std::to_string(sum);
        }
#ifdef _USE_MEM_CHECK
        line += " ptr:" + ptr2Str(this->val.get());
#endif
        return line;
    }

    F32 element(U32 index) override
    {
        U8 *res = (U8 *)this->get_ptr();
        U32 offset = bytesOf(this->desc.dt) * index;
        F32 value;
        transformToFloat(this->desc.dt, res + offset, &value, 1);
        return value;
    }

private:
    // array val's really bytes
    U32 capacitySize;
    std::shared_ptr<U8> val;

    TensorDesc desc;

    bool allocated;
};
#endif
