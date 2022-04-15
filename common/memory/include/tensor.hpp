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

#include "memory_cpu.hpp"
#ifdef _USE_GPU
#include "memory_ocl.hpp"
#include "memory_ocl_img.hpp"
#endif

class Tensor {
public:
    Tensor(MemoryType memoryType = CPUMem)
    {
        if (memoryType == CPUMem) {
            this->val = std::shared_ptr<Memory>(new CpuMemory());
        } else {
#ifdef _USE_GPU
            if (memoryType == OCLMem) {
                this->val = std::shared_ptr<Memory>(new OclMemory());
            } else if (memoryType == OCLMemImg) {
                this->val = std::shared_ptr<Memory>(new OclMemoryImg());
            } else if (memoryType == OCLMemImg2D) {
                this->val = std::shared_ptr<Memory>(new OclMemoryImg(OCLMemImg2D));
            } else if (memoryType == OCLMemImg1D) {
                this->val = std::shared_ptr<Memory>(new OclMemoryImg(OCLMemImg1D));
            } else {
                CHECK_STATUS(NOT_SUPPORTED);
                UNI_ERROR_LOG("Unsupported GPU memory type\n");
            }
#else
            UNI_ERROR_LOG("not support to create GPU Tensor\n");
#endif
        }
        this->scale = std::shared_ptr<F32>(new F32(0.0));
    }

    Tensor clone(bool allocate = true)
    {
        Tensor tensor = *this;
        tensor.val = this->val->clone(allocate);
        tensor.scale = std::shared_ptr<F32>(new F32(tensor.get_scale()));
        return tensor;
    }

    void resize(TensorDesc desc)
    {
        this->val->resize(desc);
    }

    void alloc()
    {
        this->val->alloc();
    }

    template <MemoryType type>
    static Tensor alloc_sized(TensorDesc desc)
    {
        Tensor tensor(type);
        tensor.resize(desc);
        tensor.alloc();
        return tensor;
    }

    TensorDesc get_desc()
    {
        return this->val->get_dims();
    }

    void set_scale(F32 scale)
    {
        *(this->scale) = scale;
    }

    void set_scale_ptr(std::shared_ptr<F32> scale)
    {
        this->scale = scale;
    }

    F32 get_scale()
    {
        return *(this->scale);
    }

    void reuse(Tensor *other)
    {
        this->val->reuse(other->val.get());
    }

    void copy_from(Tensor *other)
    {
        UNI_MEMCPY(this->scale.get(), other->scale.get(), sizeof(F32));
        this->val->copy_from(other->val.get());
    }

    void copy_to(Tensor *other)
    {
        other->copy_from(this);
    }

    Memory *get_memory()
    {
        return this->val.get();
    }

    std::shared_ptr<Memory> get_shared_memory()
    {
        return this->val;
    }

    void set_shared_memory(std::shared_ptr<Memory> mem)
    {
        this->val = mem;
    }

    MemoryType get_mem_type()
    {
        return this->val->get_mem_type();
    }

    U32 length()
    {
        return this->val->length();
    }

    U32 bytes()
    {
        return this->val->bytes();
    }

    void capacity(U32 *size)
    {
        this->val->capacity(size);
    }

    std::string string(int length = -1)
    {
        int num = tensorNumElements(this->val->get_dims());
        if (length >= 0 && length < num) {
            num = length;
        }
        F32 factor = this->get_scale();
        factor = (factor <= 0) ? 1 : factor;
        std::string line = this->val->string(num, factor);
        return line;
    }

    F32 element(U32 index)
    {
        F32 factor = this->get_scale();
        factor = (factor <= 0) ? 1 : factor;
        return this->val->element(index) / factor;
    }

private:
    std::shared_ptr<Memory> val;
    std::shared_ptr<F32> scale;
};
#endif  // _TENSOR_H
