// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_UT_UTIL_OCL
#define _H_UT_UTIL_OCL

#include "ut_util.h"
#include "gcl.h"
#include "libkernelsource.h"

inline GCLMem_t alloc(Tensor tensor)
{
    auto mem = (OclMemory *)tensor.get_memory();
    mem->alloc();
    return (GCLMem_t)mem->get_ptr();
}

inline GCLMem_t alloc_host_ptr(Tensor tensor, U8 *host_ptr)
{
    auto mem = (OclMemory *)tensor.get_memory();
    mem->alloc(host_ptr);
    return (GCLMem_t)mem->get_ptr();
}

inline GCLMem_t alloc_map(Tensor tensor)
{
    auto mem = (OclMemory *)tensor.get_memory();
    mem->mapped_alloc();
    return (GCLMem_t)mem->get_ptr();
}

inline GCLMem_t alloc_bytes(Tensor tensor, U32 size)
{
    auto mem = (OclMemory *)tensor.get_memory();
    GCLMem_t ptr = NULL;
    if (size > 0) {
        mem->resize(tensor1d(DT_U8, size));
        mem->alloc();
        ptr = (GCLMem_t)mem->get_ptr();
    }
    return ptr;
}

inline GCLMem_t alloc_img(Tensor tensor, U32 *str)
{
    auto mem = (OclMemoryImg *)tensor.get_memory();
    GCLMem_t ptr = NULL;
    if (str[0] > 0 && str[1] > 0 && str[2] > 0) {
        mem->alloc(str[0], str[1], str[2]);
        ptr = (GCLMem_t)mem->get_ptr();
    }
    return ptr;
}

inline GCLMem_t alloc_padding(Tensor tensor, U32 pl, U32 pr, U32 pt, U32 pb, U8 *hostPtr = nullptr)
{
    auto mem = (OclMemory *)tensor.get_memory();
    mem->padding(pl, pr, pt, pb);
    if (hostPtr) {
        mem->alloc(hostPtr);
    } else {
        mem->alloc();
    }
    return (GCLMem_t)mem->get_ptr();
}
#endif
