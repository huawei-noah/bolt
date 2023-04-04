// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_UNI_MEMORY_CPU
#define _H_UNI_MEMORY_CPU

#include "secure_c_wrapper.h"
#include <string>
#ifdef _USE_MEM_CHECK
#include <map>
extern std::map<std::string, size_t> mem_statistics;
#endif

inline void *UNI_MALLOC(size_t size)
{
    void *p = nullptr;
    if (size > 0) {
        p = malloc(size);
        if (p == nullptr) {
            UNI_ERROR_LOG("cpu malloc ptr:%p bytes:%zu error.\n", p, size);
        }
#ifdef _USE_MEM_CHECK
        UNI_DEBUG_LOG("cpu malloc ptr:%p bytes:%zu.\n", p, size);
        std::string key = ptr2Str(p) + std::string("(alloc by malloc)");
        mem_statistics[key] = size;
#endif
    }
    return p;
}

inline void UNI_FREE(void *p)
{
    if (p == nullptr) {
        return;
    }
#ifdef _USE_MEM_CHECK
    size_t size = 0;
    std::string key = ptr2Str(p) + std::string("(alloc by malloc)");
    if (mem_statistics.find(key) == mem_statistics.end()) {
        UNI_ERROR_LOG("try to free unalloc ptr:%p.\n", p);
    } else {
        size = mem_statistics[key];
        mem_statistics.erase(key);
    }
    UNI_DEBUG_LOG("cpu free ptr:%p bytes:%zu.\n", p, size);
#endif
    free(p);
}

inline void *UNI_OPERATOR_NEW(size_t size)
{
    void *p = nullptr;
    if (size > 0) {
        p = operator new(size);
        if (p == nullptr) {
            UNI_ERROR_LOG("cpu operator new ptr:%p bytes:%zu error.\n", p, size);
        }
#ifdef _USE_MEM_CHECK
        UNI_DEBUG_LOG("cpu operator new ptr:%p bytes:%zu.\n", p, size);
        std::string key = ptr2Str(p) + std::string("(alloc by operator new)");
        mem_statistics[key] = size;
#endif
    }
    return p;
}

inline void UNI_OPERATOR_DELETE(void *p)
{
    if (p == nullptr) {
        return;
    }
#ifdef _USE_MEM_CHECK
    size_t size = 0;
    std::string key = ptr2Str(p) + std::string("(alloc by operator new)");
    if (mem_statistics.find(key) == mem_statistics.end()) {
        UNI_ERROR_LOG("try to operator delete unalloc ptr:%p.\n", p);
    } else {
        size = mem_statistics[key];
        mem_statistics.erase(key);
    }
    UNI_DEBUG_LOG("cpu operator delete ptr:%p bytes:%zu.\n", p, size);
#endif
    operator delete(p);
}

inline void *UNI_ALIGNED_MALLOC(size_t alignment, size_t size)
{
    void *p = nullptr;
    if (size > 0) {
        size_t bytes = size + sizeof(void *) + alignment - 1;
        void *pp = (void *)malloc(bytes);
        if (pp == nullptr) {
            UNI_ERROR_LOG(
                "cpu aligned_alloc ptr:%p alignment:%zu bytes:%zu error.\n", pp, alignment, size);
        }
        void **ppp =
            (void **)(((uintptr_t)(pp) + sizeof(void *) + alignment - 1) & ~(alignment - 1));
        ppp[-1] = pp;
        p = ppp;
#ifdef _USE_MEM_CHECK
        UNI_DEBUG_LOG("cpu aligned_alloc ptr:%p alignment:%zu bytes:%zu.\n", p, alignment, size);
        std::string key = ptr2Str(p) + std::string("(alloc by aligned_alloc)");
        mem_statistics[key] = size;
#endif
    }
    return p;
}

inline void UNI_ALIGNED_FREE(void *p)
{
    if (p == nullptr) {
        return;
    }
#ifdef _USE_MEM_CHECK
    size_t size = 0;
    std::string key = ptr2Str(p) + std::string("(alloc by aligned_alloc)");
    if (mem_statistics.find(key) == mem_statistics.end()) {
        UNI_ERROR_LOG("try to aligned_free unalloc ptr:%p.\n", p);
    } else {
        size = mem_statistics[key];
        mem_statistics.erase(key);
    }
    UNI_DEBUG_LOG("cpu aligned_free ptr:%p bytes:%zu.\n", p, size);
#endif
    free(((void **)p)[-1]);
}

inline size_t UNI_MEM_SIZE()
{
    size_t size = 0;
#ifdef _USE_MEM_CHECK
    for (auto iter : mem_statistics) {
        size += iter.second;
    }
#endif
    return size;
}

inline void UNI_MEM_STATISTICS()
{
#ifdef _USE_MEM_CHECK
    for (auto iter : mem_statistics) {
        UNI_ERROR_LOG("ptr:%s bytes:%zu is not free.\n", iter.first.c_str(), iter.second);
    }
#endif
}
#endif
