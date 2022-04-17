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
extern std::map<std::string, unsigned int> mem_statistics;
#endif

inline std::string ptr2Str(const void *p)
{
    char b[64];
#ifdef _USE_SECURE_C
    sprintf_s(b, 64, "%p", p);
#else
    sprintf(b, "%p", p);
#endif
    return std::string(b);
}

inline void *UNI_MALLOC(unsigned int size)
{
    void *p = nullptr;
    if (size > 0) {
        p = malloc(size);
        if (p == nullptr) {
            UNI_ERROR_LOG("cpu malloc ptr:%p bytes:%u error.\n", p, size);
        }
#ifdef _USE_MEM_CHECK
        UNI_DEBUG_LOG("cpu malloc ptr:%p bytes:%u.\n", p, size);
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
    UNI_DEBUG_LOG("cpu free ptr:%p.\n", p);
    std::string key = ptr2Str(p) + std::string("(alloc by malloc)");
    if (mem_statistics.find(key) == mem_statistics.end()) {
        UNI_ERROR_LOG("try to free unalloc ptr:%p.\n", p);
    } else {
        mem_statistics.erase(key);
    }
#endif
    free(p);
}

inline void *UNI_OPERATOR_NEW(unsigned int size)
{
    void *p = nullptr;
    if (size > 0) {
        try {
            p = operator new(size);
        } catch (const std::bad_alloc &e) {
            UNI_ERROR_LOG("cpu operator new ptr:%p bytes:%u error.\n", p, size);
        }
#ifdef _USE_MEM_CHECK
        UNI_DEBUG_LOG("cpu operator new ptr:%p bytes:%u.\n", p, size);
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
    UNI_DEBUG_LOG("cpu operator delete ptr:%p.\n", p);
    std::string key = ptr2Str(p) + std::string("(alloc by operator new)");
    if (mem_statistics.find(key) == mem_statistics.end()) {
        UNI_ERROR_LOG("try to operator delete unalloc ptr:%p.\n", p);
    } else {
        mem_statistics.erase(key);
    }
#endif
    operator delete(p);
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
        UNI_ERROR_LOG("ptr:%s bytes:%u is not free.\n", iter.first.c_str(), iter.second);
    }
#endif
}
#endif
