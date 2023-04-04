// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_SECURE_C_WRAPPER
#define _H_SECURE_C_WRAPPER
#ifdef _USE_SECURE_C
#include <securec.h>
#else
#include <string.h>
#endif
#include <string>

#include "error.h"

inline void UNI_MEMCPY(void *dst, const void *src, size_t size)
{
    if (src == dst || size == 0) {
        return;
    }
    if (dst == NULL || src == NULL) {
        printf("cpu memcpy error dst:%p src:%p bytes:%d.\n", dst, src, (int)size);
    }
    //UNI_DEBUG_LOG("cpu memcpy dst:%p src:%p bytes:%d.\n", dst, src, (int)size);
#ifdef _USE_SECURE_C
    memcpy_s(dst, size, src, size);
#else
    memcpy(dst, src, size);
#endif
}

inline void UNI_MEMSET(void *dst, int c, size_t size)
{
#ifdef _USE_SECURE_C
    memset_s(dst, size, c, size);
#else
    memset(dst, c, size);
#endif
}

inline void UNI_STRCPY(char *dst, const char *src)
{
#ifdef _USE_SECURE_C
    strcpy_s(dst, strlen(src) + 1, src);
#else
    strcpy(dst, src);
#endif
}

#ifdef _USE_SECURE_C
#define UNI_FSCANF fscanf_s
#define UNI_SSCANF sscanf_s
#define UNI_SNPRINTF snprintf_truncated_s
#else
#define UNI_FSCANF fscanf
#define UNI_SSCANF sscanf
#define UNI_SNPRINTF snprintf
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
#endif
