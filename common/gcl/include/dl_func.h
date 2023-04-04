// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_DL_FUNC
#define _H_DL_FUNC

#ifndef _WIN32

#include <dlfcn.h>

#else

#include <windows.h>
#include "secure_c_wrapper.h"

#define RTLD_GLOBAL 0x100 /* do not hide entries in this module */
#define RTLD_LOCAL 0x000  /* hide entries in this module */
#define RTLD_LAZY 0x000   /* accept unresolved externs */
#define RTLD_NOW 0x001    /* abort if module has unresolved externs */

static struct {
    long error;
    const char *func;
} var = {0, NULL};

inline void *dlopen(const char *filename, int flags)
{
    HINSTANCE p = LoadLibrary(filename);
    if (p == NULL) {
        var.error = GetLastError();
        var.func = "dlopen";
    }
    return p;
}

inline int dlclose(void *handle)
{
    int rc = 0;
    BOOL ok = FreeLibrary((HINSTANCE)handle);
    if (!ok) {
        var.error = GetLastError();
        var.func = "dlclose";
        rc = -1;
    }
    return rc;
}

inline void *dlsym(void *handle, const char *name)
{
    FARPROC fp = GetProcAddress((HINSTANCE)handle, name);
    if (!fp) {
        var.error = GetLastError();
        var.func = "dlsym";
    }
    return (void *)(intptr_t)fp;
}

inline const char *dlerror(void)
{
    static char ret[128];
    if (var.error) {
        UNI_SNPRINTF(ret, sizeof(ret), "%s error #%ld", var.func, var.error);
        return ret;
    } else {
        return NULL;
    }
}
#endif

#endif
