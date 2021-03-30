// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_SYS
#define _H_SYS

#if defined(_USE_GENERAL) || defined(_USE_NEON) || defined(_USE_X86)
#define _USE_CPU
#endif
#define IS_GENERAL(arch) (arch == CPU_GENERAL)
#define IS_X86_AVX2(arch) (arch == X86_AVX2)
#define IS_ARM_V7(arch) (arch == ARM_V7)
#define IS_ARM_V8(arch) (arch == ARM_V8)
#define IS_ARM_A55(arch) (arch == ARM_A55)
#define IS_ARM_A76(arch) (arch == ARM_A76)
#define IS_ARM_LG_V8(arch) (IS_ARM_A55(arch) || IS_ARM_A76(arch))
#define IS_ARM(arch) (IS_ARM_LG_V8(arch) || IS_ARM_V8(arch) || IS_ARM_V7(arch))
#define IS_CPU(arch) (IS_GENERAL(arch) || IS_X86_AVX2(arch) || IS_ARM(arch))
#define IS_MALI_GPU(arch) (arch == MALI)

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CPU_GENERAL = 1,
    MALI = 2,
    ARM_V7 = 3,
    ARM_V8 = 4,
    ARM_A55 = 5,
    ARM_A76 = 6,
    X86_AVX2 = 7,
} Arch;

inline const char *const *ArchName()
{
    static const char *const names[] = {"UNKNOWN", "SERIAL", "MALI", "ARM_V7", "ARM_V8",
        "ARM_V8.2_LITTLE", "ARM_V8.2_BIG", "X86_AVX2"};
    return names;
}

typedef struct {
    Arch arch;
    void *archPara;
} ArchInfo;
typedef ArchInfo *ArchInfo_t;
#ifdef __cplusplus
}
#endif

#endif
