// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MMM_2X4
#define _H_MMM_2X4

void mmm_2x4(U64 offset, U64 K4, INT8 *A, INT8 *B, I32 *C)
{
    __asm__ __volatile__("ldr q16, [%[A]]\n"
                         "ld1 {v20.16b, v21.16b}, [%[B]]\n"
                         "mov x20, %[A]\n"
                         "mov x21, %[B]\n"
                         "mov x26, %[C]\n"
                         "mov x22, %[K]\n"
                         "movi v0.16b, #0x0\n"
                         "movi v4.16b, #0x0\n"

                         "cmp x22, #1\n"
                         "ble 1f\n"

                         "0:\n"
                         "ldr q24, [x20, 0x10]\n"
                         ".inst 0x4e94a600  // smmla v0.4s, v16.16b, v20.16b\n"
                         "ldr q28, [x21, 0x20]\n"
                         ".inst 0x4e95a604  // smmla v4.4s, v16.16b, v21.16b\n"
                         "ldr q29, [x21, 0x30]\n"
                         "ldr q16, [x20, 0x20]\n"
                         "ldr q20, [x21, 0x40]\n"

                         ".inst 0x4e9ca700  // smmla v0.4s, v24.16b, v28.16b\n"
                         "sub x22, x22, #2\n"
                         ".inst 0x4e9da704  // smmla v4.4s, v24.16b, v29.16b\n"
                         "add x20, x20, 0x20\n"
                         "ldr q21, [x21, 0x50]\n"
                         "add x21, x21, 0x40\n"
                         "cmp x22, #1\n"
                         "bgt 0b\n"
                         "1:\n"
                         "bne 2f\n"
                         ".inst 0x4e94a600  // smmla v0.4s, v16.16b, v20.16b\n"
                         ".inst 0x4e95a604  // smmla v4.4s, v16.16b, v21.16b\n"

                         "2:\n"
                         "ldr q30, [x26]\n"
                         "uzp1 v16.2d, v0.2d, v4.2d\n"
                         "uzp2 v18.2d, v0.2d, v4.2d\n"
                         "add v16.4s, v16.4s, v30.4s\n"
                         "str q16, [x26]\n"
                         "add x26, x26, %[offset]\n"

                         "ldr q30, [x26]\n"
                         "add v18.4s, v18.4s, v30.4s\n"
                         "str q18, [x26]\n"
                         : [A] "+r"(A), [B] "+r"(B), [C] "+r"(C)
                         : [K] "r"(K4), [offset] "r"(offset)
                         : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                         "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
                         "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
                         "v30", "v31", "x19", "x20", "x21", "x22", "x26");
}
#endif