// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MMM_2X2
#define _H_MMM_2X2

template <typename T = F16>
void mmm_2x2(U64 offset, U64 K4, F16 *A, F16 *B, F16 *C)
{
    __asm__ __volatile__("mov x20, %[A]\n"
                         "mov x21, %[B]\n"
                         "mov x22, %[K]\n"
                         "movi v0.16b, #0x0\n"

                         "cmp x22, #1\n"
                         "blt 1f\n"

                         "0:\n"
                         "ldr q16, [x20]\n"
                         "ldr q20, [x21]\n"
                         "add x20, x20, 0x10\n"
                         "add x21, x21, 0x10\n"
                         ".inst 0x6e54ee00  // bfmmla v0.4s, v16.8h, v20.8h\n"
                         "subs x22, x22, #1\n"
                         "bne 0b\n"
                         "1:\n"
                         : [A] "+r"(A), [B] "+r"(B), [C] "+r"(C)
                         : [K] "r"(K4), [offset] "r"(offset)
                         : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                         "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
                         "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
                         "v30", "v31", "x19", "x20", "x21", "x22", "x26");

    if (sizeof(T) == 4) {
        __asm__ __volatile__("mov x26, %[C]\n"
                             "ldr d30, [x26]\n"
                             "uzp1 v16.2d, v0.2d, v4.2d\n"
                             "uzp2 v18.2d, v0.2d, v4.2d\n"
                             "fadd v16.2s, v16.2s, v30.2s\n"
                             "str d16, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr d30, [x26]\n"
                             "fadd v18.2s, v18.2s, v30.2s\n"
                             "str d18, [x26]\n"
                             : [A] "+r"(A), [B] "+r"(B), [C] "+r"(C)
                             : [K] "r"(K4), [offset] "r"(offset)
                             : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                             "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                             "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                             "v29", "v30", "v31", "x19", "x20", "x21", "x22", "x26");
    } else {
        __asm__ __volatile__("mov x26, %[C]\n"
                             "ldr s28, [x26]\n"
                             "fcvtn v16.4h, v0.4s\n"
                             "uzp1 v0.2s, v16.2s, v30.2s\n"
                             "uzp2 v4.2s, v16.2s, v30.2s\n"
                             "fadd v0.4h, v0.4h, v28.4h\n"
                             "str s0, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr s29, [x26]\n"
                             "fadd v4.4h, v4.4h, v29.4h\n"
                             "str s4, [x26]\n"
                             : [A] "+r"(A), [B] "+r"(B), [C] "+r"(C)
                             : [K] "r"(K4), [offset] "r"(offset)
                             : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                             "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                             "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                             "v29", "v30", "v31", "x19", "x20", "x21", "x22", "x26");
    }
}
#endif
