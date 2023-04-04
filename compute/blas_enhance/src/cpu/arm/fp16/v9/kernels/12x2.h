// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MMM_12X2
#define _H_MMM_12X2

template <typename T = F16>
void mmm_12x2(U64 offset, U64 K4, F16 *A, F16 *B, F16 *C)
{
    __asm__ __volatile__("ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [%[A]]\n"
                         "ldr q18, [%[B]]\n"
                         "mov x20, %[A]\n"
                         "mov x21, %[B]\n"
                         "mov x22, %[K]\n"

                         "movi v0.16b, #0x0\n"
                         "movi v2.16b, #0x0\n"
                         "movi v4.16b, #0x0\n"
                         "movi v6.16b, #0x0\n"
                         "movi v8.16b, #0x0\n"
                         "movi v10.16b, #0x0\n"

                         "cmp x22, #1\n"
                         "ble 1f\n"

                         "0:\n"
                         "ldr q16, [x20, 0x40]\n"
                         "ldr q17, [x20, 0x50]\n"
                         ".inst 0x6e52ed80  // bfmmla v0.4s, v12.8h, v18.8h\n"
                         "ldr q26, [x21, 0x10]\n"
                         "ldr q20, [x20, 0x60]\n"
                         ".inst 0x6e52eda2  // bfmmla v2.4s, v13.8h, v18.8h\n"
                         "ldr q21, [x20, 0x70]\n"
                         ".inst 0x6e52edc4  // bfmmla v4.4s, v14.8h, v18.8h\n"
                         "ldr q22, [x20, 0x80]\n"
                         ".inst 0x6e52ede6  // bfmmla v6.4s, v15.8h, v18.8h\n"
                         "ldr q23, [x20, 0x90]\n"
                         ".inst 0x6e52ee08  // bfmmla v8.4s, v16.8h, v18.8h\n"
                         "ldr q24, [x20, 0xa0]\n"
                         ".inst 0x6e52ee2a  // bfmmla v10.4s, v17.8h, v18.8h\n"
                         "ldr q18, [x21, 0x20]\n"
                         "sub x22, x22, #2\n"

                         "ldr q25, [x20, 0xb0]\n"
                         ".inst 0x6e5aee80  // bfmmla v0.4s, v20.8h, v26.8h\n"
                         "ldr q12, [x20, 0xc0]\n"
                         ".inst 0x6e5aeea2  // bfmmla v2.4s, v21.8h, v26.8h\n"
                         "ldr q13, [x20, 0xd0]\n"
                         ".inst 0x6e5aeec4  // bfmmla v4.4s, v22.8h, v26.8h\n"
                         "ldr q14, [x20, 0xe0]\n"
                         ".inst 0x6e5aeee6  // bfmmla v6.4s, v23.8h, v26.8h\n"
                         "ldr q15, [x20, 0xf0]\n"
                         ".inst 0x6e5aef08  // bfmmla v8.4s, v24.8h, v26.8h\n"
                         ".inst 0x6e5aef2a  // bfmmla v10.4s, v25.8h, v26.8h\n"

                         "add x20, x20, 0xc0\n"
                         "add x21, x21, 0x20\n"
                         "cmp x22, #1\n"
                         "bgt 0b\n"
                         "1:\n"
                         "bne 2f\n"
                         "ldr q16, [x20, 0x40]\n"
                         "ldr q17, [x20, 0x50]\n"
                         ".inst 0x6e52ed80  // bfmmla v0.4s, v12.8h, v18.8h\n"
                         ".inst 0x6e52eda2  // bfmmla v2.4s, v13.8h, v18.8h\n"
                         ".inst 0x6e52edc4  // bfmmla v4.4s, v14.8h, v18.8h\n"
                         ".inst 0x6e52ede6  // bfmmla v6.4s, v15.8h, v18.8h\n"
                         ".inst 0x6e52ee08  // bfmmla v8.4s, v16.8h, v18.8h\n"
                         ".inst 0x6e52ee2a  // bfmmla v10.4s, v17.8h, v18.8h\n"

                         "2:\n"
                         : [A] "+r"(A), [B] "+r"(B), [C] "+r"(C)
                         : [K] "r"(K4), [offset] "r"(offset)
                         : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                         "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
                         "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
                         "v30", "v31", "x19", "x20", "x21", "x22", "x26");
    if (sizeof(T) == 4) {
        __asm__ __volatile__("mov x26, %[C]\n"
                             "ldr d30, [x26]\n"
                             "uzp1 v12.2d, v0.2d, v1.2d\n"
                             "uzp2 v13.2d, v0.2d, v1.2d\n"
                             "uzp1 v14.2d, v2.2d, v3.2d\n"
                             "uzp2 v15.2d, v2.2d, v3.2d\n"

                             "fadd v12.2s, v12.2s, v30.2s\n"
                             "uzp1 v16.2d, v4.2d, v5.2d\n"
                             "str d12, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d31, [x26]\n"
                             "uzp2 v17.2d, v4.2d, v5.2d\n"
                             "fadd v13.2s, v13.2s, v31.2s\n"
                             "uzp1 v18.2d, v6.2d, v7.2d\n"
                             "uzp2 v19.2d, v6.2d, v7.2d\n"
                             "str d13, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr d24, [x26]\n"
                             "uzp1 v20.2d, v8.2d, v9.2d\n"
                             "fadd v14.2s, v14.2s, v24.2s\n"
                             "uzp2 v21.2d, v8.2d, v9.2d\n"
                             "str d14, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d25, [x26]\n"
                             "fadd v15.2s, v15.2s, v25.2s\n"
                             "uzp1 v22.2d, v10.2d, v11.2d\n"
                             "uzp2 v23.2d, v10.2d, v11.2d\n"
                             "str d15, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr d24, [x26]\n"
                             "fadd v16.2s, v16.2s, v24.2s\n"
                             "str d16, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d25, [x26]\n"
                             "fadd v17.2s, v17.2s, v25.2s\n"
                             "str d17, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr d24, [x26]\n"
                             "fadd v18.2s, v18.2s, v24.2s\n"
                             "str d18, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d25, [x26]\n"
                             "fadd v19.2s, v19.2s, v25.2s\n"
                             "str d19, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr d24, [x26]\n"
                             "fadd v20.2s, v20.2s, v24.2s\n"
                             "str d20, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d25, [x26]\n"
                             "fadd v21.2s, v21.2s, v25.2s\n"
                             "str d21, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr d24, [x26]\n"
                             "fadd v22.2s, v22.2s, v24.2s\n"
                             "str d22, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d25, [x26]\n"
                             "fadd v23.2s, v23.2s, v25.2s\n"
                             "str d23, [x26]\n"
                             : [A] "+r"(A), [B] "+r"(B), [C] "+r"(C)
                             : [K] "r"(K4), [offset] "r"(offset)
                             : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                             "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                             "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                             "v29", "v30", "v31", "x19", "x20", "x21", "x22", "x26");
    } else {
        __asm__ __volatile__("mov x26, %[C]\n"
                             "ldr s28, [x26]\n"
                             "fcvtn v12.4h, v0.4s\n"
                             "uzp1 v0.2s, v12.2s, v13.2s\n"
                             "uzp2 v1.2s, v12.2s, v13.2s\n"
                             "fcvtn v14.4h, v2.4s\n"
                             "fadd v0.4h, v0.4h, v28.4h\n"
                             "uzp1 v2.2s, v14.2s, v15.2s\n"
                             "uzp2 v3.2s, v14.2s, v15.2s\n"
                             "str s0, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr s29, [x26]\n"
                             "fcvtn v16.4h, v4.4s\n"
                             "fadd v1.4h, v1.4h, v29.4h\n"
                             "uzp1 v4.2s, v16.2s, v17.2s\n"
                             "uzp2 v5.2s, v16.2s, v17.2s\n"
                             "str s1, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr s28, [x26]\n"
                             "fcvtn v18.4h, v6.4s\n"
                             "fadd v2.4h, v2.4h, v28.4h\n"
                             "uzp1 v6.2s, v18.2s, v19.2s\n"
                             "uzp2 v7.2s, v18.2s, v19.2s\n"
                             "str s2, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr s29, [x26]\n"
                             "fcvtn v20.4h, v8.4s\n"
                             "fadd v3.4h, v3.4h, v29.4h\n"
                             "uzp1 v8.2s, v20.2s, v21.2s\n"
                             "uzp2 v9.2s, v20.2s, v21.2s\n"
                             "str s3, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr s28, [x26]\n"
                             "fcvtn v22.4h, v10.4s\n"
                             "fadd v4.4h, v4.4h, v28.4h\n"
                             "uzp1 v10.2s, v22.2s, v23.2s\n"
                             "uzp2 v11.2s, v22.2s, v23.2s\n"
                             "str s4, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr s29, [x26]\n"
                             "fadd v5.4h, v5.4h, v29.4h\n"
                             "str s5, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr s28, [x26]\n"
                             "fadd v6.4h, v6.4h, v28.4h\n"
                             "str s6, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr s29, [x26]\n"
                             "fadd v7.4h, v7.4h, v29.4h\n"
                             "str s7, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr s28, [x26]\n"
                             "fadd v8.4h, v8.4h, v28.4h\n"
                             "str s8, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr s29, [x26]\n"
                             "fadd v9.4h, v9.4h, v29.4h\n"
                             "str s9, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr s28, [x26]\n"
                             "fadd v10.4h, v10.4h, v28.4h\n"
                             "str s10, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr s29, [x26]\n"
                             "fadd v11.4h, v11.4h, v29.4h\n"
                             "str s11, [x26]\n"
                             : [A] "+r"(A), [B] "+r"(B), [C] "+r"(C)
                             : [K] "r"(K4), [offset] "r"(offset)
                             : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                             "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                             "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                             "v29", "v30", "v31", "x19", "x20", "x21", "x22", "x26");
    }
}
#endif
