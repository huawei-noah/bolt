// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MMM_8X4
#define _H_MMM_8X4

template <typename T = F16>
void mmm_8x4(U64 offset, U64 K4, F16 *A, F16 *B, F16 *C)
{
    __asm__ __volatile__("ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [%[A]]\n"
                         "ld1 {v20.8h, v21.8h}, [%[B]]\n"
                         "mov x20, %[A]\n"
                         "mov x21, %[B]\n"
                         "mov x22, %[K]\n"
                         "movi v0.16b, #0x0\n"
                         "movi v1.16b, #0x0\n"
                         "movi v2.16b, #0x0\n"
                         "movi v3.16b, #0x0\n"
                         "movi v4.16b, #0x0\n"
                         "movi v5.16b, #0x0\n"
                         "movi v6.16b, #0x0\n"
                         "movi v7.16b, #0x0\n"

                         "cmp x22, #1\n"
                         "ble 1f\n"

                         "0:\n"
                         "ldr q24, [x20, 0x40]\n"
                         ".inst 0x6e54ee00  // bfmmla v0.4s, v16.8h, v20.8h\n"
                         "ldr q28, [x21, 0x20]\n"
                         ".inst 0x6e54ee21  // bfmmla v1.4s, v17.8h, v20.8h\n"
                         "ldr q25, [x20, 0x50]\n"
                         ".inst 0x6e54ee42  // bfmmla v2.4s, v18.8h, v20.8h\n"
                         ".inst 0x6e54ee63  // bfmmla v3.4s, v19.8h, v20.8h\n"
                         "ldr q26, [x20, 0x60]\n"
                         ".inst 0x6e55ee04  // bfmmla v4.4s, v16.8h, v21.8h\n"
                         ".inst 0x6e55ee25  // bfmmla v5.4s, v17.8h, v21.8h\n"
                         "ldr q27, [x20, 0x70]\n"
                         ".inst 0x6e55ee46  // bfmmla v6.4s, v18.8h, v21.8h\n"
                         ".inst 0x6e55ee67  // bfmmla v7.4s, v19.8h, v21.8h\n"
                         "ldr q29, [x21, 0x30]\n"
                         "ldr q16, [x20, 0x80]\n"
                         "ldr q20, [x21, 0x40]\n"

                         ".inst 0x6e5cef00  // bfmmla v0.4s, v24.8h, v28.8h\n"
                         "sub x22, x22, #2\n"
                         ".inst 0x6e5cef21  // bfmmla v1.4s, v25.8h, v28.8h\n"
                         "ldr q17, [x20, 0x90]\n"
                         ".inst 0x6e5cef42  // bfmmla v2.4s, v26.8h, v28.8h\n"
                         ".inst 0x6e5cef63  // bfmmla v3.4s, v27.8h, v28.8h\n"
                         "ldr q18, [x20, 0xa0]\n"
                         ".inst 0x6e5def04  // bfmmla v4.4s, v24.8h, v29.8h\n"
                         ".inst 0x6e5def25  // bfmmla v5.4s, v25.8h, v29.8h\n"
                         "ldr q19, [x20, 0xb0]\n"
                         ".inst 0x6e5def46  // bfmmla v6.4s, v26.8h, v29.8h\n"
                         "add x20, x20, 0x80\n"
                         ".inst 0x6e5def67  // bfmmla v7.4s, v27.8h, v29.8h\n"
                         "ldr q21, [x21, 0x50]\n"
                         "add x21, x21, 0x40\n"
                         "cmp x22, #1\n"
                         "bgt 0b\n"
                         "1:\n"
                         "bne 2f\n"
                         ".inst 0x6e54ee00  // bfmmla v0.4s, v16.8h, v20.8h\n"
                         ".inst 0x6e54ee21  // bfmmla v1.4s, v17.8h, v20.8h\n"
                         ".inst 0x6e54ee42  // bfmmla v2.4s, v18.8h, v20.8h\n"
                         ".inst 0x6e54ee63  // bfmmla v3.4s, v19.8h, v20.8h\n"
                         ".inst 0x6e55ee04  // bfmmla v4.4s, v16.8h, v21.8h\n"
                         ".inst 0x6e55ee25  // bfmmla v5.4s, v17.8h, v21.8h\n"
                         ".inst 0x6e55ee46  // bfmmla v6.4s, v18.8h, v21.8h\n"
                         ".inst 0x6e55ee67  // bfmmla v7.4s, v19.8h, v21.8h\n"

                         "2:\n"
                         : [A] "+r"(A), [B] "+r"(B), [C] "+r"(C)
                         : [K] "r"(K4), [offset] "r"(offset)
                         : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                         "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
                         "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
                         "v30", "v31", "x19", "x20", "x21", "x22", "x26");

    if (sizeof(T) == 4) {
        __asm__ __volatile__("mov x26, %[C]\n"
                             "ldr q30, [x26]\n"
                             "uzp1 v16.2d, v0.2d, v4.2d\n"
                             "uzp2 v18.2d, v0.2d, v4.2d\n"
                             "fadd v16.4s, v16.4s, v30.4s\n"
                             "str q16, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr q30, [x26]\n"
                             "uzp1 v20.2d, v1.2d, v5.2d\n"
                             "fadd v18.4s, v18.4s, v30.4s\n"
                             "uzp2 v22.2d, v1.2d, v5.2d\n"
                             "str q18, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr q30, [x26]\n"
                             "uzp1 v24.2d, v2.2d, v6.2d\n"
                             "fadd v20.4s, v20.4s, v30.4s\n"
                             "uzp2 v26.2d, v2.2d, v6.2d\n"
                             "str q20, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr q0, [x26]\n"
                             "uzp1 v28.2d, v3.2d, v7.2d\n"
                             "fadd v22.4s, v22.4s, v0.4s\n"
                             "str q22, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr q0, [x26]\n"
                             "uzp2 v30.2d, v3.2d, v7.2d\n"
                             "fadd v24.4s, v24.4s, v0.4s\n"
                             "str q24, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr q0, [x26]\n"
                             "fadd v26.4s, v26.4s, v0.4s\n"
                             "str q26, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr q0, [x26]\n"
                             "fadd v28.4s, v28.4s, v0.4s\n"
                             "str q28, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr q0, [x26]\n"
                             "fadd v30.4s, v30.4s, v0.4s\n"
                             "str q30, [x26]\n"
                             : [A] "+r"(A), [B] "+r"(B), [C] "+r"(C)
                             : [K] "r"(K4), [offset] "r"(offset)
                             : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                             "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                             "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                             "v29", "v30", "v31", "x19", "x20", "x21", "x22", "x26");
    } else {
        __asm__ __volatile__("mov x26, %[C]\n"
                             "ldr d28, [x26]\n"
                             "fcvtn v16.4h, v0.4s\n"
                             "fcvtn v30.4h, v4.4s\n"
                             "uzp1 v0.2s, v16.2s, v30.2s\n"
                             "uzp2 v4.2s, v16.2s, v30.2s\n"
                             "fadd v0.4h, v0.4h, v28.4h\n"
                             "str d0, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d29, [x26]\n"
                             "fadd v4.4h, v4.4h, v29.4h\n"
                             "str d4, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d28, [x26]\n"

                             "fcvtn v18.4h, v1.4s\n"
                             "fcvtn v30.4h, v5.4s\n"
                             "uzp1 v1.2s, v18.2s, v30.2s\n"
                             "uzp2 v5.2s, v18.2s, v30.2s\n"
                             "fadd v1.4h, v1.4h, v28.4h\n"
                             "str d1, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d29, [x26]\n"
                             "fadd v5.4h, v5.4h, v29.4h\n"
                             "str d5, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d28, [x26]\n"

                             "fcvtn v20.4h, v2.4s\n"
                             "fcvtn v30.4h, v6.4s\n"
                             "uzp1 v2.2s, v20.2s, v30.2s\n"
                             "uzp2 v6.2s, v20.2s, v30.2s\n"
                             "fadd v2.4h, v2.4h, v28.4h\n"
                             "str d2, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d29, [x26]\n"
                             "fadd v6.4h, v6.4h, v29.4h\n"
                             "str d6, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d28, [x26]\n"

                             "fcvtn v22.4h, v3.4s\n"
                             "fcvtn v30.4h, v7.4s\n"
                             "uzp1 v3.2s, v22.2s, v30.2s\n"
                             "uzp2 v7.2s, v22.2s, v30.2s\n"
                             "fadd v3.4h, v3.4h, v28.4h\n"
                             "str d3, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d29, [x26]\n"
                             "fadd v7.4h, v7.4h, v29.4h\n"
                             "str d7, [x26]\n"
                             : [A] "+r"(A), [B] "+r"(B), [C] "+r"(C)
                             : [K] "r"(K4), [offset] "r"(offset)
                             : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                             "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                             "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                             "v29", "v30", "v31", "x19", "x20", "x21", "x22", "x26");
    }
}
#endif
