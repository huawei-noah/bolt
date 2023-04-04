// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MMM_12X4
#define _H_MMM_12X4

template <typename T = F16>
void mmm_12x4(U64 offset, U64 K4, F16 *A, F16 *B, F16 *C)
{
    __asm__ __volatile__("ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [%[A]]\n"
                         "ld1 {v18.8h, v19.8h}, [%[B]]\n"
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
                         "movi v8.16b, #0x0\n"
                         "movi v9.16b, #0x0\n"
                         "movi v10.16b, #0x0\n"
                         "movi v11.16b, #0x0\n"

                         "cmp x22, #1\n"
                         "ble 1f\n"

                         "0:\n"
                         "ldr q16, [x20, 0x40]\n"
                         "ldr q17, [x20, 0x50]\n"
                         ".inst 0x6e52ed80  // bfmmla v0.4s, v12.8h, v18.8h\n"
                         "ldr q26, [x21, 0x20]\n"
                         "ldr q27, [x21, 0x30]\n"
                         ".inst 0x6e53ed81  // bfmmla v1.4s, v12.8h, v19.8h\n"
                         "ldr q20, [x20, 0x60]\n"
                         ".inst 0x6e52eda2  // bfmmla v2.4s, v13.8h, v18.8h\n"
                         ".inst 0x6e53eda3  // bfmmla v3.4s, v13.8h, v19.8h\n"
                         "ldr q21, [x20, 0x70]\n"
                         ".inst 0x6e52edc4  // bfmmla v4.4s, v14.8h, v18.8h\n"
                         ".inst 0x6e53edc5  // bfmmla v5.4s, v14.8h, v19.8h\n"
                         "ldr q22, [x20, 0x80]\n"
                         ".inst 0x6e52ede6  // bfmmla v6.4s, v15.8h, v18.8h\n"
                         ".inst 0x6e53ede7  // bfmmla v7.4s, v15.8h, v19.8h\n"
                         "ldr q23, [x20, 0x90]\n"
                         ".inst 0x6e52ee08  // bfmmla v8.4s, v16.8h, v18.8h\n"
                         ".inst 0x6e53ee09  // bfmmla v9.4s, v16.8h, v19.8h\n"
                         "ldr q24, [x20, 0xa0]\n"
                         ".inst 0x6e52ee2a  // bfmmla v10.4s, v17.8h, v18.8h\n"
                         "ldr q18, [x21, 0x40]\n"
                         ".inst 0x6e53ee2b  // bfmmla v11.4s, v17.8h, v19.8h\n"
                         "sub x22, x22, #2\n"

                         "ldr q25, [x20, 0xb0]\n"
                         ".inst 0x6e5aee80  // bfmmla v0.4s, v20.8h, v26.8h\n"
                         "ldr q19, [x21, 0x50]\n"
                         ".inst 0x6e5bee81  // bfmmla v1.4s, v20.8h, v27.8h\n"
                         "ldr q12, [x20, 0xc0]\n"
                         ".inst 0x6e5aeea2  // bfmmla v2.4s, v21.8h, v26.8h\n"
                         ".inst 0x6e5beea3  // bfmmla v3.4s, v21.8h, v27.8h\n"
                         "ldr q13, [x20, 0xd0]\n"
                         ".inst 0x6e5aeec4  // bfmmla v4.4s, v22.8h, v26.8h\n"
                         ".inst 0x6e5beec5  // bfmmla v5.4s, v22.8h, v27.8h\n"
                         "ldr q14, [x20, 0xe0]\n"
                         ".inst 0x6e5aeee6  // bfmmla v6.4s, v23.8h, v26.8h\n"
                         ".inst 0x6e5beee7  // bfmmla v7.4s, v23.8h, v27.8h\n"
                         "ldr q15, [x20, 0xf0]\n"
                         ".inst 0x6e5aef08  // bfmmla v8.4s, v24.8h, v26.8h\n"
                         ".inst 0x6e5bef09  // bfmmla v9.4s, v24.8h, v27.8h\n"
                         ".inst 0x6e5aef2a  // bfmmla v10.4s, v25.8h, v26.8h\n"
                         ".inst 0x6e5bef2b  // bfmmla v11.4s, v25.8h, v27.8h\n"

                         "add x20, x20, 0xc0\n"
                         "add x21, x21, 0x40\n"
                         "cmp x22, #1\n"
                         "bgt 0b\n"
                         "1:\n"
                         "bne 2f\n"
                         "ldr q16, [x20, 0x40]\n"
                         "ldr q17, [x20, 0x50]\n"
                         ".inst 0x6e52ed80  // bfmmla v0.4s, v12.8h, v18.8h\n"
                         ".inst 0x6e53ed81  // bfmmla v1.4s, v12.8h, v19.8h\n"
                         ".inst 0x6e52eda2  // bfmmla v2.4s, v13.8h, v18.8h\n"
                         ".inst 0x6e53eda3  // bfmmla v3.4s, v13.8h, v19.8h\n"
                         ".inst 0x6e52edc4  // bfmmla v4.4s, v14.8h, v18.8h\n"
                         ".inst 0x6e53edc5  // bfmmla v5.4s, v14.8h, v19.8h\n"
                         ".inst 0x6e52ede6  // bfmmla v6.4s, v15.8h, v18.8h\n"
                         ".inst 0x6e53ede7  // bfmmla v7.4s, v15.8h, v19.8h\n"
                         ".inst 0x6e52ee08  // bfmmla v8.4s, v16.8h, v18.8h\n"
                         ".inst 0x6e53ee09  // bfmmla v9.4s, v16.8h, v19.8h\n"
                         ".inst 0x6e52ee2a  // bfmmla v10.4s, v17.8h, v18.8h\n"
                         ".inst 0x6e53ee2b  // bfmmla v11.4s, v17.8h, v19.8h\n"

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
                             "uzp1 v12.2d, v0.2d, v1.2d\n"
                             "uzp2 v13.2d, v0.2d, v1.2d\n"
                             "uzp1 v14.2d, v2.2d, v3.2d\n"
                             "uzp2 v15.2d, v2.2d, v3.2d\n"

                             "fadd v12.4s, v12.4s, v30.4s\n"
                             "uzp1 v16.2d, v4.2d, v5.2d\n"
                             "str q12, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr q31, [x26]\n"
                             "uzp2 v17.2d, v4.2d, v5.2d\n"
                             "fadd v13.4s, v13.4s, v31.4s\n"
                             "uzp1 v18.2d, v6.2d, v7.2d\n"
                             "uzp2 v19.2d, v6.2d, v7.2d\n"
                             "str q13, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr q24, [x26]\n"
                             "uzp1 v20.2d, v8.2d, v9.2d\n"
                             "fadd v14.4s, v14.4s, v24.4s\n"
                             "uzp2 v21.2d, v8.2d, v9.2d\n"
                             "str q14, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr q25, [x26]\n"
                             "fadd v15.4s, v15.4s, v25.4s\n"
                             "uzp1 v22.2d, v10.2d, v11.2d\n"
                             "uzp2 v23.2d, v10.2d, v11.2d\n"
                             "str q15, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr q24, [x26]\n"
                             "fadd v16.4s, v16.4s, v24.4s\n"
                             "str q16, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr q25, [x26]\n"
                             "fadd v17.4s, v17.4s, v25.4s\n"
                             "str q17, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr q24, [x26]\n"
                             "fadd v18.4s, v18.4s, v24.4s\n"
                             "str q18, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr q25, [x26]\n"
                             "fadd v19.4s, v19.4s, v25.4s\n"
                             "str q19, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr q24, [x26]\n"
                             "fadd v20.4s, v20.4s, v24.4s\n"
                             "str q20, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr q25, [x26]\n"
                             "fadd v21.4s, v21.4s, v25.4s\n"
                             "str q21, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr q24, [x26]\n"
                             "fadd v22.4s, v22.4s, v24.4s\n"
                             "str q22, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr q25, [x26]\n"
                             "fadd v23.4s, v23.4s, v25.4s\n"
                             "str q23, [x26]\n"
                             : [A] "+r"(A), [B] "+r"(B), [C] "+r"(C)
                             : [K] "r"(K4), [offset] "r"(offset)
                             : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                             "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                             "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                             "v29", "v30", "v31", "x19", "x20", "x21", "x22", "x26");
    } else {
        __asm__ __volatile__("mov x26, %[C]\n"
                             "ldr d28, [x26]\n"
                             "fcvtn v12.4h, v0.4s\n"
                             "fcvtn v13.4h, v1.4s\n"
                             "uzp1 v0.2s, v12.2s, v13.2s\n"
                             "uzp2 v1.2s, v12.2s, v13.2s\n"
                             "fcvtn v14.4h, v2.4s\n"
                             "fcvtn v15.4h, v3.4s\n"
                             "fadd v0.4h, v0.4h, v28.4h\n"
                             "uzp1 v2.2s, v14.2s, v15.2s\n"
                             "uzp2 v3.2s, v14.2s, v15.2s\n"
                             "str d0, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d29, [x26]\n"
                             "fcvtn v16.4h, v4.4s\n"
                             "fcvtn v17.4h, v5.4s\n"
                             "fadd v1.4h, v1.4h, v29.4h\n"
                             "uzp1 v4.2s, v16.2s, v17.2s\n"
                             "uzp2 v5.2s, v16.2s, v17.2s\n"
                             "str d1, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d28, [x26]\n"
                             "fcvtn v18.4h, v6.4s\n"
                             "fcvtn v19.4h, v7.4s\n"
                             "fadd v2.4h, v2.4h, v28.4h\n"
                             "uzp1 v6.2s, v18.2s, v19.2s\n"
                             "uzp2 v7.2s, v18.2s, v19.2s\n"
                             "str d2, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d29, [x26]\n"
                             "fcvtn v20.4h, v8.4s\n"
                             "fcvtn v21.4h, v9.4s\n"
                             "fadd v3.4h, v3.4h, v29.4h\n"
                             "uzp1 v8.2s, v20.2s, v21.2s\n"
                             "uzp2 v9.2s, v20.2s, v21.2s\n"
                             "str d3, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d28, [x26]\n"
                             "fcvtn v22.4h, v10.4s\n"
                             "fcvtn v23.4h, v11.4s\n"
                             "fadd v4.4h, v4.4h, v28.4h\n"
                             "uzp1 v10.2s, v22.2s, v23.2s\n"
                             "uzp2 v11.2s, v22.2s, v23.2s\n"
                             "str d4, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d29, [x26]\n"
                             "fadd v5.4h, v5.4h, v29.4h\n"
                             "str d5, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr d28, [x26]\n"
                             "fadd v6.4h, v6.4h, v28.4h\n"
                             "str d6, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d29, [x26]\n"
                             "fadd v7.4h, v7.4h, v29.4h\n"
                             "str d7, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr d28, [x26]\n"
                             "fadd v8.4h, v8.4h, v28.4h\n"
                             "str d8, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d29, [x26]\n"
                             "fadd v9.4h, v9.4h, v29.4h\n"
                             "str d9, [x26]\n"
                             "add x26, x26, %[offset]\n"

                             "ldr d28, [x26]\n"
                             "fadd v10.4h, v10.4h, v28.4h\n"
                             "str d10, [x26]\n"
                             "add x26, x26, %[offset]\n"
                             "ldr d29, [x26]\n"
                             "fadd v11.4h, v11.4h, v29.4h\n"
                             "str d11, [x26]\n"
                             : [A] "+r"(A), [B] "+r"(B), [C] "+r"(C)
                             : [K] "r"(K4), [offset] "r"(offset)
                             : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                             "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                             "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                             "v29", "v30", "v31", "x19", "x20", "x21", "x22", "x26");
    }
}
#endif
