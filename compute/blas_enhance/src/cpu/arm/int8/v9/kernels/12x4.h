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

void mmm_12x4(U64 offset, U64 K4, INT8 *A, INT8 *B, I32 *C)
{
    __asm__ __volatile__("ld1 {v12.16b, v13.16b, v14.16b, v15.16b}, [%[A]]\n"
                         "ld1 {v18.16b, v19.16b}, [%[B]]\n"
                         "mov x20, %[A]\n"
                         "mov x21, %[B]\n"
                         "mov x26, %[C]\n"
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
                         ".inst 0x4e92a580  // smmla v0.4s, v12.16b, v18.16b\n"
                         "ldr q26, [x21, 0x20]\n"
                         "ldr q27, [x21, 0x30]\n"
                         ".inst 0x4e93a581  // smmla v1.4s, v12.16b, v19.16b\n"
                         "ldr q20, [x20, 0x60]\n"
                         ".inst 0x4e92a5a2  // smmla v2.4s, v13.16b, v18.16b\n"
                         ".inst 0x4e93a5a3  // smmla v3.4s, v13.16b, v19.16b\n"
                         "ldr q21, [x20, 0x70]\n"
                         ".inst 0x4e92a5c4  // smmla v4.4s, v14.16b, v18.16b\n"
                         ".inst 0x4e93a5c5  // smmla v5.4s, v14.16b, v19.16b\n"
                         "ldr q22, [x20, 0x80]\n"
                         ".inst 0x4e92a5e6  // smmla v6.4s, v15.16b, v18.16b\n"
                         ".inst 0x4e93a5e7  // smmla v7.4s, v15.16b, v19.16b\n"
                         "ldr q23, [x20, 0x90]\n"
                         ".inst 0x4e92a608  // smmla v8.4s, v16.16b, v18.16b\n"
                         ".inst 0x4e93a609  // smmla v9.4s, v16.16b, v19.16b\n"
                         "ldr q24, [x20, 0xa0]\n"
                         ".inst 0x4e92a62a  // smmla v10.4s, v17.16b, v18.16b\n"
                         "ldr q18, [x21, 0x40]\n"
                         ".inst 0x4e93a62b  // smmla v11.4s, v17.16b, v19.16b\n"
                         "sub x22, x22, #2\n"

                         "ldr q25, [x20, 0xb0]\n"
                         ".inst 0x4e9aa680  // smmla v0.4s, v20.16b, v26.16b\n"
                         "ldr q19, [x21, 0x50]\n"
                         ".inst 0x4e9ba681  // smmla v1.4s, v20.16b, v27.16b\n"
                         "ldr q12, [x20, 0xc0]\n"
                         ".inst 0x4e9aa6a2  // smmla v2.4s, v21.16b, v26.16b\n"
                         ".inst 0x4e9ba6a3  // smmla v3.4s, v21.16b, v27.16b\n"
                         "ldr q13, [x20, 0xd0]\n"
                         ".inst 0x4e9aa6c4  // smmla v4.4s, v22.16b, v26.16b\n"
                         ".inst 0x4e9ba6c5  // smmla v5.4s, v22.16b, v27.16b\n"
                         "ldr q14, [x20, 0xe0]\n"
                         ".inst 0x4e9aa6e6  // smmla v6.4s, v23.16b, v26.16b\n"
                         ".inst 0x4e9ba6e7  // smmla v7.4s, v23.16b, v27.16b\n"
                         "ldr q15, [x20, 0xf0]\n"
                         ".inst 0x4e9aa708  // smmla v8.4s, v24.16b, v26.16b\n"
                         ".inst 0x4e9ba709  // smmla v9.4s, v24.16b, v27.16b\n"
                         ".inst 0x4e9aa72a  // smmla v10.4s, v25.16b, v26.16b\n"
                         ".inst 0x4e9ba72b  // smmla v11.4s, v25.16b, v27.16b\n"

                         "add x20, x20, 0xc0\n"
                         "add x21, x21, 0x40\n"
                         "cmp x22, #1\n"
                         "bgt 0b\n"
                         "1:\n"
                         "bne 2f\n"
                         "ldr q16, [x20, 0x40]\n"
                         "ldr q17, [x20, 0x50]\n"
                         ".inst 0x4e92a580  // smmla v0.4s, v12.16b, v18.16b\n"
                         ".inst 0x4e93a581  // smmla v1.4s, v12.16b, v19.16b\n"
                         ".inst 0x4e92a5a2  // smmla v2.4s, v13.16b, v18.16b\n"
                         ".inst 0x4e93a5a3  // smmla v3.4s, v13.16b, v19.16b\n"
                         ".inst 0x4e92a5c4  // smmla v4.4s, v14.16b, v18.16b\n"
                         ".inst 0x4e93a5c5  // smmla v5.4s, v14.16b, v19.16b\n"
                         ".inst 0x4e92a5e6  // smmla v6.4s, v15.16b, v18.16b\n"
                         ".inst 0x4e93a5e7  // smmla v7.4s, v15.16b, v19.16b\n"
                         ".inst 0x4e92a608  // smmla v8.4s, v16.16b, v18.16b\n"
                         ".inst 0x4e93a609  // smmla v9.4s, v16.16b, v19.16b\n"
                         ".inst 0x4e92a62a  // smmla v10.4s, v17.16b, v18.16b\n"
                         ".inst 0x4e93a62b  // smmla v11.4s, v17.16b, v19.16b\n"

                         "2:\n"
                         "ldr q30, [x26]\n"
                         "uzp1 v12.2d, v0.2d, v1.2d\n"
                         "uzp2 v13.2d, v0.2d, v1.2d\n"
                         "uzp1 v14.2d, v2.2d, v3.2d\n"
                         "uzp2 v15.2d, v2.2d, v3.2d\n"

                         "add v12.4s, v12.4s, v30.4s\n"
                         "uzp1 v16.2d, v4.2d, v5.2d\n"
                         "str q12, [x26]\n"
                         "add x26, x26, %[offset]\n"
                         "ldr q31, [x26]\n"
                         "uzp2 v17.2d, v4.2d, v5.2d\n"
                         "add v13.4s, v13.4s, v31.4s\n"
                         "uzp1 v18.2d, v6.2d, v7.2d\n"
                         "uzp2 v19.2d, v6.2d, v7.2d\n"
                         "str q13, [x26]\n"
                         "add x26, x26, %[offset]\n"

                         "ldr q24, [x26]\n"
                         "uzp1 v20.2d, v8.2d, v9.2d\n"
                         "add v14.4s, v14.4s, v24.4s\n"
                         "uzp2 v21.2d, v8.2d, v9.2d\n"
                         "str q14, [x26]\n"
                         "add x26, x26, %[offset]\n"
                         "ldr q25, [x26]\n"
                         "add v15.4s, v15.4s, v25.4s\n"
                         "uzp1 v22.2d, v10.2d, v11.2d\n"
                         "uzp2 v23.2d, v10.2d, v11.2d\n"
                         "str q15, [x26]\n"
                         "add x26, x26, %[offset]\n"

                         "ldr q24, [x26]\n"
                         "add v16.4s, v16.4s, v24.4s\n"
                         "str q16, [x26]\n"
                         "add x26, x26, %[offset]\n"
                         "ldr q25, [x26]\n"
                         "add v17.4s, v17.4s, v25.4s\n"
                         "str q17, [x26]\n"
                         "add x26, x26, %[offset]\n"

                         "ldr q24, [x26]\n"
                         "add v18.4s, v18.4s, v24.4s\n"
                         "str q18, [x26]\n"
                         "add x26, x26, %[offset]\n"
                         "ldr q25, [x26]\n"
                         "add v19.4s, v19.4s, v25.4s\n"
                         "str q19, [x26]\n"
                         "add x26, x26, %[offset]\n"

                         "ldr q24, [x26]\n"
                         "add v20.4s, v20.4s, v24.4s\n"
                         "str q20, [x26]\n"
                         "add x26, x26, %[offset]\n"
                         "ldr q25, [x26]\n"
                         "add v21.4s, v21.4s, v25.4s\n"
                         "str q21, [x26]\n"
                         "add x26, x26, %[offset]\n"

                         "ldr q24, [x26]\n"
                         "add v22.4s, v22.4s, v24.4s\n"
                         "str q22, [x26]\n"
                         "add x26, x26, %[offset]\n"
                         "ldr q25, [x26]\n"
                         "add v23.4s, v23.4s, v25.4s\n"
                         "str q23, [x26]\n"
                         : [A] "+r"(A), [B] "+r"(B), [C] "+r"(C)
                         : [K] "r"(K4), [offset] "r"(offset)
                         : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                         "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
                         "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
                         "v30", "v31", "x19", "x20", "x21", "x22", "x26");
}
#endif
