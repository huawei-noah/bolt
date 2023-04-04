// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_BLAS_MATRIX_TRANSPOSE
#define _H_BLAS_MATRIX_TRANSPOSE

#include "uni.h"

inline void matrix1_trans(U32 size, U32 blockK, U32 K, F16 *src, F16 *dst)
{
    F16 *src1 = src;
    U32 offset = 8 * blockK;
    for (U32 i = 0; i < blockK; i++) {
        for (U32 j = 0; j < size; j++) {
            src1 = src + j * K;
            if (i % 32) {
                asm volatile("prfm pldl2keep, [%0, %1]\n"
                             : "+r"(src1)
                             : "r"((I64)offset)
                             : "memory", "cc");
            }
            *dst++ = *(src1 + i);
        }
    }
}

inline void matrix2_trans(U32 size, U32 blockK, U32 M, F16 *src, F16 *dst)
{
    U32 tile = size * sizeof(F16);
    for (U32 i = 0; i < blockK; i++) {
        asm volatile("prfm pldl2keep, [%0, #48]\n" : "+r"(src) : : "memory", "cc");
        UNI_MEMCPY(dst, src, tile);
        dst += size;
        src += M;
    }
}
#endif
