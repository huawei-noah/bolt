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

inline void matrix1_trans_int8(U32 size, U32 blockK, U32 K, INT8 *src, INT8 *dst)
{
    INT8 *src1 = src;
    for (U32 i = 0; i < blockK; i++) {
        for (U32 j = 0; j < size; j++) {
            src1 = src + j * K;
            if (i % 16 == 0) {
                __builtin_prefetch(src1 + 16);
            }
            *dst++ = *(src1 + i);
        }
    }
    U32 K4 = UNI_ALIGN(blockK, 4);
    for (U32 i = 0; i < K4 - blockK; i++) {
        UNI_MEMSET(dst, 0, size * sizeof(INT8));
        dst += size;
    }
}

inline void matrix2_trans_int8(U32 size, U32 blockK, U32 M, INT8 *src, INT8 *dst)
{
    for (U32 i = 0; i < blockK; i++) {
        if (i % 16 == 0) {
            __builtin_prefetch(src + 16);
        }
        UNI_MEMCPY(dst, src, size * sizeof(INT8));
        dst += size;
        src += M;
    }
    U32 K4 = UNI_ALIGN(blockK, 4);
    for (U32 i = 0; i < K4 - blockK; i++) {
        UNI_MEMSET(dst, 0, size * sizeof(INT8));
        dst += size;
    }
}
#endif
