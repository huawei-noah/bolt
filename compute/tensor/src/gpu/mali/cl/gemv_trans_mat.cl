// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "kernel_def.h"
#define MANGLE_NAME_IMPL(base, TN, C) base##TN##C
#define MANGLE_NAME(base, TN, C) MANGLE_NAME_IMPL(base, TN, C)

#define TN
#if defined(USE_TRANS_CK)
#define TN kc_
#if (C == 4)
#define STORE_VAL(ov)              \
    {                              \
        int off = idx * row + idy; \
        vstore4(ov, off, out);     \
    }
#elif (C == 8)
#define STORE_VAL(ov)                                               \
    {                                                               \
        int off = (idx >> 1) * (row << 1) + (idy << 1) + (idx & 1); \
        vstore4(ov, off, out);                                      \
    }
#elif (C == 16)
#define STORE_VAL(ov)                                               \
    {                                                               \
        int off = (idx >> 2) * (row << 2) + (idy << 2) + (idx & 3); \
        vstore4(ov, off, out);                                      \
    }
#endif
#endif

#define LOAD_VAL(iv)                              \
    {                                             \
        int idx_4 = idx << 2;                     \
        if (idx_4 + 3 < col) {                    \
            iv = vload4(idx, in + idy * col);     \
        } else {                                  \
            if (idx_4 < col) {                    \
                iv.x = in[idx_4 + idy * col];     \
            }                                     \
            if (idx_4 + 1 < col) {                \
                iv.y = in[idx_4 + idy * col + 1]; \
            }                                     \
            if (idx_4 + 2 < col) {                \
                iv.z = in[idx_4 + idy * col + 2]; \
            }                                     \
        }                                         \
    }

__kernel void MANGLE_NAME(gemv_trans_mat_, TN, C)(
    const int row, const int col, __global const T *in, __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
#if defined(USE_TRANS_CK)
    T4 val = 0;
    LOAD_VAL(val);
    STORE_VAL(val);
#else
    T4 val = 0;
    LOAD_VAL(val);
    int colAlign = ((col + 7) >> 3) << 3;
    vstore4(val, idx, out + idy * colAlign);
#endif
}
