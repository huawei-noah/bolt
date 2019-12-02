// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "error.h"
#include "type.h"
#include "cpu/general/blas_general.h"

template<typename T1, typename T2>
inline void mvm(U32 M, U32 K, bool transpose, T1* mat, T1 *vec, T2* res) {
    if (! transpose) {
        for (U32 i = 0; i < M; i++) {
            T2 out_f = 0;
            for (U32 j = 0; j < K; j++) {
                out_f += mat[i * K + j] * vec[j];
            }
            res[i] += out_f;
        }
    } else {
        for (U32 i = 0; i < K; i++) {
            T2 out_f = 0;
            for (U32 j = 0; j < M; j++) {
                out_f += mat[j * K + i] * vec[j];
            }
            res[i] += out_f;
        }
    }
}

EE mvm_general(U32 row, U32 col, DataType dt, bool transpose, const void *matrix, const void *vector, void *result) {
    EE ret = SUCCESS;
    switch (dt) {
        case DT_F16:
            mvm<F16, F16>(row, col, transpose, (F16*)matrix, (F16*)vector, (F16*)result);
            break;
        case DT_I8:
            mvm<INT8, I32>(row, col, transpose, (INT8*)matrix, (INT8*)vector, (I32*)result);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
