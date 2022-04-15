// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "blas_enhance.h"
#ifdef _USE_GENERAL
#include "cpu/general/blas_general.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/blas_arm.h"
#endif

EE vector_vector_axpby(
    F32 a, TensorDesc xDesc, const void *x, F32 b, TensorDesc yDesc, void *y, Arch arch)
{
    if (nullptr == x || nullptr == y) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType xDataType, yDataType;
    DataFormat xDataFormat, yDataFormat;
    U32 xLen, yLen;
    CHECK_STATUS(tensor1dGet(xDesc, &xDataType, &xDataFormat, &xLen));
    CHECK_STATUS(tensor1dGet(yDesc, &yDataType, &yDataFormat, &yLen));

    if (xDataType != yDataType) {
        CHECK_STATUS(NOT_MATCH);
    }

    if (xLen != yLen) {
        CHECK_STATUS(NOT_MATCH);
    }

    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = axpby_general(yLen, yDataType, a, x, b, y);
#endif
#ifdef _USE_NEON
    } else {
        ret = axpby_arm(yLen, yDataType, a, x, b, y, arch);
#endif
    }
    return ret;
}
