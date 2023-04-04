// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/tensor_computing_x86.h"
#ifdef _USE_INT8
#include "cpu/x86/int8/tensor_computing_int8.h"
#endif

EE dequantize_x86(TensorDesc qDesc,
    void *qData,
    const F32 *scale,
    TensorDesc bDesc,
    void *bData,
    TensorDesc dDesc,
    void *dData)
{
    EE ret = SUCCESS;
    if (dDesc.dt == DT_F32) {
        switch (qDesc.dt) {
#ifdef _USE_INT8
            case DT_I32: {
                ret = dequantizeI32ToF32(qDesc, (I32 *)qData, scale, dDesc, (F32 *)dData);
                break;
            }
            case DT_U8_Q: {
                ret = dequantizeU8ToF32(qDesc, (UINT8 *)qData, scale, dDesc, (F32 *)dData);
                break;
            }
#endif
            default:
                ret = NOT_SUPPORTED;
                break;
        }
    } else {
        ret = NOT_SUPPORTED;
    }
    return ret;
}