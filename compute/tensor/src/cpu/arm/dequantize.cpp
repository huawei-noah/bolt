// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/tensor_computing_arm.h"
#if defined(_USE_FP32)
#include "cpu/arm/fp32/tensor_computing_fp32.h"
#endif
#if defined(_USE_FP16)
#include "cpu/arm/fp16/tensor_computing_fp16.h"
#endif

EE dequantize_arm(TensorDesc qDesc,
    void *qData,
    const F32 *scale,
    TensorDesc bDesc,
    void *bData,
    TensorDesc dDesc,
    void *data)
{
    if (nullptr == data || nullptr == qData || nullptr == scale) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = NOT_SUPPORTED;
    switch (dDesc.dt) {
#if defined(_USE_FP32)
        case DT_F32:
            ret = dequantize_to_f32(qDesc, qData, scale, bDesc, bData, dDesc, data);
            break;
#endif
#if defined(_USE_FP16)
        case DT_F16:
            ret = dequantize_to_f16(qDesc, qData, scale, bDesc, bData, dDesc, data);
            break;
#endif
        default:
            break;
    }
    return ret;
}
