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
#ifdef _USE_FP16
#include "cpu/arm/fp16/tensor_computing_fp16.h"
#endif


EE quantize_tensor_arm(TensorDesc dDesc, const void* data, TensorDesc* qDesc, void* qData, void *scale)
{
    EE ret = SUCCESS;
    switch (dDesc.dt) {
#ifdef _USE_FP16
        case DT_F16: {
            ret = quantize_tensor_fp16(dDesc, data, qDesc, qData, (F16*)scale);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
