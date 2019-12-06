// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "sys.h"
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "cpu/arm/tensor_computing_arm.h"
#include "cpu/arm/fp16/concat_fp16.h"
#ifdef _USE_INT8
#include "cpu/arm/int8/concat_int8.h"
#endif

EE concat_arm(std::vector<TensorDesc> inputDesc, std::vector<void*> input, std::vector<F16> inputScale,
    TensorDesc outputDesc, void* output, F16* outputScale, U32 concatDim)
{
    EE ret = SUCCESS;
    switch (outputDesc.dt) {
        case DT_F16: {
            ret = concat_fp16(inputDesc, input,
                              outputDesc, output,
                              concatDim);
            break;
        }
#ifdef _USE_INT8
        case DT_I8: {
            ret = concat_int8(inputDesc, input, inputScale,
                              outputDesc, output, outputScale,
                              concatDim);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
