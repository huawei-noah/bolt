// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <cmath>
#include "sys.h"
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "tensor_computing.h"
#include "cpu/general/tensor_computing_general.h"
#include "cpu/arm/tensor_computing_arm.h"

EE pooling_infer_output_size(TensorDesc inputDesc, PoolingDesc poolingDesc, TensorDesc* outputDesc)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    }
    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS_WITH_RETURN(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    U32 stride = poolingDesc.stride;
    U32 padding = poolingDesc.padding;
    U32 kernelSize = poolingDesc.kernelSize;
    RoundMode rm = poolingDesc.rm;
    U32 oh = 0, ow = 0;
    switch (rm) {
        case CEIL: {
            oh = (U32)(ceil((double(ih + 2.0 * padding - kernelSize) / stride))) + 1;
            ow = (U32)(ceil((double(iw + 2.0 * padding - kernelSize) / stride))) + 1;
            break;
        }
        case FLOOR: {
            oh = (U32)(floor((double(ih + 2.0 * padding - kernelSize) / stride))) + 1;
            ow = (U32)(floor((double(iw + 2.0 * padding - kernelSize) / stride))) + 1;
            break;
        }
        default: {
            CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
        }
    }
    *outputDesc = tensor4df(idt, idf, in, ic, oh, ow);
    return SUCCESS;
}

EE pooling(TensorDesc inputDesc, const void* input, PoolingDesc poolingDesc, const void* scale, TensorDesc outputDesc, void* output, Arch arch)
{
    EE ret = SUCCESS;
    switch (arch) {
        case CPU_GENERAL:
            ret = pooling_general(inputDesc, input,
                                  poolingDesc,
                                  outputDesc, output);
            break;
        case ARM_A55:
            ret = pooling_arm(inputDesc, input,
                              poolingDesc, scale,
                              outputDesc, output);
            break;
        case ARM_A76:
            ret = pooling_arm(inputDesc, input,
                              poolingDesc, scale,
                              outputDesc, output);
            break;
        default:
            ret = NOT_SUPPORTED;
    }
    return ret;
}
