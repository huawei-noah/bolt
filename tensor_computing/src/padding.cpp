// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "tensor_computing.h"
#ifdef _USE_GENERAL
#include "cpu/general/tensor_computing_general.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif
#ifdef _USE_MALI 
#include "gpu/mali/tensor_computing_mali.h"
#endif

EE padding_infer_output_size(TensorDesc inputDesc, PadDesc padDesc, TensorDesc* outputDesc) 
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    int out_n = in;
    int out_c = ic;
    int out_h = ih + padDesc.top + padDesc.bottom;
    int out_w = iw + padDesc.left + padDesc.right;
    *outputDesc = tensor4df(idt, idf, out_n, out_c, out_h, out_w);
    return SUCCESS;
}

EE padding(TensorDesc inputDesc, const void* input, PadDesc padDesc, TensorDesc outputDesc, void* output, Arch arch)
{
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL) {
#ifdef _USE_GENERAL
        ret = padding_general(inputDesc, input, padDesc, outputDesc, output);
#endif
#ifdef _USE_NEON
    } else if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
        ret = padding_arm(inputDesc, input, padDesc, outputDesc, output);
#endif
    }
    return ret;
}
