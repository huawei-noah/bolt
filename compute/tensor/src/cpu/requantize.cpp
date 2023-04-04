// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/tensor_computing_cpu.h"
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif
#include "cpu/cpu_functions.h"

EE requantize_cpu(TensorDesc inputDesc,
    INT8* input,
    F32 inputScale,
    TensorDesc outputDesc,
    INT8 *output,
    F32 outputScale,
    Arch arch)
{
#ifdef _USE_NEON
    if (IS_ARM(arch)) {
        return requantize_arm(inputDesc,
                input,
                inputScale,
                outputDesc,
                output,
                outputScale);
    }
#endif
    int num = tensorNumElements(outputDesc);
    if (num <= 0) {
        return SUCCESS;
    }
    if (outputScale == inputScale) {
        UNI_MEMCPY(output, input, num * sizeof(INT8));
        return SUCCESS;
    }
    F32 rescale = outputScale / inputScale;
    ArrayScaleFunction scale_func = get_array_scale_function(arch);
    scale_func(DT_I8, input, output, num, rescale, 0);
    return SUCCESS;
}
