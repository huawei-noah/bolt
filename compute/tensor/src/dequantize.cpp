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
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif

EE dequantize(Tensor input, const F32 *scale, Tensor bias, Tensor output, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc qDesc = input.get_desc();
    void *qData = get_ptr_from_tensor(input, arch);
    TensorDesc bDesc = bias.get_desc();
    void *bData = get_ptr_from_tensor(bias, arch);
    TensorDesc dDesc = output.get_desc();
    void *data = get_ptr_from_tensor(output, arch);
    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = dequantize_general(qDesc, qData, scale, bDesc, bData, dDesc, data);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = dequantize_arm(qDesc, qData, scale, bDesc, bData, dDesc, data);
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = dequantize_x86(qDesc, qData, scale, bDesc, bData, dDesc, data);
#endif
    }
    return ret;
}
