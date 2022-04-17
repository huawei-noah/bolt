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
#ifdef _USE_FP32
#include "cpu/x86/fp32/tensor_computing_fp32.h"
#endif
#ifdef _USE_INT8
#include "cpu/x86/int8/tensor_computing_int8.h"
#endif

EE rnncell_x86(TensorDesc xDesc,
    const void *currentX,
    const TensorDesc *filterDesc,
    const void **filter,
    const TensorDesc *biasDesc,
    const void **bias,
    float *scale,
    void *state,
    U32 tmpBytes,
    void *tmp,
    RNNParamSpec rnnParamSpec,
    U32 batchStrideX,
    U32 batchStrideH,
    TensorDesc hDesc,
    void *output,
    Arch arch)
{
    EE ret = NOT_SUPPORTED;
    switch (xDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            if (0) {
#if defined(_USE_INT8) && defined(_USE_ULTRA_OPTIMIZATION)
            } else if (arch == X86_AVX512 && rnnParamSpec.mode == RNN_LSTM &&
                rnnParamSpec.num_projection == 0) {
                ret = rnncell_int8(xDesc, currentX, filterDesc, filter, biasDesc, bias, scale, state,
                    tmpBytes, tmp, rnnParamSpec, batchStrideX, batchStrideH, hDesc, output, arch);
#endif
            } else {
                ret = rnncell_fp32(xDesc, currentX, filterDesc, filter, biasDesc, bias, state,
                    tmpBytes, tmp, rnnParamSpec, batchStrideX, batchStrideH, hDesc, output, arch);
            }
            break;
        }
#endif
        default:
            break;
    }
    return ret;
}
