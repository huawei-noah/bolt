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

EE pooling_x86(TensorDesc inputDesc,
    const void *input,
    PoolingParamSpec poolingParamSpec,
    void *scale,
    TensorDesc outputDesc,
    void *output,
    void *idx)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            UNUSED(scale);
            if (inputDesc.df == DF_NCHWC8) {
                ret = pooling_fp32(
                    inputDesc, (const F32 *)input, poolingParamSpec, outputDesc, (F32 *)output, (I32 *)idx);
            } else if (inputDesc.df == DF_NCHWC16) {
                ret = pooling_c16_fp32(
                    inputDesc, (const F32 *)input, poolingParamSpec, outputDesc, (F32 *)output);
            } else if (inputDesc.df == DF_NCHW || inputDesc.df == DF_MTK || inputDesc.df == DF_NORMAL) {
                ret = pooling_nchw_fp32(
                    inputDesc, (const F32 *)input, poolingParamSpec, outputDesc, (F32 *)output, (I32 *)idx);
            } else {
                ret = NOT_SUPPORTED;
            }
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_U8_Q: {
            ret = pooling_uint8(inputDesc, (const UINT8 *)input, poolingParamSpec,
                outputDesc, (UINT8 *)output, scale);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE pooling_bp_x86(TensorDesc inputDesc,
    const void *input,
    PoolingParamSpec poolingParamSpec,
    TensorDesc outputDesc,
    void *output)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = pooling_bp_fp32(
                inputDesc, (const F32 *)input, poolingParamSpec, outputDesc, (F32 *)output);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
