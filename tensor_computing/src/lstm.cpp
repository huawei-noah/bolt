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

EE lstm_transform_filter(TensorDesc filterDesc, const void* filter,
    LSTMDesc lstmDesc,
    TensorDesc *ftmDesc, void* filterTransformed,
    Arch arch)
{
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL || arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
#if defined(_USE_GENERAL) || defined(_USE_NEON)
        ret = lstm_transform_filter_arm(filterDesc, filter, lstmDesc, ftmDesc, filterTransformed);
#endif
    }
    return ret;
}

EE lstm_transform_filter_bytes(TensorDesc filterDesc, LSTMDesc lstmDesc, U32* bytes, Arch arch)
{
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL || arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
#if defined(_USE_GENERAL) || defined(_USE_NEON)
        ret = lstm_transform_filter_bytes_arm(filterDesc, lstmDesc, bytes);
#endif
    }
    return ret;
}

EE lstm_infer_output_size(TensorDesc inputDesc, TensorDesc filterDesc,
    LSTMDesc lstmDesc,
    TensorDesc* outputDesc, U32* outputBytes)
{
    UNUSED(filterDesc);

    if (nullptr == outputDesc || nullptr == outputBytes)
        CHECK_STATUS(NULL_POINTER);
    DataType idt;
    DataFormat idf;
    U32 batch, step, xDim;
    CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &batch, &step, &xDim));
    U32 num = (lstmDesc.biDirection) ? 2 : 1;
    U32 hDim = num * lstmDesc.numOutput;
    *outputDesc = tensor3df(idt, idf, batch, step, hDim);
    *outputBytes = batch * step * hDim * bytesOf(idt);
    return SUCCESS;
}

EE lstm_infer_forward_tmp_bytes(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    LSTMDesc lstmDesc,
    U32 *bytes, Arch arch)
{
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL || arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
#if defined(_USE_GENERAL) || defined(_USE_NEON)
        ret = lstm_infer_forward_tmp_bytes_arm(inputDesc, filterDesc, outputDesc, lstmDesc, bytes, arch);
#endif
    }
    return ret;
}

EE lstm(TensorDesc inputDesc, const void* input,
    TensorDesc filterDesc, const void* filter,
    TensorDesc biasDesc, const void* bias,
    U32 tmpBytes, void* tmp,
    LSTMDesc lstmDesc,
    TensorDesc outputDesc, void* output,
    Arch arch)
{
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL) {
#ifdef _USE_GENERAL
        ret = lstm_general(inputDesc, input,
                           filterDesc, filter,
                           biasDesc, bias,
                           tmpBytes, tmp,
                           lstmDesc,
                           outputDesc, output);
#endif
#ifdef _USE_NEON
    } else if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
        ret = lstm_arm(inputDesc, input,
                       filterDesc, filter,
                       biasDesc, bias,
                       tmpBytes, tmp,
                       lstmDesc,
                       outputDesc, output,
                       arch);
#endif
    }
    return ret;
}

EE lstmcell_infer_output_size(TensorDesc inputDesc, TensorDesc filterDesc,
    LSTMDesc lstmDesc,
    TensorDesc* outputDesc, U32* outputBytes)
{
    UNUSED(filterDesc);

    if (nullptr == outputDesc || nullptr == outputBytes)
        CHECK_STATUS(NULL_POINTER);
    DataType idt;
    DataFormat idf;
    U32 batch, xDim;
    CHECK_STATUS(tensor2dfGet(inputDesc, &idt, &idf, &batch, &xDim));
    U32 hDim = lstmDesc.numOutput;
    *outputDesc = tensor2df(idt, idf, batch, hDim);
    *outputBytes = batch * hDim * bytesOf(idt);
    return SUCCESS;
}

EE lstmcell_infer_forward_tmp_bytes(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    LSTMDesc lstmDesc,
    U32 *bytes, Arch arch)
{
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL || arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
#if defined(_USE_GENERAL) || defined(_USE_NEON)
        ret = lstmcell_infer_forward_tmp_bytes_arm(inputDesc, filterDesc, outputDesc, lstmDesc, bytes, arch);
#endif
    }
    return ret;
}

EE lstmcell(TensorDesc xDesc, const void* currentX,
    TensorDesc filterDesc, const void* filter,
    TensorDesc biasDesc, const void* bias,
    void *state,
    U32 tmpBytes, void *tmp,
    LSTMDesc lstmDesc, U32 batchStrideX, U32 batchStrideH,
    TensorDesc hDesc, void* currentH,
    Arch arch)
{
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL) {
#ifdef _USE_GENERAL
        ret = lstmcell_general(xDesc, currentX,
                               filterDesc, filter,
                               biasDesc, bias,
                               state,
                               tmpBytes, tmp,
                               lstmDesc, batchStrideX, batchStrideH,
                               hDesc, currentH);
#endif
#ifdef _USE_NEON
    } else if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
        ret = lstmcell_arm(xDesc, currentX,
                           filterDesc, filter,
                           biasDesc, bias,
                           state,
                           tmpBytes, tmp,
                           lstmDesc, batchStrideX, batchStrideH,
                           hDesc, currentH,
                           arch);
#endif
    }
    return ret;
}
