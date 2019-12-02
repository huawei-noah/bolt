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
#include "cpu/arm/tensor_computing_arm.h"
#include "cpu/general/tensor_computing_general.h"

EE lstm_transform_filter(TensorDesc filterDesc, const void* filter, TensorDesc *ftmDesc, void* filterTransformed, U32 x_dim, U32 h_dim, Arch arch)
{
    EE ret = SUCCESS;
    if (arch == ARM_A55 || arch == ARM_A76)
        ret = lstm_transform_filter_arm(filterDesc, filter, ftmDesc, filterTransformed, x_dim, h_dim);
    else
        ret = NOT_SUPPORTED;
    return ret;
}

EE lstm_transform_filter_bytes(TensorDesc filterDesc, U32* bytes, Arch arch)
{
    EE ret = SUCCESS;
    if (arch == ARM_A55 || arch == ARM_A76)
        ret = lstm_transform_filter_bytes_arm(filterDesc, bytes);
    else
        ret = NOT_SUPPORTED;
    return ret;
}

EE lstm_infer_output_size(TensorDesc inputDesc, TensorDesc filterDesc, LSTMDesc lstmDesc, TensorDesc* outputDesc, U32* outputBytes)
{
    UNUSED(filterDesc);

    if (nullptr == outputDesc || nullptr == outputBytes)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    DataType idt;
    DataFormat idf;
    U32 batch, step, x_dim;
    CHECK_STATUS_WITH_RETURN(tensor3dGet(inputDesc, &idt, &idf, &batch, &step, &x_dim));
    U32 h_dim = lstmDesc.num_output;
    *outputDesc = tensor3df(idt, idf, batch, step, h_dim);
    *outputBytes = batch * step * h_dim;
    EE ret = SUCCESS;
    switch(idt) {
        case DT_F16:
            *outputBytes *= sizeof(F16);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE lstm_infer_forward_tmp_bytes(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc, LSTMDesc lstmDesc, U32 *bytes, Arch arch)
{
    EE ret = SUCCESS;
    if (arch == ARM_A55 || arch == ARM_A76)
        ret = lstm_infer_forward_tmp_bytes_arm(inputDesc, filterDesc, outputDesc, lstmDesc, bytes);
    else
        ret = NOT_SUPPORTED;
    return ret;
}

EE lstm(TensorDesc inputDesc, const void* input, TensorDesc filterDesc, const void* filter,
    LSTMDesc lstmDesc, TensorDesc biasDesc, const void* bias, U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, void* output, Arch arch)
{
    EE ret = SUCCESS;
    switch (arch) {
        case CPU_GENERAL: {
            ret = lstm_general(inputDesc, input, filterDesc, filter, lstmDesc, biasDesc, bias, tmpBytes, tmp, outputDesc, output);
            break;
        }
        case ARM_A55: {
            ret = lstm_arm(inputDesc, input, filterDesc, filter, lstmDesc, biasDesc, bias, tmpBytes, tmp, outputDesc, output);
            break;
        }
        case ARM_A76: {
            ret = lstm_arm(inputDesc, input, filterDesc, filter, lstmDesc, biasDesc, bias, tmpBytes, tmp, outputDesc, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
    }
    return ret;
}
