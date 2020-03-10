// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "cpu/general/tensor_computing_general.h"

template<typename T>
EE scale(T* alpha, T* beta, T* data, U32 in, U32 ic, U32 elements_per_channel)
{
    if (nullptr == alpha || nullptr == beta || nullptr == data)
        CHECK_STATUS(NULL_POINTER);

    U32 align_size = 8;
    ic = ic / align_size;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 i = 0; i < elements_per_channel; i++) {
                for (U32 k = 0; k < align_size; k++) {
                    T alphaValue = alpha[c * align_size + k];
                    T betaValue = (nullptr == beta) ? 0 : beta[c * align_size + k];
                    U32 index = ((n * ic + c) * elements_per_channel + i) * align_size + k;
                    data[index] = alphaValue * data[index] + betaValue;
                }
            }
        }
    }
    return SUCCESS;
}

EE scale_general(void *alpha, void *beta, TensorDesc inputDesc, void* data)
{
    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;    
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    U32 elements_per_channel = ih * iw;
    EE ret = SUCCESS;
    switch (idt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = scale<F32>((F32*)alpha, (F32*)beta, (F32*)data, in, ic, elements_per_channel);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = scale<F16>((F16*)alpha, (F16*)beta, (F16*)data, in, ic, elements_per_channel);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}
