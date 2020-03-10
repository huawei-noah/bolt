// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_COMMON_GENERAL_H
#define _H_COMMON_GENERAL_H
#include <string.h>
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "math.h"

template<typename T>
inline EE from_nchwc8_to_nchw(TensorDesc *desc, T *data) {
    if (desc == nullptr || data == nullptr)
        CHECK_STATUS(NULL_POINTER);

    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(*desc, &idt, &idf, &in, &ic, &ih, &iw));
    if (idf != DF_NCHWC8)
        CHECK_STATUS(NOT_MATCH);

    *desc = tensor4df(idt, DF_NCHW, in, ic, ih, iw);

    T *tmp = (T *)malloc(tensorNumBytes(*desc));
    ic /= 8;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 hw = 0; hw < ih*iw; hw++) {
                for (U32 c8 = 0; c8 < 8; c8++) {
                    tmp[n*ic*8*ih*iw + (c*8 + c8)*ih*iw + hw] = data[n*ic*ih*iw*8 + c*ih*iw*8 + hw*8 + c8];
                }
            }
        }
    }
    memcpy(data, tmp, tensorNumBytes(*desc));
    free(tmp);
    return SUCCESS;
}

template<typename T>
inline EE from_nchw_to_nchwc8(TensorDesc *desc, T *data) {
    if (desc == nullptr || data == nullptr)
        CHECK_STATUS(NULL_POINTER);

    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(*desc, &idt, &idf, &in, &ic, &ih, &iw));
    if (idf != DF_NCHW)
        CHECK_STATUS(NOT_MATCH);

    *desc = tensor4df(idt, DF_NCHWC8, in, ic, ih, iw);

    T *tmp = (T *)malloc(tensorNumBytes(*desc));
    ic /= 8;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 hw = 0; hw < ih*iw; hw++) {
                for (U32 c8 = 0; c8 < 8; c8++) {
                    tmp[n*ic*ih*iw*8 + c*ih*iw*8 + hw*8 + c8] = data[n*ic*8*ih*iw + (c*8 + c8)*ih*iw + hw];
                }
            }
        }
    }
    memcpy(data, tmp, tensorNumBytes(*desc));
    free(tmp);
    return SUCCESS;
}


template<typename T>
EE activation(ActivationMode activationMode, F32 input, T* output) {
    F32 value, result;
    switch (activationMode){
        case ACTIVATION_NULL:{
            result = input;
            break;
        }
        case ACTIVATION_RELU:{
            value = input;
            if(value < 0) value = 0;
            result = value;
            break;
        }
        case ACTIVATION_RELU6:{
            value = input;
            if(value < 0) value = 0;
            if(value > 6) value = 6;
            result = value;
            break;
        }
        case ACTIVATION_H_SIGMOID:{
            value = input + 3;
            if(value < 0) value = 0;
            if(value > 6) value = 6;
            result = value / 6;
            break;
        }
        case ACTIVATION_H_SWISH:{
            value = input + 3;
            if(value < 0) value = 0;
            if(value > 6) value = 6;
            result = input * (value / 6);
            break;
        }
        case ACTIVATION_GELU:{
            value = input;
            F32 two_div_PI_sqrt = sqrt(2 / 3.14159265358979323846);
            value = two_div_PI_sqrt * (value + 0.044715 * pow(value, 3));
            value = 1.0 - 2.0 / (exp(2.0 * value) + 1.0);
            value = 0.5 * (1.0 + value);
            value = input * value;
            result = value;
            break;
        }
        case ACTIVATION_TANH:{
            value = 1.0 - 2.0 / (exp(2.0 * input) + 1.0);
            result = value;
            break;
        }
        default:
            return NOT_SUPPORTED;
    }
    *output = result;
    return SUCCESS;
}

template<typename T>
F32 array_sum(const T* array, U32 length) {
    F32 sum = 0;
    for (U32 i=0; i < length; i++)
        sum += array[i];
    return sum;
}

// array mean
template<typename T>
F32 array_mean(const T *data, I32 len) {
    if(len <= 0) return 0;
    return array_sum<T>(data, len) / len;
}

// array var
template<typename T>
F32 array_var(const T *data, I32 len, F32 mean) {
    F32 sum_s = 0;
    for(I32 i = 0; i < len; i++){
        F32 in = data[i];
        F32 tmp = in - mean;
        sum_s += tmp * tmp;
    }
    return sum_s / len;
}

#endif
