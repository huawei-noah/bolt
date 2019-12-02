// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_TYPE
#define _H_TYPE

#ifdef __cplusplus
extern "C" {
#endif

#include <arm_neon.h>
    #define UNUSED(x) (void)x
    typedef enum {
        RGB_SC = 0, // scale and center crop
        RGB = 1,
        BGR = 2
    } ImageType;

    typedef enum {
        DT_U8 = 0,
        DT_I8 = 1,
        DT_U32 = 2,
        DT_I32 = 3,
        DT_F16 = 4,
        DT_F16_8Q = 5,
        DT_F32 = 6,
        DT_DOREFA = 7,
        DT_XNOR = 8,
        DT_NUM = 9
    } DataType;

    typedef unsigned char U8;
    typedef const unsigned char CU8;
    typedef char I8;
    typedef const char CI8;
    typedef int8_t INT8;
    typedef unsigned int U32;
    typedef const unsigned int CU32;
    typedef int I32;
    typedef const int CI32;
    typedef float F32;
    typedef double F64;
    typedef __fp16 F16;
    typedef unsigned char BIN8;

    inline U32 bytesOf(DataType dt) {
        U32 bytes[] = {1, 1, 4, 4, 2, 2, 4, 1, 1, 8}; // Please divide number of elements by 8 first in the case of binary data types
        return dt < DT_NUM ? bytes[dt] : 0;
    }

    typedef enum {
        Max,
        Mean,
        GLOBAL_AVG,
    } PoolingMode;
    
    typedef enum {
        CEIL,
        FLOOR,
    } RoundMode;
    
    typedef enum {
        Relu,
    } EltwiseType;
    
    typedef enum {
        ELTWISE_SUM,
        ELTWISE_MAX,
        ELTWISE_PROD
    } EltwiseMode;
    
    typedef enum {
        ACTIVATION_RELU,
        ACTIVATION_RELU6,
        ACTIVATION_H_SWISH,
        ACTIVATION_H_SIGMOID,
        ACTIVATION_SIGMOID,
        ACTIVATION_TANH,
        ACTIVATION_GELU,
        ACTIVATION_NULL,
    } ActivationMode;
    
    typedef enum {
        Convolution_Pointwise,
        Convolution_Dilation,
        Convolution_Depthwise,
        Convolution_Depthwise_Pointwise
    } ConvolutionMode;
    
    typedef enum {
        Pad_Constant,
        Pad_Reflect,
        Pad_Edge
    } PadMode;

    typedef enum {
        INT8_Q,
        NO_Q
    } QuantizationMode;

#ifdef __cplusplus
}
#endif

#endif
