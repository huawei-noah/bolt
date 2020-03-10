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

    #include <arm_neon.h>
    #define UNUSED(x) (void)x
    #define UNI_MIN(a,b) (((a)<(b))?(a):(b))
    #define UNI_MAX(a,b) (((a)>(b))?(a):(b))
    #define UNI_F16_MIN -65504.0f
    #define UNI_F16_MAX 65504.0f
    #define NAME_LEN 128

    #include <math.h>
#ifdef __clang__
    #define UNI_ISNAN(a) isnan((a))
    #define UNI_ISINF(a) isinf((a))
#else
    #define UNI_ISNAN(a) std::isnan((a))
    #define UNI_ISINF(a) std::isinf((a))
#endif

#ifdef __cplusplus
extern "C" {
#endif

    typedef enum {
        RGB_SC = 0, // scale and center crop
        RGB = 1,
        BGR = 2,
        RGB_RAW = 3,
        RGB_SC_RAW = 4,
        BGR_SC_RAW = 5
    } ImageFormat;

    typedef enum {
        DT_U8 = 0,
        DT_I8 = 1,
        DT_U32 = 2,
        DT_I32 = 3,
        DT_F16 = 4,
        DT_F16_8Q = 5,
        DT_F32 = 6,
        DT_BIN01 = 7,
        DT_BIN11 = 8,
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
    typedef long I64;
#ifdef _USE_FP16
    typedef __fp16 F16;
#endif
    typedef unsigned char BIN8;

    inline U32 bytesOf(DataType dt) {
        U32 bytes[] = {1, 1, 4, 4, 2, 2, 4, 1, 1, 8};  // Please divide number of elements by 8 first in the case of binary data types
        return dt < DT_NUM ? bytes[dt] : 0;
    }

    typedef enum {
        POOLING_MAX,
        POOLING_MEAN
    } PoolingMode;
    
    typedef enum {
        CEIL,
        FLOOR
    } RoundMode;
    
    typedef enum {
        Relu
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
        ACTIVATION_NULL
    } ActivationMode;

    typedef enum{
      BSliceApply_CONV,
      BSliceApply_NULL
    } BilateralSliceApplyMode;
    
    typedef enum {
        Convolution_Pointwise,
        Convolution_Dilation,
        Convolution_Depthwise,
        Convolution_Depthwise_Pointwise,
        Convolution_Deconvolution
    } ConvolutionMode;
    
    typedef enum {
        Pad_Constant,
        Pad_Reflect,
        Pad_Edge
    } PadMode;

    typedef enum {
        FP16,
        INT8_Q,
        FP32
    } InferencePrecision;

    typedef enum {
        CHECK_EQUAL
    } CheckMode;

#ifdef __cplusplus
}
#endif

#endif
