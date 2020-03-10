// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _CONVOLUTION_H
#define _CONVOLUTION_H

#include "weight_operator.hpp"
#include "tensor_computing.h"
#include "op_type.h"

class Convolution: public WeightOperator {
public:
    Convolution(DataType dt, U32 nf,
        U32 ksizeH, U32 ksizeW, U32 kstrideH, U32 kstrideW,
        U32 kpaddingT, U32 kpaddingB, U32 kpaddingL, U32 kpaddingR,
        ActivationMode dwActiveMode, ActivationMode pwActiveMode,
        ConvolutionMode convolutionType, U32 group, U32 dilateH, U32 dilateW)
    {
        this->dt = dt;
        this->numFilters = nf;
        this->kernelSizeH = ksizeH;
        this->kernelSizeW = ksizeW;
        this->strideH = kstrideH;
        this->strideW = kstrideW;
        this->paddingT = kpaddingT;
        this->paddingB = kpaddingB;
        this->paddingL = kpaddingL;
        this->paddingR = kpaddingR;
        this->dwActiveMode = dwActiveMode;
        this->pwActiveMode = pwActiveMode;
        this->convolutionType = convolutionType;
        this->group = group;
        this->dilateH = dilateH;
        this->dilateW = dilateW;
        this->hasBias = false;
        this->pwAlg = CONVOLUTION_ALGORITHM_NULL;
        this->dwAlg = DEPTHWISE_CONVOLUTION_ALGORITHM_NULL;
    }

    OperatorType get_op_type() override
    {
        return OT_Conv;
    }

    ConvolutionDesc create_convDesc(U32 strideH, U32 strideW, U32 paddingT, U32 paddingB, U32 paddingL, U32 paddingR, U32 dilateH, U32 dilateW)
    {
        ConvolutionDesc convDesc;
        convDesc.stride_h = strideH;
        convDesc.stride_w = strideW;
        convDesc.padding_top = paddingT;
        convDesc.padding_bottom = paddingB;
        convDesc.padding_left = paddingL;
        convDesc.padding_right = paddingR;
        convDesc.dilatedRate_h = dilateH;
        convDesc.dilatedRate_w = dilateW;
        return convDesc;
    }
    virtual EE init_weight_bias_from_model(U8** modelPtr) = 0;
    virtual EE infer_forward_algorithm(HashMap<std::string, int> &algorithmMap) = 0;
    virtual EE transform_filter() = 0;
public:
    U32 numFilters;
    U32 numChannels;
    U32 kernelSizeH;
    U32 kernelSizeW;
    U32 strideH;
    U32 strideW;
    U32 paddingT;
    U32 paddingB;
    U32 paddingL;
    U32 paddingR;
    ConvolutionMode convolutionType;
    U32 group;
    U32 dilateH;
    U32 dilateW;

    ActivationMode dwActiveMode;
    ActivationMode pwActiveMode;

    ConvolutionForwardAlgorithm pwAlg;
    DepthwiseConvolutionForwardAlgorithm dwAlg;
#ifdef _USE_FP16
    std::shared_ptr<F16> scales;
#endif
};

#endif  //_CONVOLUTION_H
