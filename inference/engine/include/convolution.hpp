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

class Convolution : public WeightOperator {
public:
    Convolution(DataType dt,
        ConvolutionParamSpec p,
        ActivationParamSpec dwActivationParamSpec,
        ActivationParamSpec pwActivationParamSpec)
    {
        this->dt = dt;
        this->p = p;
        this->dwActivationParamSpec = dwActivationParamSpec;
        this->pwActivationParamSpec = pwActivationParamSpec;
        this->hasBias = false;
        this->pwAlg = CONVOLUTION_ALGORITHM_NULL;
        this->dwAlg = DEPTHWISE_CONVOLUTION_ALGORITHM_NULL;
    }

    OperatorType get_type() override
    {
        return OT_Conv;
    }

    TensorDesc desc_process(TensorDesc inputDesc)
    {
        TensorDesc resultDesc;
        if (tensorIs3d(inputDesc)) {
            DataType idt;
            DataFormat idf;
            U32 in, ic, ih;
            CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
            resultDesc = tensor4df(idt, idf, in, ic, ih, 1);
        } else {
            resultDesc = inputDesc;
        }
        return resultDesc;
    }

public:
    ConvolutionParamSpec p;
    ActivationParamSpec dwActivationParamSpec;
    ActivationParamSpec pwActivationParamSpec;

    ConvolutionForwardAlgorithm pwAlg;
    DepthwiseConvolutionForwardAlgorithm dwAlg;
#ifdef _USE_FP16
    std::shared_ptr<F16> scales;
#endif
};

#endif  // _CONVOLUTION_H
