// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _ACTIVATION_H
#define _ACTIVATION_H

#include "operator.hpp"

class Activation : public Operator {
public:
    Activation(ActivationParamSpec activationDesc)
    {
        this->activationDesc = activationDesc;
        switch (activationDesc.mode) {
            case ACTIVATION_RELU: {
                this->opt = OT_Relu;
                break;
            }
            case ACTIVATION_RELU6: {
                this->opt = OT_Relu6;
                break;
            }
            case ACTIVATION_H_SWISH: {
                this->opt = OT_HSwish;
                break;
            }
            case ACTIVATION_H_SWISH_NODIV: {
                this->opt = OT_HSwishNoDiv;
                break;
            }
            case ACTIVATION_SIGMOID: {
                this->opt = OT_Sigmoid;
                break;
            }
            case ACTIVATION_H_SIGMOID: {
                this->opt = OT_HSigmoid;
                break;
            }
            case ACTIVATION_GELU: {
                this->opt = OT_Gelu;
                break;
            }
            case ACTIVATION_TANH: {
                this->opt = OT_TanH;
                break;
            }
            case ACTIVATION_MISH: {
                this->opt = OT_Mish;
                break;
            }
            case ACTIVATION_GREATER: {
                this->opt = OT_Greater;
                break;
            }
            case ACTIVATION_EXP: {
                this->opt = OT_Exp;
                break;
            }
            case ACTIVATION_SOFTPLUS: {
                this->opt = OT_SoftPlus;
                break;
            }
            case ACTIVATION_ABS: {
                this->opt = OT_Abs;
                break;
            }
            case ACTIVATION_SIGN: {
                this->opt = OT_Sign;
                break;
            }
            default: {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        this->lenOfTemp = 0;
    }

    OperatorType get_type() override
    {
        return this->opt;
    }

    bool can_input_output_the_same() override
    {
        return true;
    }

protected:
    ActivationParamSpec activationDesc;
    OperatorType opt;
};

#endif  // _ACTIVATION_H
