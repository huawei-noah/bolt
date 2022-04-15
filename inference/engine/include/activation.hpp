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
        std::map<ActivationMode, OperatorType> activationMap = {{ACTIVATION_RELU, OT_Relu},
            {ACTIVATION_RELU6, OT_Relu6}, {ACTIVATION_H_SWISH, OT_HSwish},
            {ACTIVATION_H_SWISH_NODIV, OT_HSwishNoDiv}, {ACTIVATION_SIGMOID, OT_Sigmoid},
            {ACTIVATION_H_SIGMOID, OT_HSigmoid}, {ACTIVATION_GELU, OT_Gelu},
            {ACTIVATION_TANH, OT_TanH}, {ACTIVATION_MISH, OT_Mish}, {ACTIVATION_GREATER, OT_Greater},
            {ACTIVATION_EXP, OT_Exp}, {ACTIVATION_SOFTPLUS, OT_SoftPlus}, {ACTIVATION_ABS, OT_Abs},
            {ACTIVATION_SIGN, OT_Sign}, {ACTIVATION_NOT, OT_Not}, {ACTIVATION_LOG, OT_Log},
            {ACTIVATION_NEG, OT_Neg}};
        if (activationMap.find(activationDesc.mode) == activationMap.end()) {
            UNI_ERROR_LOG("can not map ActivationMode to OperatorType.\n");
        } else {
            this->opt = activationMap[activationDesc.mode];
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
