// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef FRONTEND_LAYERS_H
#define FRONTEND_LAYERS_H

#include <utility>

#include "Declaration.h"
#include "Generator.h"

namespace raul::frontend
{

struct LinearDeclaration : Declaration
{
    explicit LinearDeclaration(size_t features)
        : Declaration(Type::Linear, Inputs{ "in" }, Outputs{ "out" })
        , features{ features }
    {
    }
    size_t features;
    bool bias = false;
};

struct Linear : GeneratorTyped<LinearDeclaration>
{
    explicit Linear(size_t features)
        : GeneratorTyped(features)
    {
    }

    Linear enableBias()
    {
        getDeclaration()->bias = true;
        return *this;
    }

    Linear disableBias()
    {
        getDeclaration()->bias = false;
        return *this;
    }
};

struct ReLUDeclaration : Declaration
{
    explicit ReLUDeclaration()
        : Declaration(Type::ReLU, Inputs{ "in" }, Outputs{ "out" })
    {
    }
};

struct ReLU : GeneratorTyped<ReLUDeclaration>
{
    explicit ReLU()
        : GeneratorTyped()
    {
    }
};

struct Conv1dDeclaration : Declaration
{
    explicit Conv1dDeclaration()
        : Declaration(Type::Conv1d, Inputs{ "in" }, Outputs{ "out" })
    {
    }
};

struct Conv1d : GeneratorTyped<Conv1dDeclaration>
{
    explicit Conv1d()
        : GeneratorTyped()
    {
    }
};

struct DropoutDeclaration : Declaration
{
    explicit DropoutDeclaration()
        : Declaration(Type::Dropout, Inputs{ "in" }, Outputs{ "out" })
    {
    }
};

struct Dropout : GeneratorTyped<DropoutDeclaration>
{
    explicit Dropout()
        : GeneratorTyped()
    {
    }
};

struct NormDeclaration : Declaration
{
    explicit NormDeclaration()
        : Declaration(Type::Norm, Inputs{ "in" }, Outputs{ "out" })
    {
    }
};

struct Norm : GeneratorTyped<NormDeclaration>
{
    explicit Norm()
        : GeneratorTyped()
    {
    }
};

struct SumDeclaration : Declaration
{
    explicit SumDeclaration()
        : Declaration(Type::Sum, Inputs{ "in" }, Outputs{ "out" })
    {
    }
};

struct Sum : GeneratorTyped<SumDeclaration>
{
    explicit Sum()
        : GeneratorTyped()
    {
    }
};

struct SigmoidDeclaration : Declaration
{
    explicit SigmoidDeclaration()
        : Declaration(Type::Sigmoid, Inputs{ "in" }, Outputs{ "out" })
    {
    }
};

struct Sigmoid : GeneratorTyped<SigmoidDeclaration>
{
    explicit Sigmoid()
        : GeneratorTyped()
    {
    }
};

struct TanhDeclaration : Declaration
{
    explicit TanhDeclaration()
        : Declaration(Type::Tanh, Inputs{ "in" }, Outputs{ "out" })
    {
    }
};

struct Tanh : GeneratorTyped<TanhDeclaration>
{
    explicit Tanh()
        : GeneratorTyped()
    {
    }
};

struct SoftmaxDeclaration : Declaration
{
    explicit SoftmaxDeclaration()
        : Declaration(Type::Softmax, Inputs{ "in" }, Outputs{ "out" })
    {
    }
};

struct Softmax : GeneratorTyped<SoftmaxDeclaration>
{
    explicit Softmax()
        : GeneratorTyped()
    {
    }
};

struct ReshapeDeclaration : Declaration
{
    ReshapeDeclaration(std::initializer_list<int> shape)
        : Declaration(Type::Reshape, Inputs{ "in" }, Outputs{ "out" })
        , shape{ shape }
    {
        if (shape.size() == 0)
        {
            throw std::invalid_argument("shape has to contain at least one dimension");
        }
    }

    std::vector<int> shape;
};

struct Reshape : GeneratorTyped<ReshapeDeclaration>
{
    Reshape(std::initializer_list<int> shape)
        : GeneratorTyped(shape)
    {
    }
};

struct LambdaDeclaration : Declaration
{
    using Func = std::function<void(const Inputs, Outputs)>;
    using Forward = Func;
    using Backward = Func;

    explicit LambdaDeclaration(Forward forward, Backward backward)
        : Declaration(Type::Lambda, Inputs{ "in" }, Outputs{ "out" })
        , forward{ std::move(forward) }
        , backward{ std::move(backward) }
    {
    }
    Forward forward;
    Backward backward;
};

struct Lambda : GeneratorTyped<LambdaDeclaration>
{
    explicit Lambda(const LambdaDeclaration::Forward& forward, const LambdaDeclaration::Backward& backward)
        : GeneratorTyped(forward, backward)
    {
    }
};

} // namespace raul::frontend

#endif // FRONTEND_LAYERS_H
