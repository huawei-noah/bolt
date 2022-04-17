// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef FRONTEND_PROCESSOR_H
#define FRONTEND_PROCESSOR_H

#include <training/frontend/Types.h>

namespace raul::frontend
{

struct LinearDeclaration;
struct ReLUDeclaration;
struct SigmoidDeclaration;
struct TanhDeclaration;
struct SoftmaxDeclaration;
struct ReshapeDeclaration;
struct DropoutDeclaration;
struct LambdaDeclaration;
struct GraphDeclaration;
struct Processor
{
    virtual void process(const ReLUDeclaration&, const std::optional<Path>) {}
    virtual void process(const SigmoidDeclaration&, const std::optional<Path>) {}
    virtual void process(const TanhDeclaration&, const std::optional<Path>) {}
    virtual void process(const SoftmaxDeclaration&, const std::optional<Path>) {}
    virtual void process(const ReshapeDeclaration&, const std::optional<Path>) {}
    virtual void process(const DropoutDeclaration&, const std::optional<Path>) {}
    virtual void process(const LinearDeclaration&, const std::optional<Path>) {}
    virtual void process(const LambdaDeclaration&, const std::optional<Path>) {}
    virtual void process(const GraphDeclaration&, const std::optional<Path>);

    virtual void process(const Declaration&, const std::optional<Path>) {}
};

}

#endif // FRONTEND_PROCESSOR_H
