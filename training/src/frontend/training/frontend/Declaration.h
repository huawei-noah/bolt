// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef FRONTEND_DECLARATION_H
#define FRONTEND_DECLARATION_H

#include <utility>

#include "Types.h"

namespace raul::frontend
{

/**
 * Base graph element declaration
 */
struct Declaration
{
    explicit Declaration(Type type)
        : type{ type }
    {
    }

    explicit Declaration(Type type, Inputs inputs, Outputs outputs)
        : type{ type }
        , inputs{ std::move(inputs) }
        , outputs{ std::move(outputs) }
    {
    }

    Declaration(const Declaration&) = delete;
    Declaration(Declaration&&) = default;

    Type type;
    PortNames inputs{};
    PortNames outputs{};
};

}

#endif // FRONTEND_DECLARATION_H
