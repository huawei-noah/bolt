// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef FRONTEND_TYPES_H
#define FRONTEND_TYPES_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Path.h"
#include "Port.h"

namespace raul::frontend
{

using Name = std::string;

template<class T>
using Ref = std::shared_ptr<T>;

struct Processor;
using Handler = std::function<void(Processor&, std::optional<Path>)>;

using PortNames = std::vector<Name>;
using Inputs = PortNames;
using Outputs = PortNames;

enum class Type
{
    Graph,
    // Trainable
    Linear,
    Conv1d,
    // Activations
    ReLU,
    Tanh,
    Sigmoid,
    Softmax,
    // Other
    Dropout,
    Sum,
    Norm,
    Reshape,
    Lambda
};

#define PRINT_TYPE_STR(TYPE) #TYPE
#define PRINT_TYPE(TYPE)                                                                                                                                                                               \
    case Type::TYPE:                                                                                                                                                                                   \
        out << PRINT_TYPE_STR(TYPE);                                                                                                                                                                   \
        break

inline std::ostream& operator<<(std::ostream& out, const Type& type)
{
    switch (type)
    {
        PRINT_TYPE(Graph);
        PRINT_TYPE(Linear);
        PRINT_TYPE(ReLU);
        PRINT_TYPE(Sigmoid);
        PRINT_TYPE(Tanh);
        PRINT_TYPE(Softmax);
        PRINT_TYPE(Conv1d);
        PRINT_TYPE(Dropout);
        PRINT_TYPE(Norm);
        PRINT_TYPE(Sum);
        PRINT_TYPE(Reshape);
        PRINT_TYPE(Lambda);
    }

    return out;
}

}

#endif // FRONTEND_TYPES_H
