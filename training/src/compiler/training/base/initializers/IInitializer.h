// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef IINITIALIZER_H
#define IINITIALIZER_H

#include <iostream>
#include <random>
#include <training/base/common/Common.h>
#include <training/base/common/Tensor.h>

namespace raul::initializers
{
/**
 * @brief Initializer interface
 */
struct IInitializer
{
    IInitializer() = default;
    virtual ~IInitializer() = default;

    IInitializer(const IInitializer& other) = delete;
    IInitializer(const IInitializer&& other) = delete;
    IInitializer& operator=(const IInitializer& other) = delete;
    IInitializer& operator=(const IInitializer&& other) = delete;

    virtual void operator()(Tensor& output) = 0;
    virtual void operator()(TensorFP16& output) = 0;
    friend std::ostream& operator<<(std::ostream& out, const IInitializer& instance) { return instance.as_ostream(out); }

  private:
    virtual std::ostream& as_ostream(std::ostream& out) const = 0;
};
} // raul::initializers

#endif