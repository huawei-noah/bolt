// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef XAVIER_INITIALIZER_H
#define XAVIER_INITIALIZER_H

#include "IInitializer.h"

namespace raul::initializers
{
struct XavierUniformInitializer : public IInitializer
{
    explicit XavierUniformInitializer(const size_t seed);
    XavierUniformInitializer();
    static size_t calculateFactor(const Tensor& output);
    void operator()(Tensor& output) override;
    void operator()(TensorFP16&) override { THROW_NONAME("XavierUniformInitializer", "not implemented") }

  private:
    size_t mSeed;
    std::default_random_engine mGenerator;
    std::ostream& as_ostream(std::ostream& out) const override;
};

struct XavierNormInitializer : public IInitializer
{
    explicit XavierNormInitializer(const size_t seed);
    XavierNormInitializer();
    static size_t calculateFactor(const Tensor& output);
    void operator()(Tensor& output) override;
    void operator()(TensorFP16&) override { THROW_NONAME("XavierNormInitializer", "not implemented") }

  private:
    size_t mSeed;
    std::default_random_engine mGenerator;
    std::ostream& as_ostream(std::ostream& out) const override;
};
} // raul::initializers

#endif