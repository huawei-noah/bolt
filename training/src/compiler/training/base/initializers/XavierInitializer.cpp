// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "XavierInitializer.h"
#include <math.h>

namespace raul::initializers
{
XavierUniformInitializer::XavierUniformInitializer(const size_t seed)
    : mSeed(seed)
    , mGenerator(static_cast<unsigned>(seed))
{
}

XavierUniformInitializer::XavierUniformInitializer()
    : XavierUniformInitializer(std::random_device{}())
{
}

size_t XavierUniformInitializer::calculateFactor(const Tensor& output)
{
    shape mShape = output.getShape();
    if (mShape[0] == 1 && mShape[1] == 1) return (mShape[2] + mShape[3]);
    if (mShape[0] == 1) return (mShape[1] + mShape[2]) * mShape[3];
    return (mShape[0] + mShape[1]) * mShape[2] * mShape[3];
}

void XavierUniformInitializer::operator()(Tensor& output)
{
    const auto factor = static_cast<dtype>(calculateFactor(output));
    std::uniform_real_distribution<dtype> distribution(-sqrt(6.0_dt / factor), sqrt(6.0_dt / factor));
    for (auto& d : output)
    {
        d = distribution(mGenerator);
    }
}

std::ostream& XavierUniformInitializer::as_ostream(std::ostream& out) const
{
    out << "XavierUniformInitializer(seed=" << mSeed << ")";
    return out;
}

XavierNormInitializer::XavierNormInitializer(const size_t seed)
    : mSeed(seed)
    , mGenerator(static_cast<unsigned>(seed))
{
}

XavierNormInitializer::XavierNormInitializer()
    : XavierNormInitializer(std::random_device{}())
{
}

size_t XavierNormInitializer::calculateFactor(const Tensor& output)
{
    shape mShape = output.getShape();
    if (mShape[0] == 1 && mShape[1] == 1) return (mShape[2] + mShape[3]);
    if (mShape[0] == 1) return (mShape[1] + mShape[2]) * mShape[3];
    return (mShape[0] + mShape[1]) * mShape[2] * mShape[3];
}

void XavierNormInitializer::operator()(Tensor& output)
{
    const auto factor = static_cast<dtype>(calculateFactor(output));
    std::normal_distribution<dtype> distribution(0.0_dt, sqrt(2.0_dt / factor));
    for (auto& d : output)
    {
        d = distribution(mGenerator);
    }
}

std::ostream& XavierNormInitializer::as_ostream(std::ostream& out) const
{
    out << "XavierNormInitializer(seed=" << mSeed << ")";
    return out;
}
}