// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "RandomUniformInitializer.h"

namespace raul::initializers
{
RandomUniformInitializer::RandomUniformInitializer(const size_t seed, const dtype minval, const dtype maxval)
    : mSeed(seed)
    , mGenerator(static_cast<unsigned>(seed))
    , mMinval(minval)
    , mMaxval(maxval)
{
    if (minval >= maxval)
        THROW_NONAME("RandomUniformInitializer", "lower bound should be less than upper(minval = " + Conversions::toString(minval) + " >= " + Conversions::toString(maxval) + " = maxval)");
}

RandomUniformInitializer::RandomUniformInitializer(const dtype minval, const dtype maxval)
    : RandomUniformInitializer(std::random_device{}(), minval, maxval)
{
}

void RandomUniformInitializer::operator()(Tensor& output)
{
    std::uniform_real_distribution<dtype> distribution(mMinval, mMaxval);
    for (auto& d : output)
    {
        d = distribution(mGenerator);
    }
}

void RandomUniformInitializer::operator()(TensorFP16& output)
{
    std::uniform_real_distribution<dtype> distribution(mMinval, mMaxval);
    for (auto& d : output)
    {
        d = TOHTYPE(distribution(mGenerator));
    }
}

std::ostream& RandomUniformInitializer::as_ostream(std::ostream& out) const
{
    std::ios_base::fmtflags flags(out.flags());
    out << "RandomUniformInitializer(minval=" << std::scientific << mMinval << ", maxval=" << mMaxval << std::defaultfloat << ", seed=" << mSeed << ")";
    out.flags(flags);
    return out;
}
}