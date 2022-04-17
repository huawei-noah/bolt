// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "RandomNormInitializer.h"

namespace raul::initializers
{
RandomNormInitializer::RandomNormInitializer(const size_t seed, const dtype mean, const dtype stddev)
    : mSeed(seed)
    , mGenerator(static_cast<unsigned>(seed))
    , mMean(mean)
    , mStddev(stddev)
{
}

RandomNormInitializer::RandomNormInitializer(const dtype mean, const dtype stddev)
    : RandomNormInitializer(std::random_device{}(), mean, stddev)
{
}

void RandomNormInitializer::operator()(Tensor& output)
{
    std::normal_distribution<dtype> distribution(mMean, mStddev);
    for (auto& d : output)
    {
        d = distribution(mGenerator);
    }
}

void RandomNormInitializer::operator()(TensorFP16& output)
{
    std::normal_distribution<dtype> distribution(mMean, mStddev);
    for (auto& d : output)
    {
        d = toFloat16(distribution(mGenerator));
    }
}

std::ostream& RandomNormInitializer::as_ostream(std::ostream& out) const
{
    std::ios_base::fmtflags flags(out.flags());
    out << "RandomNormInitializer(mean=" << std::scientific << mMean << ", stddev=" << mStddev << ", seed=" << mSeed << ")";
    out.flags(flags);
    return out;
}
}