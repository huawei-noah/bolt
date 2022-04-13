// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef RANDOM_NORM_INITIALIZER_H
#define RANDOM_NORM_INITIALIZER_H

#include "IInitializer.h"

namespace raul::initializers
{
struct RandomNormInitializer : public IInitializer
{
    explicit RandomNormInitializer(const size_t seed, const dtype mean = 0.0_dt, const dtype stddev = 1.0_dt);
    RandomNormInitializer(const dtype mean = 0.0_dt, const dtype stddev = 1.0_dt);

    void operator()(Tensor& output) override;
    void operator()(TensorFP16&) override;

    size_t getSeed() const { return mSeed; }
    dtype getMean() const { return mMean; }
    dtype getStddev() const { return mStddev; }

    void setMean(const dtype val) { mMean = val; }
    void setStddev(const dtype val) { mStddev = val; }

  private:
    std::ostream& as_ostream(std::ostream& out) const override;
    size_t mSeed;
    std::default_random_engine mGenerator;
    dtype mMean;
    dtype mStddev;
};
} // raul::initializers

#endif
