// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef RANDOM_TENSOR_LAYER_PARAMS_H
#define RANDOM_TENSOR_LAYER_PARAMS_H

#include "BasicParameters.h"
#include <string>
#include <training/base/common/Random.h>
#include <vector>

namespace raul
{
/**
 * @param output name of output tensor
 * @param initializer specify the distribution to use
 */
struct RandomTensorLayerParams : public BasicParams
{
    RandomTensorLayerParams() = delete;
    RandomTensorLayerParams(const raul::Name& output,
                            size_t depth = 1,
                            size_t height = 1,
                            size_t width = 1,
                            raul::dtype mean = 0.0_dt,
                            raul::dtype stddev = 1.0_dt,
                            size_t seed = random::getGlobalSeed())
        : BasicParams(Names(), Names(1, output))
        , mDepth(depth)
        , mHeight(height)
        , mWidth(width)
        , mMean(mean)
        , mStdDev(stddev)
        , mSeed(seed)
    {
    }

    // Shape
    size_t mDepth;
    size_t mHeight;
    size_t mWidth;

    // Distribution params
    raul::dtype mMean;
    raul::dtype mStdDev;
    size_t mSeed;

    void print(std::ostream& stream) const override;
};

}

#endif // raul namespace
