// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef SCALING_STRATEGY_H
#define SCALING_STRATEGY_H

namespace raul
{

struct ScalingStrategy
{
    explicit ScalingStrategy(dtype scale)
        : mScale(scale)
        , mMin(std::numeric_limits<dtype>::min())
        , mMax(std::numeric_limits<dtype>::max())
    {
    }

    ScalingStrategy setMax(dtype max)
    {
        mMax = max;
        return *this;
    }

    ScalingStrategy setMin(dtype min)
    {
        mMin = min;
        return *this;
    }

    template<typename T>
    void scale(T& tensor)
    {
        tensor.unscale();
        tensor.scale(static_cast<typename T::type>(mScale));
    }

  private:
    dtype mScale;
    dtype mMin;
    dtype mMax;
};

}

#endif // SCALING_STRATEGY_H
