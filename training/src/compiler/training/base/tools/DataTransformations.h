// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef DATATRANSFORMATIONS_H
#define DATATRANSFORMATIONS_H

#include <fstream>
#include <training/base/common/Tensor.h>

namespace raul
{

class Transform
{
  public:
    virtual std::unique_ptr<Tensor> operator()(const Tensor&) = 0;

    virtual ~Transform() = default;
};

class Normalize : public Transform
{
  public:
    Normalize(dtype coefficient)
        : mNormalizationCoefficient(coefficient)
    {
    }

    std::unique_ptr<Tensor> operator()(const Tensor& tensor) final;
 ~Normalize(){}
  private:
    dtype mNormalizationCoefficient;
};

class BuildOneHotVector : public Transform
{
  public:
    explicit BuildOneHotVector(size_t numberOfClasses)
        : mNumberOfClasses(numberOfClasses)
    {
    }

    std::unique_ptr<Tensor> operator()(const Tensor& tensor) final;
~BuildOneHotVector(){}
  private:
    bool hasFractionalPart(dtype v);

  private:
    size_t mNumberOfClasses;
};

class Resize : public Transform
{
  public:
    Resize(size_t newHeight, size_t newWidth)
        : mNewHeight(newHeight)
        , mNewWidth(newWidth)
    {
    }

    std::unique_ptr<Tensor> operator()(const Tensor&) final;
~Resize(){}
  private:
    size_t mNewHeight;
    size_t mNewWidth;
};

} // !namespace raul

#endif // DATATRANSFORMATIONS_H
