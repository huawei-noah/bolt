// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>

#include <training/base/tools/DataTransformations.h>

using namespace raul;

std::unique_ptr<Tensor> Normalize::operator()(const Tensor& tsr)
{
    auto result = std::make_unique<Tensor>(tsr.getBatchSize(), tsr.getDepth(), tsr.getHeight(), tsr.getWidth());
    std::transform(tsr.begin(), tsr.end(), result->begin(), [nc = mNormalizationCoefficient](dtype v) { return v / nc; });
    return result;
}

std::unique_ptr<Tensor> BuildOneHotVector::operator()(const Tensor& tsr)
{
    auto result = std::make_unique<Tensor>(tsr.getBatchSize(), tsr.getDepth(), tsr.getHeight(), mNumberOfClasses);
    for (size_t i = 0; i < tsr.size(); ++i)
    {
        if (tsr[i] < 0_dt || tsr[i] >= static_cast<dtype>(mNumberOfClasses))
        {
            THROW_NONAME("DataTransformations", "Incorrect value for one-hot vector (less then 0 or exceeds max value - number of classes)");
        }

        if (hasFractionalPart(tsr[i]))
        {
            THROW_NONAME("DataTransformations", "Incorrect value for one-hot vector (has fractional part but should not)");
        }

        (*result)[static_cast<int32_t>(tsr[i]) + mNumberOfClasses * i] = 1_dt;
    }
    return result;
}

bool BuildOneHotVector::hasFractionalPart(dtype v)
{
    return static_cast<dtype>(static_cast<int32_t>(v)) != v;
}

std::unique_ptr<Tensor> Resize::operator()(const Tensor& tsr)
{
    auto result = std::make_unique<Tensor>(tsr.getBatchSize(), tsr.getDepth(), mNewHeight, mNewWidth);

    float ratioX = static_cast<float>(tsr.getWidth()) / static_cast<float>(result->getWidth());
    float ratioY = static_cast<float>(tsr.getHeight()) / static_cast<float>(result->getHeight());

    auto buffer4D = result->get4DView();
    auto image4D = tsr.get4DView();
    for (size_t i = 0; i < result->getBatchSize(); ++i)
    {
        for (size_t j = 0; j < result->getDepth(); ++j)
        {
            for (size_t k = 0; k < result->getHeight(); ++k)
            {
                for (size_t l = 0; l < result->getWidth(); ++l)
                {
                    const size_t oldX = static_cast<size_t>(std::floor(static_cast<dtype>(l) * ratioX));
                    const size_t oldY = static_cast<size_t>(std::floor(static_cast<dtype>(k) * ratioY));
                    buffer4D[i][j][k][l] = image4D[i][j][oldY][oldX];
                }
            }
        }
    }
    return result;
}
