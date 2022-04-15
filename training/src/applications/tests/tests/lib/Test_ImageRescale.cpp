// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <tests/tools/TestTools.h>

#include <training/tools/Datasets.h>

namespace UT
{

TEST(TestImageTest, Rescale)
{
    PROFILE_TEST
    raul::MNIST mnist;
    ASSERT_EQ(mnist.loadingData(tools::getTestAssetsDir() / "MNIST"), true);

    ASSERT_EQ(mnist.getTestImages().size() / 28 / 28, mnist.getTestImageAmount());

    raul::TensorU8 original("", 1, 1, 28, 28);
    std::copy(&mnist.getTestImages()[0], &mnist.getTestImages()[28 * 28], original.begin());

    raul::TensorU8 reshaped("", 1, 1, 4 * 56, 4 * 56);
    ASSERT_NO_THROW(raul::reshapeTensor(original, reshaped));
    raul::TensorU8 rereshaped("", 1, 1, 28, 28);
    ASSERT_NO_THROW(raul::reshapeTensor(reshaped, rereshaped));

    for (size_t i = 0; i < 28 * 28; ++i)
    {
        ASSERT_EQ(original[i], rereshaped[i]);
    }
}

} // UT namespace