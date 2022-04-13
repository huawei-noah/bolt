// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>
#include <iostream>

#include <training/base/common/Common.h>
#include <training/base/common/MemoryManager.h>
#include <training/base/optimizers/SGD.h>

namespace UT
{

TEST(TestOptimizerSGD, StreamUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    auto learning_rate = TODTYPE(0.01);
    {
        raul::optimizers::SGD optimizer{ learning_rate };
        stream << optimizer;
        ASSERT_STREQ(stream.str().c_str(), "SGD(lr=1.000000e-02)");
    }
}

TEST(TestOptimizerSGD, AssertLRNegativeUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    {
        auto learning_rate = TODTYPE(-0.5);
        ASSERT_THROW(raul::optimizers::SGD{ learning_rate }, raul::Exception);
    }
}

TEST(TestOptimizerSGD, AssertLRZeroUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    {
        auto learning_rate = TODTYPE(.0);
        ASSERT_THROW(raul::optimizers::SGD{ learning_rate }, raul::Exception);
    }
}

TEST(TestOptimizerSGD, OptimizerStepUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto learning_rate = TODTYPE(0.1);
    auto amount_of_element = 10U;
    {
        raul::optimizers::SGD optimizer{ learning_rate };
        auto& params = *memory_manager.createTensor("params", 1, amount_of_element, 1, 1, 1.0_dt);
        auto& gradients = *memory_manager.createTensor("gradients", 1, amount_of_element, 1, 1, 1.0_dt);
        optimizer(memory_manager, params, gradients);

        EXPECT_EQ(params.size(), amount_of_element);
        EXPECT_EQ(gradients.size(), amount_of_element);
        for (raul::dtype d : params)
            EXPECT_EQ(d, 1.0_dt - learning_rate);
    }
}

} // UT namespace