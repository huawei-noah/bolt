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
#include <training/base/common/MemoryManager.h>
#include <training/base/initializers/RandomNormInitializer.h>

namespace UT
{

TEST(TestInitializerRandomNorm, StreamUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    const size_t seed = 42;
    {
        raul::initializers::RandomNormInitializer initializer{ seed };
        stream << initializer;
        ASSERT_STREQ(stream.str().c_str(), "RandomNormInitializer(mean=0.000000e+00, stddev=1.000000e+00, seed=42)");
    }
}

TEST(TestInitializerRandomNorm, InitializeUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const size_t width = 4;
    const size_t height = 3;
    const size_t depth = 2;
    const size_t batch_size = 1;
    const size_t seed = 1;
    {
        auto& output = *memory_manager.createTensor(batch_size, depth, height, width);
        raul::initializers::RandomNormInitializer initializer{ seed };
        EXPECT_NO_THROW(initializer(output));
    }
}

TEST(TestInitializerRandomNorm, InitializeBigTensorUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const raul::dtype mean = TODTYPE(1.0);
    const raul::dtype stddev = TODTYPE(5.0);
    const size_t seed = 1;
    const size_t width = 16;
    const size_t height = 16;
    const size_t depth = 3;
    const size_t batch_size = 128;
    {
        auto& output = *memory_manager.createTensor(batch_size, depth, height, width);
        raul::initializers::RandomNormInitializer initializer{ seed, mean, stddev };
        initializer(output);
        EXPECT_EQ(output.size(), batch_size * width * height * depth);
        raul::dtype average = std::accumulate(output.begin(), output.end(), 0.0_dt) / (batch_size * width * height * depth);
        printf("Average of elements is = %f\n", average);
        raul::dtype bias = 0.0_dt;
        for (auto d : output)
        {
            bias += (d - average) * (d - average);
        }
        printf("Standard deviation of elements is = %f\n", sqrt(bias / static_cast<raul::dtype>(output.size())));
    }
}

TEST(TestInitializerRandomNorm, InitializeDifferentSeedsUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const size_t width = 3;
    const size_t height = 3;
    const size_t depth = 3;
    const size_t seed1 = 1;
    const size_t seed2 = 2;
    {
        auto& output1 = *memory_manager.createTensor(1, depth, height, width);
        auto& output2 = *memory_manager.createTensor(1, depth, height, width);
        raul::initializers::RandomNormInitializer initializer1{ seed1 };
        raul::initializers::RandomNormInitializer initializer2{ seed2 };
        initializer1(output1);
        initializer2(output2);
        EXPECT_EQ(output1.size(), output2.size());
        for (size_t i = 0; i < output1.size(); i++)
        {
            EXPECT_NE(output1[i], output2[i]);
        }
    }
}

TEST(TestInitializerRandomNorm, InitializeSameSeedsUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const size_t width = 3;
    const size_t height = 3;
    const size_t depth = 3;
    const size_t seed = 1;
    {
        auto& output1 = *memory_manager.createTensor(1, depth, height, width);
        auto& output2 = *memory_manager.createTensor(1, depth, height, width);
        raul::initializers::RandomNormInitializer initializer1{ seed };
        raul::initializers::RandomNormInitializer initializer2{ seed };
        initializer1(output1);
        initializer2(output2);
        EXPECT_EQ(output1.size(), output2.size());
        for (size_t i = 0; i < output1.size(); i++)
        {
            EXPECT_EQ(output1[i], output2[i]);
        }
    }
}

} // UT namespace