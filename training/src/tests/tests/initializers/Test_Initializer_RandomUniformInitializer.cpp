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
#include <training/base/initializers/RandomUniformInitializer.h>

namespace UT
{

TEST(TestInitializerRandomUniform, StreamUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    const raul::dtype minval = TODTYPE(0.0);
    const raul::dtype maxval = TODTYPE(1.0);
    const size_t seed = 42;
    {
        raul::initializers::RandomUniformInitializer initializer{ seed, minval, maxval };
        stream << initializer;
        ASSERT_STREQ(stream.str().c_str(), "RandomUniformInitializer(minval=0.000000e+00, maxval=1.000000e+00, seed=42)");
    }
}

TEST(TestInitializerRandomUniform, AssertIncorrectBoundariesUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    const raul::dtype minval = TODTYPE(1.0);
    const raul::dtype maxval = TODTYPE(0.0);
    {
        ASSERT_THROW(raul::initializers::RandomUniformInitializer(minval, maxval), raul::Exception);
    }
}

TEST(TestInitializerRandomUniform, InitializeUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const raul::dtype minval = TODTYPE(0.0);
    const raul::dtype maxval = TODTYPE(1.0);
    const size_t seed = 1;
    const size_t width = 3;
    const size_t height = 3;
    const size_t depth = 3;
    {
        auto& output = *memory_manager.createTensor(1, depth, height, width);
        raul::initializers::RandomUniformInitializer initializer{ seed, minval, maxval };
        initializer(output);
        EXPECT_EQ(output.size(), width * height * depth);
        for (raul::dtype d : output)
        {
            EXPECT_TRUE(d >= minval);
            EXPECT_TRUE(d < maxval);
        }
    }
}

TEST(TestInitializerRandomUniform, InitializeBigTensorUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const raul::dtype minval = TODTYPE(0.0);
    const raul::dtype maxval = TODTYPE(1.0);
    const size_t seed = 1;
    const size_t width = 16;
    const size_t height = 16;
    const size_t depth = 3;
    const size_t batch_size = 128;
    {
        auto& output = *memory_manager.createTensor(batch_size, depth, height, width);
        raul::initializers::RandomUniformInitializer initializer{ seed, minval, maxval };
        initializer(output);
        EXPECT_EQ(output.size(), batch_size * width * height * depth);
        raul::dtype average = std::accumulate(output.begin(), output.end(), 0.0_dt) / (batch_size * width * height * depth);
        printf("Average of elements is = %f\n", average);
    }
}

TEST(TestInitializerRandomUniform, InitializeDifferentSeedsUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const raul::dtype minval = TODTYPE(0.0);
    const raul::dtype maxval = TODTYPE(1.0);
    const size_t width = 3;
    const size_t height = 3;
    const size_t depth = 3;
    const size_t seed1 = 1;
    const size_t seed2 = 2;
    {
        auto& output1 = *memory_manager.createTensor(1, depth, height, width);
        auto& output2 = *memory_manager.createTensor(1, depth, height, width);
        raul::initializers::RandomUniformInitializer initializer1{ seed1, minval, maxval };
        raul::initializers::RandomUniformInitializer initializer2{ seed2, minval, maxval };
        initializer1(output1);
        initializer2(output2);
        EXPECT_EQ(output1.size(), output2.size());
        for (size_t i = 0; i < output1.size(); i++)
        {
            EXPECT_NE(output1[i], output2[i]);
        }
    }
}

TEST(TestInitializerRandomUniform, InitializeSameSeedsUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const raul::dtype minval = TODTYPE(0.0);
    const raul::dtype maxval = TODTYPE(1.0);
    const size_t width = 3;
    const size_t height = 3;
    const size_t depth = 3;
    const size_t seed = 1;
    {
        auto& output1 = *memory_manager.createTensor(1, depth, height, width);
        auto& output2 = *memory_manager.createTensor(1, depth, height, width);
        raul::initializers::RandomUniformInitializer initializer1{ seed, minval, maxval };
        raul::initializers::RandomUniformInitializer initializer2{ seed, minval, maxval };
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