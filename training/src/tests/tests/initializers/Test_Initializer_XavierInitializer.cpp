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
#include <training/base/initializers/XavierInitializer.h>

namespace UT
{

TEST(TestInitializerXavierUniform, StreamUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    const size_t seed = 42;
    {
        raul::initializers::XavierUniformInitializer initializer{ seed };
        stream << initializer;
        ASSERT_STREQ(stream.str().c_str(), "XavierUniformInitializer(seed=42)");
    }
}

TEST(TestInitializerXavierNorm, StreamUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    const size_t seed = 42;
    {
        raul::initializers::XavierNormInitializer initializer{ seed };
        stream << initializer;
        ASSERT_STREQ(stream.str().c_str(), "XavierNormInitializer(seed=42)");
    }
}

TEST(TestInitializerXavierNorm, CalculateFactorUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const size_t dim11 = 1;
    const size_t dim12 = 1;
    const size_t dim13 = 3;
    const size_t dim14 = 4;
    const size_t torch_factor1 = dim13 + dim14;

    const size_t dim21 = 1;
    const size_t dim22 = 2;
    const size_t dim23 = 3;
    const size_t dim24 = 4;
    const size_t torch_factor2 = (dim22 + dim23) * dim24;

    const size_t dim31 = 2;
    const size_t dim32 = 3;
    const size_t dim33 = 4;
    const size_t dim34 = 5;
    const size_t torch_factor3 = (dim31 + dim32) * dim33 * dim34;
    {
        auto& output1 = *memory_manager.createTensor(dim11, dim12, dim13, dim14);
        auto& output2 = *memory_manager.createTensor(dim21, dim22, dim23, dim24);
        auto& output3 = *memory_manager.createTensor(dim31, dim32, dim33, dim34);
        EXPECT_EQ(raul::initializers::XavierNormInitializer::calculateFactor(output1), torch_factor1);
        EXPECT_EQ(raul::initializers::XavierNormInitializer::calculateFactor(output2), torch_factor2);
        EXPECT_EQ(raul::initializers::XavierNormInitializer::calculateFactor(output3), torch_factor3);
    }
}

TEST(TestInitializerXavierUniform, InitializeUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const size_t width = 3;
    const size_t height = 3;
    const size_t depth = 3;
    const size_t batch_size = 1;
    const raul::dtype minval = TODTYPE(-sqrt(6.0_dt / ((depth + height) * width)));
    const raul::dtype maxval = TODTYPE(sqrt(6.0_dt / ((depth + height) * width)));
    {
        auto& output = *memory_manager.createTensor(batch_size, depth, height, width);
        raul::initializers::XavierUniformInitializer initializer;
        initializer(output);

        EXPECT_EQ(output.size(), width * height * depth);
        for (raul::dtype d : output)
        {
            EXPECT_TRUE(d >= minval);
            EXPECT_TRUE(d < maxval);
        }
    }
}

TEST(TestInitializerXavierNorm, InitializeUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const size_t width = 3;
    const size_t height = 3;
    const size_t depth = 3;
    const size_t batch_size = 1;
    const size_t seed = 1;
    {
        auto& output = *memory_manager.createTensor(batch_size, depth, height, width);
        raul::initializers::XavierNormInitializer initializer{ seed };
        EXPECT_NO_THROW(initializer(output));
    }
}

TEST(TestInitializerXavierNorm, InitializeBigTensorUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const size_t seed = 1;
    const size_t width = 16;
    const size_t height = 16;
    const size_t depth = 3;
    const size_t batch_size = 128;
    {
        auto& output = *memory_manager.createTensor(batch_size, depth, height, width);
        raul::initializers::XavierNormInitializer initializer{ seed };
        initializer(output);
        EXPECT_EQ(output.size(), batch_size * width * height * depth);
        raul::dtype average = std::accumulate(output.begin(), output.end(), 0.0_dt) / static_cast<raul::dtype>(output.size());
        raul::dtype stddev = 0.0_dt;
        printf("Average of elements is = %f\n", average);
        for (auto d : output)
        {
            stddev += (d - average) * (d - average);
        }
        printf("Dispersion of elements is = %f\n", stddev / static_cast<raul::dtype>(output.size()));
    }
}

TEST(TestInitializerXavierNorm, InitializeDifferentSeedsUnit)
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
        raul::initializers::XavierNormInitializer initializer1{ seed1 };
        raul::initializers::XavierNormInitializer initializer2{ seed2 };
        initializer1(output1);
        initializer2(output2);
        EXPECT_EQ(output1.size(), output2.size());
        for (size_t i = 0; i < output1.size(); i++)
        {
            EXPECT_NE(output1[i], output2[i]);
        }
    }
}

TEST(TestInitializerXavierNorm, InitializeSameSeedsUnit)
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
        raul::initializers::XavierNormInitializer initializer1{ seed };
        raul::initializers::XavierNormInitializer initializer2{ seed };
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