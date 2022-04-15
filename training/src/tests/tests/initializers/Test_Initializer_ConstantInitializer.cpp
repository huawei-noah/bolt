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
#include <training/base/initializers/ConstantInitializer.h>

namespace UT
{

TEST(TestInitializerConstantInitializer, StreamUnit)
{
    PROFILE_TEST
    std::ostringstream stream;
    const raul::dtype value = TODTYPE(0.0);
    {
        raul::initializers::ConstantInitializer initializer{ value };
        stream << initializer;
        ASSERT_STREQ(stream.str().c_str(), "ConstantInitializer(value=0)");
    }
}

TEST(TestInitializerConstantInitializer, InitializeZerosUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const size_t batch_size = 1;
    const size_t width = 2;
    const size_t height = 3;
    const size_t depth = 4;
    const raul::dtype value = TODTYPE(0.0);
    {
        auto& output = *memory_manager.createTensor(batch_size, depth, height, width);
        raul::initializers::ConstantInitializer initializer{ value };
        initializer(output);

        for (raul::dtype d : output)
        {
            EXPECT_EQ(d, 0.0_dt);
        }
    }
}

TEST(TestInitializerConstantInitializer, InitializeOnesUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    const size_t batch_size = 1;
    const size_t width = 2;
    const size_t height = 3;
    const size_t depth = 4;
    const raul::dtype value = TODTYPE(1.0);
    {
        auto& output = *memory_manager.createTensor(batch_size, depth, height, width);
        raul::initializers::ConstantInitializer initializer{ value };
        initializer(output);

        for (raul::dtype d : output)
        {
            EXPECT_EQ(d, 1.0_dt);
        }
    }
}

}