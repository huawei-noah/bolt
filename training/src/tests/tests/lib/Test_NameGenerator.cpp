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

#include <training/system/NameGenerator.h>

namespace UT
{

TEST(TestNameGenerator, GeneralUnit)
{
    PROFILE_TEST
    raul::NameGenerator nameGenerator("Test_");

    EXPECT_EQ(nameGenerator.generate(), "Test_1");
    EXPECT_EQ(nameGenerator.getNext(), static_cast<size_t>(2));
    nameGenerator.reset();
    EXPECT_EQ(nameGenerator.generate(), "Test_1");
    nameGenerator.setNext(10);
    EXPECT_EQ(nameGenerator.generate(), "Test_10");
    EXPECT_EQ(nameGenerator.getNext(), static_cast<size_t>(11));

    nameGenerator.setPrefix("TT_");
    EXPECT_EQ(nameGenerator.generate(), "TT_11");
}

TEST(TestNameGenerator, RandomDefaultLengthUnit)
{
    PROFILE_TEST
    const auto default_length = DEFAULT_RANDOM_PREFIX_LENGTH;
    raul::NameGenerator nameGenerator{};
    EXPECT_EQ(nameGenerator.getPrefix().length(), default_length);
}

TEST(TestNameGenerator, RandomCustomLengthUnit)
{
    PROFILE_TEST
    const auto length = 50U;
    raul::NameGenerator nameGenerator{ length };
    EXPECT_EQ(nameGenerator.getPrefix().length(), length);
}

} // UT namespace