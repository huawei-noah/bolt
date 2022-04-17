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

#include <unordered_set>

#include <training/base/tools/ElementSequence.h>

using namespace raul;

struct TestElementSequenceRandomized : public testing::Test
{
};

TEST_F(TestElementSequenceRandomized, ShouldReturnNumbersInRandomOrderFromRangeRangeWithoutRepetitionsUnit)
{
    PROFILE_TEST
    RandomSequence rs(0, 60000u / 16u);

    std::unordered_set<uint32_t> generatedNumbers;
    for (size_t i = 0; i < 60000u / 16u + 1; ++i)
    {
        uint32_t rv = rs.getElement();
        ASSERT_TRUE(rv <= 60000u / 16u);
        auto [_, isNewOne] = generatedNumbers.insert(rv);
        ASSERT_TRUE(isNewOne);
    }
    ASSERT_TRUE(generatedNumbers.find(0) != generatedNumbers.end());
    ASSERT_TRUE(generatedNumbers.find(60000u / 16u) != generatedNumbers.end());
}

TEST_F(TestElementSequenceRandomized, ShouldGenerateTwoDifferentSequencesUnit)
{
    PROFILE_TEST
    RandomSequence rs(0, 60000u / 16u);

    std::vector<uint32_t> s1;
    for (size_t i = 0; i < 60000u / 16u + 1; ++i)
    {
        s1.push_back(rs.getElement());
    }
    std::vector<uint32_t> s2;
    for (size_t i = 0; i < 60000u / 16u + 1; ++i)
    {
        s2.push_back(rs.getElement());
    }

    ASSERT_FALSE(std::is_sorted(s1.begin(), s1.end()));
    ASSERT_FALSE(std::is_sorted(s2.begin(), s2.end()));
    ASSERT_FALSE(std::equal(s1.begin(), s1.end(), s2.begin()));
}

struct TestElementSequenceMonotonic : public testing::Test
{
};

TEST_F(TestElementSequenceMonotonic, ShouldReturnNumbersInMonotonicIncreasingOrderWithStepOneUnit)
{
    PROFILE_TEST
    std::vector<uint32_t> ref(60000u / 16u + 1, uint32_t());
    std::generate(ref.begin(), ref.end(), [n = 0]() mutable { return n++; });

    MonotonicSequence ms(0, 60000u / 16u);
    std::vector<uint32_t> s1;
    for (size_t i = 0; i < 60000u / 16u + 1; ++i)
    {
        s1.push_back(ms.getElement());
    }
    std::vector<uint32_t> s2;
    for (size_t i = 0; i < 60000u / 16u + 1; ++i)
    {
        s2.push_back(ms.getElement());
    }

    ASSERT_TRUE(std::equal(s1.begin(), s1.end(), ref.begin()));
    ASSERT_TRUE(std::equal(s2.begin(), s2.end(), ref.begin()));
}