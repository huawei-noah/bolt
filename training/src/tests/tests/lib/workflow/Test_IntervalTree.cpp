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

#include <training/compiler/IntervalTree.h>

namespace UT
{

TEST(TestIntervalTree, FindUnit)
{
    PROFILE_TEST

    raul::IntervalTree tree(6u);

    typedef raul::WorkflowPool<raul::MemoryManager>::Interval Interval;

    std::vector<Interval> intervals;
    intervals.push_back(Interval(15, 20, ""));
    intervals.push_back(Interval(10, 30, ""));
    intervals.push_back(Interval(17, 19, ""));
    intervals.push_back(Interval(5, 20, ""));
    intervals.push_back(Interval(12, 15, ""));
    intervals.push_back(Interval(30, 40, ""));

    for (size_t q = 0; q < intervals.size(); ++q)
    {
        tree.insert(&intervals[q]);
    }

    EXPECT_THROW(tree.insert(&intervals[0]), raul::Exception);

    Interval interval(17, 18, "");
    const Interval* found = tree.find(&interval);

    EXPECT_NE(found, nullptr);
    EXPECT_EQ(found->start, 15u);
    EXPECT_EQ(found->finish, 20u);
}

TEST(TestIntervalTree, FindAllUnit)
{
    PROFILE_TEST

    raul::IntervalTree tree(6u);

    typedef raul::WorkflowPool<raul::MemoryManager>::Interval Interval;

    std::vector<Interval> intervals;
    intervals.push_back(Interval(15, 20, ""));
    intervals.push_back(Interval(10, 30, ""));
    intervals.push_back(Interval(17, 19, ""));
    intervals.push_back(Interval(5, 20, ""));
    intervals.push_back(Interval(12, 15, ""));
    intervals.push_back(Interval(30, 40, ""));

    for (size_t q = 0; q < intervals.size(); ++q)
    {
        tree.insert(&intervals[q]);
    }

    Interval interval(17, 18, "");

    std::vector<const Interval*> foundAll = tree.findAll(&interval);

    EXPECT_EQ(foundAll.size(), 4u);

    EXPECT_EQ(foundAll[0]->start, 5u);
    EXPECT_EQ(foundAll[0]->finish, 20u);

    EXPECT_EQ(foundAll[1]->start, 10u);
    EXPECT_EQ(foundAll[1]->finish, 30u);

    EXPECT_EQ(foundAll[2]->start, 17u);
    EXPECT_EQ(foundAll[2]->finish, 19u);

    EXPECT_EQ(foundAll[3]->start, 15u);
    EXPECT_EQ(foundAll[3]->finish, 20u);
}

} // UT namespace