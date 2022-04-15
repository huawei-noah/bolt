// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <training/common/DataLoader.h>

#include <tests/tools/TestTools.h>

#include <tests/tools/TestTools.h>
namespace UT
{

TEST(TestDataLoader, BuildOneHotVector)
{
    PROFILE_TEST
    raul::DataLoader dataLoader;
    auto& res = dataLoader.buildOneHotVector({ 1, 2, 3 }, 10);

    ASSERT_EQ(res.size(), 30u);
    ASSERT_EQ(dataLoader.getMemoryManager().size(), 1u);
    ASSERT_EQ(res[1], 1.0_dt);
    ASSERT_EQ(res[12], 1.0_dt);
    ASSERT_EQ(res[23], 1.0_dt);
}

} // UT namespace