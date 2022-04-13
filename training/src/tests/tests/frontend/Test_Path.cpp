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

#include <training/frontend/Path.h>

using namespace raul::frontend;

namespace UT
{

TEST(TestPath, InitializationUnit)
{
    const char* fromCharArr = "char";
    std::string fromString = "str";

    EXPECT_NO_THROW(frontend::Path{ fromCharArr });
    EXPECT_NO_THROW(frontend::Path{ fromString });

    EXPECT_NO_THROW(frontend::Path("x"));
    EXPECT_NO_THROW(frontend::Path("x"s));

    EXPECT_NO_THROW(frontend::Path({ "x", "y", "z" }));
    EXPECT_NO_THROW(frontend::Path({ "x"s, "y"s, "z"s }));
}

TEST(TestPath, PathUnit)
{
    frontend::Path name{ "x" };

    auto& nameParts = name.parts();

    ASSERT_EQ(nameParts.size(), 1);
    ASSERT_EQ(name.depth(), 1);

    name /= "y";

    ASSERT_EQ(nameParts.size(), 2);
    ASSERT_EQ(name.depth(), 2);

    name /= "z";

    ASSERT_EQ(nameParts.size(), 3);
    ASSERT_EQ(name.depth(), 3);

    frontend::Path another = "x"_name / "y" / "z";
    ASSERT_EQ(another.depth(), 3);

    ASSERT_EQ(name, another);

    {
        testing::internal::CaptureStdout();
        std::cout << name.str();
        std::string output = testing::internal::GetCapturedStdout();

        ASSERT_STREQ(output.c_str(), "z");
    }

    {
        testing::internal::CaptureStdout();
        std::cout << name.fullname();
        std::string output = testing::internal::GetCapturedStdout();

        ASSERT_STREQ(output.c_str(), "x/y/z");
    }

    {
        testing::internal::CaptureStdout();
        std::cout << name;
        std::string output = testing::internal::GetCapturedStdout();

        ASSERT_STREQ(output.c_str(), "z");
    }
}

TEST(TestPath, CompareUnit)
{
    EXPECT_TRUE("x"_name == "x"_name);
    EXPECT_FALSE("x"_name == "y"_name);
    EXPECT_TRUE("x"_name != "y"_name);

    EXPECT_EQ("x"_name > "y"_name, "x"s > "y"s);
    EXPECT_EQ("x"_name < "y"_name, "x"s < "y"s);
}

TEST(TestPath, ComparePathUnit)
{
    EXPECT_TRUE(frontend::Path({ "x", "y" }) == "x"_name / "y");
    EXPECT_FALSE(frontend::Path({ "x", "y" }) == "x"_name);
    EXPECT_FALSE(frontend::Path({ "x", "y" }) == "y"_name);
    EXPECT_FALSE(frontend::Path({ "x", "y" }) == "y"_name / "x");

    EXPECT_TRUE("x"_name < "x"_name / "y");
    EXPECT_TRUE("x"_name / "y" > "x"_name);
}

} // UT namespace