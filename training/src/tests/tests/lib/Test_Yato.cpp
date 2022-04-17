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

#include <yato/array_view.h>

namespace UT
{

TEST(TestYato, ReshapeUnit)
{
    PROFILE_TEST

    std::vector<int> raw = { 1, 2, 3, 4, 5, 6 };

    yato::array_view_1d<int> plain_view = yato::view(raw);
    EXPECT_EQ(6U, plain_view.size(0));
    EXPECT_EQ(6U, plain_view.total_size());

    yato::array_view_2d<int> view_2x3 = plain_view.reshape(yato::dims(2, 3));
    EXPECT_EQ(2U, view_2x3.size(0));
    EXPECT_EQ(3U, view_2x3.size(1));
    EXPECT_EQ(6U, view_2x3.total_size());
    EXPECT_EQ(1, view_2x3[0][0]);
    EXPECT_EQ(2, view_2x3[0][1]);
    EXPECT_EQ(3, view_2x3[0][2]);
    EXPECT_EQ(4, view_2x3[1][0]);
    EXPECT_EQ(5, view_2x3[1][1]);
    EXPECT_EQ(6, view_2x3[1][2]);

    yato::array_view_2d<int> view_3x2 = view_2x3.reshape(yato::dims(3, 2));
    EXPECT_EQ(3U, view_3x2.size(0));
    EXPECT_EQ(2U, view_3x2.size(1));
    EXPECT_EQ(6U, view_3x2.total_size());
    EXPECT_EQ(1, view_3x2[0][0]);
    EXPECT_EQ(2, view_3x2[0][1]);
    EXPECT_EQ(3, view_3x2[1][0]);
    EXPECT_EQ(4, view_3x2[1][1]);
    EXPECT_EQ(5, view_3x2[2][0]);
    EXPECT_EQ(6, view_3x2[2][1]);

    yato::array_view_1d<int> view_6 = view_3x2.reshape(yato::dims(6));
    EXPECT_EQ(6U, view_6.size(0));
    EXPECT_EQ(6U, view_6.total_size());
    EXPECT_EQ(1, view_6[0]);
    EXPECT_EQ(2, view_6[1]);
    EXPECT_EQ(3, view_6[2]);
    EXPECT_EQ(4, view_6[3]);
    EXPECT_EQ(5, view_6[4]);
    EXPECT_EQ(6, view_6[5]);

    yato::array_view_3d<int> view_1x2x3 = plain_view.reshape(yato::dims(1, 2, 3));
    EXPECT_EQ(1U, view_1x2x3.size(0));
    EXPECT_EQ(2U, view_1x2x3.size(1));
    EXPECT_EQ(3U, view_1x2x3.size(2));
    EXPECT_EQ(6U, view_1x2x3.total_size());
    EXPECT_EQ(1, view_1x2x3[0][0][0]);
    EXPECT_EQ(2, view_1x2x3[0][0][1]);
    EXPECT_EQ(3, view_1x2x3[0][0][2]);
    EXPECT_EQ(4, view_1x2x3[0][1][0]);
    EXPECT_EQ(5, view_1x2x3[0][1][1]);
    EXPECT_EQ(6, view_1x2x3[0][1][2]);

    yato::array_view_3d<int> view_2x3x1 = plain_view.reshape(yato::dims(2, 3, 1));
    EXPECT_EQ(2U, view_2x3x1.size(0));
    EXPECT_EQ(3U, view_2x3x1.size(1));
    EXPECT_EQ(1U, view_2x3x1.size(2));
    EXPECT_EQ(6U, view_2x3x1.total_size());
    EXPECT_EQ(1, view_2x3x1[0][0][0]);
    EXPECT_EQ(2, view_2x3x1[0][1][0]);
    EXPECT_EQ(3, view_2x3x1[0][2][0]);
    EXPECT_EQ(4, view_2x3x1[1][0][0]);
    EXPECT_EQ(5, view_2x3x1[1][1][0]);
    EXPECT_EQ(6, view_2x3x1[1][2][0]);

    yato::array_view_4d<int> view_1x1x2x3 = plain_view.reshape(yato::dims(1, 1, 2, 3));
    EXPECT_EQ(1U, view_1x1x2x3.size(0));
    EXPECT_EQ(1U, view_1x1x2x3.size(1));
    EXPECT_EQ(2U, view_1x1x2x3.size(2));
    EXPECT_EQ(3U, view_1x1x2x3.size(3));
    EXPECT_EQ(6U, view_1x1x2x3.total_size());
    EXPECT_EQ(1, view_1x1x2x3[0][0][0][0]);
    EXPECT_EQ(2, view_1x1x2x3[0][0][0][1]);
    EXPECT_EQ(3, view_1x1x2x3[0][0][0][2]);
    EXPECT_EQ(4, view_1x1x2x3[0][0][1][0]);
    EXPECT_EQ(5, view_1x1x2x3[0][0][1][1]);
    EXPECT_EQ(6, view_1x1x2x3[0][0][1][2]);

    view_1x1x2x3[0][0][1][2] = 7;

    EXPECT_EQ(7, view_2x3[1][2]);
    EXPECT_EQ(7, view_3x2[2][1]);
    EXPECT_EQ(7, view_6[5]);
    EXPECT_EQ(7, view_1x2x3[0][1][2]);
    EXPECT_EQ(7, view_2x3x1[1][2][0]);
    EXPECT_EQ(7, view_1x1x2x3[0][0][1][2]);

    EXPECT_THROW(plain_view.at(6), yato::out_of_range_error);
    EXPECT_THROW(view_2x3.at(0, 3), yato::out_of_range_error);
    EXPECT_THROW(view_2x3.at(2, 0), yato::out_of_range_error);
}

TEST(TestYato, WidthWiseViewUnit)
{
    PROFILE_TEST
    int arr[2][3][2][4] = { {
                                { { 1111, 1112, 1113, 1114 }, { 1121, 1122, 1123, 1124 } },
                                { { 1211, 1212, 1213, 1214 }, { 1221, 1222, 1223, 1224 } },
                                { { 1311, 1312, 1313, 1314 }, { 1321, 1322, 1323, 1324 } },
                            },
                            {
                                { { 2111, 2112, 2113, 2114 }, { 2121, 2122, 2123, 2124 } },
                                { { 2211, 2212, 2213, 2214 }, { 2221, 2222, 2223, 2224 } },
                                { { 2311, 2312, 2313, 2314 }, { 2321, 2322, 2323, 2324 } },
                            } };

    yato::array_view_4d<int> view(&arr[0][0][0][1], yato::dims(2, 3, 2, 2), yato::dims(3, 2, 4));

    EXPECT_EQ(1112, view[0][0][0][0]);
    EXPECT_EQ(1113, view[0][0][0][1]);
    EXPECT_EQ(1122, view[0][0][1][0]);
    EXPECT_EQ(1123, view[0][0][1][1]);
    EXPECT_EQ(1212, view[0][1][0][0]);
    EXPECT_EQ(1213, view[0][1][0][1]);
    EXPECT_EQ(1222, view[0][1][1][0]);
    EXPECT_EQ(1223, view[0][1][1][1]);
    EXPECT_EQ(1312, view[0][2][0][0]);
    EXPECT_EQ(1313, view[0][2][0][1]);
    EXPECT_EQ(1322, view[0][2][1][0]);
    EXPECT_EQ(1323, view[0][2][1][1]);
    EXPECT_EQ(2112, view[1][0][0][0]);
    EXPECT_EQ(2113, view[1][0][0][1]);
    EXPECT_EQ(2122, view[1][0][1][0]);
    EXPECT_EQ(2123, view[1][0][1][1]);
    EXPECT_EQ(2212, view[1][1][0][0]);
    EXPECT_EQ(2213, view[1][1][0][1]);
    EXPECT_EQ(2222, view[1][1][1][0]);
    EXPECT_EQ(2223, view[1][1][1][1]);
    EXPECT_EQ(2312, view[1][2][0][0]);
    EXPECT_EQ(2313, view[1][2][0][1]);
    EXPECT_EQ(2322, view[1][2][1][0]);
    EXPECT_EQ(2323, view[1][2][1][1]);
}

TEST(TestYato, WidthWiseView2Unit)
{
    PROFILE_TEST
    int arr[2 * 3 * 2 * 4] = {
        1111, 1112, 1113, 1114, 1121, 1122, 1123, 1124, 1211, 1212, 1213, 1214, 1221, 1222, 1223, 1224, 1311, 1312, 1313, 1314, 1321, 1322, 1323, 1324,
        2111, 2112, 2113, 2114, 2121, 2122, 2123, 2124, 2211, 2212, 2213, 2214, 2221, 2222, 2223, 2224, 2311, 2312, 2313, 2314, 2321, 2322, 2323, 2324,
    };

    yato::array_view_4d<int> arr4(arr, yato::dims(2, 3, 2, 4));

    yato::array_view_4d<int> view(&arr4[0][0][0][1], yato::dims(2, 3, 2, 2), yato::dims(3, 2, 4));

    EXPECT_EQ(1112, view[0][0][0][0]);
    EXPECT_EQ(1113, view[0][0][0][1]);
    EXPECT_EQ(1122, view[0][0][1][0]);
    EXPECT_EQ(1123, view[0][0][1][1]);
    EXPECT_EQ(1212, view[0][1][0][0]);
    EXPECT_EQ(1213, view[0][1][0][1]);
    EXPECT_EQ(1222, view[0][1][1][0]);
    EXPECT_EQ(1223, view[0][1][1][1]);
    EXPECT_EQ(1312, view[0][2][0][0]);
    EXPECT_EQ(1313, view[0][2][0][1]);
    EXPECT_EQ(1322, view[0][2][1][0]);
    EXPECT_EQ(1323, view[0][2][1][1]);
    EXPECT_EQ(2112, view[1][0][0][0]);
    EXPECT_EQ(2113, view[1][0][0][1]);
    EXPECT_EQ(2122, view[1][0][1][0]);
    EXPECT_EQ(2123, view[1][0][1][1]);
    EXPECT_EQ(2212, view[1][1][0][0]);
    EXPECT_EQ(2213, view[1][1][0][1]);
    EXPECT_EQ(2222, view[1][1][1][0]);
    EXPECT_EQ(2223, view[1][1][1][1]);
    EXPECT_EQ(2312, view[1][2][0][0]);
    EXPECT_EQ(2313, view[1][2][0][1]);
    EXPECT_EQ(2322, view[1][2][1][0]);
    EXPECT_EQ(2323, view[1][2][1][1]);
}

TEST(TestYato, DepthWiseViewUnit)
{
    PROFILE_TEST
    int arr[2][3][2][4] = { {
                                { { 1111, 1112, 1113, 1114 }, { 1121, 1122, 1123, 1124 } },
                                { { 1211, 1212, 1213, 1214 }, { 1221, 1222, 1223, 1224 } },
                                { { 1311, 1312, 1313, 1314 }, { 1321, 1322, 1323, 1324 } },
                            },
                            {
                                { { 2111, 2112, 2113, 2114 }, { 2121, 2122, 2123, 2124 } },
                                { { 2211, 2212, 2213, 2214 }, { 2221, 2222, 2223, 2224 } },
                                { { 2311, 2312, 2313, 2314 }, { 2321, 2322, 2323, 2324 } },
                            } };

    yato::array_view_4d<int> view(&arr[0][1][0][1], yato::dims(2, 1, 2, 2), yato::dims(3, 2, 4));

    EXPECT_EQ(1212, view[0][0][0][0]);
    EXPECT_EQ(1213, view[0][0][0][1]);
    EXPECT_EQ(1222, view[0][0][1][0]);
    EXPECT_EQ(1223, view[0][0][1][1]);
    EXPECT_EQ(2212, view[1][0][0][0]);
    EXPECT_EQ(2213, view[1][0][0][1]);
    EXPECT_EQ(2222, view[1][0][1][0]);
    EXPECT_EQ(2223, view[1][0][1][1]);
}

TEST(TestYato, DepthWise2ViewUnit)
{
    PROFILE_TEST
    int arr[2 * 3 * 2 * 4] = {
        1111, 1112, 1113, 1114, 1121, 1122, 1123, 1124, 1211, 1212, 1213, 1214, 1221, 1222, 1223, 1224, 1311, 1312, 1313, 1314, 1321, 1322, 1323, 1324,
        2111, 2112, 2113, 2114, 2121, 2122, 2123, 2124, 2211, 2212, 2213, 2214, 2221, 2222, 2223, 2224, 2311, 2312, 2313, 2314, 2321, 2322, 2323, 2324,
    };

    yato::array_view_4d<int> arr4(arr, yato::dims(2, 3, 2, 4));

    yato::array_view_4d<int> view(&arr4[0][1][0][1], yato::dims(2, 1, 2, 2), yato::dims(3, 2, 4));

    EXPECT_EQ(1212, view[0][0][0][0]);
    EXPECT_EQ(1213, view[0][0][0][1]);
    EXPECT_EQ(1222, view[0][0][1][0]);
    EXPECT_EQ(1223, view[0][0][1][1]);
    EXPECT_EQ(2212, view[1][0][0][0]);
    EXPECT_EQ(2213, view[1][0][0][1]);
    EXPECT_EQ(2222, view[1][0][1][0]);
    EXPECT_EQ(2223, view[1][0][1][1]);
}

} // UT namespace