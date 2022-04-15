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

#include <training/base/common/Common.h>
#include <training/base/common/MemoryManager.h>
#include <training/base/layers/basic/ConcatenationLayer.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/SlicerLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

using namespace raul;
using namespace std;

TEST(TestSlicing, SlicingConcatUnit)
{
    PROFILE_TEST

    size_t BATCH_SIZE = 2;
    size_t DEPTH = 3;
    size_t HEIGHT = 2;
    size_t WIDTH = 4;
    size_t SLICES[] = { 2u, 2u, 3u };
    string dirs[] = { "width", "height", "depth" };

    Tensor raw = {
        1111._dt, 1112._dt, 1113._dt, 1114._dt, 1121._dt, 1122._dt, 1123._dt, 1124._dt, 1211._dt, 1212._dt, 1213._dt, 1214._dt, 1221._dt, 1222._dt, 1223._dt, 1224._dt,
        1311._dt, 1312._dt, 1313._dt, 1314._dt, 1321._dt, 1322._dt, 1323._dt, 1324._dt, 2111._dt, 2112._dt, 2113._dt, 2114._dt, 2121._dt, 2122._dt, 2123._dt, 2124._dt,
        2211._dt, 2212._dt, 2213._dt, 2214._dt, 2221._dt, 2222._dt, 2223._dt, 2224._dt, 2311._dt, 2312._dt, 2313._dt, 2314._dt, 2321._dt, 2322._dt, 2323._dt, 2324._dt,
    };

    Tensor rawSliced[] = {
        { 1111._dt, 1112._dt, 1121._dt, 1122._dt, 1211._dt, 1212._dt, 1221._dt, 1222._dt, 1311._dt, 1312._dt, 1321._dt, 1322._dt,
          2111._dt, 2112._dt, 2121._dt, 2122._dt, 2211._dt, 2212._dt, 2221._dt, 2222._dt, 2311._dt, 2312._dt, 2321._dt, 2322._dt },

        { 1113._dt, 1114._dt, 1123._dt, 1124._dt, 1213._dt, 1214._dt, 1223._dt, 1224._dt, 1313._dt, 1314._dt, 1323._dt, 1324._dt,
          2113._dt, 2114._dt, 2123._dt, 2124._dt, 2213._dt, 2214._dt, 2223._dt, 2224._dt, 2313._dt, 2314._dt, 2323._dt, 2324._dt },

        { 1111._dt, 1112._dt, 1113._dt, 1114._dt, 1211._dt, 1212._dt, 1213._dt, 1214._dt, 1311._dt, 1312._dt, 1313._dt, 1314._dt,
          2111._dt, 2112._dt, 2113._dt, 2114._dt, 2211._dt, 2212._dt, 2213._dt, 2214._dt, 2311._dt, 2312._dt, 2313._dt, 2314._dt },

        { 1121._dt, 1122._dt, 1123._dt, 1124._dt, 1221._dt, 1222._dt, 1223._dt, 1224._dt, 1321._dt, 1322._dt, 1323._dt, 1324._dt,
          2121._dt, 2122._dt, 2123._dt, 2124._dt, 2221._dt, 2222._dt, 2223._dt, 2224._dt, 2321._dt, 2322._dt, 2323._dt, 2324._dt },

        { 1111._dt, 1112._dt, 1113._dt, 1114._dt, 1121._dt, 1122._dt, 1123._dt, 1124._dt, 2111._dt, 2112._dt, 2113._dt, 2114._dt, 2121._dt, 2122._dt, 2123._dt, 2124._dt },
        { 1211._dt, 1212._dt, 1213._dt, 1214._dt, 1221._dt, 1222._dt, 1223._dt, 1224._dt, 2211._dt, 2212._dt, 2213._dt, 2214._dt, 2221._dt, 2222._dt, 2223._dt, 2224._dt },
        { 1311._dt, 1312._dt, 1313._dt, 1314._dt, 1321._dt, 1322._dt, 1323._dt, 1324._dt, 2311._dt, 2312._dt, 2313._dt, 2314._dt, 2321._dt, 2322._dt, 2323._dt, 2324._dt },

    };

    for (size_t k = 0, curRow = 0; k < size(dirs); curRow += SLICES[k], ++k)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });

        auto dir = dirs[k];
        Names outputs;
        for (size_t n = 0; n < SLICES[k]; ++n)
        {
            outputs.push_back("slice[" + to_string(n) + "]");
        }
        SlicerLayer slicer("slicing", { "in", outputs, dir }, networkParameters);
        ConcatenationLayer concat("concat", { outputs, { "out" }, dir }, networkParameters);
        TENSORS_CREATE(BATCH_SIZE);
        memory_manager["in"] = TORANGE(raw);
        slicer.forwardCompute(NetworkMode::Train);

        for (size_t n = 0; n < SLICES[k]; ++n)
        {
            auto& slice = memory_manager.getTensor(outputs[n]);
            EXPECT_EQ(slice.size(), rawSliced[curRow + n].size());
            for (size_t i = 0; i < slice.size(); ++i)
            {
                EXPECT_EQ(rawSliced[curRow + n][i], slice[i]);
            }
        }

        cout << " - Slicer [" + dir + "] forward is Ok." << endl;

        concat.forwardCompute(NetworkMode::Train);

        auto& out = memory_manager.getTensor("out");
        EXPECT_EQ(out.size(), raw.size());
        for (size_t i = 0; i < out.size(); ++i)
        {
            EXPECT_EQ(raw[i], out[i]);
        }

        cout << " - Concat [" + dir + "] forward is Ok." << endl;

        memory_manager[Name("out").grad()].memAllocate(nullptr);
        memory_manager[Name("out").grad()] = TORANGE(raw);
        concat.backwardCompute();

        for (size_t n = 0; n < SLICES[k]; ++n)
        {
            auto& sliceMabla = memory_manager.getTensor(outputs[n].grad());
            EXPECT_EQ(sliceMabla.size(), rawSliced[curRow + n].size());
            for (size_t i = 0; i < sliceMabla.size(); ++i)
            {
                EXPECT_EQ(rawSliced[curRow + n][i], sliceMabla[i]);
            }
        }

        cout << " - Concat [" + dir + "] backward is Ok." << endl;

        slicer.backwardCompute();

        auto& inNabla = memory_manager.getTensor(Name("in").grad());
        EXPECT_EQ(raw.size(), inNabla.size());
        for (size_t i = 0; i < inNabla.size(); ++i)
        {
            EXPECT_EQ(raw[i], inNabla[i]);
        }

        cout << " - Slicer [" + dir + "] backward is Ok." << endl;

        memory_manager.clear();
    }
}

TEST(TestSlicing, SlicingSizeSplits1Unit)
{
    PROFILE_TEST

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    size_t BATCH_SIZE = 2;
    size_t DEPTH = 1;
    size_t HEIGHT = 1;
    size_t WIDTH = 6;
    vector<int> slices = { 1, 1, 1 };
    vector<size_t> realSlices = { 1, 1, 1 };
    vector<vector<dtype>> realValues = { { 11._dt, 21._dt }, { 12._dt, 22._dt }, { 13._dt, 23._dt } };

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    SlicerLayer slicer("slicing", { "in", { "out[0]", "out[1]", "out[2]" }, "width", slices }, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = { 11._dt, 12._dt, 13._dt, 14._dt, 15._dt, 16._dt, 21._dt, 22._dt, 23._dt, 24._dt, 25._dt, 26._dt };
    slicer.forwardCompute(NetworkMode::Train);
    for (size_t q = 0; q < realSlices.size(); ++q)
    {
        auto str = "out[" + to_string(q) + "]";
        ASSERT_TRUE(memory_manager.tensorExists(str));
        auto& out = memory_manager[str];
        ASSERT_EQ(out.getBatchSize(), BATCH_SIZE);
        ASSERT_EQ(out.getDepth(), DEPTH);
        ASSERT_EQ(out.getHeight(), HEIGHT);
        ASSERT_EQ(out.getWidth(), realSlices[q]);

        ASSERT_EQ(out.size(), realValues[q].size());
        for (size_t i = 0; i < out.size(); ++i)
        {
            ASSERT_EQ(out[i], realValues[q][i]);
        }
    }
}

TEST(TestSlicing, SlicingConcatSizeSplits2Unit)
{
    PROFILE_TEST

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    size_t BATCH_SIZE = 2;
    size_t DEPTH = 1;
    size_t HEIGHT = 1;
    size_t WIDTH = 6;
    vector<int> slices = { -1, 4 };
    vector<size_t> realSlices = { 2, 4 };
    vector<vector<dtype>> realValues = { { 11._dt, 12._dt, 21._dt, 22._dt }, { 13._dt, 14._dt, 15._dt, 16._dt, 23._dt, 24._dt, 25._dt, 26._dt } };

    Tensor raw({ 11._dt, 12._dt, 13._dt, 14._dt, 15._dt, 16._dt, 21._dt, 22._dt, 23._dt, 24._dt, 25._dt, 26._dt });

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    SlicerLayer slicer("slicing", { "in", { "out[0]", "out[1]" }, "width", slices }, networkParameters);
    ConcatenationLayer concat("concat", { { "out[0]", "out[1]" }, { "concat" }, "width" }, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = TORANGE(raw);
    slicer.forwardCompute(NetworkMode::Train);
    for (size_t q = 0; q < realSlices.size(); ++q)
    {
        auto str = "out[" + to_string(q) + "]";
        ASSERT_TRUE(memory_manager.tensorExists(str));
        auto& out = memory_manager[str];
        EXPECT_EQ(out.getBatchSize(), BATCH_SIZE);
        EXPECT_EQ(out.getDepth(), DEPTH);
        EXPECT_EQ(out.getHeight(), HEIGHT);
        EXPECT_EQ(out.getWidth(), realSlices[q]);

        ASSERT_EQ(out.size(), realValues[q].size());
        for (size_t i = 0; i < out.size(); ++i)
        {
            EXPECT_EQ(out[i], realValues[q][i]);
        }
    }

    concat.forwardCompute(NetworkMode::Train);
    const auto& c = memory_manager["concat"];
    EXPECT_EQ(c.getShape(), memory_manager["in"].getShape());
    for (size_t i = 0; i < c.size(); ++i)
    {
        EXPECT_EQ(c[i], raw[i]);
    }

    memory_manager[Name("concat").grad()].memAllocate(nullptr);
    memory_manager[Name("concat").grad()] = TORANGE(raw);
    concat.backwardCompute();
    slicer.backwardCompute();

    ASSERT_TRUE(memory_manager.tensorExists(Name("in").grad()));

    auto& in_nabla = memory_manager[Name("in").grad()];

    EXPECT_EQ(in_nabla.getShape(), memory_manager["in"].getShape());
    for (size_t i = 0; i < c.size(); ++i)
    {
        ASSERT_EQ(in_nabla[i], raw[i]);
    }
}

TEST(TestSlicing, SlicingSizeSplits3Unit)
{
    PROFILE_TEST

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    size_t BATCH_SIZE = 2;
    size_t DEPTH = 1;
    size_t HEIGHT = 1;
    size_t WIDTH = 6;
    vector<int> slices = { 2, -1 };
    vector<size_t> realSlices = { 2, 4 };
    vector<vector<dtype>> realValues = { { 11._dt, 12._dt, 21._dt, 22._dt }, { 13._dt, 14._dt, 15._dt, 16._dt, 23._dt, 24._dt, 25._dt, 26._dt } };

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    SlicerLayer slicer("slicing", { "in", { "out[0]", "out[1]" }, "width", slices }, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = { 11._dt, 12._dt, 13._dt, 14._dt, 15._dt, 16._dt, 21._dt, 22._dt, 23._dt, 24._dt, 25._dt, 26._dt };
    slicer.forwardCompute(NetworkMode::Train);
    for (size_t q = 0; q < realSlices.size(); ++q)
    {
        auto str = "out[" + to_string(q) + "]";
        ASSERT_TRUE(memory_manager.tensorExists(str));
        auto& out = memory_manager[str];
        ASSERT_EQ(out.getBatchSize(), BATCH_SIZE);
        ASSERT_EQ(out.getDepth(), DEPTH);
        ASSERT_EQ(out.getHeight(), HEIGHT);
        ASSERT_EQ(out.getWidth(), realSlices[q]);

        ASSERT_EQ(out.size(), realValues[q].size());
        for (size_t i = 0; i < out.size(); ++i)
        {
            ASSERT_EQ(out[i], realValues[q][i]);
        }
    }
}

TEST(TestSlicing, SlicingSizeSplits4Unit)
{
    PROFILE_TEST

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    size_t BATCH_SIZE = 2;
    size_t DEPTH = 1;
    size_t HEIGHT = 1;
    size_t WIDTH = 6;
    vector<int> slices = { 2, -1, 2 };
    vector<size_t> realSlices = { 2, 2, 2 };
    vector<vector<dtype>> realValues = { { 11._dt, 12._dt, 21._dt, 22._dt }, { 13._dt, 14._dt, 23._dt, 24._dt }, { 15._dt, 16._dt, 25._dt, 26._dt } };

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    SlicerLayer slicer("slicing", { "in", { "out[0]", "out[1]", "out[2]" }, "width", slices }, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = { 11._dt, 12._dt, 13._dt, 14._dt, 15._dt, 16._dt, 21._dt, 22._dt, 23._dt, 24._dt, 25._dt, 26._dt };
    slicer.forwardCompute(NetworkMode::Train);
    for (size_t q = 0; q < realSlices.size(); ++q)
    {
        auto str = "out[" + to_string(q) + "]";
        ASSERT_TRUE(memory_manager.tensorExists(str));
        auto& out = memory_manager[str];
        ASSERT_EQ(out.getBatchSize(), BATCH_SIZE);
        ASSERT_EQ(out.getDepth(), DEPTH);
        ASSERT_EQ(out.getHeight(), HEIGHT);
        ASSERT_EQ(out.getWidth(), realSlices[q]);

        ASSERT_EQ(out.size(), realValues[q].size());
        for (size_t i = 0; i < out.size(); ++i)
        {
            ASSERT_EQ(out[i], realValues[q][i]);
        }
    }
}

TEST(TestSlicing, SlicingSizeSplits5Unit)
{
    PROFILE_TEST

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    size_t BATCH_SIZE = 2;
    size_t DEPTH = 4;
    size_t HEIGHT = 1;
    size_t WIDTH = 2;
    vector<int> slices = { -1, 1 };
    vector<size_t> realSlices = { 3, 1 };
    vector<vector<dtype>> realValues = { { 111._dt, 112._dt, 121._dt, 122._dt, 131._dt, 132._dt, 211._dt, 212._dt, 221._dt, 222._dt, 231._dt, 232._dt }, { 141._dt, 142._dt, 241._dt, 242._dt } };

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    SlicerLayer slicer("slicing", { "in", { "out[0]", "out[1]" }, "depth", slices }, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = { 111_dt, 112_dt, 121_dt, 122_dt, 131_dt, 132_dt, 141_dt, 142_dt, 211_dt, 212_dt, 221_dt, 222_dt, 231_dt, 232_dt, 241_dt, 242_dt };
    slicer.forwardCompute(NetworkMode::Train);
    for (size_t q = 0; q < realSlices.size(); ++q)
    {
        auto str = "out[" + to_string(q) + "]";
        ASSERT_TRUE(memory_manager.tensorExists(str));
        auto& out = memory_manager[str];
        ASSERT_EQ(out.getBatchSize(), BATCH_SIZE);
        ASSERT_EQ(out.getDepth(), realSlices[q]);
        ASSERT_EQ(out.getHeight(), HEIGHT);
        ASSERT_EQ(out.getWidth(), WIDTH);

        ASSERT_EQ(out.size(), realValues[q].size());
        for (size_t i = 0; i < out.size(); ++i)
        {
            ASSERT_EQ(out[i], realValues[q][i]);
        }
    }
}

TEST(TestSlicing, SkipGradientUnit)
{
    PROFILE_TEST

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    size_t BATCH_SIZE = 2;
    size_t DEPTH = 1;
    size_t HEIGHT = 2;
    size_t WIDTH = 2;
    vector<int> slices = { 1, -1 };

    vector<dtype> realGradient = {
        0._dt, 1._dt, 0._dt, 2._dt, 0._dt, 3._dt, 0._dt, 4._dt,
    };

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    SlicerLayer slicer("slicing", { "in", { "out[0]", "out[1]" }, "width", slices }, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = { 111._dt, 112._dt, 121._dt, 122._dt, 211._dt, 212._dt, 221._dt, 222._dt };
    slicer.forwardCompute(NetworkMode::Train);

    memory_manager[Name("out[1]").grad()].memAllocate(nullptr);
    memory_manager[Name("out[1]").grad()] = { 1._dt, 2._dt, 3._dt, 4._dt };
    slicer.backwardCompute();

    ASSERT_TRUE(memory_manager.tensorExists(Name("in").grad()));

    auto& in_nabla = memory_manager[Name("in").grad()];

    ASSERT_EQ(memory_manager["in"].getShape(), in_nabla.getShape());

    for (size_t q = 0; q < realGradient.size(); ++q)
    {
        ASSERT_EQ(in_nabla[q], realGradient[q]);
    }
}

TEST(TestSlicing, SkipGradient2Unit)
{
    PROFILE_TEST

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    size_t BATCH_SIZE = 2;
    size_t DEPTH = 1;
    size_t HEIGHT = 2;
    size_t WIDTH = 2;
    vector<int> slices = { 1, -1 };

    vector<dtype> realGradient = {
        0._dt, 1._dt, 0._dt, 2._dt, 0._dt, 3._dt, 0._dt, 4._dt,
    };

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    SlicerLayer slicer("slicing", { "in", { "out[0]", "out[1]" }, "width", slices }, networkParameters);
    ConcatenationLayer concat("concat", { { "out[0]", "out[1]" }, { "out" }, "width" }, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = { 111._dt, 112._dt, 121._dt, 122._dt, 211._dt, 212._dt, 221._dt, 222._dt };

    slicer.forwardCompute(NetworkMode::Train);
    concat.forwardCompute(NetworkMode::Train);

    memory_manager[Name("out").grad()].memAllocate(nullptr);
    memory_manager[Name("out").grad()] = { 1_dt, 1_dt, 2_dt, 2_dt, 3_dt, 3_dt, 4_dt, 4_dt };

    concat.backwardCompute();

    // EXPECT_FALSE(memory_manager.tensorExists(Name("out[0]").grad()));
    memory_manager[Name("out[0]").grad()] = 0_dt;
    ASSERT_TRUE(memory_manager.tensorExists(Name("out[1]").grad()));

    slicer.backwardCompute();

    ASSERT_TRUE(memory_manager.tensorExists(Name("in").grad()));

    auto& in_nabla = memory_manager[Name("in").grad()];

    ASSERT_EQ(memory_manager["in"].getShape(), in_nabla.getShape());

    for (size_t q = 0; q < realGradient.size(); ++q)
    {
        ASSERT_EQ(in_nabla[q], realGradient[q]);
    }
}

TEST(TestSlicing, ExtractHSliceUnit)
{
    PROFILE_TEST

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    size_t BATCH_SIZE = 2;
    size_t DEPTH = 1;
    size_t HEIGHT = 2;
    size_t WIDTH = 3;
    vector<int> slices = { 1 };
    vector<size_t> realSlices = { 1 };
    vector<vector<dtype>> realValues = { { 111._dt, 112._dt, 113._dt, 211._dt, 212._dt, 213._dt } };

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    SlicerLayer slicer("slicing", { "in", { "out" }, "height", slices }, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = { 111_dt, 112_dt, 113_dt, 121_dt, 122_dt, 123_dt, 211_dt, 212_dt, 213_dt, 221_dt, 222_dt, 223_dt };
    slicer.forwardCompute(NetworkMode::Train);
    for (size_t q = 0; q < realSlices.size(); ++q)
    {
        string str = "out";
        ASSERT_TRUE(memory_manager.tensorExists(str));
        auto& out = memory_manager[str];
        ASSERT_EQ(out.getBatchSize(), BATCH_SIZE);
        ASSERT_EQ(out.getDepth(), DEPTH);
        ASSERT_EQ(out.getHeight(), realSlices[q]);
        ASSERT_EQ(out.getWidth(), WIDTH);

        ASSERT_EQ(out.size(), realValues[q].size());
        for (size_t i = 0; i < out.size(); ++i)
        {
            ASSERT_EQ(out[i], realValues[q][i]);
        }
    }
}

TEST(TestSlicing, SingleSliceUnit)
{
    PROFILE_TEST

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    size_t BATCH_SIZE = 2;
    size_t DEPTH = 1;
    size_t HEIGHT = 2;
    size_t WIDTH = 3;
    vector<int> slices = { -1 };
    vector<size_t> realSlices = { 2 };

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    SlicerLayer slicer("slicing", { "in", { "out" }, "height", slices }, networkParameters);

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = { 111_dt, 112_dt, 113_dt, 121_dt, 122_dt, 123_dt, 211_dt, 212_dt, 213_dt, 221_dt, 222_dt, 223_dt };
    slicer.forwardCompute(NetworkMode::Train);
    for (size_t q = 0; q < realSlices.size(); ++q)
    {
        string str = "out";
        ASSERT_TRUE(memory_manager.tensorExists(str));
        auto& out = memory_manager[str];
        ASSERT_EQ(out.getBatchSize(), BATCH_SIZE);
        ASSERT_EQ(out.getDepth(), DEPTH);
        ASSERT_EQ(out.getHeight(), realSlices[q]);
        ASSERT_EQ(out.getWidth(), WIDTH);

        ASSERT_EQ(out.size(), memory_manager["in"].size());
        for (size_t i = 0; i < out.size(); ++i)
        {
            ASSERT_EQ(out[i], memory_manager["in"][i]);
        }
    }
}

} // UT namespace