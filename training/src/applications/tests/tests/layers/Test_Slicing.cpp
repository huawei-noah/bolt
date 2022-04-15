// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <tests/tools/TestTools.h>

#include <training/api/API.h>
#include <training/common/Common.h>
#include <training/common/MemoryManager.h>
#include <training/layers/basic/ConcatenationLayer.h>
#include <training/layers/basic/DataLayer.h>
#include <training/layers/basic/SlicerLayer.h>
#include <training/network/Workflow.h>

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

TEST(TestSlicing, SlicingConcatGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    size_t BATCH_SIZE = 2;
    size_t DEPTH = 3;
    size_t HEIGHT = 2;
    size_t WIDTH = 4;
    size_t SLICES[] = { 2u, 2u };
    string dirs[] = { "width", "height" };

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
        WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);
        work.getKernelManager().setExecutionPolicy(raul::KernelExecutionPolicy::DefaultParams);

        work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });

        auto dir = dirs[k];
        Names outputs;
        for (size_t n = 0; n < SLICES[k]; ++n)
        {
            outputs.push_back("slice[" + to_string(n) + "]");
        }
        work.add<SlicerLayer>("slice", SlicingParams{ "in", outputs, dir });
        work.add<ConcatenationLayer>("concat", BasicParamsWithDim{ outputs, { "out" }, dir });

        TENSORS_CREATE(BATCH_SIZE);
        MemoryManagerGPU& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

        memory_manager["in"] = TORANGE(raw);
        work.forwardPassTraining();

        for (size_t n = 0; n < SLICES[k]; ++n)
        {
            const Tensor& slice = memory_manager[outputs[n]];
            EXPECT_EQ(slice.size(), rawSliced[curRow + n].size());
            for (size_t i = 0; i < slice.size(); ++i)
            {
                EXPECT_EQ(rawSliced[curRow + n][i], slice[i]);
            }
        }

        const Tensor& out = memory_manager["out"];
        EXPECT_EQ(out.size(), raw.size());
        for (size_t i = 0; i < out.size(); ++i)
        {
            EXPECT_EQ(raw[i], out[i]);
        }

        cout << " - Slicer and Concat [" + dir + "] forward is Ok." << endl;

        memory_manager[Name("out").grad()] = TORANGE(raw);
        work.backwardPassTraining();

        for (size_t n = 0; n < SLICES[k]; ++n)
        {
            const Tensor& sliceNabla = memory_manager[outputs[n].grad()];
            EXPECT_EQ(sliceNabla.size(), rawSliced[curRow + n].size());
            for (size_t i = 0; i < sliceNabla.size(); ++i)
            {
                EXPECT_EQ(rawSliced[curRow + n][i], sliceNabla[i]);
            }
        }

        const Tensor& inNabla = memory_manager[Name("in").grad()];
        EXPECT_EQ(raw.size(), inNabla.size());
        for (size_t i = 0; i < inNabla.size(); ++i)
        {
            EXPECT_EQ(raw[i], inNabla[i]);
        }

        cout << " - Slicer and Concat [" + dir + "] backward is Ok." << endl;
    }
}

TEST(TestSlicing, DifferentSlicesGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    const size_t BATCH_SIZE = 2;

    const size_t DEPTH0 = 3;
    const size_t HEIGHT0 = 4;
    const size_t WIDTH0 = 5;

    const size_t DEPTH1 = 3;
    const size_t HEIGHT1 = 2;
    const size_t WIDTH1 = 1;

    const size_t DEPTH2 = 7;
    const size_t HEIGHT2 = 4;
    const size_t WIDTH2 = 6;

    const std::pair<dtype, dtype> range = std::make_pair(0.0_dt, 1.0_dt);

    std::unordered_map<std::string, std::array<size_t, 3>> inputSizes{ { "depth", { DEPTH0, DEPTH1, DEPTH2 } }, { "height", { HEIGHT0, HEIGHT1, HEIGHT2 } }, { "width", { WIDTH0, WIDTH1, WIDTH2 } } };
    string dirs[] = { "depth", "height", "width" };

    for (size_t i = 0; i < std::size(dirs); ++i)
    {
        WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU);

        work.add<DataLayer>("data0", DataParams{ { "in0" }, DEPTH0, HEIGHT0, WIDTH0 });
        work.add<DataLayer>("data1", DataParams{ { "in1" }, DEPTH1, HEIGHT1, WIDTH1 });
        work.add<DataLayer>("data2", DataParams{ { "in2" }, DEPTH2, HEIGHT2, WIDTH2 });

        auto dir = dirs[i];
        std::array<Names, 3> outputs;
        for (size_t j = 0; j < 3; ++j)
        {
            for (size_t n = 0; n < inputSizes[dir][j]; ++n)
            {
                outputs[j].push_back("slice_" + to_string(j) + "[" + to_string(n) + "]");
            }
            work.add<SlicerLayer>("slice" + to_string(j), SlicingParams{ "in" + to_string(j), outputs[j], dir });
            work.add<ConcatenationLayer>("concat" + to_string(j), BasicParamsWithDim{ outputs[j], { "out" + to_string(j) }, dir });
        }

        TENSORS_CREATE(BATCH_SIZE);
        MemoryManagerGPU& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

        for (size_t j = 0; j < 3; ++j)
        {
            tools::init_rand_tensor("in" + to_string(j), range, memory_manager);
            tools::init_rand_tensor(Name("out" + to_string(j)).grad(), range, memory_manager);
        }

        work.forwardPassTraining();

        for (size_t j = 0; j < 3; ++j)
        {
            const Tensor& in = memory_manager["in" + to_string(j)];
            const Tensor& out = memory_manager["out" + to_string(j)];
            EXPECT_EQ(out.size(), in.size());
            for (size_t q = 0; q < out.size(); ++q)
            {
                EXPECT_EQ(in[q], out[q]);
            }
        }

        work.backwardPassTraining();

        for (size_t j = 0; j < 3; ++j)
        {
            const Tensor& outNabla = memory_manager[Name("out" + to_string(j)).grad()];
            const Tensor& inNabla = memory_manager[Name("in" + to_string(j)).grad()];
            EXPECT_EQ(outNabla.size(), inNabla.size());
            for (size_t q = 0; q < inNabla.size(); ++q)
            {
                EXPECT_EQ(outNabla[q], inNabla[q]);
            }
        }
    }
}

struct TestLayerSlicer : public testing::TestWithParam<tuple<size_t, size_t, size_t, size_t, size_t, std::string>>
{
    const size_t BATCH = get<0>(GetParam());
    const size_t DEPTH = get<1>(GetParam());
    const size_t HEIGHT = get<2>(GetParam());
    const size_t WIDTH = get<3>(GetParam());
    const size_t SLICES = get<4>(GetParam());
    const std::string dimension = get<5>(GetParam());
    const std::pair<dtype, dtype> range = std::make_pair(0.0_dt, 1.0_dt);
};

TEST_P(TestLayerSlicer, GpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU };

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });

    Names outputs;
    for (size_t n = 0; n < SLICES; ++n)
    {
        outputs.push_back("slice[" + to_string(n) + "]");
    }
    work.add<SlicerLayer>("slice", SlicingParams{ "in", outputs, dimension });
    work.add<ConcatenationLayer>("concat", BasicParamsWithDim{ outputs, { "out" }, dimension });
    TENSORS_CREATE(BATCH);

    MemoryManagerGPU& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    tools::init_rand_tensor("in", range, memory_manager);
    tools::init_rand_tensor(Name("out").grad(), range, memory_manager);
    work.forwardPassTraining();

    const Tensor& in = memory_manager["in"];
    const Tensor& out = memory_manager["out"];
    EXPECT_EQ(out.size(), in.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_EQ(in[i], out[i]);
    }

    cout << " - Slicer and Concat [" + dimension + "] forward is Ok." << endl;

    work.backwardPassTraining();

    const Tensor& outNabla = memory_manager[Name("out").grad()];
    const Tensor& inNabla = memory_manager[Name("in").grad()];
    EXPECT_EQ(outNabla.size(), inNabla.size());
    for (size_t i = 0; i < inNabla.size(); ++i)
    {
        EXPECT_EQ(outNabla[i], inNabla[i]);
    }

    cout << " - Slicer and Concat [" + dimension + "] backward is Ok." << endl;
}

INSTANTIATE_TEST_SUITE_P(TestGpu,
                         TestLayerSlicer,
                         testing::Values(make_tuple(2, 6, 15, 8, 6, "depth"),
                                         make_tuple(2, 6, 15, 8, 3, "depth"),
                                         make_tuple(2, 6, 15, 8, 2, "depth"),
                                         make_tuple(2, 6, 15, 8, 15, "height"),
                                         make_tuple(2, 6, 15, 8, 5, "height"),
                                         make_tuple(2, 6, 15, 8, 3, "height"),
                                         make_tuple(2, 6, 15, 8, 8, "width"),
                                         make_tuple(2, 6, 15, 8, 4, "width"),
                                         make_tuple(2, 6, 15, 8, 2, "width"),
                                         make_tuple(2, 1, 1, 2048, 4, "width"),
                                         make_tuple(2, 1, 85, 20, 85, "height")));

} // UT namespace
