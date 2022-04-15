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
#include <utility>

#include <training/base/common/Common.h>
#include <training/base/common/MemoryManager.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestDataLayer, DataLayerUnit)
{
    PROFILE_TEST
    constexpr auto batch_size = 1000U;
    constexpr auto depth = 5u;
    constexpr auto height = 4u;
    constexpr auto width = 3u;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const auto params = raul::DataParams{ { "out" }, depth, height, width };

    raul::DataLayer data("data", params, networkParameters);

    TENSORS_CREATE(batch_size);

    ASSERT_TRUE(memory_manager.tensorExists("out"));

    auto& out = memory_manager["out"];

    EXPECT_EQ(out.getBatchSize(), batch_size);
    EXPECT_EQ(out.getDepth(), depth);
    EXPECT_EQ(out.getHeight(), height);
    EXPECT_EQ(out.getWidth(), width);
}

TEST(TestDataLayer, DataLayerWithLabelsUnit)
{
    PROFILE_TEST
    constexpr auto batch_size = 1000U;
    constexpr auto depth = 5u;
    constexpr auto height = 4u;
    constexpr auto width = 3u;
    constexpr auto labelCnt = 2u;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const auto params = raul::DataParams{ { "out", "labels" }, depth, height, width, labelCnt };

    raul::DataLayer data("data", params, networkParameters);

    TENSORS_CREATE(batch_size);

    ASSERT_TRUE(memory_manager.tensorExists("out"));
    ASSERT_TRUE(memory_manager.tensorExists("labels"));

    auto& out = memory_manager["out"];

    EXPECT_EQ(out.getBatchSize(), batch_size);
    EXPECT_EQ(out.getDepth(), depth);
    EXPECT_EQ(out.getHeight(), height);
    EXPECT_EQ(out.getWidth(), width);

    auto& labels = memory_manager["labels"];

    EXPECT_EQ(labels.getBatchSize(), batch_size);
    EXPECT_EQ(labels.getDepth(), 1u);
    EXPECT_EQ(labels.getHeight(), 1u);
    EXPECT_EQ(labels.getWidth(), labelCnt);
}

TEST(TestDataLayer, DataLayerWithLabels2Unit)
{
    PROFILE_TEST
    constexpr auto batch_size = 1000U;
    constexpr auto depth = 5u;
    constexpr auto height = 4u;
    constexpr auto width = 3u;
    constexpr auto labelCnt = 2u;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const auto params1 = raul::DataParams{ { "out" }, depth, height, width };
    const auto params2 = raul::DataParams{ { "labels" }, 1u, 1u, labelCnt };

    raul::DataLayer data1("data", params1, networkParameters);
    raul::DataLayer data2("labels", params2, networkParameters);

    TENSORS_CREATE(batch_size);

    ASSERT_TRUE(memory_manager.tensorExists("out"));
    ASSERT_TRUE(memory_manager.tensorExists("labels"));

    auto& out = memory_manager["out"];

    EXPECT_EQ(out.getBatchSize(), batch_size);
    EXPECT_EQ(out.getDepth(), depth);
    EXPECT_EQ(out.getHeight(), height);
    EXPECT_EQ(out.getWidth(), width);

    auto& labels = memory_manager["labels"];

    EXPECT_EQ(labels.getBatchSize(), batch_size);
    EXPECT_EQ(labels.getDepth(), 1u);
    EXPECT_EQ(labels.getHeight(), 1u);
    EXPECT_EQ(labels.getWidth(), labelCnt);
}

TEST(TestDataLayer, DataLayerNoLabelsUnit)
{
    PROFILE_TEST
    constexpr auto batch_size = 1000U;
    constexpr auto depth = 5u;
    constexpr auto height = 4u;
    constexpr auto width = 3u;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const auto params = raul::DataParams{ { "out", "labels" }, depth, height, width };

    raul::DataLayer data("data", params, networkParameters);

    TENSORS_CREATE(batch_size);

    ASSERT_TRUE(memory_manager.tensorExists("out"));
    ASSERT_TRUE(memory_manager.tensorExists("labels"));

    auto& out = memory_manager["out"];

    EXPECT_EQ(out.getBatchSize(), batch_size);
    EXPECT_EQ(out.getDepth(), depth);
    EXPECT_EQ(out.getHeight(), height);
    EXPECT_EQ(out.getWidth(), width);

    auto& labels = memory_manager["labels"];

    EXPECT_EQ(labels.getBatchSize(), batch_size);
    EXPECT_EQ(labels.getDepth(), depth);
    EXPECT_EQ(labels.getHeight(), height);
    EXPECT_EQ(labels.getWidth(), width);
}

TEST(TestDataLayer, DataLayerOnlyLabelsUnit)
{
    PROFILE_TEST
    constexpr auto batch_size = 1000U;
    constexpr auto depth = 5u;
    constexpr auto height = 4u;
    constexpr auto width = 3u;
    constexpr auto labelCnt = 2u;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const auto params = raul::DataParams{ { "labels" }, depth, height, width, labelCnt };

    raul::DataLayer data("data", params, networkParameters);

    TENSORS_CREATE(batch_size);

    ASSERT_TRUE(memory_manager.tensorExists("labels"));

    auto& labels = memory_manager["labels"];

    EXPECT_EQ(labels.getBatchSize(), batch_size);
    EXPECT_EQ(labels.getDepth(), 1u);
    EXPECT_EQ(labels.getHeight(), 1u);
    EXPECT_EQ(labels.getWidth(), labelCnt);
}

} // UT namespace