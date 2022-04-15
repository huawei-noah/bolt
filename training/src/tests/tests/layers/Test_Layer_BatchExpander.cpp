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

#include <training/base/common/MemoryManager.h>
#include <training/base/layers/basic/BatchExpanderLayer.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/TensorLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{
using namespace raul;

TEST(TestLayerBatchExpander, IncorrectInputSizeUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto params = ViewParams{ { "x" }, { "out" } };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<TensorLayer>("input", TensorParams{ { "x" }, 2, 1, 1, 1 });
    BatchExpanderLayer expander("expander", params, networkParameters);
    ASSERT_THROW(expander.forwardCompute(NetworkMode::Test), raul::Exception);
}

TEST(TestLayerBatchExpander, IncorrectInputShapeUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto params = ViewParams{ { "x" }, { "out" } };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<DataLayer>("input", DataParams{ { "x" }, 1, 1, 1 });
    ASSERT_THROW(BatchExpanderLayer expander("expander", params, networkParameters), raul::Exception);
}

TEST(TestLayerBatchExpander, ForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batchSize = 3;
    const auto depth = 4;
    const auto height = 5;
    const auto width = 6;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<TensorLayer>("input", TensorParams{ { "x" }, 1, depth, height, width });
    // Apply function
    const auto params = ViewParams{ { "x" }, { "out" }, 4, 5, 6 };
    BatchExpanderLayer expander("expander", params, networkParameters);
    TENSORS_CREATE(batchSize);
    tools::init_rand_tensor("x", random_range, memory_manager);
    expander.forwardCompute(NetworkMode::Test);

    // Checks
    const auto& xTensor = memory_manager["x"];
    const auto& outTensor = memory_manager["out"];

    EXPECT_EQ(outTensor.size(), xTensor.size() * batchSize);
    auto outTensor2D = outTensor.reshape(yato::dims(batchSize, depth * height * width));
    for (size_t i = 0; i < batchSize; ++i)
    {
        for (size_t j = 0; j < depth * height * width; ++j)
        {
            EXPECT_EQ(outTensor2D[i][j], xTensor[j]);
        }
    }
}

TEST(TestLayerBatchExpander, BackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batchSize = 3;
    const auto depth = 4;
    const auto height = 5;
    const auto width = 6;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<TensorLayer>("input", TensorParams{ { "x" }, 1, depth, height, width });
    // Apply function
    const auto params = ViewParams{ { "x" }, { "out" }, 4, 5, 6 };
    BatchExpanderLayer expander("expander", params, networkParameters);
    TENSORS_CREATE(batchSize);

    expander.forwardCompute(NetworkMode::Test);

    memory_manager[Name("out").grad()].memAllocate(nullptr);
    memory_manager[Name("out").grad()] = 1_dt;

    expander.backwardCompute();

    // Checks
    const auto& xTensor = memory_manager["x"];
    const auto& xNablaTensor = memory_manager[Name("x").grad()];

    EXPECT_EQ(xNablaTensor.size(), xTensor.size());
    for (size_t i = 0; i < xNablaTensor.size(); ++i)
    {
        EXPECT_EQ(xNablaTensor[i], static_cast<dtype>(batchSize));
    }
}

}