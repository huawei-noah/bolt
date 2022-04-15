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
#include <training/base/layers/basic/RandomTensorLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerRandomTensor, ForwardUnit)
{
    PROFILE_TEST
    const size_t batch = 2;
    const size_t depth = 30;
    const size_t height = 40;
    const size_t width = 50;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<raul::RandomTensorLayer>("random", raul::RandomTensorLayerParams{ { "out" }, depth, height, width });
    TENSORS_CREATE(batch);

    work.forwardPassTraining();

    const auto& outTensor = memory_manager["out"];

    EXPECT_EQ(outTensor.size(), batch * depth * height * width);
    {
        raul::dtype average = std::accumulate(outTensor.begin(), outTensor.end(), 0.0_dt) / static_cast<raul::dtype>(outTensor.size());
        printf("Average of elements is = %f\n", average);
        raul::dtype bias = 0.0_dt;
        for (auto d : outTensor)
        {
            bias += (d - average) * (d - average);
        }
        printf("Standard deviation of elements is = %f\n", sqrt(bias / static_cast<raul::dtype>(outTensor.size())));
    }
}

TEST(TestLayerRandomTensor, ForwardSeedUnit)
{
    PROFILE_TEST
    const size_t batch = 2;
    const size_t depth = 30;
    const size_t height = 40;
    const size_t width = 50;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    raul::random::setGlobalSeed(42);

    // Apply function
    work.add<raul::RandomTensorLayer>("random", raul::RandomTensorLayerParams{ { "out" }, depth, height, width });
    TENSORS_CREATE(batch);

    work.forwardPassTraining();

    const auto& outTensor = memory_manager["out"];

    EXPECT_EQ(outTensor.size(), batch * depth * height * width);

    {
        raul::dtype average = std::accumulate(outTensor.begin(), outTensor.end(), 0.0_dt) / static_cast<raul::dtype>(outTensor.size());
        printf("Average of elements is = %f\n", average);
        raul::dtype bias = 0.0_dt;
        for (auto d : outTensor)
        {
            bias += (d - average) * (d - average);
        }
        printf("Standard deviation of elements is = %f\n", sqrt(bias / static_cast<raul::dtype>(outTensor.size())));
    }
}

TEST(TestLayerRandomTensor, BackwardUnit)
{
    PROFILE_TEST
    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<raul::RandomTensorLayer>("random", raul::RandomTensorLayerParams{ { "out" }, 1u, 1u, 1u });
    TENSORS_CREATE(1);

    work.forwardPassTraining();
    ASSERT_NO_THROW(work.backwardPassTraining());
}

}