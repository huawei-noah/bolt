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

#include <training/layers/basic/DataLayer.h>
#include <training/layers/basic/ReduceNonZeroLayer.h>
#include <training/network/Workflow.h>

#include <random>

namespace UT
{

namespace
{

std::tuple<size_t, size_t, size_t, size_t> reassign(raul::Dimension dim, size_t i, size_t j, size_t k, size_t q)
{
    if (dim == raul::Dimension::Depth)
    {
        return std::make_tuple(i, q, j, k);
    }
    if (dim == raul::Dimension::Height)
    {
        return std::make_tuple(i, j, q, k);
    }
    if (dim == raul::Dimension::Width)
    {
        return std::make_tuple(i, j, k, q);
    }
    return std::make_tuple(q, i, j, k);
}

}

TEST(TestLayerReduceNonZero, InputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::ReduceNonZeroLayer("nonzero", raul::BasicParamsWithDim{ { "x", "y" }, { "x_out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerReduceNonZero, OutputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::ReduceNonZeroLayer("nonzero", raul::BasicParamsWithDim{ { "x", "y" }, { "x_out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerReduceNonZero, ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 1;
    const auto depth = 2;
    const auto height = 3;
    const auto width = 4;

    // See reduce_sum.py
    const raul::Tensor x{ 0.1_dt, 0.0_dt,        0.2_dt,        0.13203049_dt, 0.0_dt,        0.63407868_dt, 0.49009341_dt, 0.89644474_dt, 0.45562798_dt, 0.63230628_dt, 0.34889346_dt, 0.51909077_dt,
                          0.0_dt, 0.81018829_dt, 0.98009706_dt, 0.11468822_dt, 0.31676513_dt, 0.69650495_dt, 0.91427469_dt, 0.0_dt,        0.94117838_dt, 0.59950727_dt, 0.0_dt,        0.0_dt };

    std::string dimensions[] = { "default", "batch", "depth", "height", "width" };

    raul::Tensor realOutputs[] = { { 18.0_dt },
                                   { 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                                     0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt },
                                   { 1.0_dt, 1.0_dt, 2.0_dt, 2.0_dt, 1.0_dt, 2.0_dt, 2.0_dt, 1.0_dt, 2.0_dt, 2.0_dt, 1.0_dt, 1.0_dt },
                                   { 2.0_dt, 2.0_dt, 3.0_dt, 3.0_dt, 2.0_dt, 3.0_dt, 2.0_dt, 1.0_dt },
                                   { 3.0_dt, 3.0_dt, 4.0_dt, 3.0_dt, 3.0_dt, 2.0_dt } };

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
        work.add<raul::ReduceNonZeroLayer>("rnonzero", raul::BasicParamsWithDim{ { "x" }, { "out" }, dimensions[iter] });
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);

        work.forwardPassTraining();

        // Checks
        const auto& outTensor = memory_manager["out"];
        EXPECT_EQ(outTensor.size(), realOutputs[iter].size());
        for (size_t i = 0; i < outTensor.size(); ++i)
        {
            EXPECT_EQ(outTensor[i], realOutputs[iter][i]);
        }
    }
}

TEST(TestLayerReduceNonZero, BackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto batch = 1;
    const auto depth = 2;
    const auto height = 3;
    const auto width = 4;

    // See reduce_sum.py
    const raul::Tensor x{ 0.1_dt, 0.0_dt,        0.2_dt,        0.13203049_dt, 0.0_dt,        0.63407868_dt, 0.49009341_dt, 0.89644474_dt, 0.45562798_dt, 0.63230628_dt, 0.34889346_dt, 0.51909077_dt,
                          0.0_dt, 0.81018829_dt, 0.98009706_dt, 0.11468822_dt, 0.31676513_dt, 0.69650495_dt, 0.91427469_dt, 0.0_dt,        0.94117838_dt, 0.59950727_dt, 0.0_dt,        0.0_dt };

    std::string dimensions[] = { "default", "batch", "depth", "height", "width" };

    const raul::Tensor realGrad = raul::Tensor("realGrad", batch, depth, height, width, 1.0_dt);

    for (size_t iter = 0; iter < std::size(dimensions); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
        work.add<raul::ReduceNonZeroLayer>("rnonzero", raul::BasicParamsWithDim{ { "x" }, { "out" }, dimensions[iter] });
        TENSORS_CREATE(batch);
        memory_manager["x"] = TORANGE(x);
        memory_manager[raul::Name("out").grad()] = 1.0_dt;

        work.forwardPassTraining();
        work.backwardPassTraining();

        // Checks
        const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
        EXPECT_EQ(x_tensor_grad.getShape(), memory_manager["x"].getShape());
        for (size_t i = 0; i < x_tensor_grad.size(); ++i)
        {
            EXPECT_EQ(x_tensor_grad[i], realGrad[i]);
        }
    }
}

struct TestReduceNonZeroLayerGPU : public testing::TestWithParam<tuple<size_t, size_t, size_t, size_t, raul::dtype, raul::Dimension>>
{
    const size_t batch = get<0>(GetParam());
    const size_t depth = get<1>(GetParam());
    const size_t height = get<2>(GetParam());
    const size_t width = get<3>(GetParam());
    const raul::dtype secondBound = get<4>(GetParam());
    const raul::Dimension dim = get<5>(GetParam());
    const raul::dtype eps = TODTYPE(1e-4);
    const std::pair<raul::dtype, raul::dtype> range = std::make_pair(1.0_dt, secondBound);
};

TEST_P(TestReduceNonZeroLayerGPU, ForwardRandGpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU };
    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, depth, height, width });
    work.add<raul::ReduceNonZeroLayer>("rnonzero", raul::BasicParamsWithDim{ { "x" }, { "out" }, dim });
    TENSORS_CREATE(batch);
        
    auto& memory_manager = work.getMemoryManager<raul::MemoryManagerGPU>();
    tools::init_rand_tensor("x", range, memory_manager);
    tools::init_rand_tensor(raul::Name("out").grad(), range, memory_manager);

    ASSERT_NO_THROW(work.forwardPassTraining());
    ASSERT_NO_THROW(work.backwardPassTraining());

    // Checks
    const raul::Tensor& in = memory_manager["x"];
    const raul::Tensor& out = memory_manager["out"];
    const raul::Tensor& inGrad = memory_manager[raul::Name("x").grad()];
    const raul::Tensor& outGrad = memory_manager[raul::Name("out").grad()];

    if (dim == raul::Dimension::Default)
    {
        // Forward checks
        ASSERT_TRUE(tools::expect_near_relative(out[0], static_cast<raul::dtype>(std::count_if(in.begin(), in.end(), [](raul::dtype val) { return val != 0.0_dt; })), eps));

        // Backward checks
        for (size_t i = 0; i < inGrad.size(); ++i)
        {
            ASSERT_TRUE(tools::expect_near_relative(inGrad[i], outGrad[0], eps));
        }
    }
    else
    {
        // Forward checks
        const auto in4D = in.get4DView();
        const auto out4D = out.get4DView();
        auto sum = 0.0_dt;
        auto inputShape = in.getShape();
        // Pick chosen dimension
        const auto chosenDimSize = inputShape[static_cast<size_t>(dim)];
        // Delete it
        std::vector<size_t> otherDims;
        for (size_t i = 0; i < inputShape.dimensions_num(); ++i)
        {
            if (i != static_cast<size_t>(dim))
            {
                otherDims.push_back(inputShape[i]);
            }
        }

        for (size_t i = 0; i < otherDims[0]; ++i)
        {
            for (size_t j = 0; j < otherDims[1]; ++j)
            {
                for (size_t k = 0; k < otherDims[2]; ++k)
                {
                    sum = 0.0_dt;
                    for (size_t q = 0; q < chosenDimSize; ++q)
                    {
                        auto [realI, realJ, realK, realQ] = reassign(dim, i, j, k, q);
                        if (in4D[realI][realJ][realK][realQ] != 0.0_dt)
                        {
                            sum += 1.0_dt;
                        }
                    }
                    auto [realIO, realJO, realKO, realQO] = reassign(dim, i, j, k, 0);
                    ASSERT_TRUE(tools::expect_near_relative(out4D[realIO][realJO][realKO][realQO], sum, eps));
                }
            }
        }

        // Backward checks
        const auto inGrad4D = inGrad.get4DView();
        const auto outGrad4D = outGrad.get4DView();

        for (size_t i = 0; i < otherDims[0]; ++i)
        {
            for (size_t j = 0; j < otherDims[1]; ++j)
            {
                for (size_t k = 0; k < otherDims[2]; ++k)
                {
                    for (size_t q = 0; q < chosenDimSize; ++q)
                    {
                        auto [realI, realJ, realK, realQ] = reassign(dim, i, j, k, q);
                        auto [realIO, realJO, realKO, realQO] = reassign(dim, i, j, k, 0);
                        ASSERT_TRUE(tools::expect_near_relative(inGrad4D[realI][realJ][realK][realQ], outGrad4D[realIO][realJO][realKO][realQO], eps));
                    }
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(TestGpu,
                         TestReduceNonZeroLayerGPU,
                         testing::Values(make_tuple(1, 1, 1, 1, 0.0_dt, raul::Dimension::Default),
                                         make_tuple(3, 1, 1, 1, 1.0_dt, raul::Dimension::Default),
                                         make_tuple(1, 3, 1, 1, 0.0_dt, raul::Dimension::Default),
                                         make_tuple(1, 1, 3, 1, 1.0_dt, raul::Dimension::Default),
                                         make_tuple(1, 1, 1, 3, 1.0_dt, raul::Dimension::Default),
                                         make_tuple(2, 3, 4, 5, 0.0_dt, raul::Dimension::Default),
                                         make_tuple(3, 7, 11, 9, 0.0_dt, raul::Dimension::Default),
                                         make_tuple(3, 1, 66, 128, 1.0_dt, raul::Dimension::Default),
                                         make_tuple(1, 1, 1, 1, 1.0_dt, raul::Dimension::Batch),
                                         make_tuple(3, 1, 1, 1, 1.0_dt, raul::Dimension::Batch),
                                         make_tuple(1, 3, 1, 1, 0.0_dt, raul::Dimension::Batch),
                                         make_tuple(1, 1, 3, 1, 0.0_dt, raul::Dimension::Batch),
                                         make_tuple(1, 1, 1, 3, 1.0_dt, raul::Dimension::Batch),
                                         make_tuple(2, 3, 4, 5, 1.0_dt, raul::Dimension::Batch),
                                         make_tuple(3, 7, 11, 9, 0.0_dt, raul::Dimension::Batch),
                                         make_tuple(3, 1, 66, 128, 1.0_dt, raul::Dimension::Batch),
                                         make_tuple(1, 1, 1, 1, 0.0_dt, raul::Dimension::Depth),
                                         make_tuple(3, 1, 1, 1, 1.0_dt, raul::Dimension::Depth),
                                         make_tuple(1, 3, 1, 1, 0.0_dt, raul::Dimension::Depth),
                                         make_tuple(1, 1, 3, 1, 1.0_dt, raul::Dimension::Depth),
                                         make_tuple(1, 1, 1, 3, 1.0_dt, raul::Dimension::Depth),
                                         make_tuple(2, 3, 4, 5, 1.0_dt, raul::Dimension::Depth),
                                         make_tuple(3, 7, 11, 9, 0.0_dt, raul::Dimension::Depth),
                                         make_tuple(3, 1, 66, 128, 0.0_dt, raul::Dimension::Depth),
                                         make_tuple(1, 1, 1, 1, 1.0_dt, raul::Dimension::Height),
                                         make_tuple(3, 1, 1, 1, 0.0_dt, raul::Dimension::Height),
                                         make_tuple(1, 3, 1, 1, 1.0_dt, raul::Dimension::Height),
                                         make_tuple(1, 1, 3, 1, 0.0_dt, raul::Dimension::Height),
                                         make_tuple(1, 1, 1, 3, 1.0_dt, raul::Dimension::Height),
                                         make_tuple(2, 3, 4, 5, 1.0_dt, raul::Dimension::Height),
                                         make_tuple(3, 7, 11, 9, 1.0_dt, raul::Dimension::Height),
                                         make_tuple(3, 1, 66, 128, 0.0_dt, raul::Dimension::Height),
                                         make_tuple(1, 1, 1, 1, 0.0_dt, raul::Dimension::Width),
                                         make_tuple(3, 1, 1, 1, 0.0_dt, raul::Dimension::Width),
                                         make_tuple(1, 3, 1, 1, 1.0_dt, raul::Dimension::Width),
                                         make_tuple(1, 1, 3, 1, 0.0_dt, raul::Dimension::Width),
                                         make_tuple(1, 1, 1, 3, 1.0_dt, raul::Dimension::Width),
                                         make_tuple(2, 3, 4, 5, 0.0_dt, raul::Dimension::Width),
                                         make_tuple(3, 7, 11, 9, 0.0_dt, raul::Dimension::Width),
                                         make_tuple(3, 1, 66, 128, 1.0_dt, raul::Dimension::Width)));

}
