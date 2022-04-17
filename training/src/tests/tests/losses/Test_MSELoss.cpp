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
#include <tests/tools/callbacks/TensorChecker.h>

#include <training/base/common/Common.h>
#include <training/base/common/MemoryManager.h>
#include <training/base/loss/MSELoss.h>
#include <training/compiler/Workflow.h>

#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/TensorLayer.h>

namespace UT
{

using namespace raul;

namespace
{

dtype golden_mse_loss(const dtype x, const dtype y)
{
    return (x - y) * (x - y);
}

dtype golden_mse_loss_grad(const dtype x, const dtype y, const dtype g = 1.0_dt)
{
    return 2 * (x - y) * g;
}

}

TEST(TestLoss, MSELossOutputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(MSELoss("loss", LossParams{ { "x", "y" }, { "x_out", "y_out" } }, networkParameters), raul::Exception);
}

TEST(TestLoss, MSELossNoneForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    networkParameters.mLossReductionCoefficient = tensor_size;

    work.add<DataLayer>("data", DataParams{ { "input", "target" }, 1u, 1u, 1u });

    // Apply function
    MSELoss mse("loss", LossParams{ { "input", "target" }, { "out" }, "none" }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("input", random_range, memory_manager);
    tools::init_rand_tensor("target", random_range, memory_manager);

    mse.forwardCompute(NetworkMode::Train);

    // Checks
    const auto& x_tensor = memory_manager["input"];
    const auto& y_tensor = memory_manager["target"];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), x_tensor.size());
    EXPECT_EQ(out_tensor.size(), y_tensor.size());

    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto golden_out_value = golden_mse_loss(x_tensor[i], y_tensor[i]);
        ASSERT_TRUE(tools::expect_near_relative(out_tensor[i], golden_out_value, eps_rel));
    }
}

TEST(TestLoss, MSELossNoneBackwardRandUnit)
{
    PROFILE_TEST

    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    networkParameters.mLossReductionCoefficient = tensor_size;

    work.add<DataLayer>("data", DataParams{ { "input", "target" }, 1u, 1u, 1u });

    // Apply function
    MSELoss mse("loss", LossParams{ { "input", "target" }, { "out" }, "none" }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("input", random_range, memory_manager);
    tools::init_rand_tensor("target", random_range, memory_manager);
    std::fill(memory_manager[Name("out").grad()].begin(), memory_manager[Name("out").grad()].end(), 1.0_dt);

    mse.forwardCompute(NetworkMode::Train);
    mse.backwardCompute();

    // Checks
    const auto& x_tensor = memory_manager["input"];
    const auto& y_tensor = memory_manager["target"];
    const auto& x_tensor_grad = memory_manager[Name("input").grad()];

    for (size_t i = 0; i < x_tensor.size(); ++i)
    {
        const auto golden_out_value_x = golden_mse_loss_grad(x_tensor[i], y_tensor[i]);
        ASSERT_TRUE(tools::expect_near_relative(x_tensor_grad[i], golden_out_value_x, eps_rel));
    }
}

TEST(TestLoss, MSELossUnit)
{
    PROFILE_TEST
    constexpr size_t batch = 2;
    constexpr dtype eps = TODTYPE(1e-4);

    const Tensor inputs = { 0.8822693_dt, 0.9150040_dt, 0.3828638_dt, 0.9593056_dt, 0.3904482_dt, 0.6008953_dt };
    const Tensor targets = { 0.2565725_dt, 0.7936413_dt, 0.9407715_dt, 0.1331859_dt, 0.9345981_dt, 0.5935796_dt };

    const std::string reduction[] = { "none", "mean", "sum" };

    const Tensor realLoss[] = { { 0.39149645_dt, 0.014728887_dt, 0.311261_dt, 0.68247378_dt, 0.2960991_dt, 0.000053519398_dt }, { 0.2826854_dt }, { 1.6961126_dt } };

    const Tensor realInGrad[] = { { 1.2513936_dt, 0.2427254_dt, -1.1158154_dt, 1.6522393_dt, -1.0882998_dt, 0.0146314_dt },
                                  { 0.2085656_dt, 0.0404542_dt, -0.1859692_dt, 0.2753732_dt, -0.1813833_dt, 0.0024386_dt },
                                  { 1.2513936_dt, 0.2427254_dt, -1.1158154_dt, 1.6522393_dt, -1.0882998_dt, 0.0146314_dt } };

    // See mseloss.py
    for (size_t iter = 0; iter < std::size(reduction); ++iter)
    {
        Workflow work;

        work.getNetworkParameters().mLossReductionCoefficient = batch;

        work.add<DataLayer>("data", DataParams{ { "in", "target", "realInGrad" }, 1u, 1u, inputs.size() / batch });
        work.add<DataLayer>("data_grad", DataParams{ { "inGradient" }, 1u, 1u, inputs.size() / batch });
        if (iter == 0)
        {
            work.add<DataLayer>("data_loss", DataParams{ { "realLoss" }, 1u, 1u, inputs.size() / batch });
        }
        else
        {
            work.add<TensorLayer>("data_loss", TensorParams{ { "realLoss" }, 1u, 1u, 1u, 1u });
        }
        work.add<MSELoss>("loss", LossParams{ { "in", "target" }, { "loss" }, reduction[iter].c_str() });

        if (iter == 0)
        {
            work.add<TensorLayer>("grad",
                                  TensorParams{ { Name("loss").grad() }, WShape{ BS(), 1u, 1u, inputs.size() / batch }, 1_dt, Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read, true, true });
        }

        TENSORS_CREATE(batch);
        auto& memory_manager = work.getMemoryManager();
        memory_manager["in"] = TORANGE(inputs);
        memory_manager["target"] = TORANGE(targets);
        memory_manager["realInGrad"] = TORANGE(realInGrad[iter]);
        memory_manager["realLoss"] = TORANGE(realLoss[iter]);
        UT::tools::callbacks::TensorChecker checker({ { "loss", "realLoss" } }, { {} }, eps);

        work.getNetworkParameters().mCallback = checker;

        // Forward
        work.forwardPassTraining();
        printf(" - MSELoss[reduction=%s] forward is Ok.\n", reduction[iter].c_str());

        // Backward
        work.backwardPassTraining();

        const auto inGrad = memory_manager[Name("in").grad()];
        for (size_t i = 0; i < inGrad.size(); ++i)
        {
            ASSERT_TRUE(tools::expect_near_relative(inGrad[i], realInGrad[iter][i], eps));
        }
        printf(" - MSELoss[reduction=%s] backward is Ok.\n", reduction[iter].c_str());
    }
}

}