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
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/loss/BinaryCrossEntropyLoss.h>
#include <training/compiler/Workflow.h>

namespace UT
{

namespace
{

raul::dtype golden_bce_loss(const raul::dtype x, const raul::dtype y)
{
    return -(y * std::log(x) + (1.0_dt - y) * std::log(1.0_dt - x));
}

raul::dtype golden_bce_loss_grad(const raul::dtype x, const raul::dtype y, const raul::dtype g = 1.0_dt)
{
    return  -(y - x) / ((1.0_dt - x) * x) * g;
}

}

TEST(TestLoss, BCELossSimpleUnit)
{
    PROFILE_TEST

    using namespace raul;

    constexpr size_t batch = 2;
    constexpr dtype eps = 1.0e-4_dt;

    const Tensor inputs = { 0.95963228_dt, 0.17414898_dt, 0.79030937_dt, 0.70006239_dt, 0.60727251_dt, 0.81080395_dt };
    const Tensor targets = { 0.27816898_dt, 0.22943836_dt, 0.15281659_dt, 0.22011006_dt, 0.92237228_dt, 0.17413777_dt };

    const Tensor realLoss = { 2.32834077_dt, 0.54846245_dt, 1.35936630_dt, 1.01761663_dt, 0.53261262_dt, 1.41155887_dt };
    const Tensor realInGrad = { 17.59152031_dt, -0.38443166_dt, 3.84679580_dt, 2.28575897_dt, -1.32121396_dt, 4.15034199_dt };

    // See bce_loss.py
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data", DataParams{ { "in", "target", "realInGrad" }, 1u, 1u, inputs.size() / batch });
    work.add<BinaryCrossEntropyLoss>("loss", LossParams{ { "in", "target" }, { "loss" }, "none" });

    TENSORS_CREATE(batch);
    
    memory_manager["in"] = TORANGE(inputs);
    memory_manager["target"] = TORANGE(targets);

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());

    const auto& loss = memory_manager["loss"];
    EXPECT_EQ(inputs.size(), loss.size());
    for (size_t i = 0; i < loss.size(); ++i)
    {
        EXPECT_NEAR(loss[i], realLoss[i], eps);
    }

    // Backward
    memory_manager[Name("loss").grad()] = 1.0_dt;
    ASSERT_NO_THROW(work.backwardPassTraining());

    const auto& inGrad = memory_manager[Name("in").grad()];
    EXPECT_EQ(inGrad.size(), realInGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], realInGrad[i], eps);
    }
}

TEST(TestLoss, BCELossRandomUnit)
{
    PROFILE_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 5;
    constexpr size_t WIDTH = 19;
    constexpr size_t HEIGHT = 2;
    constexpr size_t DEPTH = 3;
    constexpr dtype eps = 1.0e-4_dt;
    constexpr auto range = std::make_pair(0.0001_dt, 0.9999_dt);
    
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data", DataParams{ { "in", "target" }, DEPTH, HEIGHT, WIDTH });
    work.add<BinaryCrossEntropyLoss>("loss", LossParams{ { "in", "target" }, { "loss" }, "none" });

    TENSORS_CREATE(BATCH_SIZE)

    tools::init_rand_tensor("in", range, memory_manager);
    tools::init_rand_tensor("target", range, memory_manager);
    tools::init_rand_tensor("lossGradient", range, memory_manager);

    ASSERT_NO_THROW(work.forwardPassTraining());

    const auto& loss = memory_manager["loss"];
    const auto& in = memory_manager["in"];
    const auto& target = memory_manager["target"];
    EXPECT_EQ(in.size(), loss.size());
    for (size_t i = 0; i < loss.size(); ++i)
    {
        EXPECT_NEAR(loss[i], golden_bce_loss(in[i], target[i]), eps);
    }

    ASSERT_NO_THROW(work.backwardPassTraining());

    const auto& inGrad = memory_manager[raul::Name("in").grad()];
    const auto& lossGrad = memory_manager[raul::Name("loss").grad()];
    EXPECT_EQ(inGrad.size(), in.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], golden_bce_loss_grad(in[i], target[i], lossGrad[i]), eps);
    }
}

TEST(TestLoss, BCELossCornerCasesUnit)
{
    PROFILE_TEST

    using namespace raul;

    constexpr size_t batch = 4;
    constexpr dtype eps = 1.0e-4_dt;

    const Tensor inputs = { 0.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 0.998_dt, 0.002_dt, 1.0_dt, 0.999_dt, 0.0001_dt, 0.11_dt };
    const Tensor targets = { 0.0_dt, 1.0_dt, 0.99_dt, 0.0_dt, 1.0_dt, 0.5_dt, 0.0001_dt, 0.997_dt, 1.0_dt, 0.999_dt, 0.001_dt, 0.00001_dt };

    const Tensor realLoss = { 0.0_dt, 100.0_dt, 99.0_dt, 100.0_dt, 0.0_dt, 50.0_dt, 6.214_dt, 6.196_dt, 0.0_dt, 0.0079072_dt, 0.0093103_dt, 0.11655_dt };

    const Tensor realInGrad = { 0.0_dt, -9.99999996e+11_dt, -9.89999989e+11_dt, 9.99999996e+11_dt,  0.0_dt,  4.99999998e+11_dt, 4.99956299e+02_dt, -4.98496979e+02_dt, 0.0_dt, 0.0_dt, -9.00090122_dt,  1.12349343_dt };

    // See bce_loss.py
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data", DataParams{ { "in", "target", "realInGrad" }, 1u, 1u, inputs.size() / batch });
    work.add<BinaryCrossEntropyLoss>("loss", LossParams{ { "in", "target" }, { "loss" }, "none" });

    TENSORS_CREATE(batch);
    
    memory_manager["in"] = TORANGE(inputs);
    memory_manager["target"] = TORANGE(targets);

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());

    const auto& loss = memory_manager["loss"];
    EXPECT_EQ(inputs.size(), loss.size());
    for (size_t i = 0; i < loss.size(); ++i)
    {
        EXPECT_NEAR(loss[i], realLoss[i], eps);
    }

    // Backward
    memory_manager[Name("loss").grad()] = 1.0_dt;
    ASSERT_NO_THROW(work.backwardPassTraining());

    const auto& inGrad = memory_manager[Name("in").grad()];
    EXPECT_EQ(inGrad.size(), realInGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], realInGrad[i], eps);
    }
}

}