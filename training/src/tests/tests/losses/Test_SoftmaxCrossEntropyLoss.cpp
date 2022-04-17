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
#include <training/base/layers/activations/SoftMaxActivation.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/ReduceSumLayer.h>
#include <training/base/loss/CrossEntropyLoss.h>
#include <training/base/loss/SoftmaxCrossEntropyLoss.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLoss, SoftmaxCELossOutputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::SoftmaxCrossEntropyLoss("loss", raul::LossParams{ { "x", "y" }, { "x_out", "y_out" } }, networkParameters), raul::Exception);
}

TEST(TestLoss, SoftmaxCELossElementWiseForwardRandUnit)
{
    PROFILE_TEST
    using namespace raul;

    // Test parameters
    const auto eps = TODTYPE(1e-6);
    const auto batchSize = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<DataLayer>("data", DataParams{ { "input", "target" }, 2u, 3u, 4u });
    work.add<SoftmaxCrossEntropyLoss>("loss", LossParams{ { "input", "target" }, { "out" }, "none" });
    // To compare
    work.add<SoftMaxActivation>("sf", BasicParamsWithDim{ { "input" }, { "inputSF" }, "width" });
    work.add<CrossEntropyLoss>("lossGolden", LossParams{ { "inputSF", "target" }, { "outGolden" }, "none" });

    TENSORS_CREATE(batchSize);
    tools::init_rand_tensor("input", random_range, memory_manager);
    tools::init_rand_tensor("target", random_range, memory_manager);

    work.forwardPassTraining();

    // Checks
    const auto& outTensor = memory_manager["out"];
    const auto& outTensorGolden = memory_manager["outGolden"];

    for (size_t i = 0; i < outTensor.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(outTensor[i], outTensorGolden[i], eps));
    }
}

TEST(TestLoss, SoftmaxCELossElementWiseSimpleUnit)
{
    PROFILE_TEST
    using namespace raul;

    // See tf_softmax_cross_entropy_with_logits.py

    // Test parameters
    const auto eps = TODTYPE(1e-6);
    const auto batchSize = 1U;
    const auto depth = 1U;
    const auto height = 1U;
    const auto width = 3U;

    const Tensor logits{ 0.81269646_dt, 0.07857466_dt, 0.8916855_dt };
    const Tensor targets{ 1.0_dt, 0.0_dt, 0.0_dt };

    // Output
    const Tensor realLoss{ 0.94083476_dt };
    const Tensor realInputGrad{ -0.6096981_dt, 0.18731631_dt, 0.42238176_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<DataLayer>("data", DataParams{ { "input", "target" }, depth, height, width });
    work.add<SoftmaxCrossEntropyLoss>("loss", LossParams{ { "input", "target" }, { "loss" }, "none" });
    work.add<ReduceSumLayer>("final_loss", BasicParamsWithDim{ { "loss" }, { "finalLoss" } });

    TENSORS_CREATE(batchSize);
    memory_manager["input"] = TORANGE(logits);
    memory_manager["target"] = TORANGE(targets);
    memory_manager[Name("finalLoss").grad()] = 1.0_dt;

    ASSERT_NO_THROW(work.forwardPassTraining());

    // Forward checks
    const auto& loss = memory_manager["finalLoss"];
    for (size_t i = 0; i < loss.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(loss[i], realLoss[i], eps));
    }

    ASSERT_NO_THROW(work.backwardPassTraining());

    // Backward checks
    const auto& inputGrad = memory_manager[Name("input").grad()];
    for (size_t i = 0; i < inputGrad.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(inputGrad[i], realInputGrad[i], eps));
    }
}

TEST(TestLoss, SoftmaxCELossElementWiseUnit)
{
    PROFILE_TEST
    using namespace raul;

    // See tf_softmax_cross_entropy_with_logits.py

    // Test parameters
    const auto eps = TODTYPE(1e-6);
    const auto batchSize = 2U;
    const auto depth = 1U;
    const auto height = 3U;
    const auto width = 4U;

    const Tensor logits{ 0.81269646_dt, 0.07857466_dt, 0.8916855_dt,  0.16925514_dt, 0.06311357_dt, 0.54531074_dt, 0.5037316_dt,  0.9248222_dt,
                         0.66955376_dt, 0.9281193_dt,  0.12239242_dt, 0.8532245_dt,  0.90477383_dt, 0.7104306_dt,  0.40681756_dt, 0.5755513_dt,
                         0.8547678_dt,  0.59606934_dt, 0.77619946_dt, 0.97301054_dt, 0.06244731_dt, 0.33562684_dt, 0.22166848_dt, 0.32035887_dt };
    const Tensor targets{ 1.0_dt, 0.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 1.0_dt, 1.0_dt,
                          0.0_dt, 0.0_dt, 1.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 2.0_dt, 2.0_dt, 2.0_dt };

    // Output
    const Tensor realLoss{ 2.8982296_dt, 0.0_dt, 5.9832497_dt, 4.5992084_dt, 5.8805046_dt, 8.007221_dt };
    const Tensor realInputGrad{ -0.67612386_dt, 0.15543681_dt, 0.3504963_dt,   -0.82980925_dt, 0.15289354_dt, 0.24763085_dt, 0.23754568_dt, 0.3619299_dt,
                                -1.7546182_dt,  0.31778693_dt, -0.85802454_dt, -0.7051442_dt,  0.31739688_dt, 0.26133674_dt, -0.8070952_dt, -1.7716384_dt,
                                -1.7383577_dt,  -1.7979975_dt, 0.24187233_dt,  0.29448295_dt,  0.20916311_dt, -1.7251312_dt, -1.754736_dt,  -1.729296_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<DataLayer>("data", DataParams{ { "input", "target" }, depth, height, width });
    work.add<SoftmaxCrossEntropyLoss>("loss", LossParams{ { "input", "target" }, { "loss" }, "none" });
    work.add<ReduceSumLayer>("final_loss", BasicParamsWithDim{ { "loss" }, { "finalLoss" }, "width" });

    TENSORS_CREATE(batchSize);
    memory_manager["input"] = TORANGE(logits);
    memory_manager["target"] = TORANGE(targets);
    memory_manager[Name("finalLoss").grad()] = 1.0_dt;

    ASSERT_NO_THROW(work.forwardPassTraining());

    // Forward checks
    const auto& loss = memory_manager["finalLoss"];
    for (size_t i = 0; i < loss.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(loss[i], realLoss[i], eps));
    }

    ASSERT_NO_THROW(work.backwardPassTraining());

    // Backward checks
    const auto& inputGrad = memory_manager[Name("input").grad()];
    for (size_t i = 0; i < inputGrad.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(inputGrad[i], realInputGrad[i], eps));
    }
}

}