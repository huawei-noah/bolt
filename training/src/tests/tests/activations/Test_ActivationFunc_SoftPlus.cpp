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

#include <training/base/layers/activations/SoftPlusActivation.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestActivationFuncSoftPlus, PositiveBetaAndThresholdUnit)
{
    PROFILE_TEST

    // Test parameters
    constexpr auto eps = 1e-6_dt;
    constexpr auto BATCH = 1U;
    constexpr auto DEPTH = 2U;
    constexpr auto HEIGHT = 3U;
    constexpr auto WIDTH = 4U;
    constexpr auto beta = 5.0_dt;
    constexpr auto threshold = 2.5_dt;

    const raul::Tensor input{ 0.49625659_dt, 0.76822180_dt, 0.08847743_dt, 0.13203049_dt, 0.30742282_dt, 0.63407868_dt, 0.49009341_dt, 0.89644474_dt,
                              0.45562798_dt, 0.63230628_dt, 0.34889346_dt, 0.40171731_dt, 0.02232575_dt, 0.16885895_dt, 0.29388845_dt, 0.51852179_dt,
                              0.69766760_dt, 0.80001140_dt, 0.16102946_dt, 0.28226858_dt, 0.68160856_dt, 0.91519397_dt, 0.39709991_dt, 0.87415588_dt };

    // Outputs
    const raul::Tensor realOut{ 0.51232100_dt, 0.76822180_dt, 0.18772143_dt, 0.21534744_dt, 0.34637174_dt, 0.63407868_dt, 0.50664032_dt, 0.89644474_dt,
                                0.47513944_dt, 0.63230628_dt, 0.38110250_dt, 0.42689896_dt, 0.15010367_dt, 0.24037433_dt, 0.33530003_dt, 0.51852179_dt,
                                0.69766760_dt, 0.80001140_dt, 0.23493099_dt, 0.32590535_dt, 0.68160856_dt, 0.91519397_dt, 0.42283344_dt, 0.87415588_dt };
    const raul::Tensor realInGrad{ 1.00000000_dt, 1.00000000_dt, 0.60882771_dt, 0.65929461_dt, 0.82304484_dt, 1.00000000_dt, 1.00000000_dt, 1.00000000_dt,
                                   0.90705031_dt, 1.00000000_dt, 0.85125363_dt, 0.88169563_dt, 0.52787828_dt, 0.69936901_dt, 0.81297261_dt, 1.00000000_dt,
                                   1.00000000_dt, 1.00000000_dt, 0.69107443_dt, 0.80397767_dt, 1.00000000_dt, 1.00000000_dt, 0.87926620_dt, 1.00000000_dt };

    // Initialization
    raul::WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPU);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<raul::SoftPlusActivation>("softplus", raul::SoftPlusActivationParams{ { "in" }, { "out" }, beta, threshold });

    TENSORS_CREATE(BATCH);
    auto& memory_manager = work.getMemoryManager();
    memory_manager["in"] = TORANGE(input);
    memory_manager["outGradient"] = 1.0_dt;

    ASSERT_NO_THROW(work.forwardPassTraining());

    // Forward checks
    const auto& out = memory_manager["out"];
    EXPECT_EQ(out.size(), realOut.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], realOut[i], eps);
    }

    ASSERT_NO_THROW(work.backwardPassTraining());

    // Backward checks
    const auto& inGrad = memory_manager["inGradient"];
    EXPECT_EQ(inGrad.size(), realInGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], realInGrad[i], eps);
    }
}

TEST(TestActivationFuncSoftPlus, NegativeBetaAndThresholdUnit)
{
    PROFILE_TEST

    // Test parameters
    constexpr auto eps = 1e-6_dt;
    constexpr auto BATCH = 1U;
    constexpr auto DEPTH = 2U;
    constexpr auto HEIGHT = 3U;
    constexpr auto WIDTH = 4U;
    constexpr auto beta = -4.0_dt;
    constexpr auto threshold = -2.5_dt;

    const raul::Tensor input{ 0.41940832_dt, 0.55290705_dt, 0.95273811_dt, 0.03616482_dt, 0.18523103_dt, 0.37341738_dt, 0.30510002_dt, 0.93200040_dt,
                              0.17591017_dt, 0.26983356_dt, 0.15067977_dt, 0.03171951_dt, 0.20812976_dt, 0.92979902_dt, 0.72310919_dt, 0.74233627_dt,
                              0.52629578_dt, 0.24365824_dt, 0.58459234_dt, 0.03315264_dt, 0.13871688_dt, 0.24223500_dt, 0.81546897_dt, 0.79316062_dt };

    // Output
    const raul::Tensor realOut{ 0.41940832_dt, 0.55290705_dt, -0.00547146_dt, 0.03616482_dt, 0.18523103_dt, 0.37341738_dt,  0.30510002_dt,  -0.00593910_dt,
                                0.17591017_dt, 0.26983356_dt, 0.15067977_dt,  0.03171951_dt, 0.20812976_dt, -0.00599100_dt, -0.01348966_dt, -0.01251565_dt,
                                0.52629578_dt, 0.24365824_dt, 0.58459234_dt,  0.03315264_dt, 0.13871688_dt, 0.24223500_dt,  -0.00940015_dt, -0.01025975_dt };

    // Initialization
    raul::WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPU);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<raul::SoftPlusActivation>("softplus", raul::SoftPlusActivationParams{ { "in" }, { "out" }, beta, threshold });

    TENSORS_CREATE(BATCH);
    auto& memory_manager = work.getMemoryManager();
    memory_manager["in"] = TORANGE(input);
    memory_manager["outGradient"] = 1.0_dt;

    ASSERT_NO_THROW(work.forwardPassTraining());

    // Forward checks
    const auto& out = memory_manager["out"];
    EXPECT_EQ(out.size(), realOut.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], realOut[i], eps);
    }

    ASSERT_NO_THROW(work.backwardPassTraining());

    // Backward checks
    const auto& inGrad = memory_manager["inGradient"];
    EXPECT_EQ(inGrad.size(), input.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_EQ(inGrad[i], 1.0_dt);
    }
}

} // UT namespace