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
#include <training/base/loss/SigmoidCrossEntropyLoss.h>
#include <training/compiler/Workflow.h>

namespace UT
{

namespace
{

raul::dtype golden_sigmoid_cs_loss(const raul::dtype x, const raul::dtype y)
{
    return std::max(x, 0.0_dt) - x * y + std::log(1.0_dt + std::exp(-abs(x)));
}

raul::dtype golden_sigmoid_cs_loss_grad(const raul::dtype x, const raul::dtype y, const raul::dtype g = 1.0_dt)
{
    return (1_dt - y - std::exp(-x) / (1_dt + std::exp(-x))) * g;
}

}

TEST(TestLoss, SigmoidCELossOutputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::SigmoidCrossEntropyLoss("loss", raul::LossParams{ { "x", "y" }, { "x_out", "y_out" } }, networkParameters), raul::Exception);
}

TEST(TestLoss, SigmoidCELossNoneForwardRandUnit)
{
    PROFILE_TEST
    using namespace raul;

    // Test parameters
    const auto eps = TODTYPE(1e-6);
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<DataLayer>("data", DataParams{ { "input", "target" }, 1u, 1u, 1u });
    work.add<SigmoidCrossEntropyLoss>("loss", LossParams{ { "input", "target" }, { "out" }, "none" });
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("input", random_range, memory_manager);
    tools::init_rand_tensor("target", random_range, memory_manager);

    work.forwardPassTraining();

    // Checks
    const auto& xTensor = memory_manager["input"];
    const auto& yTensor = memory_manager["target"];
    const auto& outTensor = memory_manager["out"];

    for (size_t i = 0; i < outTensor.size(); ++i)
    {
        const auto goldenValue = golden_sigmoid_cs_loss(xTensor[i], yTensor[i]);
        ASSERT_TRUE(tools::expect_near_relative(outTensor[i], goldenValue, eps));
    }
}

TEST(TestLoss, SigmoidCELossNoneBackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps = TODTYPE(1e-6);
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(1.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    work.add<raul::DataLayer>("data", raul::DataParams{ { "input", "target" }, 1u, 1u, 1u });
    work.add<raul::SigmoidCrossEntropyLoss>("loss", raul::LossParams{ { "input", "target" }, { "out" }, "none" });
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("input", random_range, memory_manager);
    tools::init_rand_tensor("target", random_range, memory_manager);
    memory_manager[raul::Name("out").grad()] = 1.0_dt;

    work.forwardPassTraining();
    work.backwardPassTraining();

    // Checks
    const auto& xTensor = memory_manager["input"];
    const auto& yTensor = memory_manager["target"];
    const auto& inNablaTensor = memory_manager[raul::Name("input").grad()];

    EXPECT_EQ(inNablaTensor.size(), xTensor.size());
    EXPECT_EQ(inNablaTensor.size(), yTensor.size());

    for (size_t i = 0; i < inNablaTensor.size(); ++i)
    {
        const auto goldenGradValue = golden_sigmoid_cs_loss_grad(xTensor[i], yTensor[i]);
        ASSERT_TRUE(tools::expect_near_relative(inNablaTensor[i], goldenGradValue, eps));
    }
}

TEST(TestLoss, SigmoidCELossForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps = TODTYPE(1e-6);
    const size_t batch = 1;
    const size_t depth = 3;
    const size_t height = 3;
    const size_t width = 3;

    const raul::Tensor x{ -1.0_dt, 2.0_dt,   3.0_dt,  -4.0_dt, 5.0_dt,   6.0_dt,  -7.0_dt, 8.0_dt,   9.0_dt,  -10.0_dt, 11.0_dt,  12.0_dt, -13.0_dt, 14.0_dt,
                          15.0_dt, -16.0_dt, 17.0_dt, 18.0_dt, -19.0_dt, 20.0_dt, 21.0_dt, -22.0_dt, 23.0_dt, 24.0_dt,  -25.0_dt, 26.0_dt, 27.0_dt };
    const raul::Tensor y{ 1.0_dt, 1.0_dt,  1.0_dt,  2.0_dt,  2.0_dt,  2.0_dt,  3.0_dt,  3.0_dt, 3.0_dt, 5.0_dt, 5.0_dt, 5.0_dt, 8.0_dt, 8.0_dt,
                          8.0_dt, 17.0_dt, 16.0_dt, 15.0_dt, 14.0_dt, 21.0_dt, 22.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 6.0_dt, 7.0_dt };

    const raul::Tensor realOut[]{ {
                                      1.3132617_dt, 0.126928_dt,   0.048587352_dt, 8.01815_dt, -4.9932847_dt, -5.9975243_dt, 21.000912_dt, -15.999664_dt, -17.999876_dt,
                                      50.000046_dt, -43.999985_dt, -47.999992_dt,  104.0_dt,   -98.0_dt,      -105.0_dt,     272.0_dt,     -255.0_dt,     -252.0_dt,
                                      266.0_dt,     -400.0_dt,     -441.0_dt,      44.0_dt,    -46.0_dt,      -72.0_dt,      125.0_dt,     -130.0_dt,     -162.0_dt,
                                  },
                                  { -1206.4824_dt },
                                  { -44.684536_dt } };
    std::string reduction[] = { "none", "sum", "mean" };

    for (size_t iter = 0; iter < std::size(reduction); ++iter)
    {
        // Initialization
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        // Apply function
        work.add<raul::DataLayer>("data", raul::DataParams{ { "input", "target" }, depth, height, width });
        work.add<raul::SigmoidCrossEntropyLoss>("loss", raul::LossParams{ { "input", "target" }, { "out" }, reduction[iter] });
        TENSORS_CREATE(batch);
        memory_manager["input"] = TORANGE(x);
        memory_manager["target"] = TORANGE(y);

        work.forwardPassTraining();

        // Checks
        const auto& outTensor = memory_manager["out"];

        EXPECT_EQ(outTensor.size(), realOut[iter].size());

        for (size_t i = 0; i < outTensor.size(); ++i)
        {
            ASSERT_TRUE(tools::expect_near_relative(outTensor[i], realOut[iter][i], eps));
        }
    }
}

TEST(TestLoss, SigmoidCELossBackwardUnit)
{
    PROFILE_TEST
    using namespace raul;

    // Test parameters
    const auto eps = TODTYPE(1e-6);
    const size_t batch = 1;
    const size_t depth = 3;
    const size_t height = 3;
    const size_t width = 3;

    const Tensor x{ -1.0_dt, 2.0_dt,   3.0_dt,  -4.0_dt, 5.0_dt,   6.0_dt,  -7.0_dt, 8.0_dt,   9.0_dt,  -10.0_dt, 11.0_dt,  12.0_dt, -13.0_dt, 14.0_dt,
                    15.0_dt, -16.0_dt, 17.0_dt, 18.0_dt, -19.0_dt, 20.0_dt, 21.0_dt, -22.0_dt, 23.0_dt, 24.0_dt,  -25.0_dt, 26.0_dt, 27.0_dt };
    const Tensor y{ 1.0_dt, 1.0_dt,  1.0_dt,  2.0_dt,  2.0_dt,  2.0_dt,  3.0_dt,  3.0_dt, 3.0_dt, 5.0_dt, 5.0_dt, 5.0_dt, 8.0_dt, 8.0_dt,
                    8.0_dt, 17.0_dt, 16.0_dt, 15.0_dt, 14.0_dt, 21.0_dt, 22.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 6.0_dt, 7.0_dt };

    const Tensor realGrad[]{ { -0.7310586_dt, -0.11920291_dt, -0.047425874_dt, -1.9820138_dt, -1.0066929_dt, -1.0024726_dt, -2.999089_dt, -2.0003355_dt, -2.0001235_dt,
                               -4.9999547_dt, -4.0000167_dt,  -4.000006_dt,    -7.9999976_dt, -7.000001_dt,  -7.0000005_dt, -17.0_dt,     -15.0_dt,      -14.0_dt,
                               -14.0_dt,      -20.0_dt,       -21.0_dt,        -2.0_dt,       -2.0_dt,       -3.0_dt,       -5.0_dt,      -5.0_dt,       -6.0_dt },
                             { -0.7310586_dt, -0.11920291_dt, -0.047425874_dt, -1.9820138_dt, -1.0066929_dt, -1.0024726_dt, -2.999089_dt, -2.0003355_dt, -2.0001235_dt,
                               -4.9999547_dt, -4.0000167_dt,  -4.000006_dt,    -7.9999976_dt, -7.000001_dt,  -7.0000005_dt, -17.0_dt,     -15.0_dt,      -14.0_dt,
                               -14.0_dt,      -20.0_dt,       -21.0_dt,        -2.0_dt,       -2.0_dt,       -3.0_dt,       -5.0_dt,      -5.0_dt,       -6.0_dt },
                             { -0.027076244_dt, -0.004414923_dt, -0.0017565137_dt, -0.07340792_dt,  -0.03728492_dt,  -0.037128616_dt, -0.11107737_dt, -0.074086495_dt, -0.07407864_dt,
                               -0.18518351_dt,  -0.14814878_dt,  -0.14814837_dt,   -0.2962962_dt,   -0.25925928_dt,  -0.25925925_dt,  -0.6296296_dt,  -0.5555556_dt,   -0.51851857_dt,
                               -0.5185185_dt,   -0.7407408_dt,   -0.7777778_dt,    -0.074074075_dt, -0.074074075_dt, -0.11111111_dt,  -0.1851852_dt,  -0.1851852_dt,   -0.22222221_dt } };
    std::string reduction[] = { "none", "sum", "mean" };

    for (size_t iter = 0; iter < std::size(reduction); ++iter)
    {
        // Initialization
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "input", "target" }, depth, height, width });
        work.add<raul::SigmoidCrossEntropyLoss>("loss", raul::LossParams{ { "input", "target" }, { "out" }, reduction[iter] });
        TENSORS_CREATE(batch);
        memory_manager["input"] = TORANGE(x);
        memory_manager["target"] = TORANGE(y);

        if (iter == 0)
        {
            memory_manager[raul::Name("out").grad()] = 1.0_dt;
        }
        work.forwardPassTraining();
        work.backwardPassTraining();

        // Checks
        const auto& xNablaTensor = memory_manager[Name("input").grad()];

        EXPECT_EQ(xNablaTensor.size(), realGrad[iter].size());

        for (size_t i = 0; i < xNablaTensor.size(); ++i)
        {
            ASSERT_TRUE(tools::expect_near_relative(xNablaTensor[i], realGrad[iter][i], eps));
        }
    }
}

}