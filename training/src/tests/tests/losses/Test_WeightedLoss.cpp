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
#include <training/base/loss/L1Loss.h>
#include <training/base/loss/LossWrapper.h>
#include <training/base/loss/SigmoidCrossEntropyLoss.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLoss, WeightedSigmoidCrossEntropyUnit)
{
    PROFILE_TEST
    const raul::dtype eps = TODTYPE(1e-4);

    const raul::Tensor inputs = { 1.3_dt, 1.2_dt, 0.1_dt, -4.0_dt, -0.3_dt, -10.0_dt, 1.0_dt, -1.0_dt, 2.0_dt, -2.3_dt };
    const raul::Tensor targets = { 0.1_dt, 1.2_dt, 1.0_dt, 0.1_dt, 7.7_dt, 0.2_dt, 0.2_dt, 0.2_dt, -1.3_dt, -2.3_dt };
    const raul::Tensor weights = {
        1.0_dt, 1.0_dt, 0.0_dt, 0.5_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
    };

    const size_t batch = 2;

    raul::LossParams::Reduction reduction[] = { raul::LossParams::Reduction::None,
                                                raul::LossParams::Reduction::Sum,
                                                raul::LossParams::Reduction::Mean,
                                                raul::LossParams::Reduction::Sum_Over_Weights,
                                                raul::LossParams::Reduction::Sum_Over_Nonzero_Weights,
                                                raul::LossParams::Reduction::None,
                                                raul::LossParams::Reduction::Sum,
                                                raul::LossParams::Reduction::Mean };
    std::string reductionName[] = { "none", "sum", "mean", "sum_over_weights", "sum_over_nonzero_weights", "none", "sum", "mean" };

    yato::dimensionality<4> expectedShapes[] = { yato::dims(batch, 1, 1, 5), yato::dims(1, 1, 1, 1),     yato::dims(1, 1, 1, 1), yato::dims(1, 1, 1, 1),
                                                 yato::dims(1, 1, 1, 1),     yato::dims(batch, 1, 1, 5), yato::dims(1, 1, 1, 1), yato::dims(1, 1, 1, 1) };
    raul::Tensor realLoss[] = { { 1.4110085_dt, 0.02328247_dt, 0.0_dt, 0.2090749_dt, 0.0_dt, 2.0000453_dt, 1.1132617_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                { 4.75667197_dt },
                                { 0.475667197_dt },
                                { 1.05703822_dt },
                                { 0.95133439_dt },
                                { 1.4110085_dt, 0.02328247_dt, 0.64439676_dt, 0.41814995_dt, 2.864355_dt, 2.0000453_dt, 1.1132617_dt, 0.5132617_dt, 4.7269278_dt, -5.194454_dt },
                                { 8.520235_dt },
                                { 0.8520235_dt } };

    raul::Tensor realInGrad[] = { { 0.68583494_dt, -0.43147528_dt, 0.0_dt, -0.04100689_dt, 0.0_dt, -0.1999546_dt, 0.53105855_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                  { 0.68583494_dt, -0.43147528_dt, 0.0_dt, -0.04100689_dt, 0.0_dt, -0.1999546_dt, 0.53105855_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                  { 0.068583494_dt, -0.043147528_dt, 0.0_dt, -0.004100689_dt, 0.0_dt, -0.01999546_dt, 0.053105855_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                  { 0.15240776_dt, -0.0958834_dt, 0.0_dt, -0.00911264_dt, 0.0_dt, -0.04443436_dt, 0.11801301_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                  { 0.13716699_dt, -0.08629506_dt, 0.0_dt, -0.00820138_dt, 0.0_dt, -0.03999092_dt, 0.10621171_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                  { 0.68583494_dt, -0.43147528_dt, -0.47502077_dt, -0.08201379_dt, -7.274442_dt, -0.1999546_dt, 0.53105855_dt, 0.06894143_dt, 2.180797_dt, 2.3911228_dt },
                                  { 0.68583494_dt, -0.43147528_dt, -0.47502077_dt, -0.08201379_dt, -7.274442_dt, -0.1999546_dt, 0.53105855_dt, 0.06894143_dt, 2.180797_dt, 2.3911228_dt },
                                  { 0.068583494_dt, -0.043147528_dt, -0.047502077_dt, -0.008201379_dt, -0.7274442_dt, -0.01999546_dt, 0.053105855_dt, 0.006894143_dt, 0.2180797_dt, 0.23911228_dt } };

    for (size_t iter = 0; iter < std::size(reduction); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        if (iter < 5)
        {
            work.add<raul::DataLayer>("data", raul::DataParams{ { "in", "target", "weight" }, expectedShapes[0][1], expectedShapes[0][2], expectedShapes[0][3] });
            work.add<raul::SigmoidCrossEntropyLoss>("loss", raul::LossParams{ { "in", "target", "weight" }, { "loss" }, reduction[iter] });
        }
        else
        {
            work.add<raul::DataLayer>("data", raul::DataParams{ { "in", "target" }, expectedShapes[0][1], expectedShapes[0][2], expectedShapes[0][3] });
            work.add<raul::SigmoidCrossEntropyLoss>("loss", raul::LossParams{ { "in", "target" }, { "loss" }, reduction[iter] });
        }
        TENSORS_CREATE(batch);
        memory_manager["in"] = TORANGE(inputs);
        memory_manager["target"] = TORANGE(targets);
        if (iter < 5)
        {
            memory_manager["weight"] = TORANGE(weights);
        }
        if (iter == 5)
        {
            memory_manager[raul::Name("loss").grad()] = 1.0_dt;
        }

        work.forwardPassTraining();
        const raul::Tensor& loss = memory_manager["loss"];

        for (size_t i = 0; i < realLoss[iter].size(); ++i)
        {
            EXPECT_NEAR(loss[i], realLoss[iter][i], eps);
        }
        printf(" - Weighted SigmoidCrossEntropyLoss[reduction=%s] forward is Ok.\n", reductionName[iter].c_str());

        work.backwardPassTraining();
        const raul::Tensor& in_nabla = memory_manager[raul::Name("in").grad()];

        for (size_t i = 0; i < in_nabla.size(); ++i)
        {
            EXPECT_NEAR(in_nabla[i], realInGrad[iter][i], eps);
        }
        printf(" - Weighted SigmoidCrossEntropyLoss[reduction=%s] backward is Ok.\n", reductionName[iter].c_str());
    }
}

TEST(TestLoss, WeightedL1LossUnit)
{
    PROFILE_TEST

    const raul::dtype eps = TODTYPE(1e-4);

    const raul::Tensor inputs = { 1.3_dt, 1.2_dt, 0.1_dt, -4.0_dt, -0.3_dt, -10.0_dt, 1.0_dt, -1.0_dt, 2.0_dt, -2.3_dt };
    const raul::Tensor targets = { 0.1_dt, 1.2_dt, 1.0_dt, 0.1_dt, 7.7_dt, 0.2_dt, 0.2_dt, 0.2_dt, -1.3_dt, -2.3_dt };
    const raul::Tensor weights = {
        1.0_dt, 1.0_dt, 0.0_dt, 0.5_dt, 0.0_dt, 1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
    };

    const size_t batch = 2;

    raul::LossParams::Reduction reduction[] = { raul::LossParams::Reduction::None,
                                                raul::LossParams::Reduction::Sum,
                                                raul::LossParams::Reduction::Mean,
                                                raul::LossParams::Reduction::Sum_Over_Weights,
                                                raul::LossParams::Reduction::Sum_Over_Nonzero_Weights,
                                                raul::LossParams::Reduction::None,
                                                raul::LossParams::Reduction::Sum,
                                                raul::LossParams::Reduction::Mean };
    std::string reductionName[] = { "none", "sum", "mean", "sum_over_weights", "sum_over_nonzero_weights", "none", "sum", "mean" };

    yato::dimensionality<4> expectedShapes[] = { yato::dims(batch, 1, 1, 5), yato::dims(1, 1, 1, 1),     yato::dims(1, 1, 1, 1), yato::dims(1, 1, 1, 1),
                                                 yato::dims(1, 1, 1, 1),     yato::dims(batch, 1, 1, 5), yato::dims(1, 1, 1, 1), yato::dims(1, 1, 1, 1) };

    raul::Tensor realLoss[] = { { 1.1999999285_dt, 0.0_dt, 0.0_dt, 2.0499999523_dt, 0.0_dt, 10.1999998093_dt, 0.8000000119_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                { 14.24999974493_dt },
                                { 1.424999974493_dt },
                                { 3.166666609984_dt },
                                { 2.849999948986_dt },
                                { 1.1999999285_dt, 0.0_dt, 0.9_dt, 4.1_dt, 8.0_dt, 10.1999998093_dt, 0.8000000119_dt, 1.2_dt, 3.3_dt, 0.0_dt },
                                { 29.7_dt },
                                { 2.97_dt } };
    raul::Tensor realInGrad[] = { { 1.0_dt, 0.0_dt, 0.0_dt, -0.5_dt, 0.0_dt, -1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                  { 1.0_dt, 0.0_dt, 0.0_dt, -0.5_dt, 0.0_dt, -1.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                  { 0.1_dt, 0.0_dt, 0.0_dt, -0.05_dt, 0.0_dt, -0.1_dt, 0.1_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                  { 0.222222_dt, 0.0_dt, 0.0_dt, -0.111111_dt, 0.0_dt, -0.222222_dt, 0.222222_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                  { 0.2_dt, 0.0_dt, 0.0_dt, -0.1_dt, 0.0_dt, -0.2_dt, 0.2_dt, 0.0_dt, 0.0_dt, 0.0_dt },
                                  { 1.0_dt, 0.0_dt, -1.0_dt, -1.0_dt, -1.0_dt, -1.0_dt, 1.0_dt, -1.0_dt, 1.0_dt, 0.0_dt },
                                  { 1.0_dt, 0.0_dt, -1.0_dt, -1.0_dt, -1.0_dt, -1.0_dt, 1.0_dt, -1.0_dt, 1.0_dt, 0.0_dt },
                                  { 0.1_dt, 0.0_dt, -0.1_dt, -0.1_dt, -0.1_dt, -0.1_dt, 0.1_dt, -0.1_dt, 0.1_dt, 0.0_dt } };

    for (size_t iter = 0; iter < std::size(reduction); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        if (iter < 5)
        {
            work.add<raul::DataLayer>("data", raul::DataParams{ { "in", "target", "weight" }, expectedShapes[0][1], expectedShapes[0][2], expectedShapes[0][3] });
            work.add<raul::L1Loss>("loss", raul::LossParams{ { "in", "target", "weight" }, { "loss" }, reduction[iter] });
        }
        else
        {
            work.add<raul::DataLayer>("data", raul::DataParams{ { "in", "target" }, expectedShapes[0][1], expectedShapes[0][2], expectedShapes[0][3] });
            work.add<raul::L1Loss>("loss", raul::LossParams{ { "in", "target" }, { "loss" }, reduction[iter] });
        }
        TENSORS_CREATE(batch);
        memory_manager["in"] = TORANGE(inputs);
        memory_manager["target"] = TORANGE(targets);
        if (iter < 5)
        {
            memory_manager["weight"] = TORANGE(weights);
        }
        if (iter == 5)
        {
            memory_manager[raul::Name("loss").grad()] = 1.0_dt;
        }

        work.forwardPassTraining();

        const raul::Tensor& loss = memory_manager["loss"];

        for (size_t i = 0; i < realLoss[iter].size(); ++i)
        {
            EXPECT_NEAR(loss[i], realLoss[iter][i], eps);
        }
        printf(" - Weighted L1Loss[reduction=%s] forward is Ok.\n", reductionName[iter].c_str());

        work.backwardPassTraining();

        const raul::Tensor& in_nabla = memory_manager[raul::Name("in").grad()];

        for (size_t i = 0; i < in_nabla.size(); ++i)
        {
            EXPECT_NEAR(in_nabla[i], realInGrad[iter][i], eps);
        }
        printf(" - Weighted L1Loss[reduction=%s] backward is Ok.\n", reductionName[iter].c_str());
    }
}

}