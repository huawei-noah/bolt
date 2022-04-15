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

#include <training/base/layers/activations/LogSoftMaxActivation.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/loss/KLDivLoss.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLoss, KLDivLossUnit)
{
    PROFILE_TEST
    using namespace raul;

    dtype eps = TODTYPE(1e-4);

    const Tensor raw = { 1, 2, 5, -1, -2, 5, -1, 3 };

    const Tensor targetProbabilities = { 0.7f, 0.0f, 0.2f, 0.1f, 0.2f, 0.2f, 0.2f, 0.4f };
    const size_t batch = 2;

    std::string reduction[] = { "none", "sum", "batch_mean", "mean" };
    const Tensor realOut = { -4.0682f, -3.0682f, -0.0682f, -6.0682f, -7.1299f, -0.1299f, -6.1299f, -2.1299f };
    const Tensor realLoss[] = { { 2.5981f, 0.0000f, -0.3082f, 0.3766f, 1.1041f, -0.2959f, 0.9041f, 0.4854f }, { 4.8641f }, { 2.4321f }, { 0.6080f } };

    const Tensor realOutGrad[] = { { 0.0_dt },
                                   { -0.7000f, 0.0000f, -0.2000f, -0.1000f, -0.2000f, -0.2000f, -0.2000f, -0.4000f },
                                   { -0.3500f, 0.0000f, -0.1000f, -0.0500f, -0.1000f, -0.1000f, -0.1000f, -0.2000f },
                                   { -0.0875f, 0.0000f, -0.0250f, -0.0125f, -0.0250f, -0.0250f, -0.0250f, -0.0500f } };

    const Tensor realInGrad[] = { { 0.0_dt },
                                  { -0.6829f, 0.0465f, 0.7341f, -0.0977f, -0.1992f, 0.6782f, -0.1978f, -0.2812f },
                                  { -0.3414f, 0.0233f, 0.3670f, -0.0488f, -0.0996f, 0.3391f, -0.0989f, -0.1406f },
                                  { -0.0854f, 0.0058f, 0.0918f, -0.0122f, -0.0249f, 0.0848f, -0.0247f, -0.0352f } };

    for (size_t iter = 0; iter < std::size(reduction); ++iter)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);
        networkParameters.mLossReductionCoefficient = batch;

        work.add<DataLayer>("data", DataParams{ { "in", "labels" }, 1u, 1u, raw.size() / batch });
        work.add<LogSoftMaxActivation>("logsoftmax", BasicParamsWithDim{ { "in" }, { "out" } });
        work.add<KLDivLoss>("loss", LossParams{ { "out", "labels" }, { "loss" }, reduction[iter].c_str() });

        TENSORS_CREATE(batch);
        memory_manager["in"] = TORANGE(raw);
        memory_manager["labels"] = TORANGE(targetProbabilities);

        work.forwardPassTraining();
        const Tensor& out = memory_manager["out"];
        const Tensor& loss = memory_manager["loss"];

        if (iter == 0)
        {
            for (size_t i = 0; i < out.size(); ++i)
            {
                EXPECT_NEAR(out[i], realOut[i], eps);
            }
        }

        for (size_t i = 0; i < loss.size(); ++i)
        {
            EXPECT_NEAR(loss[i], realLoss[iter][i], eps);
        }
        printf(" - KLDivLoss[reduction=%s] forward is Ok.\n", reduction[iter].c_str());

        if (iter > 0)
        {
            work.backwardPassTraining();

            const Tensor& out_nabla = memory_manager[raul::Name("out").grad()];
            const Tensor& in_nabla = memory_manager[raul::Name("in").grad()];

            for (size_t i = 0; i < out_nabla.size(); ++i)
            {
                EXPECT_NEAR(out_nabla[i], realOutGrad[iter][i], eps);
            }
            printf(" - KLDivLoss[reduction=%s] backward is Ok.\n", reduction[iter].c_str());

            for (size_t i = 0; i < in_nabla.size(); ++i)
            {
                EXPECT_NEAR(in_nabla[i], realInGrad[iter][i], eps);
            }
            printf(" - LogSoftMax backward is Ok.\n");
        }
    }
}

}