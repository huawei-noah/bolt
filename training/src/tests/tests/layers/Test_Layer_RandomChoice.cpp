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
#include <training/base/layers/basic/RandomChoiceLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerRandomChoice, BadParamsUnit)
{
    PROFILE_TEST
    const size_t batch = 2;
    const size_t depth = 3;
    const size_t height = 4;
    const size_t width = 5;

    const raul::RandomChoiceParams params[] = {
        { { "1", "2" }, "out", { 0.3f, 0.5f, 2.f } },
        { { "1", "2" }, "out", { 1.3f } },
        { { "1", "2", "3" }, "out", { 0.5f } },
    };

    for (auto& p : params)
    {
        // Initialization
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        for (size_t i = 0; i < p.getInputs().size(); ++i)
        {
            work.tensorNeeded(std::to_string(i + 1), std::to_string(i + 1), raul::WShape{ raul::BS(), depth, height, width }, DEC_FORW_READ_NOMEMOPT);
        }

        TENSORS_CREATE(batch);

        // Apply function
        EXPECT_THROW(raul::RandomChoiceLayer random("random", p, networkParameters), raul::Exception);
    }
}

TEST(TestLayerRandomChoice, RandomChoiceUnit)
{
    PROFILE_TEST
    const size_t batch = 2;
    const size_t depth = 3;
    const size_t height = 4;
    const size_t width = 5;

    const raul::RandomChoiceParams params[] = {
        { { "1", "2" }, "out", { 0.3f } },
        { { "1", "2" }, "out", { 2.f, 3.f } },
        { { "1", "2", "3" }, "out", { 2.f, 3.f, 5.f } },
    };

    for (auto& p : params)
    {
        // Initialization
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        for (size_t i = 0; i < p.getInputs().size(); ++i)
        {
            work.tensorNeeded(std::to_string(i + 1), std::to_string(i + 1), raul::WShape{ raul::BS(), depth, height, width }, DEC_FORW_READ_NOMEMOPT);
        }

        // Apply function
        raul::RandomChoiceLayer random("random", p, networkParameters);
        TENSORS_CREATE(batch);
        for (size_t i = 0; i < p.getInputs().size(); ++i)
        {
            memory_manager[std::to_string(i + 1)] = TORANGE(*memory_manager.createTensor("temp_" + std::to_string(i + 1), batch, depth, height, width, TODTYPE(i + 1)));
        }

        random.forwardCompute(raul::NetworkMode::Test);

        const auto& outTensor = memory_manager["out"];

        EXPECT_EQ(outTensor.size(), batch * depth * height * width);

        auto outVal = outTensor[0];

        for (auto v : outTensor)
        {
            EXPECT_EQ(outVal, v);
        }

        memory_manager[raul::Name("out").grad()] = TORANGE(*memory_manager.createTensor("gradient", batch, depth, height, width, 1.0_dt));

        random.backwardCompute();
        size_t ind = static_cast<size_t>(outVal) - 1;

        for (size_t i = 0; i < 2; ++i)
        {
            const auto& t = memory_manager[raul::Name(std::to_string(i + 1)).grad()];
            auto grad = 0_dt;
            if (i == ind)
            {
                grad = 1_dt;
            }

            for (auto v : t)
            {
                EXPECT_EQ(grad, v);
            }
        }
    }
}

TEST(TestLayerRandomChoice, UniformRandUnit)
{
    PROFILE_TEST
    const auto eps = 1e-5_dt;
    const auto tensor_cnt = 10U;
    const size_t batch = 1;
    const size_t depth = 1;
    const size_t height = 1;
    const size_t width = 1;
    const size_t repeat = 1'000'000;
    const raul::dtype prob_val = raul::random::uniform::rand<raul::dtype>(0., 100.);
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    raul::Names tensors;

    for (size_t i = 0; i < tensor_cnt; ++i)
    {
        const size_t value = i;
        const auto name = std::to_string(value);
        work.tensorNeeded(name, name, raul::WShape{ raul::BS(), depth, height, width }, DEC_FORW_READ_NOMEMOPT);
        memory_manager.createTensor(name, batch, depth, height, width, TODTYPE(value));
        tensors.push_back(name);
    }

    raul::RandomChoiceLayer random("random", raul::RandomChoiceParams(tensors, "out", std::vector<raul::dtype>(tensor_cnt, prob_val)), networkParameters);
    TENSORS_CREATE(batch);
    for (size_t i = 0; i < tensor_cnt; ++i)
    {
        const size_t value = i;
        const auto name = std::to_string(value);
        memory_manager[name] = TORANGE(*memory_manager.createTensor("temp_" + name, batch, depth, height, width, TODTYPE(value)));
    }
    // Apply
    std::vector<raul::dtype> distr(tensor_cnt);

    for (size_t i = 0; i < repeat; ++i)
    {
        random.forwardCompute(raul::NetworkMode::Train);
        const size_t val = static_cast<size_t>(memory_manager["out"][0]);
        ++distr[val];
    }

    std::transform(distr.cbegin(), distr.cend(), distr.begin(), [&](raul::dtype x) { return x / TODTYPE(repeat); });

    // Check
    std::vector<raul::dtype> golden_distr(tensor_cnt, 1.0_dt / TODTYPE(tensor_cnt));

    raul::dtype kld = 0.0_dt;

    for (size_t i = 0; i < tensor_cnt; ++i)
    {
        const auto p = distr[i];
        const auto q = golden_distr[i];
        kld += p * std::log(p / q);
    }

    EXPECT_NEAR(kld, 0.0_dt, eps);
}

}