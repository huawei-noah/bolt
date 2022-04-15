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

#include <training/base/initializers/RandomUniformInitializer.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/DropoutLayer.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerDropout, TestForward1RandUnit)
{
    PROFILE_TEST
    constexpr auto eps = 1e-6_dt;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    raul::Tensor raw = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    size_t batch = 1;

    constexpr raul::dtype probability = 0.3_dt;
    work.add<raul::DataLayer>("data", raul::DataParams{ { "in", "labels" }, 1, 3, 3, 1 });
    raul::DropoutLayer drop("drop", raul::DropoutParams{ { "in" }, { "drop" }, probability }, networkParameters);
    TENSORS_CREATE(batch);
    memory_manager["in"] = TORANGE(raw);

    const raul::Tensor& out = memory_manager["drop"];
    drop.forwardCompute(raul::NetworkMode::Train);
    constexpr auto scale = 1.0_dt / (1.0_dt - probability);
    for (size_t i = 0; i < out.size(); ++i)
    {
        if (out[i] != 0.0_dt)
        {
            EXPECT_NEAR(out[i], raw[i] * scale, eps);
        }
    }

    printf(" - Dropout forward is Ok.\n");
}

TEST(TestLayerDropout, TestForwardRandSeedUnit)
{
    PROFILE_TEST

    raul::random::setGlobalSeed(48);

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const raul::Tensor raw = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    size_t batch = 2;

    constexpr auto probability = 0.3_dt;
    work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, 1u, 3u, 3u });
    work.add<raul::DropoutLayer>("drop", raul::DropoutParams{ { "in" }, { "drop" }, probability });
    TENSORS_CREATE(batch);
    memory_manager["in"] = TORANGE(raw);

    const raul::Tensor& out = memory_manager["drop"];
    work.forwardPassTraining();
    auto avg = 0_dt;
    for (size_t i = 0; i < out.size(); ++i)
    {
        avg += out[i];
    }
    avg /= TODTYPE(out.size());

    printf(" - Dropout mean: %f\n", avg);
}

TEST(TestLayerDropout, TestForward2RandUnit)
{
    PROFILE_TEST
    constexpr auto probability = 0.35_dt;
    const size_t batch = 1;
    const auto random_range = raul::random::dtypeRange{ 1.0_dt, 100.0_dt };

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    raul::Tensor raw(1, 1, 1, 1000);
    auto filler = [&random_range]() { return raul::random::uniform::rand<raul::dtype>(random_range); };
    std::generate(raw.begin(), raw.end(), filler);

    work.add<raul::DataLayer>("data", raul::DataParams{ { "in", "labels" }, 1, 10, 100, 1 });
    raul::DropoutLayer drop("drop", raul::DropoutParams{ { "in" }, { "drop" }, probability }, networkParameters);
    TENSORS_CREATE(batch);
    memory_manager["in"] = TORANGE(raw);

    const raul::Tensor& out = memory_manager["drop"];
    drop.forwardCompute(raul::NetworkMode::Train);
    size_t count = 0;
    for (raul::dtype x : out)
    {
        if (x == 0.0_dt) count++;
    }
    raul::dtype variance = probability * (1.0_dt - probability);
    raul::dtype mean = TODTYPE(count) / TODTYPE(out.size());
    printf(" - Probability = 0.35, mean in output is %f\n", mean);
    EXPECT_NEAR(mean, probability, variance);
}

TEST(TestLayerDropout, TestBackwardRandUnit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    const auto probability = raul::random::uniform::rand<raul::dtype>(0.0_dt, 1.0_dt);
    const size_t n = 1'000'000;
    const size_t batch = 1;

    std::cout << "Test with p=" << probability << std::endl;

    raul::initializers::RandomUniformInitializer initializer_uniform{ -1e3_dt, 1e3_dt };

    work.add<raul::DataLayer>("data", raul::DataParams{ { "in", "labels" }, 1, 1, n, 1 });
    raul::DropoutLayer drop("drop", raul::DropoutParams{ { "in" }, { "drop" }, 0_dt }, networkParameters);
    TENSORS_CREATE(batch);
    initializer_uniform(memory_manager["in"]);
    initializer_uniform(memory_manager[raul::Name("drop").grad()]);

    drop.forwardCompute(raul::NetworkMode::Train);
    drop.backwardCompute();

    // Check
    const raul::Tensor& out = memory_manager["drop"];
    const raul::Tensor& out_grad = memory_manager[raul::Name("drop").grad()];
    const raul::Tensor& in_grad = memory_manager[raul::Name("in").grad()];
    for (size_t i = 0; i < out.size(); ++i)
    {
        if (out[i] == 0.0_dt)
        {
            EXPECT_TRUE(in_grad[i] == 0.0_dt);
        }
        else
        {
            EXPECT_TRUE(in_grad[i] == out_grad[i]);
        }
    }
}

} // namespace UT