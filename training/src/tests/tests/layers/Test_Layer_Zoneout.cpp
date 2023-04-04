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

#include <training/base/initializers/ConstantInitializer.h>
#include <training/base/initializers/RandomUniformInitializer.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/composite/rnn/ZoneoutLayer.h>
#include <training/compiler/Layers.h>

namespace UT
{

using namespace std;
using namespace raul;

/*namespace
{

dtype outputs_test_golden(const dtype prev, const dtype curr, const dtype prob)
{
    return prob * prev + (1.0_dt - prob) * curr;
}

}*/

TEST(TestZoneout, SimpleTestForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    [[maybe_unused]] const auto eps_rel = 1e-5_dt;
    const auto probability = random::uniform::rand<raul::dtype>(0., 1.);
    const auto input_size = (size_t)random::uniform::rand<int>(1, 1000);
    const auto batch_size = random::uniform::rand<int>(1, 1000);

    cout << "Test with p=" << probability << " and shape (" << batch_size << "," << input_size << ")" << endl;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Network
    work.add<DataLayer>("input", DataParams{ { "current", "previous" }, 1, 1, input_size });
    const auto params = ZoneoutParams{ { "current", "previous" }, { "out" }, probability };
    ZoneoutLayer zo("zoneout", params, networkParameters);
    TENSORS_CREATE(batch_size)
    initializers::RandomUniformInitializer initializer{ -1e3_dt, 1e3_dt };
    initializer(memory_manager["current"]);
    initializer(memory_manager["previous"]);
    // Apply
    EXPECT_THROW(zo.forwardCompute(NetworkMode::Test), raul::Exception);

    // Checks
    /*const auto& outputs = memory_manager["out"];
    const auto& current = memory_manager["current"];
    const auto& previous = memory_manager["previous"];

    EXPECT_EQ(outputs.size(), current.size());
    EXPECT_EQ(outputs.size(), previous.size());

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        const auto val = outputs[i];
        const auto golden_val = outputs_test_golden(previous[i], current[i], probability);
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }*/
}

TEST(TestZoneout, SimpleTrainForwardP0RandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto probability = 0.0_dt;
    const auto input_size = 1'000'000;
    const auto batch_size = 1U;

    cout << "Test with p=" << probability << " and shape (" << batch_size << "," << input_size << ")" << endl;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Network
    work.add<DataLayer>("input", DataParams{ { "current", "previous" }, 1, 1, input_size });
    const auto params = ZoneoutParams{ { "current", "previous" }, { "out" }, probability };
    ZoneoutLayer zo("zoneout", params, networkParameters);
    TENSORS_CREATE(batch_size)
    initializers::RandomUniformInitializer initializer{ -1e3_dt, 1e3_dt };
    initializer(memory_manager["current"]);
    initializer(memory_manager["previous"]);

    // Apply
    zo.forwardCompute(NetworkMode::Train);

    const auto& outputs = memory_manager["out"];
    const auto& current = memory_manager["current"];

    EXPECT_EQ(outputs.size(), current.size());

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        EXPECT_EQ(outputs[i], current[i]);
    }
}

TEST(TestZoneout, SimpleTrainForwardP1RandUnit)
{
    PROFILE_TEST
    const auto probability = 1.0_dt;
    const auto input_size = 1'000'000;
    const auto batch_size = 1U;

    cout << "Test with p=" << probability << " and shape (" << batch_size << "," << input_size << ")" << endl;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Network
    work.add<DataLayer>("input", DataParams{ { "current", "previous" }, 1, 1, input_size });
    const auto params = ZoneoutParams{ { "current", "previous" }, { "out" }, probability };
    EXPECT_NO_THROW(ZoneoutLayer("zoneout", params, networkParameters));
}

TEST(TestZoneout, SimpleTrainForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto probability = random::uniform::rand<raul::dtype>(0., 1.);
    const auto input_size = 1'000'000;
    const auto batch_size = 1U;

    cout << "Test with p=" << probability << " and shape (" << batch_size << "," << input_size << ")" << endl;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);
    // Network
    work.add<DataLayer>("input", DataParams{ { "current", "previous" }, 1, 1, input_size });
    const auto params = ZoneoutParams{ { "current", "previous" }, { "out" }, probability };
    ZoneoutLayer zo("zoneout", params, networkParameters);
    TENSORS_CREATE(batch_size)

    size_t curr_cnt = 0U;
    size_t prev_cnt = 0U;
    initializers::RandomUniformInitializer initializer{ -1e3_dt, 1e3_dt };
    initializer(memory_manager["current"]);
    initializer(memory_manager["previous"]);

    // Apply
    zo.forwardCompute(NetworkMode::Train);

    const auto& outputs = memory_manager["out"];
    const auto& current = memory_manager["current"];
    const auto& previous = memory_manager["previous"];

    EXPECT_EQ(outputs.size(), current.size());
    EXPECT_EQ(outputs.size(), previous.size());

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        if (outputs[i] == current[i])
        {
            ++curr_cnt;
        }

        if (outputs[i] == previous[i])
        {
            ++prev_cnt;
        }
    }

    // Checks
    dtype curr_prob = static_cast<dtype>(curr_cnt) / static_cast<dtype>(outputs.size());
    dtype prev_prob = static_cast<dtype>(prev_cnt) / static_cast<dtype>(outputs.size());

    // Assumption
    ASSERT_TRUE(TODTYPE(outputs.size()) * curr_prob * (1.0_dt - curr_prob) >= 10.0_dt);
    ASSERT_TRUE(TODTYPE(outputs.size()) * prev_prob * (1.0_dt - prev_prob) >= 10.0_dt);

    // The confident interval for p estimation
    const auto z_ci = 4.417_dt; // 99.999%
    const auto prev_ci = z_ci * sqrt(prev_prob * (1.0_dt - prev_prob) / TODTYPE(outputs.size()));
    const auto curr_ci = z_ci * sqrt(curr_prob * (1.0_dt - curr_prob) / TODTYPE(outputs.size()));

    cout << "[prev prob] expected: " << probability << ", got: " << prev_prob << ", ci: " << prev_ci << endl;
    cout << "[curr prob] expected: " << 1.0_dt - probability << ", got: " << curr_prob << ", ci: " << curr_ci << endl;
    EXPECT_NEAR(prev_prob, probability, prev_ci);
    EXPECT_NEAR(curr_prob, 1.0_dt - probability, curr_ci);
}

TEST(TestZoneout, SimpleTrainBackwardRandUnit)
{
    PROFILE_TEST
    using namespace raul;

    // Test parameters
    const auto probability = random::uniform::rand<raul::dtype>(0., 1.);
    const auto input_size = 10'000;
    const auto batch_size = 1U;

    cout << "Test with p=" << probability << " and shape (" << batch_size << "," << input_size << ")" << endl;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);
    memory_manager.createTensor(Name("out").grad(), batch_size, 1, 1, input_size);

    // Network
    work.add<DataLayer>("input", DataParams{ { "current", "previous" }, 1, 1, input_size });
    const auto params = ZoneoutParams{ { "current", "previous" }, { "out" }, probability };
    ZoneoutLayer zo("zoneout", params, networkParameters);
    TENSORS_CREATE(batch_size)

    // Apply
    initializers::RandomUniformInitializer initializer{ -1e3_dt, 1e3_dt };
    initializer(memory_manager["current"]);
    initializer(memory_manager["previous"]);

    auto& outputs = memory_manager["out"];
    auto& current = memory_manager["current"];
    auto& previous = memory_manager["previous"];

    zo.forwardCompute(NetworkMode::Train);

    Tensor outputs_buffer = TORANGE(outputs);
    Tensor current_buffer = TORANGE(current);
    Tensor previous_buffer = TORANGE(previous);

    memory_manager[Name("out").grad()].memAllocate(nullptr);
    initializer(memory_manager[Name("out").grad()]);

    zo.backwardCompute();

    const auto& outputs_grad = memory_manager[Name("out").grad()];
    const auto& current_grad = memory_manager[Name("current").grad()];
    const auto& previous_grad = memory_manager[Name("previous").grad()];

    EXPECT_EQ(outputs_grad.size(), current_grad.size());
    EXPECT_EQ(outputs_grad.size(), previous_grad.size());

    for (size_t i = 0; i < outputs_buffer.size(); ++i)
    {
        //        cout << i << " prev: " << previous_grad[i] << ", curr: " << current_grad[i] << ", out: " << outputs_grad[i] << endl;

        if (outputs_buffer[i] == current_buffer[i])
        {
            EXPECT_TRUE(current_grad[i] == outputs_grad[i]);
            EXPECT_TRUE(previous_grad[i] == 0.0_dt);
        }

        if (outputs_buffer[i] == previous_buffer[i])
        {
            EXPECT_TRUE(previous_grad[i] == outputs_grad[i]);
            EXPECT_TRUE(current_grad[i] == 0.0_dt);
        }
    }
}

#ifdef ANDROID
TEST(TestZoneout, SimpleTrainForwardP0RandFP16Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto probability = 0.0_dt;
    const auto input_size = 1'000'000;
    const auto batch_size = 1U;

    cout << "Test with p=" << probability << " and shape (" << batch_size << "," << input_size << ")" << endl;

    // Initialization
    raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16 };
    auto& memory_manager = work.getMemoryManager<raul::MemoryManagerFP16>();
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Network
    work.add<DataLayer>("input", DataParams{ { "current", "previous" }, 1, 1, input_size });
    const auto params = ZoneoutParams{ { "current", "previous" }, { "out" }, probability };
    ZoneoutLayer zo("zoneout", params, networkParameters);
    TENSORS_CREATE(batch_size)

    tools::init_rand_tensor("current", { TOHTYPE(-1e3_hf), 1e3_hf }, memory_manager);
    tools::init_rand_tensor("previous", { TOHTYPE(-1e3_hf), 1e3_hf }, memory_manager);

    // Apply
    zo.forwardCompute(NetworkMode::Train);

    const auto& outputs = memory_manager["out"];
    const auto& current = memory_manager["current"];

    EXPECT_EQ(outputs.size(), current.size());

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        EXPECT_TRUE(outputs[i] == current[i]);
    }
}
#endif // ANDROID

} // UT namespace
