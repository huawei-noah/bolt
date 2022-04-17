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

#include <training/base/layers/basic/RandomSelectLayer.h>

namespace UT
{

namespace
{

template<typename T>
constexpr auto golden_select_layer(const bool cond, const T x, const T y)
{
    return cond ? x : y;
}

template<typename T>
constexpr auto golden_select_layer_grad(const bool cond, const T grad)
{
    return std::make_pair(static_cast<T>(cond) * grad, grad * (static_cast<T>(1) - static_cast<T>(cond)));
}

}

TEST(TestLayerRandomSelect, OutputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::RandomSelectLayer("select", raul::RandomSelectParams{ { "x", "y" }, { "x_out", "y_out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerRandomSelect, IncorrectInputNumUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::RandomSelectLayer("select", raul::RandomSelectParams{ { "cond", "x", "y" }, { "out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerRandomSelect, ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto tensor_size = 30U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    std::vector<raul::dtype> probas = { 1.0_dt, 0.0_dt, 0.5_dt };

    for (size_t i = 0; i < probas.size(); ++i)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
        work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

        // Apply function
        raul::RandomSelectLayer select("select", raul::RandomSelectParams{ { "x", "y" }, { "out" }, probas[i] }, networkParameters);
        TENSORS_CREATE(tensor_size);
        tools::init_rand_tensor("x", random_range, memory_manager);
        tools::init_rand_tensor("y", random_range, memory_manager);

        select.forwardCompute(raul::NetworkMode::Train);

        // Checks
        const auto& x_tensor = memory_manager["x"];
        const auto& y_tensor = memory_manager["y"];
        const auto& out_tensor = memory_manager["out"];

        EXPECT_EQ(out_tensor.size(), x_tensor.size());
        EXPECT_EQ(out_tensor.size(), y_tensor.size());

        if (i != probas.size() - 1)
        {
            for (size_t j = 0; j < out_tensor.size(); ++j)
            {
                const auto golden_out_value = golden_select_layer(static_cast<bool>(probas[i]), x_tensor[j], y_tensor[j]);
                EXPECT_EQ(golden_out_value, out_tensor[j]);
            }
        }
        else
        {
            raul::dtype part = 0.0_dt;
            for (size_t j = 0; j < out_tensor.size(); ++j)
            {
                part += static_cast<raul::dtype>(out_tensor[j] == x_tensor[j]);
            }
            std::cout << "Elements chosen from first tensor: " << part / static_cast<raul::dtype>(out_tensor.size()) << "% (probability = 50%)" << std::endl;
        }
    }
}

TEST(TestLayerRandomSelect, BackwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto tensor_size = 30U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    const raul::Tensor deltas(tensor_size, 1.0_dt);

    std::vector<raul::dtype> probas = { 1.0_dt, 0.0_dt, 0.5_dt };

    for (size_t i = 0; i < probas.size(); ++i)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
        work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

        // Apply function
        raul::RandomSelectLayer select("select", raul::RandomSelectParams{ { "x", "y" }, { "out" }, probas[i] }, networkParameters);
        TENSORS_CREATE(tensor_size);
        tools::init_rand_tensor("x", random_range, memory_manager);
        tools::init_rand_tensor("y", random_range, memory_manager);
        memory_manager[raul::Name("out").grad()] = TORANGE(deltas);

        select.forwardCompute(raul::NetworkMode::Train);
        select.backwardCompute();

        // Checks
        const auto& x_tensor = memory_manager["x"];
        const auto& y_tensor = memory_manager["y"];
        const auto& out_tensor = memory_manager["out"];

        const auto& x_nabla = memory_manager["xGradient"];
        const auto& y_nabla = memory_manager["yGradient"];
        const auto& out_nabla = memory_manager["outGradient"];

        EXPECT_EQ(x_tensor.size(), x_nabla.size());
        EXPECT_EQ(y_tensor.size(), y_nabla.size());

        if (i != probas.size() - 1)
        {
            for (size_t j = 0; j < out_tensor.size(); ++j)
            {
                const auto [golden_x_nabla, golden_y_nabla] = golden_select_layer_grad(static_cast<bool>(probas[i]), out_nabla[j]);
                EXPECT_EQ(golden_x_nabla, x_nabla[j]);
                EXPECT_EQ(golden_y_nabla, y_nabla[j]);
            }
        }
    }
}

#ifdef ANDROID

TEST(TestLayerRandomSelect, ForwardFP16Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto tensor_size = 60U;
    const auto random_range = std::make_pair(0.0_hf, 100.0_hf);

    std::vector<raul::dtype> probas = { 1.0_dt, 0.0_dt, 0.5_dt };

    for (size_t i = 0; i < probas.size(); ++i)
    {
        raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16 };
        auto& memory_manager = work.getMemoryManager<raul::MemoryManagerFP16>();
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
        work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

        // Apply function
        raul::RandomSelectLayer select("select", raul::RandomSelectParams{ { "x", "y" }, { "out" }, probas[i] }, networkParameters);
        TENSORS_CREATE(tensor_size);
        tools::init_rand_tensor("x", random_range, memory_manager);
        tools::init_rand_tensor("y", random_range, memory_manager);

        select.forwardCompute(raul::NetworkMode::Train);

        // Checks
        const auto& x_tensor = memory_manager["x"];
        const auto& y_tensor = memory_manager["y"];
        const auto& out_tensor = memory_manager["out"];

        EXPECT_EQ(out_tensor.size(), x_tensor.size());
        EXPECT_EQ(out_tensor.size(), y_tensor.size());

        if (i != probas.size() - 1)
        {
            for (size_t j = 0; j < out_tensor.size(); ++j)
            {
                const auto golden_out_value = golden_select_layer(static_cast<bool>(probas[i]), x_tensor[j], y_tensor[j]);
                EXPECT_TRUE(golden_out_value == out_tensor[j]);
            }
        }
        else
        {
            raul::dtype part = 0.0_dt;
            for (size_t j = 0; j < out_tensor.size(); ++j)
            {
                part += static_cast<raul::dtype>(out_tensor[j] == x_tensor[j]);
            }
            std::cout << "Elements chosen from first tensor: " << part / static_cast<raul::dtype>(out_tensor.size()) << "% (probability = 50%)" << std::endl;
        }
    }
}

TEST(TestLayerRandomSelect, BackwardFP16Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto tensor_size = 30U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    const raul::Tensor deltas(tensor_size, 1.0_hf);

    std::vector<raul::dtype> probas = { 1.0_dt, 0.0_dt, 0.5_dt };

    for (size_t i = 0; i < probas.size(); ++i)
    {
        raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16 };
        auto& memory_manager = work.getMemoryManager<raul::MemoryManagerFP16>();
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
        work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

        // Apply function
        raul::RandomSelectLayer select("select", raul::RandomSelectParams{ { "x", "y" }, { "out" }, probas[i] }, networkParameters);
        TENSORS_CREATE(tensor_size);
        tools::init_rand_tensor("x", random_range, memory_manager);
        tools::init_rand_tensor("y", random_range, memory_manager);
        memory_manager[raul::Name("out").grad()] = TORANGE(deltas);

        select.forwardCompute(raul::NetworkMode::Train);
        select.backwardCompute();

        // Checks
        const auto& x_tensor = memory_manager["x"];
        const auto& y_tensor = memory_manager["y"];
        const auto& out_tensor = memory_manager["out"];

        const auto& x_nabla = memory_manager["xGradient"];
        const auto& y_nabla = memory_manager["yGradient"];
        const auto& out_nabla = memory_manager["outGradient"];

        EXPECT_EQ(x_tensor.size(), x_nabla.size());
        EXPECT_EQ(y_tensor.size(), y_nabla.size());

        if (i != probas.size() - 1)
        {
            for (size_t j = 0; j < out_tensor.size(); ++j)
            {
                const auto [golden_x_nabla, golden_y_nabla] = golden_select_layer_grad(static_cast<bool>(probas[i]), out_nabla[j]);
                EXPECT_TRUE(golden_x_nabla == x_nabla[j]);
                EXPECT_TRUE(golden_y_nabla == y_nabla[j]);
            }
        }
    }
}

#endif // ANDROID

}