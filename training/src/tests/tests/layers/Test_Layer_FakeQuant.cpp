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
#include <training/base/common/Conversions.h>
#include <training/base/common/MemoryManager.h>
#include <training/base/common/quantization/SymmetricQuantizer.h>
#include <training/base/layers/basic/FakeQuantLayer.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>
#include <training/system/NameGenerator.h>

namespace
{

constexpr auto high_value = 127.0_dt;
constexpr auto low_value = -127.0_dt;

/**
 * @brief Quantization function directly reproduced from slides
 * @param x
 * @param scale
 * @return quntized float value
 */
raul::dtype golden_fake_quant_layer(raul::dtype x, raul::dtype scale)
{
    auto value = scale * x;
    value = (value > 0.0_dt) ? std::floor(value) : std::ceil(value);
    if (value < low_value)
    {
        value = low_value;
    }
    if (value > high_value)
    {
        value = low_value;
    }
    return value / scale;
}

raul::dtype golden_backward_fake_quant_layer(raul::dtype x, raul::dtype grad, raul::dtype scale)
{
    auto value = scale * x;
    value = (value > 0.0_dt) ? std::floor(value) : std::ceil(value);
    if (value < low_value || value > high_value)
    {
        return 0.0_dt;
    }
    return grad;
}

} // namespace

namespace UT
{

TEST(TestLayerFakeQuant, ForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto tensor_size = 1000U;
    const auto digits = 8U;
    const auto round_algo = static_cast<raul::dtype (*)(raul::dtype)>(std::trunc);
    const auto mode = raul::quantization::SymmetricQuantizer::Mode::restricted_range;
    const auto fake_quant_mode = raul::QuantizationMode::over_full_tensor;
    const auto random_range = std::make_pair(-200.0_dt, 200.0_dt);

    auto quantizer = raul::quantization::SymmetricQuantizer(round_algo, digits, mode);

    // Initialization
    raul::WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPU, false, &quantizer);
    raul::MemoryManager& memory_manager = work.getMemoryManager();
    NETWORK_PARAMS_DEFINE(net_params)

    auto params = raul::FakeQuantParams{ { "x" }, { "y" }, fake_quant_mode };

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, 1, 1, 1 });

    // Apply function
    work.add<raul::FakeQuantLayer>("fk", params);
    TENSORS_CREATE(tensor_size)

    tools::init_rand_tensor("x", random_range, memory_manager);

    work.forwardPassTesting();

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& y_tensor = memory_manager["y"];

    EXPECT_EQ(y_tensor.size(), x_tensor.size());

    const auto max_val = *std::max_element(x_tensor.cbegin(), x_tensor.cend(), [](raul::dtype a, raul::dtype b) { return std::abs(a) < std::abs(b); });
    const auto scale = high_value / std::abs(max_val);

    for (size_t i = 0; i < y_tensor.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        const auto y_value = y_tensor[i];
        const auto golden_out_value = golden_fake_quant_layer(x_value, scale);
        ASSERT_TRUE(tools::expect_near_relative(y_value, golden_out_value, eps_rel)) << "from: " << x_value << ", expected: " << golden_out_value << ", got: " << y_value;
    }
}

TEST(TestLayerFakeQuant, BackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto tensor_size = 1000U;
    const auto digits = 8U;
    const auto round_algo = static_cast<raul::dtype (*)(raul::dtype)>(std::trunc);
    const auto mode = raul::quantization::SymmetricQuantizer::Mode::restricted_range;
    const auto fake_quant_mode = raul::QuantizationMode::over_full_tensor;
    const auto random_range = std::make_pair(-200.0_dt, 200.0_dt);

    auto quantizer = raul::quantization::SymmetricQuantizer(round_algo, digits, mode);

    // Initialization
    raul::WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPU, false, &quantizer);
    raul::MemoryManager& memory_manager = work.getMemoryManager();
    NETWORK_PARAMS_DEFINE(net_params)

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, 1, 1, 1 });

    auto params = raul::FakeQuantParams{ { "x" }, { "y" }, fake_quant_mode };

    // Apply function

    work.add<raul::FakeQuantLayer>("fk", params);
    TENSORS_CREATE(tensor_size)

    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor(raul::Name("y").grad(), random_range, memory_manager);

    work.forwardPassTraining();
    work.backwardPassTraining();

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
    const auto& y_tensor_grad = memory_manager[raul::Name("y").grad()];

    EXPECT_EQ(y_tensor_grad.size(), x_tensor_grad.size());

    const auto max_val = *std::max_element(x_tensor.cbegin(), x_tensor.cend(), [](raul::dtype a, raul::dtype b) { return std::abs(a) < std::abs(b); });
    const auto scale = high_value / std::abs(max_val);

    for (size_t i = 0; i < x_tensor.size(); ++i)
    {
        const auto x_value = x_tensor[i];
        const auto x_grad_value = x_tensor_grad[i];
        const auto y_grad_value = y_tensor_grad[i];

        const auto golden_out_value = golden_backward_fake_quant_layer(x_value, y_grad_value, scale);
        ASSERT_TRUE(tools::expect_near_relative(x_grad_value, golden_out_value, eps_rel)) << "from: " << x_value << ", expected: " << golden_out_value << ", got: " << x_grad_value;
    }
}

TEST(TestLayerFakeQuant, ForwardRandOverBatchUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto tensor_size = 5U;
    const auto batch_size = 2U;
    const auto digits = 8U;
    const auto round_algo = static_cast<raul::dtype (*)(raul::dtype)>(std::trunc);
    const auto mode = raul::quantization::SymmetricQuantizer::Mode::restricted_range;
    const auto fake_quant_mode = raul::QuantizationMode::over_batch;
    const auto random_range = std::make_pair(-200.0_dt, 200.0_dt);

    auto quantizer = raul::quantization::SymmetricQuantizer(round_algo, digits, mode);

    // Initialization
    raul::WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPU, false, &quantizer);
    raul::MemoryManager& memory_manager = work.getMemoryManager();
    NETWORK_PARAMS_DEFINE(net_params)

    auto params = raul::FakeQuantParams{ { "x" }, { "y" }, fake_quant_mode };

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, tensor_size, 1, 1 });

    // Apply function
    work.add<raul::FakeQuantLayer>("fk", params);
    TENSORS_CREATE(batch_size)

    tools::init_rand_tensor("x", random_range, memory_manager);

    work.forwardPassTesting();

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& y_tensor = memory_manager["y"];

    EXPECT_EQ(y_tensor.size(), x_tensor.size());

    auto x_tensor2D = x_tensor.reshape(yato::dims(batch_size, tensor_size));
    auto y_tensor2D = y_tensor.reshape(yato::dims(batch_size, tensor_size));

    for (size_t b = 0U; b < batch_size; ++b)
    {
        const auto max_val = *std::max_element(x_tensor2D[b].cbegin(), x_tensor2D[b].cend(), [](raul::dtype a, raul::dtype b) { return std::abs(a) < std::abs(b); });
        const auto scale = high_value / std::abs(max_val);

        for (size_t i = 0; i < tensor_size; ++i)
        {
            const auto x_value = x_tensor2D[b][i];
            const auto y_value = y_tensor2D[b][i];
            const auto golden_out_value = golden_fake_quant_layer(x_value, scale);
            ASSERT_TRUE(tools::expect_near_relative(y_value, golden_out_value, eps_rel)) << "from: " << x_value << ", expected: " << golden_out_value << ", got: " << y_value;
        }
    }
}

TEST(TestLayerFakeQuant, BackwardRandOverBatchUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-6_dt;
    const auto tensor_size = 5U;
    const auto batch_size = 2U;
    const auto digits = 8U;
    const auto round_algo = static_cast<raul::dtype (*)(raul::dtype)>(std::trunc);
    const auto mode = raul::quantization::SymmetricQuantizer::Mode::restricted_range;
    const auto fake_quant_mode = raul::QuantizationMode::over_batch;
    const auto random_range = std::make_pair(-200.0_dt, 200.0_dt);

    auto quantizer = raul::quantization::SymmetricQuantizer(round_algo, digits, mode);

    // Initialization
    raul::WorkflowEager work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPU, false, &quantizer);
    raul::MemoryManager& memory_manager = work.getMemoryManager();
    NETWORK_PARAMS_DEFINE(net_params)

    work.add<raul::DataLayer>("data", raul::DataParams{ { "x" }, tensor_size, 1, 1 });

    auto params = raul::FakeQuantParams{ { "x" }, { "y" }, fake_quant_mode };

    // Apply function
    work.add<raul::FakeQuantLayer>("fk", params);
    TENSORS_CREATE(batch_size)

    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor(raul::Name("y").grad(), random_range, memory_manager);

    work.forwardPassTraining();
    work.backwardPassTraining();

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& x_tensor_grad = memory_manager[raul::Name("x").grad()];
    const auto& y_tensor_grad = memory_manager[raul::Name("y").grad()];

    EXPECT_EQ(y_tensor_grad.size(), x_tensor_grad.size());

    auto x_tensor2D = x_tensor.reshape(yato::dims(batch_size, tensor_size));
    auto x_tensor_grad2D = x_tensor_grad.reshape(yato::dims(batch_size, tensor_size));
    auto y_tensor_grad2D = y_tensor_grad.reshape(yato::dims(batch_size, tensor_size));

    for (size_t b = 0U; b < batch_size; ++b)
    {
        const auto max_val = *std::max_element(x_tensor2D[b].cbegin(), x_tensor2D[b].cend(), [](raul::dtype a, raul::dtype b) { return std::abs(a) < std::abs(b); });
        const auto scale = high_value / std::abs(max_val);

        for (size_t i = 0; i < tensor_size; ++i)
        {
            const auto x_value = x_tensor2D[b][i];
            const auto x_grad_value = x_tensor_grad2D[b][i];
            const auto y_grad_value = y_tensor_grad2D[b][i];

            const auto golden_out_value = golden_backward_fake_quant_layer(x_value, y_grad_value, scale);
            ASSERT_TRUE(tools::expect_near_relative(x_grad_value, golden_out_value, eps_rel)) << "from: " << x_value << ", expected: " << golden_out_value << ", got: " << x_grad_value;
        }
    }
}

} // UT namespace