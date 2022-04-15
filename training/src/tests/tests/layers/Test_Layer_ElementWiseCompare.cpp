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

#include <training/base/layers/basic/ElementWiseCompareLayer.h>
#include <training/base/layers/basic/ElementWiseSumLayer.h>
#include <training/base/layers/parameters/LayerParameters.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>

namespace UT
{

namespace
{

bool golden_equal_layer(const raul::dtype x, const raul::dtype y, const raul::dtype tol = 1e-6_dt)
{
    return std::abs(x - y) <= tol;
}

bool golden_less_layer(const raul::dtype x, const raul::dtype y, const raul::dtype tol = 0.0_dt)
{
    return y - x > tol;
}

bool golden_le_layer(const raul::dtype x, const raul::dtype y, const raul::dtype tol = 0.0_dt)
{
    return y - x >= tol;
}

bool golden_greater_layer(const raul::dtype x, const raul::dtype y, const raul::dtype tol = 0.0_dt)
{
    return x - y > tol;
}

bool golden_ge_layer(const raul::dtype x, const raul::dtype y, const raul::dtype tol = 0.0_dt)
{
    return x - y >= tol;
}

}

TEST(TestLayerElementWiseCompare, InputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::ElementWiseCompareLayer("comp", raul::ElementWiseComparisonLayerParams{ { "x", "y", "z" }, { "x_out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerElementWiseCompare, OutputNumExceedsUnit)
{
    PROFILE_TEST

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Apply function
    ASSERT_THROW(raul::ElementWiseCompareLayer("comp", raul::ElementWiseComparisonLayerParams{ { "x", "y" }, { "x_out", "y_out" } }, networkParameters), raul::Exception);
}

TEST(TestLayerElementWiseCompare, BackwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseCompareLayer elementwise_eq("mul", raul::ElementWiseComparisonLayerParams{ { "x", "y" }, { "out" } }, networkParameters);
    TENSORS_CREATE(tensor_size);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    elementwise_eq.forwardCompute(raul::NetworkMode::Train);
    ASSERT_NO_THROW(elementwise_eq.backwardCompute());
}

TEST(TestLayerElementWiseCompare, ForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto tensor_size = 1000U;
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);
    const auto enable_broadcast = false;

    std::vector<raul::ElementWiseComparisonLayerParams> params = { raul::ElementWiseComparisonLayerParams{ { "x", "y" }, { "out" }, enable_broadcast },
                                                                   raul::ElementWiseComparisonLayerParams{ { "x", "y" }, { "out" }, enable_broadcast, "less" },
                                                                   raul::ElementWiseComparisonLayerParams{ { "x", "y" }, { "out" }, enable_broadcast, "greater" },
                                                                   raul::ElementWiseComparisonLayerParams{ { "x", "y" }, { "out" }, enable_broadcast, "le" },
                                                                   raul::ElementWiseComparisonLayerParams{ { "x", "y" }, { "out" }, enable_broadcast, "ge" } };

    // Check comparators
    std::array<std::function<bool(const raul::dtype x, const raul::dtype y, const raul::dtype tol)>, 5> golden_layers = {
        golden_equal_layer, golden_less_layer, golden_greater_layer, golden_le_layer, golden_ge_layer
    };
    std::array<std::string, 5> names = { "equal", "less", "greater", "less_or_equal", "greater_or_equal" };
    std::array<raul::dtype, 5> tols = { 1e-6_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt };
    for (size_t i = 0; i < params.size(); i++)
    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
        work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

        raul::ElementWiseCompareLayer elementwise_comp(names[i], params[i], networkParameters);
        TENSORS_CREATE(tensor_size);
        tools::init_rand_tensor("x", random_range, memory_manager);
        tools::init_rand_tensor("y", random_range, memory_manager);
        elementwise_comp.forwardCompute(raul::NetworkMode::Test);

        // Checks
        const auto& x_tensor = memory_manager["x"];
        const auto& y_tensor = memory_manager["y"];
        const auto& out_tensor = memory_manager["out"];

        EXPECT_EQ(out_tensor.size(), x_tensor.size());
        EXPECT_EQ(out_tensor.size(), y_tensor.size());

        for (size_t j = 0; j < out_tensor.size(); j++)
        {
            const auto golden_out_value = golden_layers[i](x_tensor[j], y_tensor[j], tols[i]);
            EXPECT_EQ(out_tensor[j], golden_out_value);
        }
    }
}

TEST(TestLayerElementWiseCompare, EqualBroadcastForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto random_range = std::make_pair(-100.0_dt, 100.0_dt);
    const auto enable_broadcast = true;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.tensorNeeded("x", "x", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);
    work.tensorNeeded("y", "y", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_READ_NOMEMOPT);

    // Apply function
    raul::ElementWiseCompareLayer elementwise_eq("equal", raul::ElementWiseComparisonLayerParams{ { "x", "y" }, { "out" }, enable_broadcast }, networkParameters);
    TENSORS_CREATE(1);
    tools::init_rand_tensor("x", random_range, memory_manager);
    tools::init_rand_tensor("y", random_range, memory_manager);

    elementwise_eq.forwardCompute(raul::NetworkMode::Test);

    // Checks
    const auto& x_tensor = memory_manager["x"];
    const auto& y_tensor = memory_manager["y"];
    const auto& out_tensor = memory_manager["out"];

    EXPECT_EQ(out_tensor.size(), y_tensor.size());

    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        const auto x_value = x_tensor[0];
        const auto y_value = y_tensor[i];
        const auto out_value = out_tensor[i];
        const auto golden_out_value = golden_equal_layer(x_value, y_value);
        EXPECT_EQ(out_value, golden_out_value);
    }
}

TEST(TestLayerElementWiseCompare, ComplexForwardBackwardUnit)
{
    PROFILE_TEST
    using namespace raul;

    // Test parameters
    const auto eps_rel = TODTYPE(1e-6);
    const Tensor x{ 0.8822692633_dt, 0.9150039554_dt, 0.3828637600_dt, 0.9593056440_dt, 0.3904482126_dt, 0.6008953452_dt, 0.2565724850_dt, 0.7936413288_dt,
                    0.9407714605_dt, 0.1331859231_dt, 0.9345980883_dt, 0.5935796499_dt, 0.8694044352_dt, 0.5677152872_dt, 0.7410940528_dt, 0.4294044971_dt,
                    0.8854429126_dt, 0.5739044547_dt, 0.2665800452_dt, 0.6274491549_dt, 0.2696316838_dt, 0.4413635731_dt, 0.2969208360_dt, 0.8316854835_dt };
    const Tensor y{ 0.1053149104_dt, 0.2694948316_dt, 0.3588126302_dt, 0.1993637681_dt, 0.5471915603_dt, 0.0061604381_dt, 0.9515545368_dt, 0.0752658844_dt,
                    0.8860136867_dt, 0.5832095742_dt, 0.3376477361_dt, 0.8089749813_dt, 0.5779253840_dt, 0.9039816856_dt, 0.5546598434_dt, 0.3423134089_dt,
                    0.6343418360_dt, 0.3644102812_dt, 0.7104287744_dt, 0.9464110732_dt, 0.7890297771_dt, 0.2814137340_dt, 0.7886323333_dt, 0.5894631147_dt };
    const Tensor z{ 0.9875841737_dt, 1.1844987869_dt, 0.7416763902_dt, 1.1586694717_dt, 1.9376397133_dt, 0.6070557833_dt, 2.2081270218_dt, 0.8689072132_dt,
                    1.8267850876_dt, 1.7163954973_dt, 1.2722458839_dt, 2.4025545120_dt, 1.4473297596_dt, 2.4716968536_dt, 1.2957539558_dt, 0.7717179060_dt,
                    1.5197846889_dt, 0.9383147359_dt, 1.9770088196_dt, 2.5738601685_dt, 2.0586614609_dt, 0.7227773070_dt, 2.0855531693_dt, 1.4211485386_dt };

    const Tensor x_grad{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                         1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };
    const Tensor y_grad{ 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt,
                         1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt, 1.0_dt };
    const size_t BATCH_SIZE = 1;

    // See element-wise_compare_and_sum.py
    raul::Workflow work;
    work.add<raul::DataLayer>("x", raul::DataParams{ { "x" }, 2, 3, 4 });
    work.add<raul::DataLayer>("y", raul::DataParams{ { "y" }, 2, 3, 4 });
    work.add<raul::ElementWiseSumLayer>("sum", ElementWiseLayerParams{ { "x", "y" }, { "out1" } });
    work.add<raul::ElementWiseCompareLayer>("less", ElementWiseComparisonLayerParams{ { "x", "y" }, { "out2" } });
    work.add<raul::ElementWiseSumLayer>("final_sum", ElementWiseLayerParams{ { "out1", "out2" }, { "out" } });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    raul::MemoryManager& memory_manager = work.getMemoryManager();
    memory_manager["x"] = TORANGE(x);
    memory_manager["y"] = TORANGE(y);

    // Apply functions
    work.forwardPassTraining();
    work.backwardPassTraining();

    // Forward checks
    const auto& out_tensor = memory_manager["out"];
    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(out_tensor[i], z[i], eps_rel));
    }

    // Backward Checks
    const auto& x_tensor_grad = memory_manager[Name("x").grad()];
    const auto& y_tensor_grad = memory_manager[Name("y").grad()];

    EXPECT_EQ(x_tensor_grad.getShape(), memory_manager["x"].getShape());
    EXPECT_EQ(y_tensor_grad.getShape(), memory_manager["y"].getShape());

    for (size_t i = 0; i < x_tensor_grad.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(x_tensor_grad[i], x_grad[i], eps_rel)) << "expected: " << x_grad[i] << ", got: " << x_tensor_grad[i];
    }

    for (size_t i = 0; i < y_tensor_grad.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(y_tensor_grad[i], y_grad[i], eps_rel)) << "expected: " << y_grad[i] << ", got: " << y_tensor_grad[i];
    }
}

}