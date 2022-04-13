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

#include <training/base/layers/composite/rnn/LSTMFusedLayer.h>
#include <training/compiler/Layers.h>

namespace UT
{

TEST(TestLSTMFused, SharedForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-3_dt;
    const auto input_size = 3U;
    const auto hidden_size = 2U;
    const auto batch_size = 2U;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", DataParams{ { "in" }, 1, 1, input_size });
    work.add<raul::DataLayer>("data2", DataParams{ { "hidden" }, 1, 1, hidden_size });
    work.add<raul::DataLayer>("data3", DataParams{ { "cell" }, 1, 1, hidden_size });

    // Network
    const auto params1 = raul::LSTMParams( { "in" }, {"out"}, hidden_size, true, true, false, false, 0.0_dt, false, 0.0_dt, false );
    auto params2 = raul::LSTMParams{ { "in" }, { "out2" }, hidden_size, true, true, false, false, 0.0_dt, false, 0.0_dt, false };
    params2.getSharedLayer() = raul::Name("lstm::cell");
    work.add<raul::LSTMFusedLayer>("lstm_cell", raul::LSTMParams(params1), "lstm", "hidden", "cell");
    work.add<raul::LSTMFusedLayer>("lstm_cell2", params2, "lstm2", "hidden[0]", "cell[0]");
    TENSORS_CREATE(batch_size)

    memory_manager["in"] = { -0.0209, -0.7185, 0.5186, -1.3125,  0.1920, 0.5428 };
    memory_manager["hidden"] = { -2.2188,  0.2590, -1.0297, -0.5008 };
    memory_manager["cell"] = { 0.2734, -0.9181, -0.0404,  0.2881 };

    memory_manager[Name("lstm") / ("cell") / "linear_ih" / "Weights"] = { 
        -0.0053,  0.3793, -0.5820, -0.5204, -0.2723,  0.1896, 
        -0.0140,  0.5607, -0.0628,  0.1871, -0.2137, -0.1390, 
        -0.6755, -0.4683, -0.2915,  0.0262,  0.2795,  0.4243, 
        -0.4794, -0.3079,  0.2568,  0.5872, -0.1455,  0.5291 };

    memory_manager[Name("lstm") / ("cell") / "linear_ih" / "Biases"] = { 0.0372, -0.3625, 0.1196, -0.6602, -0.5109, -0.3645, 0.4461, 0.4146 };

    memory_manager[Name("lstm") / ("cell") / "linear_hh" / "Weights"] = { 
        -0.1140,  0.0748,  0.6403, -0.6560, -0.4452, -0.1790, -0.2756, 0.6109, 
        -0.4583, -0.3255, -0.4940, -0.6622, -0.4128,  0.6078,  0.3155, 0.3427 };

    memory_manager[Name("lstm") / ("cell") / "linear_hh" / "Biases"] = { -0.3136, -0.0255, 0.4522, 0.7030, 0.2806, 0.0955, 0.4741, -0.4163 };

    // Apply
    work.forwardPassTesting();

    // Checks
    const raul::Tensor new_hidden_golden(batch_size, 1, 1, hidden_size, { 0.3941_dt, -0.2222_dt, 0.2289_dt, 0.1148_dt });
    const raul::Tensor new_cell_golden(batch_size, 1, 1, hidden_size, { 0.4616_dt, -0.5580_dt, 0.2754_dt, 0.4600_dt });

    const raul::Tensor new_hidden_golden2(batch_size, 1, 1, hidden_size, { 0.1377_dt, -0.2434_dt, 0.2198_dt, 0.0282_dt });
    const raul::Tensor new_cell_golden2(batch_size, 1, 1, hidden_size, { 0.1913_dt, -0.4290_dt, 0.2704_dt, 0.0704_dt });

    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& cell_input_tensor = memory_manager["cell"];

    const auto& hidden_new_tensor = memory_manager["hidden[0]"];
    const auto& cell_new_tensor = memory_manager["cell[0]"];

    const auto& hidden_new_tensor2 = memory_manager["hidden[0][0]"];
    const auto& cell_new_tensor2 = memory_manager["cell[0][0]"];

    EXPECT_EQ(hidden_input_tensor.size(), hidden_new_tensor.size());
    EXPECT_EQ(cell_input_tensor.size(), cell_new_tensor.size());

    EXPECT_EQ(hidden_input_tensor.size(), hidden_new_tensor2.size());
    EXPECT_EQ(cell_input_tensor.size(), cell_new_tensor2.size());

    for (size_t i = 0; i < hidden_input_tensor.size(); ++i)
    {
        const auto cell_val = cell_new_tensor[i];
        const auto golden_cell_val = new_cell_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(cell_val, golden_cell_val, eps_rel)) << "at " << i << ", expected: " << golden_cell_val << ", got: " << cell_val;

        const auto hidden_val = hidden_new_tensor[i];
        const auto golden_hidden_val = new_hidden_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(hidden_val, golden_hidden_val, eps_rel)) << "at " << i << ", expected: " << golden_hidden_val << ", got: " << hidden_val;
    }

    for (size_t i = 0; i < hidden_input_tensor.size(); ++i)
    {
        const auto cell_val = cell_new_tensor2[i];
        const auto golden_cell_val = new_cell_golden2[i];
        ASSERT_TRUE(tools::expect_near_relative(cell_val, golden_cell_val, eps_rel)) << "at " << i << ", expected: " << golden_cell_val << ", got: " << cell_val;

        const auto hidden_val = hidden_new_tensor2[i];
        const auto golden_hidden_val = new_hidden_golden2[i];
        ASSERT_TRUE(tools::expect_near_relative(hidden_val, golden_hidden_val, eps_rel)) << "at " << i << ", expected: " << golden_hidden_val << ", got: " << hidden_val;
    }
}

TEST(TestLSTMFused, SingleParamsForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto batch_size = 2U;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", DataParams{ { "in" }, 1, 1, input_size });
    work.add<raul::DataLayer>("data2", DataParams{ { "hidden" }, 1, 1, hidden_size });
    work.add<raul::DataLayer>("data3", DataParams{ { "cell" }, 1, 1, hidden_size });

    // Network
    const auto params1 = raul::LSTMParams( { "in" }, {"out"}, hidden_size, true, true, false, false, 0.0_dt, true, 0.0_dt, false );
    work.add<raul::LSTMFusedLayer>("lstm_cell", raul::LSTMParams(params1), "lstm", "hidden", "cell");
    TENSORS_CREATE(batch_size)

    const Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    const Tensor hidden_init{ 1.892910e+00_dt, 3.111044e+00_dt, -4.583958e-01_dt, -3.359881e-01_dt, -1.569986e+00_dt, 1.231500e+00_dt };
    const Tensor cell_init{ 1.394632e+00_dt, 1.171102e+00_dt, 4.335119e-01_dt, -1.734250e+00_dt, -1.336049e+00_dt, 8.870960e-01_dt };
    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);
    memory_manager["cell"] = TORANGE(cell_init);

    memory_manager[Name("lstm") / "cell" / "linear" / "Weights"] = 1.0_dt;
    memory_manager[Name("lstm") / "cell" / "linear" / "Biases"] = 2.0_dt;

    // Apply
    work.forwardPassTesting();

    // Checks
    const Tensor hidden_golden{ 9.557552e-01_dt, 9.459500e-01_dt, 8.612298e-01_dt, -5.386918e-01_dt, -2.835236e-01_dt, 8.465633e-01_dt };
    const Tensor cell_golden{ 2.330955e+00_dt, 2.113240e+00_dt, 1.394835e+00_dt, -6.845742e-01_dt, -3.237444e-01_dt, 1.690755e+00_dt };

    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& cell_input_tensor = memory_manager["cell"];

    const auto& hidden_new_tensor = memory_manager["hidden[0]"];
    const auto& cell_new_tensor = memory_manager["cell[0]"];

    EXPECT_EQ(hidden_input_tensor.size(), hidden_new_tensor.size());
    EXPECT_EQ(cell_input_tensor.size(), cell_new_tensor.size());

    for (size_t i = 0; i < hidden_input_tensor.size(); ++i)
    {
        const auto cell_val = cell_new_tensor[i];
        const auto golden_cell_val = cell_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(cell_val, golden_cell_val, eps_rel)) << "at " << i << ", expected: " << golden_cell_val << ", got: " << cell_val;

        const auto hidden_val = hidden_new_tensor[i];
        const auto golden_hidden_val = hidden_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(hidden_val, golden_hidden_val, eps_rel)) << "at " << i << ", expected: " << golden_hidden_val << ", got: " << hidden_val;
    }
}

} // UT namespace