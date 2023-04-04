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
#include <training/base/layers/composite/rnn/LSTMCellLayer.h>
#include <training/compiler/Layers.h>

namespace
{

using namespace raul;

auto buildLSTMCellOnPrimitives(const size_t batch_size, const size_t input_size, const size_t hidden_size)
{
    auto work = std::make_unique<WorkflowEager>();
    const size_t parts = 4U;

    work->add<raul::DataLayer>("fake_data_in", DataParams{ { "in", "labels" }, 1, 1, input_size, 0 });
    work->add<raul::DataLayer>("fake_data_state", DataParams{ { "hidden", "cell" }, 1, 1, hidden_size, hidden_size });
    work->add<raul::LinearLayer>("linear_ih", LinearParams({ "in" }, { "linear_ih" }, hidden_size * parts));
    work->add<raul::LinearLayer>("linear_hh", LinearParams({ "hidden" }, { "linear_hh" }, hidden_size * parts));
    work->add<raul::ElementWiseSumLayer>("gates", ElementWiseLayerParams({ "linear_ih", "linear_hh" }, { "gates" }));
    work->add<raul::SlicerLayer>("slice", SlicingParams("gates", { "gates[0]", "gates[1]", "gates[2]", "gates[3]" }, "width"));
    work->add<raul::SigmoidActivation>("sigmoid_input", BasicParams({ "gates[0]" }, { "sigmoid_input" }));
    work->add<raul::SigmoidActivation>("sigmoid_forget", BasicParams({ "gates[1]" }, { "sigmoid_forget" }));
    work->add<raul::TanhActivation>("tanh_gates", BasicParams({ "gates[2]" }, { "tanh_gates" }));
    work->add<raul::SigmoidActivation>("sigmoid_output", BasicParams({ "gates[3]" }, { "sigmoid_output" }));
    work->add<raul::ElementWiseMulLayer>("mul_input", ElementWiseLayerParams({ "sigmoid_input", "tanh_gates" }, { "mul_input" }, false));
    work->add<raul::ElementWiseMulLayer>("mul_forget", ElementWiseLayerParams({ "sigmoid_forget", "cell" }, { "mul_forget" }, false));
    work->add<raul::ElementWiseSumLayer>("sum_new_cell_state", ElementWiseLayerParams({ "mul_input", "mul_forget" }, { "sum_new_cell_state" }));
    work->add<raul::SplitterLayer>("splitter", BasicParams({ "sum_new_cell_state" }, { "internal_new_cell", "new_cell" }));
    work->add<raul::TanhActivation>("tanh_new_cell_state", BasicParams({ "internal_new_cell" }, { "tanh_new_cell_state" }));
    work->add<raul::ElementWiseMulLayer>("mul_new_hidden_state", ElementWiseLayerParams({ "sigmoid_output", "tanh_new_cell_state" }, { "new_hidden" }, false));

    work->preparePipelines();
    work->setBatchSize(batch_size);
    work->prepareMemoryForTraining();

    return work;
}

[[maybe_unused]] dtype zoneout_outputs_test_golden(const dtype prev, const dtype curr, const dtype prob)
{
    return prob * prev + (1.0_dt - prob) * curr;
}

}

namespace UT
{

using namespace std;

TEST(TestLSTMCell, BuildUnit)
{
    PROFILE_TEST
    Workflow netdef;
    netdef.add<raul::DataLayer>("fake_data_in", DataParams{ { "in", "labels" }, 1, 1, 1, 1 });
    netdef.add<raul::DataLayer>("fake_data_state", DataParams{ { "hidden", "cell" }, 1, 1, 1, 1 });
    LSTMCellLayer("lstm", LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {} }, netdef.getNetworkParameters());
    netdef.preparePipelines();
    netdef.setBatchSize(1u);
    netdef.prepareMemoryForTraining();
    netdef.printInfo(std::cout);
}

TEST(TestLSTMCell, PrimitiveBlocksForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto batch_size = 2U;

    // Network
    auto network = buildLSTMCellOnPrimitives(batch_size, input_size, hidden_size);
    network->printInfo(std::cout);

    // Initialization
    auto& memory_manager = network->getMemoryManager();

    for (auto& [param, grad] : network->getTrainableParameters())
    {
        param = 1.0_dt;
    }

    const raul::Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    const raul::Tensor hidden_init{ 1.892910e+00_dt, 3.111044e+00_dt, -4.583958e-01_dt, -3.359881e-01_dt, -1.569986e+00_dt, 1.231500e+00_dt };
    const raul::Tensor cell_init{ 1.394632e+00_dt, 1.171102e+00_dt, 4.335119e-01_dt, -1.734250e+00_dt, -1.336049e+00_dt, 8.870960e-01_dt };
    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);
    memory_manager["cell"] = TORANGE(cell_init);

    // Apply
    network->forwardPassTesting();

    // Checks
    const raul::Tensor hidden_golden{ 9.557552e-01_dt, 9.459500e-01_dt, 8.612298e-01_dt, -5.386918e-01_dt, -2.835236e-01_dt, 8.465633e-01_dt };
    const raul::Tensor cell_golden{ 2.330955e+00_dt, 2.113240e+00_dt, 1.394835e+00_dt, -6.845742e-01_dt, -3.237444e-01_dt, 1.690755e+00_dt };

    // Intermediate values
    const raul::Tensor gatess_golden{ 3.622849e+00_dt, 3.622849e+00_dt, 3.622849e+00_dt, 3.622849e+00_dt, 3.622849e+00_dt, 3.622849e+00_dt, 3.622849e+00_dt, 3.622849e+00_dt,
                                      3.622849e+00_dt, 3.622849e+00_dt, 3.622849e+00_dt, 3.622849e+00_dt, 2.267491e+00_dt, 2.267491e+00_dt, 2.267491e+00_dt, 2.267491e+00_dt,
                                      2.267491e+00_dt, 2.267491e+00_dt, 2.267491e+00_dt, 2.267491e+00_dt, 2.267491e+00_dt, 2.267491e+00_dt, 2.267491e+00_dt, 2.267491e+00_dt };

    const raul::Tensor i_t_golden{ 9.739882e-01_dt, 9.739882e-01_dt, 9.739882e-01_dt, 9.061487e-01_dt, 9.061487e-01_dt, 9.061487e-01_dt };
    const raul::Tensor f_t_golden{ 9.739882e-01_dt, 9.739882e-01_dt, 9.739882e-01_dt, 9.061487e-01_dt, 9.061487e-01_dt, 9.061487e-01_dt };
    const raul::Tensor g_t_golden{ 9.985746e-01_dt, 9.985746e-01_dt, 9.985746e-01_dt, 9.787735e-01_dt, 9.787735e-01_dt, 9.787735e-01_dt };
    const raul::Tensor o_h_golden{ 9.739882e-01_dt, 9.739882e-01_dt, 9.739882e-01_dt, 9.061487e-01_dt, 9.061487e-01_dt, 9.061487e-01_dt };

    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& cell_input_tensor = memory_manager["cell"];

    const auto& hidden_new_tensor = memory_manager["new_hidden"];
    const auto& cell_new_tensor = memory_manager["new_cell"];

    const auto& gates_tensor = memory_manager["gates"];
    const auto& sigmoid_input_tensor = memory_manager["sigmoid_input"];
    const auto& sigmoid_forget_tensor = memory_manager["sigmoid_forget"];
    const auto& tanh_gates_tensor = memory_manager["tanh_gates"];
    const auto& sigmoid_output_tensor = memory_manager["sigmoid_output"];

    EXPECT_EQ(hidden_input_tensor.size(), hidden_new_tensor.size());
    EXPECT_EQ(cell_input_tensor.size(), cell_new_tensor.size());

    for (size_t i = 0; i < gates_tensor.size(); ++i)
    {
        const auto val = gates_tensor[i];
        const auto golden_val = gatess_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < sigmoid_input_tensor.size(); ++i)
    {
        const auto val = sigmoid_input_tensor[i];
        const auto golden_val = i_t_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < sigmoid_input_tensor.size(); ++i)
    {
        const auto val = sigmoid_forget_tensor[i];
        const auto golden_val = f_t_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < sigmoid_input_tensor.size(); ++i)
    {
        const auto val = tanh_gates_tensor[i];
        const auto golden_val = g_t_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < sigmoid_input_tensor.size(); ++i)
    {
        const auto val = sigmoid_output_tensor[i];
        const auto golden_val = o_h_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

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

TEST(TestLSTMCell, PrimitiveBlocksBackwardUnit)
{
    PROFILE_TEST
    using namespace raul;

    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto batch_size = 2U;

    // Network
    auto network = buildLSTMCellOnPrimitives(batch_size, input_size, hidden_size);

    // Initialization
    auto& memory_manager = network->getMemoryManager();

    for (auto& [param, grad] : network->getTrainableParameters())
    {
        param = 1.0_dt;
    }

    const Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    const Tensor hidden_init{ 1.892910e+00_dt, 3.111044e+00_dt, -4.583958e-01_dt, -3.359881e-01_dt, -1.569986e+00_dt, 1.231500e+00_dt };
    const Tensor cell_init{ 1.394632e+00_dt, 1.171102e+00_dt, 4.335119e-01_dt, -1.734250e+00_dt, -1.336049e+00_dt, 8.870960e-01_dt };
    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);
    memory_manager["cell"] = TORANGE(cell_init);

    memory_manager[Name("new_hidden").grad()] = 1.0_dt;
    memory_manager[Name("new_cell").grad()] = 1.0_dt;

    // Apply
    network->forwardPassTraining();
    network->backwardPassTraining();

    // Checks
    const Tensor inputs_grad_golden{ 2.458567e-01_dt, 2.458567e-01_dt, 2.458567e-01_dt, 2.458567e-01_dt, 1.941207e-01_dt, 1.941207e-01_dt, 1.941207e-01_dt, 1.941207e-01_dt };

    const auto& inputs_grad = memory_manager[Name("in").grad()];
    const auto& inputs = memory_manager["in"];

    EXPECT_EQ(inputs.size(), inputs_grad.size());

    for (size_t i = 0; i < inputs_grad.size(); ++i)
    {
        const auto cell_val = inputs_grad[i];
        const auto golden_cell_val = inputs_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(cell_val, golden_cell_val, eps_rel)) << "at " << i << ", expected: " << golden_cell_val << ", got: " << cell_val;
    }
}

TEST(TestLSTMCell, SimpleForwardUnit)
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
    const auto params = raul::LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {} };
    raul::LSTMCellLayer("lstm_cell", params, networkParameters);
    TENSORS_CREATE(batch_size)

    const raul::Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    const raul::Tensor hidden_init{ 1.892910e+00_dt, 3.111044e+00_dt, -4.583958e-01_dt, -3.359881e-01_dt, -1.569986e+00_dt, 1.231500e+00_dt };
    const raul::Tensor cell_init{ 1.394632e+00_dt, 1.171102e+00_dt, 4.335119e-01_dt, -1.734250e+00_dt, -1.336049e+00_dt, 8.870960e-01_dt };
    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);
    memory_manager["cell"] = TORANGE(cell_init);

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Apply
    work.forwardPassTesting();

    // Checks
    const raul::Tensor hidden_golden{ 9.557552e-01_dt, 9.459500e-01_dt, 8.612298e-01_dt, -5.386918e-01_dt, -2.835236e-01_dt, 8.465633e-01_dt };
    const raul::Tensor cell_golden{ 2.330955e+00_dt, 2.113240e+00_dt, 1.394835e+00_dt, -6.845742e-01_dt, -3.237444e-01_dt, 1.690755e+00_dt };

    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& cell_input_tensor = memory_manager["cell"];

    const auto& hidden_new_tensor = memory_manager["new_hidden"];
    const auto& cell_new_tensor = memory_manager["new_cell"];

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

TEST(TestLSTMCell, SharedForwardUnit)
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
    const auto params1 = raul::LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {} };
    const auto params2 = raul::LSTMCellParams{ {{"in", "new_hidden", "new_cell"}, {"new_hidden2", "new_cell2"}, raul::Name("lstm_cell")} };
    raul::LSTMCellLayer("lstm_cell", params1, networkParameters);
    raul::LSTMCellLayer("lstm_cell2", params2, networkParameters);
    TENSORS_CREATE(batch_size)

    memory_manager["in"] = { -0.0209, -0.7185, 0.5186, -1.3125,  0.1920, 0.5428 };
    memory_manager["hidden"] = { -2.2188,  0.2590, -1.0297, -0.5008 };
    memory_manager["cell"] = { 0.2734, -0.9181, -0.0404,  0.2881 };

    memory_manager[Name("lstm_cell") / "linear_ih" / "Weights"] = { 
        -0.0053,  0.3793, -0.5820, -0.5204, -0.2723,  0.1896, 
        -0.0140,  0.5607, -0.0628,  0.1871, -0.2137, -0.1390, 
        -0.6755, -0.4683, -0.2915,  0.0262,  0.2795,  0.4243, 
        -0.4794, -0.3079,  0.2568,  0.5872, -0.1455,  0.5291 };

    memory_manager[Name("lstm_cell") / "linear_ih" / "Biases"] = { 0.0372, -0.3625, 0.1196, -0.6602, -0.5109, -0.3645, 0.4461, 0.4146 };

    memory_manager[Name("lstm_cell") / "linear_hh" / "Weights"] = { 
        -0.1140,  0.0748,  0.6403, -0.6560, -0.4452, -0.1790, -0.2756, 0.6109, 
        -0.4583, -0.3255, -0.4940, -0.6622, -0.4128,  0.6078,  0.3155, 0.3427 };

    memory_manager[Name("lstm_cell") / "linear_hh" / "Biases"] = { -0.3136, -0.0255, 0.4522, 0.7030, 0.2806, 0.0955, 0.4741, -0.4163 };

    // Apply
    work.forwardPassTesting();

    // Checks
    const raul::Tensor new_hidden_golden(batch_size, 1, 1, hidden_size, { 0.3941_dt, -0.2222_dt, 0.2289_dt, 0.1148_dt });
    const raul::Tensor new_cell_golden(batch_size, 1, 1, hidden_size, { 0.4616_dt, -0.5580_dt, 0.2754_dt, 0.4600_dt });

    const raul::Tensor new_hidden_golden2(batch_size, 1, 1, hidden_size, { 0.1377_dt, -0.2434_dt, 0.2198_dt, 0.0282_dt });
    const raul::Tensor new_cell_golden2(batch_size, 1, 1, hidden_size, { 0.1913_dt, -0.4290_dt, 0.2704_dt, 0.0704_dt });

    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& cell_input_tensor = memory_manager["cell"];

    const auto& hidden_new_tensor = memory_manager["new_hidden"];
    const auto& cell_new_tensor = memory_manager["new_cell"];

    const auto& hidden_new_tensor2 = memory_manager["new_hidden2"];
    const auto& cell_new_tensor2 = memory_manager["new_cell2"];

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

TEST(TestLSTMCell, SimpleBackwardUnit)
{
    PROFILE_TEST
    using namespace raul;

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
    const auto params = LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {} };
    LSTMCellLayer("lstm_cell", params, networkParameters);
    TENSORS_CREATE(batch_size)

    const Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    const Tensor hidden_init{ 1.892910e+00_dt, 3.111044e+00_dt, -4.583958e-01_dt, -3.359881e-01_dt, -1.569986e+00_dt, 1.231500e+00_dt };
    const Tensor cell_init{ 1.394632e+00_dt, 1.171102e+00_dt, 4.335119e-01_dt, -1.734250e+00_dt, -1.336049e+00_dt, 8.870960e-01_dt };
    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);
    memory_manager["cell"] = TORANGE(cell_init);

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Apply
    work.forwardPassTraining();

    memory_manager[Name("new_hidden").grad()] = 1.0_dt;
    memory_manager[Name("new_cell").grad()] = 1.0_dt;

    work.backwardPassTraining();

    // Checks
    const Tensor inputs_grad_golden{ 2.458567e-01_dt, 2.458567e-01_dt, 2.458567e-01_dt, 2.458567e-01_dt, 1.941207e-01_dt, 1.941207e-01_dt, 1.941207e-01_dt, 1.941207e-01_dt };

    const auto& inputs_grad = memory_manager[Name("in").grad()];
    const auto& inputs = memory_manager["in"];

    EXPECT_EQ(inputs.size(), inputs_grad.size());

    for (size_t i = 0; i < inputs_grad.size(); ++i)
    {
        const auto val = inputs_grad[i];
        const auto golden_val = inputs_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

TEST(TestLSTMCell, SimpleBackwardHiddenOnlyGradUnit)
{
    PROFILE_TEST
    using namespace raul;

    // Test parameters
    const auto eps_rel = 1e-4_dt;
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
    const auto params = LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {} };
    LSTMCellLayer("lstm_cell", params, networkParameters);
    TENSORS_CREATE(batch_size)

    const Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    const Tensor hidden_init{ 1.892910e+00_dt, 3.111044e+00_dt, -4.583958e-01_dt, -3.359881e-01_dt, -1.569986e+00_dt, 1.231500e+00_dt };
    const Tensor cell_init{ 1.394632e+00_dt, 1.171102e+00_dt, 4.335119e-01_dt, -1.734250e+00_dt, -1.336049e+00_dt, 8.870960e-01_dt };
    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);
    memory_manager["cell"] = TORANGE(cell_init);

    memory_manager[Name("new_hidden").grad()] = 1.0_dt;
    memory_manager[Name("new_cell").grad()] = 0.0_dt;

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Apply
    work.forwardPassTraining();
    work.backwardPassTraining();

    // Checks
    const Tensor inputs_grad_golden{ 8.564898e-02_dt, 8.564898e-02_dt, 8.564898e-02_dt, 8.564898e-02_dt, 1.589159e-02_dt, 1.589159e-02_dt, 1.589159e-02_dt, 1.589159e-02_dt };

    const auto& inputs_grad = memory_manager[Name("in").grad()];
    const auto& inputs = memory_manager["in"];

    EXPECT_EQ(inputs.size(), inputs_grad.size());

    for (size_t i = 0; i < inputs_grad.size(); ++i)
    {
        const auto val = inputs_grad[i];
        const auto golden_val = inputs_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

TEST(TestLSTMCell, ZoneoutP0ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto batch_size = 2U;
    const auto zonenout_prob = 0.0_dt;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", DataParams{ { "in" }, 1, 1, input_size });
    work.add<raul::DataLayer>("data2", DataParams{ { "hidden" }, 1, 1, hidden_size });
    work.add<raul::DataLayer>("data3", DataParams{ { "cell" }, 1, 1, hidden_size });

    const raul::Tensor hidden_golden{ 9.557552e-01_dt, 9.459500e-01_dt, 8.612298e-01_dt, -5.386918e-01_dt, -2.835236e-01_dt, 8.465633e-01_dt };
    const raul::Tensor cell_golden{ 2.330955e+00_dt, 2.113240e+00_dt, 1.394835e+00_dt, -6.845742e-01_dt, -3.237444e-01_dt, 1.690755e+00_dt };

    // Network with zoneout
    const auto params = raul::LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {}, true, zonenout_prob };
    raul::LSTMCellLayer("lstm_cell_zoneout", params, networkParameters);
    TENSORS_CREATE(batch_size)

    const raul::Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    const raul::Tensor hidden_init{ 1.892910e+00_dt, 3.111044e+00_dt, -4.583958e-01_dt, -3.359881e-01_dt, -1.569986e+00_dt, 1.231500e+00_dt };
    const raul::Tensor cell_init{ 1.394632e+00_dt, 1.171102e+00_dt, 4.335119e-01_dt, -1.734250e+00_dt, -1.336049e+00_dt, 8.870960e-01_dt };
    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);
    memory_manager["cell"] = TORANGE(cell_init);

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Apply
    work.forwardPassTesting();

    // Checks
    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& cell_input_tensor = memory_manager["cell"];

    const auto& hidden_new_tensor = memory_manager["new_hidden"];
    const auto& cell_new_tensor = memory_manager["new_cell"];

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

TEST(TestLSTMCell, ZoneoutP1ForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto zonenout_prob = 1.0_dt;

    // Initialization
    Workflow work;

    work.add<raul::DataLayer>("data1", DataParams{ { "in" }, 1, 1, input_size });
    work.add<raul::DataLayer>("data2", DataParams{ { "hidden" }, 1, 1, hidden_size });
    work.add<raul::DataLayer>("data3", DataParams{ { "cell" }, 1, 1, hidden_size });

    const auto params = LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {}, true, zonenout_prob };
    EXPECT_NO_THROW(LSTMCellLayer("lstm_cell_zoneout", params, work.getNetworkParameters()));
}

TEST(TestLSTMCell, ZoneoutTrainForwardRandUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto probability = raul::random::uniform::rand<raul::dtype>({ 0.0_dt, 1.0_dt });
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto batch_size = 10'000;

    std::cout << "Test with p=" << probability << std::endl;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", DataParams{ { "in" }, 1, 1, input_size });
    work.add<raul::DataLayer>("data2", DataParams{ { "hidden" }, 1, 1, hidden_size });
    work.add<raul::DataLayer>("data3", DataParams{ { "cell" }, 1, 1, hidden_size });

    // Network
    const auto params = raul::LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {}, true, probability };
    LSTMCellLayer("lstm_cell", params, networkParameters);
    TENSORS_CREATE(batch_size)

    raul::initializers::RandomUniformInitializer initializer{ -1e3_dt, 1e3_dt };

    initializer(memory_manager["in"]);
    initializer(memory_manager["hidden"]);
    initializer(memory_manager["cell"]);

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Apply
    work.forwardPassTraining();

    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& cell_input_tensor = memory_manager["cell"];

    const auto& hidden_new_tensor = memory_manager["new_hidden"];
    const auto& cell_new_tensor = memory_manager["new_cell"];

    EXPECT_EQ(hidden_input_tensor.size(), hidden_new_tensor.size());
    EXPECT_EQ(cell_input_tensor.size(), cell_new_tensor.size());

    {
        std::cout << "Hidden state:" << std::endl;

        size_t curr_cnt = 0U;
        size_t prev_cnt = 0U;

        for (size_t i = 0; i < hidden_new_tensor.size(); ++i)
        {
            if (hidden_new_tensor[i] == hidden_input_tensor[i])
            {
                ++prev_cnt;
            }
            else
            {
                ++curr_cnt;
            }
        }

        // Checks
        raul::dtype curr_prob = static_cast<raul::dtype>(curr_cnt) / static_cast<raul::dtype>(hidden_new_tensor.size());
        raul::dtype prev_prob = static_cast<raul::dtype>(prev_cnt) / static_cast<raul::dtype>(hidden_new_tensor.size());

        // Assumption
        ASSERT_TRUE(TODTYPE(hidden_new_tensor.size()) * curr_prob * (1.0_dt - curr_prob) >= 10.0_dt);
        ASSERT_TRUE(TODTYPE(hidden_new_tensor.size()) * prev_prob * (1.0_dt - prev_prob) >= 10.0_dt);

        // The confident interval for p estimation
        const auto z_ci = 4.417_dt; // 99.999%
        const auto prev_ci = z_ci * std::sqrt(prev_prob * (1.0_dt - prev_prob) / TODTYPE(hidden_new_tensor.size()));
        const auto curr_ci = z_ci * std::sqrt(curr_prob * (1.0_dt - curr_prob) / TODTYPE(hidden_new_tensor.size()));

        std::cout << "[prev prob] expected: " << probability << ", got: " << prev_prob << ", ci: " << prev_ci << std::endl;
        std::cout << "[curr prob] expected: " << 1.0_dt - probability << ", got: " << curr_prob << ", ci: " << curr_ci << std::endl;
        EXPECT_NEAR(prev_prob, probability, prev_ci);
        EXPECT_NEAR(curr_prob, 1.0_dt - probability, curr_ci);
    }

    {
        std::cout << "Cell state:" << std::endl;

        size_t curr_cnt = 0U;
        size_t prev_cnt = 0U;

        for (size_t i = 0; i < cell_new_tensor.size(); ++i)
        {
            if (cell_new_tensor[i] == cell_input_tensor[i])
            {
                ++prev_cnt;
            }
            else
            {
                ++curr_cnt;
            }
        }

        // Checks
        raul::dtype curr_prob = static_cast<raul::dtype>(curr_cnt) / static_cast<raul::dtype>(cell_new_tensor.size());
        raul::dtype prev_prob = static_cast<raul::dtype>(prev_cnt) / static_cast<raul::dtype>(cell_new_tensor.size());

        // Assumption
        ASSERT_TRUE(TODTYPE(cell_new_tensor.size()) * curr_prob * (1.0_dt - curr_prob) >= 10.0_dt);
        ASSERT_TRUE(TODTYPE(cell_new_tensor.size()) * prev_prob * (1.0_dt - prev_prob) >= 10.0_dt);

        // The confident interval for p estimation
        const auto z_ci = 4.417_dt; // 99.999%
        const auto prev_ci = z_ci * std::sqrt(prev_prob * (1.0_dt - prev_prob) / TODTYPE(cell_new_tensor.size()));
        const auto curr_ci = z_ci * std::sqrt(curr_prob * (1.0_dt - curr_prob) / TODTYPE(cell_new_tensor.size()));

        std::cout << "[prev prob] expected: " << probability << ", got: " << prev_prob << ", ci: " << prev_ci << std::endl;
        std::cout << "[curr prob] expected: " << 1.0_dt - probability << ", got: " << curr_prob << ", ci: " << curr_ci << std::endl;
        EXPECT_NEAR(prev_prob, probability, prev_ci);
        EXPECT_NEAR(curr_prob, 1.0_dt - probability, curr_ci);
    }
}

TEST(TestLSTMCell, ZoneoutTestForwardRandUnit)
{
    PROFILE_TEST
    
    // Test parameters
    [[maybe_unused]] const auto eps_rel = 1e-5_dt;
    const auto probability = raul::random::uniform::rand<raul::dtype>({ 0.0_dt, 1.0_dt });
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto batch_size = 1U;

    std::cout << "Test with p=" << probability << std::endl;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    // Network
    {
        work.add<raul::DataLayer>("data1", DataParams{ { "in" }, 1, 1, input_size });
        work.add<raul::DataLayer>("data2", DataParams{ { "hidden" }, 1, 1, hidden_size });
        work.add<raul::DataLayer>("data3", DataParams{ { "cell" }, 1, 1, hidden_size });

        const auto params = raul::LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {}, true, probability };
        LSTMCellLayer("lstm_cell", params, networkParameters);
    }

    // No zoneout
    {
        work.add<raul::DataLayer>("data4", DataParams{ { "in_no_zoneout" }, 1, 1, input_size });
        work.add<raul::DataLayer>("data5", DataParams{ { "hidden_no_zoneout" }, 1, 1, hidden_size });
        work.add<raul::DataLayer>("data6", DataParams{ { "cell_no_zoneout" }, 1, 1, hidden_size });

        const auto params = raul::LSTMCellParams{ "in_no_zoneout", "hidden_no_zoneout", "cell_no_zoneout", "new_hidden_no_zoneout", "new_cell_no_zoneout", {} };
        LSTMCellLayer("lstm_cell_no_zoneout", params, networkParameters);
    }

    TENSORS_CREATE(batch_size)

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    raul::initializers::RandomUniformInitializer initializer{ -1e3_dt, 1e3_dt };

    initializer(memory_manager["in"]);
    initializer(memory_manager["hidden"]);
    initializer(memory_manager["cell"]);

    memory_manager["in_no_zoneout"] = TORANGE(memory_manager["in"]);
    memory_manager["hidden_no_zoneout"] = TORANGE(memory_manager["hidden"]);
    memory_manager["cell_no_zoneout"] = TORANGE(memory_manager["cell"]);

    EXPECT_THROW(work.forwardPassTesting(), raul::Exception);

    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& cell_input_tensor = memory_manager["cell"];

    const auto& hidden_new_tensor_no_zoneout = memory_manager["new_hidden_no_zoneout"];
    const auto& cell_new_tensor_no_zoneout = memory_manager["new_cell_no_zoneout"];

    const auto& hidden_new_tensor = memory_manager["new_hidden"];
    const auto& cell_new_tensor = memory_manager["new_cell"];

    EXPECT_EQ(hidden_input_tensor.size(), hidden_new_tensor.size());
    EXPECT_EQ(cell_input_tensor.size(), cell_new_tensor.size());
    EXPECT_EQ(hidden_input_tensor.size(), hidden_new_tensor_no_zoneout.size());
    EXPECT_EQ(cell_input_tensor.size(), cell_new_tensor_no_zoneout.size());

#if 0
    for (size_t i = 0; i < hidden_input_tensor.size(); ++i)
    {
        const auto val = hidden_new_tensor[i];
        const auto golden_val = zoneout_outputs_test_golden(hidden_input_tensor[i], hidden_new_tensor_no_zoneout[i], probability);
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < cell_input_tensor.size(); ++i)
    {
        const auto val = cell_new_tensor[i];
        const auto golden_val = zoneout_outputs_test_golden(cell_input_tensor[i], cell_new_tensor_no_zoneout[i], probability);
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
#endif
}

TEST(TestLSTMCell, SingleParamsForwardUnit)
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
    const auto params = LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {}, true, 0.0_dt, true };
    LSTMCellLayer("lstm_cell", params, networkParameters);
    TENSORS_CREATE(batch_size)

    const Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    const Tensor hidden_init{ 1.892910e+00_dt, 3.111044e+00_dt, -4.583958e-01_dt, -3.359881e-01_dt, -1.569986e+00_dt, 1.231500e+00_dt };
    const Tensor cell_init{ 1.394632e+00_dt, 1.171102e+00_dt, 4.335119e-01_dt, -1.734250e+00_dt, -1.336049e+00_dt, 8.870960e-01_dt };
    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);
    memory_manager["cell"] = TORANGE(cell_init);

    memory_manager[Name("lstm_cell") / "linear" / "Weights"] = 1.0_dt;
    memory_manager[Name("lstm_cell") / "linear" / "Biases"] = 2.0_dt;

    // Apply
    work.forwardPassTesting();

    // Checks
    const Tensor hidden_golden{ 9.557552e-01_dt, 9.459500e-01_dt, 8.612298e-01_dt, -5.386918e-01_dt, -2.835236e-01_dt, 8.465633e-01_dt };
    const Tensor cell_golden{ 2.330955e+00_dt, 2.113240e+00_dt, 1.394835e+00_dt, -6.845742e-01_dt, -3.237444e-01_dt, 1.690755e+00_dt };

    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& cell_input_tensor = memory_manager["cell"];

    const auto& hidden_new_tensor = memory_manager["new_hidden"];
    const auto& cell_new_tensor = memory_manager["new_cell"];

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

TEST(TestLSTMCell, BackwardHiddenOnlyGradSingleWeightTensorUnit)
{
    PROFILE_TEST
    using namespace raul;

    // Test parameters
    const auto eps_rel = 1e-4_dt;
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
    const auto params = LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {}, true, 0, true };
    LSTMCellLayer("lstm_cell", params, networkParameters);
    TENSORS_CREATE(batch_size)

    memory_manager[Name("new_hidden").grad()] = 1.0_dt;
    memory_manager[Name("new_cell").grad()] = 0.0_dt;

    const Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    const Tensor hidden_init{ 1.892910e+00_dt, 3.111044e+00_dt, -4.583958e-01_dt, -3.359881e-01_dt, -1.569986e+00_dt, 1.231500e+00_dt };
    const Tensor cell_init{ 1.394632e+00_dt, 1.171102e+00_dt, 4.335119e-01_dt, -1.734250e+00_dt, -1.336049e+00_dt, 8.870960e-01_dt };
    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);
    memory_manager["cell"] = TORANGE(cell_init);

    memory_manager[Name("lstm_cell") / "linear" / "Weights"] = 1.0_dt;
    memory_manager[Name("lstm_cell") / "linear" / "Biases"] = 2.0_dt;

    // Apply
    work.forwardPassTraining();
    work.backwardPassTraining();

    // Checks
    const Tensor inputs_grad_golden{ 8.564898e-02_dt, 8.564898e-02_dt, 8.564898e-02_dt, 8.564898e-02_dt, 1.589159e-02_dt, 1.589159e-02_dt, 1.589159e-02_dt, 1.589159e-02_dt };

    const auto& inputs_grad = memory_manager[Name("in").grad()];
    const auto& inputs = memory_manager["in"];

    EXPECT_EQ(inputs.size(), inputs_grad.size());

    for (size_t i = 0; i < inputs_grad.size(); ++i)
    {
        const auto val = inputs_grad[i];
        const auto golden_val = inputs_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

TEST(TestLSTMCell, SingleParamsLoadedNoBiasForgetForwardUnit)
{
    PROFILE_TEST
    /// @note(ck): Tensorflow -> PyTorch (Raul) conversion:
    /// 1. Split the parameters into 4 parts (i, j, f, o) (axis 1 for weights, 0 for bias)
    ///    i,j,f,o = np.split(parameter, 4, axis)
    /// 2. Concatenate a param redodering parts: i, f, j, o
    ///    np.concatenate([i,f,j,o], axis)
    /// 3. If we need two tensors ih and hh, we have to split the weights tensor into 2 parts the weights tensor (axis=0).
    ///    They must be dumped in transposed mode.
    ///    weights_ih, weights_hh = np.split(weights, [input_size], axis=0)
    /// 4. If we need one tensor,  we have to dump concatenate weights in in transposed mode.

    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto batch_size = 1U;
    const auto input_size = 2U;
    const auto hidden_size = 3U;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", DataParams{ { "in" }, 1, 1, input_size });
    work.add<raul::DataLayer>("data2", DataParams{ { "hidden" }, 1, 1, hidden_size });
    work.add<raul::DataLayer>("data3", DataParams{ { "cell" }, 1, 1, hidden_size });

    // Network
    const auto params = LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {}, true, 0.0_dt, true, 0.0_dt };
    LSTMCellLayer("lstm_cell", params, networkParameters);
    TENSORS_CREATE(batch_size)

    const Tensor input_init{ -0.862954_dt, 0.515829_dt };
    const Tensor hidden_init{ 0.60593426_dt, -2.328151_dt, 0.79729331_dt };
    const Tensor cell_init{ -0.1931977_dt, -1.6983756_dt, -0.19214356_dt };
    const Tensor weights_init{ -0.441001_dt,    -0.3623873_dt,  -0.17952624_dt, 0.11902428_dt,  0.02685672_dt,  0.42737567_dt,  -0.078520119_dt, -0.21646628_dt, 0.40090138_dt,   -0.27012825_dt,
                               -0.33669758_dt,  -0.38707477_dt, -0.58897257_dt, 0.09874326_dt,  -0.50579959_dt, 0.26682276_dt,  -0.37865555_dt,  -0.40907878_dt, -0.14220604_dt,  -0.16274127_dt,
                               0.25235814_dt,   -0.16358814_dt, -0.36718047_dt, -0.44681144_dt, 0.39746886_dt,  -0.33145159_dt, 0.56001222_dt,   -0.12500507_dt, -0.10351944_dt,  0.33464533_dt,
                               -0.029248476_dt, -0.54003751_dt, -0.5581485_dt,  -0.32761213_dt, 0.44943726_dt,  0.18686062_dt,  0.4887594_dt,    -0.40617755_dt, 0.58165085_dt,   -0.042889237_dt,
                               -0.55177462_dt,  0.33988893_dt,  0.13956285_dt,  0.062051535_dt, 0.048577607_dt, -0.57949477_dt, -0.56993592_dt,  -0.23829651_dt, 0.51333058_dt,   0.064030409_dt,
                               -0.43928957_dt,  0.5261904_dt,   0.20812571_dt,  0.34141016_dt,  -0.49648321_dt, -0.55918819_dt, -0.57103509_dt,  0.44186842_dt,  -0.062169135_dt, -0.57692778_dt };

    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);
    memory_manager["cell"] = TORANGE(cell_init);

    memory_manager[Name("lstm_cell") / "linear" / "Weights"] = TORANGE(weights_init);
    memory_manager[Name("lstm_cell") / "linear" / "Biases"] = 0.0_dt;

    // Apply
    work.forwardPassTesting();

    // Checks
    const Tensor hidden_golden{ 0.037091959_dt, -0.34549943_dt, 0.011092417_dt };
    const Tensor cell_golden{ 0.14763787_dt, -1.3258458_dt, 0.020733863_dt };

    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& cell_input_tensor = memory_manager["cell"];

    const auto& hidden_new_tensor = memory_manager["new_hidden"];
    const auto& cell_new_tensor = memory_manager["new_cell"];

    EXPECT_EQ(hidden_input_tensor.size(), hidden_new_tensor.size());
    EXPECT_EQ(cell_input_tensor.size(), cell_new_tensor.size());

    for (size_t i = 0; i < cell_new_tensor.size(); ++i)
    {
        const auto cell_val = cell_new_tensor[i];
        const auto golden_cell_val = cell_golden[i];
        EXPECT_TRUE(tools::expect_near_relative(cell_val, golden_cell_val, eps_rel)) << "at " << i << ", expected: " << golden_cell_val << ", got: " << cell_val;
    }

    for (size_t i = 0; i < hidden_input_tensor.size(); ++i)
    {
        const auto hidden_val = hidden_new_tensor[i];
        const auto golden_hidden_val = hidden_golden[i];
        EXPECT_TRUE(tools::expect_near_relative(hidden_val, golden_hidden_val, eps_rel)) << "at " << i << ", expected: " << golden_hidden_val << ", got: " << hidden_val;
    }
}

TEST(TestLSTMCell, SingleParamsLoadedWithBiasForgetForwardUnit)
{
    PROFILE_TEST
    /// @note(ck): Tensorflow -> PyTorch (Raul) conversion:
    /// 1. Split the parameters into 4 parts (i, j, f, o) (axis 1 for weights, 0 for bias)
    ///    i,j,f,o = np.split(parameter, 4, axis)
    /// 2. Concatenate a param redodering parts: i, f, j, o
    ///    np.concatenate([i,f,j,o], axis)
    /// 3. If we need two tensors ih and hh, we have to split the weights tensor into 2 parts the weights tensor (axis=0).
    ///    They must be dumped in transposed mode.
    ///    weights_ih, weights_hh = np.split(weights, [input_size], axis=0)
    /// 4. If we need one tensor,  we have to dump concatenate weights in in transposed mode.

    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto batch_size = 1U;
    const auto input_size = 2U;
    const auto hidden_size = 3U;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", DataParams{ { "in" }, 1, 1, input_size });
    work.add<raul::DataLayer>("data2", DataParams{ { "hidden" }, 1, 1, hidden_size });
    work.add<raul::DataLayer>("data3", DataParams{ { "cell" }, 1, 1, hidden_size });

    // Network
    const auto params = LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {}, true, 0.0_dt, true, 1.0_dt };
    LSTMCellLayer("lstm_cell", params, networkParameters);
    TENSORS_CREATE(batch_size)

    const Tensor input_init{ -0.53056484_dt, -1.2516655_dt };
    const Tensor hidden_init{ -1.3469669_dt, -1.7043214_dt, -0.55913144_dt };
    const Tensor cell_init{ 0.59965169_dt, -0.2301345_dt, -1.0851291_dt };
    const Tensor weights_init{ -0.17065069_dt, 0.44004658_dt,  0.85328132_dt,   -0.76646215_dt,  -1.2310894_dt,  -0.56331462_dt, 0.48660466_dt,  -0.067891933_dt, -0.19195049_dt, 2.1599874_dt,
                               0.61151052_dt,  1.2376982_dt,   -0.015872575_dt, -1.3109018_dt,   -0.65890425_dt, -0.32336307_dt, -0.74042344_dt, -1.1113732_dt,   -0.5390079_dt,  -0.22297508_dt,
                               2.0183513_dt,   -0.22923379_dt, -0.61300713_dt,  -0.35666472_dt,  -0.9824217_dt,  0.8425144_dt,   0.91593617_dt,  1.8140028_dt,    1.0173856_dt,   0.39566979_dt,
                               0.99805921_dt,  1.4657131_dt,   0.34357622_dt,   -0.094996393_dt, 0.71959883_dt,  1.3679156_dt,   -1.430184_dt,   -0.88232619_dt,  -0.58268321_dt, 0.11771394_dt,
                               -0.24999142_dt, 1.7739073_dt,   1.8358583_dt,    0.49045855_dt,   2.9916673_dt,   1.3255521_dt,   0.55856633_dt,  -0.41269496_dt,  0.62596655_dt,  0.37756023_dt,
                               0.0746601_dt,   -0.52719849_dt, 1.4623779_dt,    -0.092216626_dt, -0.35693371_dt, 0.064323775_dt, 1.0745957_dt,   -0.32174867_dt,  0.51347214_dt,  0.33080587_dt };
    const Tensor bias_init{ 2.0_dt, 2.0_dt, 2.0_dt, 3.0_dt, 3.0_dt, 3.0_dt, 2.0_dt, 2.0_dt, 2.0_dt, 3.0_dt, 3.0_dt, 3.0_dt };

    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);
    memory_manager["cell"] = TORANGE(cell_init);

    memory_manager[Name("lstm_cell") / "linear" / "Weights"] = TORANGE(weights_init);
    memory_manager[Name("lstm_cell") / "linear" / "Biases"] = TORANGE(bias_init);

    // Apply
    work.forwardPassTesting();

    // Checks
    const Tensor hidden_golden{ -0.086121075_dt, 0.39492172_dt, -0.57626414_dt };
    const Tensor cell_golden{ -0.12261444_dt, 0.48217469_dt, -1.0701199_dt };

    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& cell_input_tensor = memory_manager["cell"];

    const auto& hidden_new_tensor = memory_manager["new_hidden"];
    const auto& cell_new_tensor = memory_manager["new_cell"];

    EXPECT_EQ(hidden_input_tensor.size(), hidden_new_tensor.size());
    EXPECT_EQ(cell_input_tensor.size(), cell_new_tensor.size());

    for (size_t i = 0; i < cell_new_tensor.size(); ++i)
    {
        const auto cell_val = cell_new_tensor[i];
        const auto golden_cell_val = cell_golden[i];
        EXPECT_TRUE(tools::expect_near_relative(cell_val, golden_cell_val, eps_rel)) << "at " << i << ", expected: " << golden_cell_val << ", got: " << cell_val;
    }

    for (size_t i = 0; i < hidden_input_tensor.size(); ++i)
    {
        const auto hidden_val = hidden_new_tensor[i];
        const auto golden_hidden_val = hidden_golden[i];
        EXPECT_TRUE(tools::expect_near_relative(hidden_val, golden_hidden_val, eps_rel)) << "at " << i << ", expected: " << golden_hidden_val << ", got: " << hidden_val;
    }
}

TEST(TestLSTMCell, FusionOnSimpleForwardBackwardUnit)
{
    PROFILE_TEST
    using namespace raul;

    // Test parameters
    const auto eps = 1e-5_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto batch_size = 2U;

    const Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    const Tensor hidden_init{ 1.892910e+00_dt, 3.111044e+00_dt, -4.583958e-01_dt, -3.359881e-01_dt, -1.569986e+00_dt, 1.231500e+00_dt };
    const Tensor cell_init{ 1.394632e+00_dt, 1.171102e+00_dt, 4.335119e-01_dt, -1.734250e+00_dt, -1.336049e+00_dt, 8.870960e-01_dt };

    const Tensor hidden_golden{ 0.955755_dt, 0.94595_dt, 0.86123_dt, -0.538692_dt, -0.283524_dt, 0.846563_dt };
    const Tensor cell_golden{ 2.33095_dt, 2.11324_dt, 1.39484_dt, -0.684574_dt, -0.323745_dt, 1.69076_dt };

    const Tensor inputs_grad_golden{ 0.296787_dt, 0.296787_dt, 0.296787_dt, 0.296787_dt, 0.389521_dt, 0.389521_dt, 0.389521_dt, 0.389521_dt };
    const Tensor hidden_grad_golden{ 0.296787_dt, 0.296787_dt, 0.296787_dt, 0.389521_dt, 0.389521_dt, 0.389521_dt };
    const Tensor cell_grad_golden{ 1.42496_dt, 1.30811_dt, 0.327377_dt, -1.74987_dt, -2.37358_dt, 0.932454_dt };

    const Tensor biases_ih_grad_golden{ -0.123729_dt, -0.184057_dt, 0.094158_dt, 0.336505_dt, 0.337471_dt, 0.0813232_dt,
        -0.0694392_dt, -0.0959693_dt, 0.0400979_dt, 0.0640461_dt, 0.118326_dt, 0.087575_dt };
    const Tensor weights_ih_grad_golden{ -0.206206_dt, -0.0404442_dt, -0.00992588_dt, -0.00301529_dt, -0.282549_dt,
        -0.0397434_dt, 0.0109667_dt, 0.00663612_dt, 0.114829_dt, -0.00474223_dt, -0.0372058_dt, -0.0170506_dt, 0.385924_dt,
        -0.0375827_dt, -0.158969_dt, -0.072174_dt, 0.4008_dt, -0.0260736_dt, -0.144787_dt, -0.0660544_dt, 0.103369_dt,
        -0.00055761_dt, -0.0276757_dt, -0.0127994_dt, -0.0965421_dt, -0.00650937_dt, 0.0148289_dt, 0.00712453_dt, -0.131265_dt,
        -0.00717158_dt, 0.0227939_dt, 0.0108404_dt, 0.051987_dt, 0.000584807_dt, -0.0125626_dt, -0.00584275_dt, 0.0307254_dt,
        -0.0432066_dt, -0.0756874_dt, -0.0333723_dt, 0.0686959_dt, -0.0697576_dt, -0.127148_dt, -0.056173_dt, 0.127667_dt,
        0.0131973_dt, -0.0124167_dt, -0.00626879_dt };
    const Tensor biases_hh_grad_golden{ -0.123729_dt, -0.184057_dt, 0.094158_dt, 0.336505_dt, 0.337471_dt, 0.0813232_dt,
        -0.0694392_dt, -0.0959693_dt, 0.0400979_dt, 0.0640461_dt, 0.118326_dt, 0.087575_dt };
    const Tensor weights_hh_grad_golden{ 0.12407_dt, 0.367511_dt, -0.21492_dt, 0.137574_dt, 0.448019_dt, -0.284085_dt,
        -0.0126824_dt, -0.108021_dt, 0.101585_dt, 0.00215716_dt, -0.286331_dt, 0.32705_dt, -0.0245682_dt, -0.343293_dt,
        0.348256_dt, -0.0190953_dt, -0.110396_dt, 0.093911_dt, 0.032379_dt, 0.128021_dt, -0.0923745_dt, 0.0405508_dt,
        0.168115_dt, -0.124484_dt, -0.0113936_dt, -0.0585874_dt, 0.0478045_dt, 0.0833724_dt, 0.119736_dt, -0.000653027_dt,
        0.130866_dt, 0.172563_dt, 0.0163568_dt, -0.0523129_dt, -0.185561_dt, 0.125202_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", DataParams{ { "in" }, 1, 1, input_size });
    work.add<raul::DataLayer>("data2", DataParams{ { "hidden" }, 1, 1, hidden_size });
    work.add<raul::DataLayer>("data3", DataParams{ { "cell" }, 1, 1, hidden_size });

    // Network
    const auto params = LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {}, true, 0.0_dt, false, 0.0_dt, false, true };
    LSTMCellLayer("lstm_cell", params, networkParameters);
    TENSORS_CREATE(batch_size)

    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);
    memory_manager["cell"] = TORANGE(cell_init);

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Forward checks
    ASSERT_NO_THROW(work.forwardPassTraining());

    const auto& hidden_new_tensor = memory_manager["new_hidden"];
    const auto& cell_new_tensor = memory_manager["new_cell"];
    EXPECT_EQ(hidden_init.size(), hidden_new_tensor.size());
    EXPECT_EQ(cell_init.size(), cell_new_tensor.size());

    for (size_t i = 0; i < hidden_new_tensor.size(); ++i)
    {
        const auto hidden_val = hidden_new_tensor[i];
        const auto golden_hidden_val = hidden_golden[i];
        EXPECT_NEAR(hidden_val, golden_hidden_val, eps);
    }
    
    for (size_t i = 0; i < cell_new_tensor.size(); ++i)
    {
        const auto cell_val = cell_new_tensor[i];
        const auto golden_cell_val = cell_golden[i];
        EXPECT_NEAR(cell_val, golden_cell_val, eps);
    }

    memory_manager[Name("new_hidden").grad()] = TORANGE(hidden_init);
    memory_manager[Name("new_cell").grad()] = TORANGE(cell_init);

    // Backward checks
    ASSERT_NO_THROW(work.backwardPassTraining());

    const auto& inputs_grad = memory_manager[Name("in").grad()];
    const auto& inputs = memory_manager["in"];

    EXPECT_EQ(inputs.size(), inputs_grad.size());

    for (size_t i = 0; i < inputs_grad.size(); ++i)
    {
        const auto val = inputs_grad[i];
        const auto golden_val = inputs_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    const auto& hidden_grad = memory_manager[Name("hidden").grad()];
    EXPECT_EQ(hidden_grad_golden.size(), hidden_grad.size());
    for (size_t i = 0; i < hidden_grad.size(); ++i)
    {
        const auto val = hidden_grad[i];
        const auto golden_val = hidden_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    const auto& cell_grad = memory_manager[Name("cell").grad()];
    EXPECT_EQ(cell_grad_golden.size(), cell_grad.size());
    for (size_t i = 0; i < cell_grad.size(); ++i)
    {
        const auto val = cell_grad[i];
        const auto golden_val = cell_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    // Check trainable params grads
    const auto paramGrad = work.getTrainableParameters();
    
    const auto biases_hh_grad = paramGrad[0].Gradient;
    EXPECT_EQ(biases_hh_grad_golden.size(), biases_hh_grad.size());
    for (size_t i = 0; i < biases_hh_grad.size(); ++i)
    {
        const auto val = biases_hh_grad[i];
        const auto golden_val = biases_hh_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    const auto weights_hh_grad = paramGrad[1].Gradient;
    EXPECT_EQ(weights_hh_grad_golden.size(), weights_hh_grad.size());
    for (size_t i = 0; i < weights_hh_grad.size(); ++i)
    {
        const auto val = weights_hh_grad[i];
        const auto golden_val = weights_hh_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    const auto biases_ih_grad = paramGrad[2].Gradient;
    EXPECT_EQ(biases_ih_grad_golden.size(), biases_ih_grad.size());
    for (size_t i = 0; i < biases_ih_grad.size(); ++i)
    {
        const auto val = biases_ih_grad[i];
        const auto golden_val = biases_ih_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    const auto weights_ih_grad = paramGrad[3].Gradient;
    EXPECT_EQ(weights_ih_grad_golden.size(), weights_ih_grad.size());
    for (size_t i = 0; i < weights_ih_grad.size(); ++i)
    {
        const auto val = weights_ih_grad[i];
        const auto golden_val = weights_ih_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

TEST(TestLSTMCell, SingleParamsFusionOnForwardBackwardUnit)
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
    const auto params = LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {}, true, 0.0_dt, true, 0.0_dt, false, true };
    LSTMCellLayer("lstm_cell", params, networkParameters);
    TENSORS_CREATE(batch_size)

    const Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    const Tensor hidden_init{ 1.892910e+00_dt, 3.111044e+00_dt, -4.583958e-01_dt, -3.359881e-01_dt, -1.569986e+00_dt, 1.231500e+00_dt };
    const Tensor cell_init{ 1.394632e+00_dt, 1.171102e+00_dt, 4.335119e-01_dt, -1.734250e+00_dt, -1.336049e+00_dt, 8.870960e-01_dt };
    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);
    memory_manager["cell"] = TORANGE(cell_init);

    dtype initValue = 2.0_dt;
    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = initValue;
        initValue -= 1.0_dt;
    }

    // Apply
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Checks
    const Tensor hidden_golden{ 9.557552e-01_dt, 9.459500e-01_dt, 8.612298e-01_dt, -5.386918e-01_dt, -2.835236e-01_dt, 8.465633e-01_dt };
    const Tensor cell_golden{ 2.330955e+00_dt, 2.113240e+00_dt, 1.394835e+00_dt, -6.845742e-01_dt, -3.237444e-01_dt, 1.690755e+00_dt };

    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& cell_input_tensor = memory_manager["cell"];

    const auto& hidden_new_tensor = memory_manager["new_hidden"];
    const auto& cell_new_tensor = memory_manager["new_cell"];

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

    memory_manager[Name("new_hidden").grad()] = TORANGE(hidden_init);
    memory_manager[Name("new_cell").grad()] = TORANGE(cell_init);

    // Apply
    ASSERT_NO_THROW(work.backwardPassTraining());

    // Checks
    const Tensor inputs_grad_golden{ 0.296787_dt, 0.296787_dt, 0.296787_dt, 0.296787_dt, 0.389521_dt, 0.389521_dt, 0.389521_dt, 0.389521_dt };
    const Tensor hidden_grad_golden{ 0.296787_dt, 0.296787_dt, 0.296787_dt, 0.389521_dt, 0.389521_dt, 0.389521_dt };
    const Tensor cell_grad_golden{ 1.42496_dt, 1.30811_dt, 0.327377_dt, -1.74987_dt, -2.37358_dt, 0.932454_dt };

    const Tensor biases_grad_golden{ -0.123729_dt, -0.184057_dt, 0.094158_dt, 0.336505_dt, 0.337471_dt, 0.0813232_dt,
        -0.0694392_dt, -0.0959693_dt, 0.0400979_dt, 0.0640461_dt, 0.118326_dt, 0.087575_dt };
    const Tensor weights_grad_golden{ -0.206206_dt, -0.0404442_dt, -0.00992588_dt, -0.00301529_dt, 0.12407_dt, 0.367511_dt,
        -0.21492_dt, -0.282549_dt, -0.0397434_dt, 0.0109667_dt, 0.00663612_dt, 0.137574_dt, 0.448019_dt, -0.284085_dt,
        0.114829_dt, -0.00474223_dt, -0.0372058_dt, -0.0170506_dt, -0.0126824_dt, -0.108021_dt, 0.101585_dt, 0.385924_dt,
        -0.0375827_dt, -0.158969_dt, -0.072174_dt, 0.00215716_dt, -0.286331_dt, 0.32705_dt, 0.4008_dt, -0.0260736_dt,
        -0.144787_dt, -0.0660544_dt, -0.0245682_dt, -0.343293_dt, 0.348256_dt, 0.103369_dt, -0.00055761_dt, -0.0276757_dt,
        -0.0127994_dt, -0.0190953_dt, -0.110396_dt, 0.093911_dt, -0.0965421_dt, -0.00650937_dt, 0.0148289_dt, 0.00712453_dt,
        0.032379_dt, 0.128021_dt, -0.0923745_dt, -0.131265_dt, -0.00717158_dt, 0.0227939_dt, 0.0108404_dt, 0.0405508_dt,
        0.168115_dt, -0.124484_dt, 0.051987_dt, 0.000584807_dt, -0.0125626_dt, -0.00584275_dt, -0.0113936_dt, -0.0585874_dt,
        0.0478045_dt, 0.0307254_dt, -0.0432066_dt, -0.0756874_dt, -0.0333723_dt, 0.0833724_dt, 0.119736_dt, -0.000653027_dt,
        0.0686959_dt, -0.0697576_dt, -0.127148_dt, -0.056173_dt, 0.130866_dt, 0.172563_dt, 0.0163568_dt, 0.127667_dt,
        0.0131973_dt, -0.0124167_dt, -0.00626879_dt, -0.0523129_dt, -0.185561_dt, 0.125202_dt };

    const auto& inputs_grad = memory_manager[Name("in").grad()];
    const auto& inputs = memory_manager["in"];

    EXPECT_EQ(inputs.size(), inputs_grad.size());

    for (size_t i = 0; i < inputs_grad.size(); ++i)
    {
        const auto val = inputs_grad[i];
        const auto golden_val = inputs_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    const auto& hidden_grad = memory_manager[Name("hidden").grad()];
    EXPECT_EQ(hidden_grad_golden.size(), hidden_grad.size());
    for (size_t i = 0; i < hidden_grad.size(); ++i)
    {
        const auto val = hidden_grad[i];
        const auto golden_val = hidden_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    const auto& cell_grad = memory_manager[Name("cell").grad()];
    EXPECT_EQ(cell_grad_golden.size(), cell_grad.size());
    for (size_t i = 0; i < cell_grad.size(); ++i)
    {
        const auto val = cell_grad[i];
        const auto golden_val = cell_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    // Check trainable params
    const auto paramGrad = work.getTrainableParameters();
    
    const auto biases_grad = paramGrad[0].Gradient;
    EXPECT_EQ(biases_grad_golden.size(), biases_grad.size());
    for (size_t i = 0; i < biases_grad.size(); ++i)
    {
        const auto val = biases_grad[i];
        const auto golden_val = biases_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    const auto weights_grad = paramGrad[1].Gradient;
    EXPECT_EQ(weights_grad_golden.size(), weights_grad.size());
    for (size_t i = 0; i < weights_grad.size(); ++i)
    {
        const auto val = weights_grad[i];
        const auto golden_val = weights_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

TEST(TestLSTMCell, SingleParamsZoneoutP1FusionOnForwardBackwardUnit)
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
    const auto params = LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {}, true, 1.0_dt, true, 0.0_dt, false, true };
    LSTMCellLayer("lstm_cell", params, networkParameters);
    TENSORS_CREATE(batch_size)

    const Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    const Tensor hidden_init{ 1.892910e+00_dt, 3.111044e+00_dt, -4.583958e-01_dt, -3.359881e-01_dt, -1.569986e+00_dt, 1.231500e+00_dt };
    const Tensor cell_init{ 1.394632e+00_dt, 1.171102e+00_dt, 4.335119e-01_dt, -1.734250e+00_dt, -1.336049e+00_dt, 8.870960e-01_dt };
    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);
    memory_manager["cell"] = TORANGE(cell_init);

    dtype initValue = 2.0_dt;
    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = initValue;
        initValue -= 1.0_dt;
    }

    // Apply
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Checks
    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& cell_input_tensor = memory_manager["cell"];

    const auto& hidden_new_tensor = memory_manager["new_hidden"];
    const auto& cell_new_tensor = memory_manager["new_cell"];

    EXPECT_EQ(hidden_input_tensor.size(), hidden_new_tensor.size());
    EXPECT_EQ(cell_input_tensor.size(), cell_new_tensor.size());

    for (size_t i = 0; i < hidden_input_tensor.size(); ++i)
    {
        const auto cell_val = cell_new_tensor[i];
        const auto golden_cell_val = cell_input_tensor[i];
        ASSERT_TRUE(tools::expect_near_relative(cell_val, golden_cell_val, eps_rel)) << "at " << i << ", expected: " << golden_cell_val << ", got: " << cell_val;

        const auto hidden_val = hidden_new_tensor[i];
        const auto golden_hidden_val = hidden_input_tensor[i];
        ASSERT_TRUE(tools::expect_near_relative(hidden_val, golden_hidden_val, eps_rel)) << "at " << i << ", expected: " << golden_hidden_val << ", got: " << hidden_val;
    }

    memory_manager[Name("new_hidden").grad()] = TORANGE(hidden_init);
    memory_manager[Name("new_cell").grad()] = TORANGE(cell_init);

    // Apply
    ASSERT_NO_THROW(work.backwardPassTraining());

    // Checks
    const Tensor biases_grad_golden{ -0.123729_dt, -0.184057_dt, 0.094158_dt, 0.336505_dt, 0.337471_dt, 0.0813232_dt,
        -0.0694392_dt, -0.0959693_dt, 0.0400979_dt, 0.0640461_dt, 0.118326_dt, 0.087575_dt };
    const Tensor weights_grad_golden{ -0.206206_dt, -0.0404442_dt, -0.00992588_dt, -0.00301529_dt, 0.12407_dt, 0.367511_dt,
        -0.21492_dt, -0.282549_dt, -0.0397434_dt, 0.0109667_dt, 0.00663612_dt, 0.137574_dt, 0.448019_dt, -0.284085_dt,
        0.114829_dt, -0.00474223_dt, -0.0372058_dt, -0.0170506_dt, -0.0126824_dt, -0.108021_dt, 0.101585_dt, 0.385924_dt,
        -0.0375827_dt, -0.158969_dt, -0.072174_dt, 0.00215716_dt, -0.286331_dt, 0.32705_dt, 0.4008_dt, -0.0260736_dt,
        -0.144787_dt, -0.0660544_dt, -0.0245682_dt, -0.343293_dt, 0.348256_dt, 0.103369_dt, -0.00055761_dt, -0.0276757_dt,
        -0.0127994_dt, -0.0190953_dt, -0.110396_dt, 0.093911_dt, -0.0965421_dt, -0.00650937_dt, 0.0148289_dt, 0.00712453_dt,
        0.032379_dt, 0.128021_dt, -0.0923745_dt, -0.131265_dt, -0.00717158_dt, 0.0227939_dt, 0.0108404_dt, 0.0405508_dt,
        0.168115_dt, -0.124484_dt, 0.051987_dt, 0.000584807_dt, -0.0125626_dt, -0.00584275_dt, -0.0113936_dt, -0.0585874_dt,
        0.0478045_dt, 0.0307254_dt, -0.0432066_dt, -0.0756874_dt, -0.0333723_dt, 0.0833724_dt, 0.119736_dt, -0.000653027_dt,
        0.0686959_dt, -0.0697576_dt, -0.127148_dt, -0.056173_dt, 0.130866_dt, 0.172563_dt, 0.0163568_dt, 0.127667_dt,
        0.0131973_dt, -0.0124167_dt, -0.00626879_dt, -0.0523129_dt, -0.185561_dt, 0.125202_dt };

    const auto& inputs_grad = memory_manager[Name("in").grad()];
    const auto& inputs = memory_manager["in"];

    EXPECT_EQ(inputs.size(), inputs_grad.size());

    for (size_t i = 0; i < inputs_grad.size(); ++i)
    {
        const auto val = inputs_grad[i];
        EXPECT_EQ(val, 0.0_dt);
    }

    const auto& hidden_grad = memory_manager[Name("hidden").grad()];
    EXPECT_EQ(hidden_init.size(), hidden_grad.size());
    for (size_t i = 0; i < hidden_grad.size(); ++i)
    {
        const auto val = hidden_grad[i];
        const auto golden_val = hidden_init[i];
        EXPECT_EQ(val, golden_val);
    }

    const auto& cell_grad = memory_manager[Name("cell").grad()];
    EXPECT_EQ(cell_init.size(), cell_grad.size());
    for (size_t i = 0; i < cell_grad.size(); ++i)
    {
        const auto val = cell_grad[i];
        const auto golden_val = cell_init[i];
        EXPECT_EQ(val, golden_val);
    }

    // Check trainable params
    for (auto& [param, grad] : work.getTrainableParameters())
    {
        for (size_t i = 0; i < grad.size(); ++i)
        {
            EXPECT_EQ(grad[i], 0.0_dt);
        }
    }
}

TEST(TestLSTMCell, SingleParamsForgetBiasFusionOnForwardBackwardUnit)
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
    const auto params = LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {}, true, 0.0_dt, true, 1.0_dt, false, true };
    LSTMCellLayer("lstm_cell", params, networkParameters);
    TENSORS_CREATE(batch_size)

    const Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    const Tensor hidden_init{ 1.892910e+00_dt, 3.111044e+00_dt, -4.583958e-01_dt, -3.359881e-01_dt, -1.569986e+00_dt, 1.231500e+00_dt };
    const Tensor cell_init{ 1.394632e+00_dt, 1.171102e+00_dt, 4.335119e-01_dt, -1.734250e+00_dt, -1.336049e+00_dt, 8.870960e-01_dt };
    memory_manager["in"] = TORANGE(input_init);
    memory_manager["hidden"] = TORANGE(hidden_init);
    memory_manager["cell"] = TORANGE(cell_init);

    dtype initValue = 2.0_dt;
    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = initValue;
        initValue -= 1.0_dt;
    }

    // Apply
    ASSERT_NO_THROW(work.forwardPassTraining());

    // Checks
    const Tensor hidden_golden{ 0.956557_dt, 0.946984_dt, 0.86272_dt, -0.59336_dt, -0.344365_dt, 0.852137_dt };
    const Tensor cell_golden{ 2.35366_dt, 2.13231_dt, 1.40189_dt, -0.783683_dt, -0.400097_dt, 1.74145_dt };

    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& cell_input_tensor = memory_manager["cell"];

    const auto& hidden_new_tensor = memory_manager["new_hidden"];
    const auto& cell_new_tensor = memory_manager["new_cell"];

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

    memory_manager[Name("new_hidden").grad()] = TORANGE(hidden_init);
    memory_manager[Name("new_cell").grad()] = TORANGE(cell_init);

    // Apply
    ASSERT_NO_THROW(work.backwardPassTraining());

    // Checks
    const Tensor inputs_grad_golden{ 0.237544_dt, 0.237544_dt, 0.237544_dt, 0.237544_dt, 0.0195209_dt, 0.0195209_dt, 0.0195209_dt, 0.0195209_dt };
    const Tensor hidden_grad_golden{ 0.237544_dt, 0.237544_dt, 0.237544_dt, 0.0195209_dt, 0.0195209_dt, 0.0195209_dt };
    const Tensor cell_grad_golden{ 1.44583_dt, 1.32379_dt, 0.334047_dt, -1.83812_dt, -2.45951_dt, 0.978865_dt };

    const Tensor biases_grad_golden{ -0.121894_dt, -0.178706_dt, 0.0931175_dt, 0.13662_dt, 0.135692_dt, 0.0332802_dt, -0.0685741_dt, -0.0934675_dt, 0.0396115_dt, 0.0658095_dt, 0.127375_dt, 0.0882014_dt };
    const Tensor weights_grad_golden{ -0.203692_dt, -0.040303_dt, -0.0103565_dt, -0.00322031_dt, 0.123285_dt, 0.364278_dt,
        -0.212533_dt, -0.275286_dt, -0.0393915_dt, 0.00963532_dt, 0.00600555_dt, 0.135424_dt, 0.438877_dt, -0.277228_dt,
        0.113417_dt, -0.00481048_dt, -0.0369466_dt, -0.0169279_dt, -0.0122646_dt, -0.106245_dt, 0.100252_dt, 0.158256_dt,
        -0.013932_dt, -0.0628694_dt, -0.02858_dt, -0.00217656_dt, -0.12266_dt, 0.135095_dt, 0.162233_dt, -0.00957381_dt,
        -0.0570698_dt, -0.0260638_dt, -0.0119724_dt, -0.14243_dt, 0.141616_dt, 0.0424192_dt, -0.000129548_dt, -0.0112015_dt,
        -0.00518423_dt, -0.00804145_dt, -0.0456544_dt, 0.0386037_dt, -0.0953879_dt, -0.00646925_dt, 0.0145925_dt, 0.00701345_dt,
        0.0320699_dt, 0.126624_dt, -0.0912952_dt, -0.127934_dt, -0.00706192_dt, 0.0221023_dt, 0.0105157_dt, 0.0396716_dt,
        0.164106_dt, -0.121374_dt, 0.0513396_dt, 0.000563504_dt, -0.0124282_dt, -0.00577961_dt, -0.0112227_dt, -0.057808_dt,
        0.0471998_dt, 0.0330133_dt, -0.0431794_dt, -0.076238_dt, -0.0336284_dt, 0.0828679_dt, 0.117152_dt, 0.00145179_dt,
        0.0805727_dt, -0.0695029_dt, -0.129828_dt, -0.0574246_dt, 0.128013_dt, 0.158749_dt, 0.0273584_dt, 0.128517_dt,
        0.0132377_dt, -0.0125735_dt, -0.00634302_dt, -0.052563_dt, -0.186628_dt, 0.126004_dt };

    const auto& inputs_grad = memory_manager[Name("in").grad()];
    const auto& inputs = memory_manager["in"];

    EXPECT_EQ(inputs.size(), inputs_grad.size());

    for (size_t i = 0; i < inputs_grad.size(); ++i)
    {
        const auto val = inputs_grad[i];
        const auto golden_val = inputs_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    const auto& hidden_grad = memory_manager[Name("hidden").grad()];
    EXPECT_EQ(hidden_grad_golden.size(), hidden_grad.size());
    for (size_t i = 0; i < hidden_grad.size(); ++i)
    {
        const auto val = hidden_grad[i];
        const auto golden_val = hidden_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    const auto& cell_grad = memory_manager[Name("cell").grad()];
    EXPECT_EQ(cell_grad_golden.size(), cell_grad.size());
    for (size_t i = 0; i < cell_grad.size(); ++i)
    {
        const auto val = cell_grad[i];
        const auto golden_val = cell_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    // Check trainable params
    const auto paramGrad = work.getTrainableParameters();
    
    const auto biases_grad = paramGrad[0].Gradient;
    EXPECT_EQ(biases_grad_golden.size(), biases_grad.size());
    for (size_t i = 0; i < biases_grad.size(); ++i)
    {
        const auto val = biases_grad[i];
        const auto golden_val = biases_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    const auto weights_grad = paramGrad[1].Gradient;
    EXPECT_EQ(weights_grad_golden.size(), weights_grad.size());
    for (size_t i = 0; i < weights_grad.size(); ++i)
    {
        const auto val = weights_grad[i];
        const auto golden_val = weights_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

#ifdef ANDROID
TEST(TestLSTMCell, FP16FusionOffAndOnUnit)
{
    PROFILE_TEST
    using namespace raul;

    // Test parameters
    const auto eps_rel = 1e-1_hf;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto batch_size = 2U;

    const TensorFP16 input_init{ 1.752833e-01_hf, -9.315211e-01_hf, -1.505490e+00_hf, -6.609825e-01_hf, 1.323202e+00_hf, 3.711430e-02_hf, -2.849093e-01_hf, -1.334417e-01_hf };
    const TensorFP16 hidden_init{ 1.892910e+00_hf, 3.111044e+00_hf, -4.583958e-01_hf, -3.359881e-01_hf, -1.569986e+00_hf, 1.231500e+00_hf };
    const TensorFP16 cell_init{ 1.394632e+00_hf, 1.171102e+00_hf, 4.335119e-01_hf, -1.734250e+00_hf, -1.336049e+00_hf, 8.870960e-01_hf };

    const TensorFP16 hidden_golden{ 0.955755_hf, 0.94595_hf, 0.86123_hf, -0.538692_hf, -0.283524_hf, 0.846563_hf };
    const TensorFP16 cell_golden{ 2.33095_hf, 2.11324_hf, 1.39484_hf, -0.684574_hf, -0.323745_hf, 1.69076_hf };

    const TensorFP16 inputs_grad_golden{ 0.296787_hf, 0.296787_hf, 0.296787_hf, 0.296787_hf, 0.389521_hf, 0.389521_hf, 0.389521_hf, 0.389521_hf };
    const TensorFP16 hidden_grad_golden{ 0.296787_hf, 0.296787_hf, 0.296787_hf, 0.389521_hf, 0.389521_hf, 0.389521_hf };
    const TensorFP16 cell_grad_golden{ 1.42496_hf, 1.30811_hf, 0.327377_hf, -1.74987_hf, -2.37358_hf, 0.932454_hf };

    const TensorFP16 biases_ih_grad_golden{ -0.123729_hf, -0.184057_hf, 0.094158_hf, 0.336505_hf, 0.337471_hf, 0.0813232_hf,
        -0.0694392_hf, -0.0959693_hf, 0.0400979_hf, 0.0640461_hf, 0.118326_hf, 0.087575_hf };
    const TensorFP16 weights_ih_grad_golden{ -0.206206_hf, -0.0404442_hf, -0.00992588_hf, -0.00301529_hf, -0.282549_hf,
        -0.0397434_hf, 0.0109667_hf, 0.00663612_hf, 0.114829_hf, -0.00474223_hf, -0.0372058_hf, -0.0170506_hf, 0.385924_hf,
        -0.0375827_hf, -0.158969_hf, -0.072174_hf, 0.4008_hf, -0.0260736_hf, -0.144787_hf, -0.0660544_hf, 0.103369_hf,
        -0.00055761_hf, -0.0276757_hf, -0.0127994_hf, -0.0965421_hf, -0.00650937_hf, 0.0148289_hf, 0.00712453_hf, -0.131265_hf,
        -0.00717158_hf, 0.0227939_hf, 0.0108404_hf, 0.051987_hf, 0.000584807_hf, -0.0125626_hf, -0.00584275_hf, 0.0307254_hf,
        -0.0432066_hf, -0.0756874_hf, -0.0333723_hf, 0.0686959_hf, -0.0697576_hf, -0.127148_hf, -0.056173_hf, 0.127667_hf,
        0.0131973_hf, -0.0124167_hf, -0.00626879_hf };

    // Initialization
    for (size_t q = 0; q < 2; ++q)
    {
        raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16 };
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data1", DataParams{ { "in" }, 1, 1, input_size });
        work.add<raul::DataLayer>("data2", DataParams{ { "hidden" }, 1, 1, hidden_size });
        work.add<raul::DataLayer>("data3", DataParams{ { "cell" }, 1, 1, hidden_size });

        // Network
        bool useFusion = (q == 1 ? true : false);
        const auto params = raul::LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {}, true, 0.0_dt, false, 0.0_dt, false, useFusion };
        raul::LSTMCellLayer("lstm_cell", params, networkParameters);
        TENSORS_CREATE(batch_size)
            
        auto& memory_manager = work.getMemoryManager<raul::MemoryManagerFP16>();
        memory_manager["in"] = TORANGE_FP16(input_init);
        memory_manager["hidden"] = TORANGE_FP16(hidden_init);
        memory_manager["cell"] = TORANGE_FP16(cell_init);

        for (auto& [param, grad] : work.getTrainableParameters<MemoryManagerFP16>())
        {
            param = 1.0_hf;
        }

        EXPECT_NO_THROW(work.forwardPassTraining());

        // Checks
        const auto& hidden_new_tensor = memory_manager["new_hidden"];
        const auto& cell_new_tensor = memory_manager["new_cell"];
        EXPECT_EQ(hidden_init.size(), hidden_new_tensor.size());
        EXPECT_EQ(cell_init.size(), cell_new_tensor.size());

        for (size_t i = 0; i < hidden_new_tensor.size(); ++i)
        {
            const auto val = hidden_new_tensor[i];
            const auto golden_val = hidden_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << TODTYPE(golden_val) << ", got: " << TODTYPE(val);
        }
        
        for (size_t i = 0; i < cell_new_tensor.size(); ++i)
        {
            const auto val = cell_new_tensor[i];
            const auto golden_val = cell_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << TODTYPE(golden_val) << ", got: " << TODTYPE(val);
        }

        memory_manager[Name("new_hidden").grad()] = TORANGE_FP16(hidden_init);
        memory_manager[Name("new_cell").grad()] = TORANGE_FP16(cell_init);

        EXPECT_NO_THROW(work.backwardPassTraining());
        const auto& inputs_grad = memory_manager[Name("in").grad()];
        const auto& inputs = memory_manager["in"];

        EXPECT_EQ(inputs.size(), inputs_grad.size());

        for (size_t i = 0; i < inputs_grad.size(); ++i)
        {
            const auto val = inputs_grad[i];
            const auto golden_val = inputs_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << TODTYPE(golden_val) << ", got: " << TODTYPE(val);
        }

        const auto& hidden_grad = memory_manager[Name("hidden").grad()];
        EXPECT_EQ(hidden_grad_golden.size(), hidden_grad.size());
        for (size_t i = 0; i < hidden_grad.size(); ++i)
        {
            const auto val = hidden_grad[i];
            const auto golden_val = hidden_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << TODTYPE(golden_val) << ", got: " << TODTYPE(val);
        }

        const auto& cell_grad = memory_manager[Name("cell").grad()];
        EXPECT_EQ(cell_grad_golden.size(), cell_grad.size());
        for (size_t i = 0; i < cell_grad.size(); ++i)
        {
            const auto val = cell_grad[i];
            const auto golden_val = cell_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << TODTYPE(golden_val) << ", got: " << TODTYPE(val);
        }

        // Check trainable params grads
        const auto paramGrad = work.getTrainableParameters<MemoryManagerFP16>();
        
        const auto biases_ih_grad = paramGrad[(useFusion ? 2 : 0)].Gradient;
        EXPECT_EQ(biases_ih_grad_golden.size(), biases_ih_grad.size());
        for (size_t i = 0; i < biases_ih_grad.size(); ++i)
        {
            const auto val = biases_ih_grad[i];
            const auto golden_val = biases_ih_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << TODTYPE(golden_val) << ", got: " << TODTYPE(val);
        }

        const auto weights_ih_grad = paramGrad[(useFusion ? 3: 1)].Gradient;
        EXPECT_EQ(weights_ih_grad_golden.size(), weights_ih_grad.size());
        for (size_t i = 0; i < weights_ih_grad.size(); ++i)
        {
            const auto val = weights_ih_grad[i];
            const auto golden_val = weights_ih_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << TODTYPE(golden_val) << ", got: " << TODTYPE(val);
        }
    }
}
#endif // ANDROID

} // UT namespace
