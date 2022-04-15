// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <chrono>
#include <cstdio>
#include <tests/tools/TestTools.h>
#include <training/initializers/RandomUniformInitializer.h>
#include <training/layers/composite/rnn/LSTMCellLayer.h>
#include <training/network/Layers.h>
#include <tests/tools/TestTools.h>
///@todo(ck): For what?
#include <tests/topologies/TacotronTestTools.h>

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
    /// @todo(ck): fix one output tensor in slicer layer (bug AEE-209)
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

    /// @todo(ck): do we really need 2 steps to assign values?
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

    /// @todo(ck): do we really need 2 steps to assign values?
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

    /// @todo(ck): do we really need 2 steps to assign values?
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

    /// @todo(ck): do we really need 2 steps to assign values?
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

    /// @todo(ck): do we really need 2 steps to assign values?
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

    /// @todo(ck): do we really need 2 steps to assign values?
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
        // See https://www.wolframalpha.com/input/?i=confidence+99.999%25
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
        // See https://www.wolframalpha.com/input/?i=confidence+99.999%25
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

    // d.polubotko(TODO): add when ZoneoutLayer test is ready
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

    /// @todo(ck): do we really need 2 steps to assign values?
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

    /// @todo(ck): do we really need 2 steps to assign values?
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

    /// @todo(ck): do we really need 2 steps to assign values?
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

    /// @todo(ck): do we really need 2 steps to assign values?
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

TEST(TestLSTMCell, SingleParamsLoadedFromFilesWithBiasForgetForwardUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps = 1e-4_dt;
    const auto batch_size = 8U;
    const auto input_size = 896U;
    const auto hidden_size = 512U;
    const auto root = tools::getTestAssetsDir() / "lstm" / "vanilla";

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

    ASSERT_TRUE(loadTFData(root / "vanilla_lstm_inputs.data", memory_manager["in"]));
    ASSERT_TRUE(loadTFData(root / "vanilla_lstm_h0.data", memory_manager["hidden"]));
    ASSERT_TRUE(loadTFData(root / "vanilla_lstm_c0.data", memory_manager["cell"]));

    auto& bias = memory_manager[Name("lstm_cell") / "linear" / "Biases"];
    auto& weights = memory_manager[Name("lstm_cell") / "linear" / "Weights"];
    ASSERT_TRUE(loadTFData(root / "vanilla_lstm_weight1.data", bias));
    ASSERT_TRUE(loadTFData(root / "vanilla_lstm_weight0.data", weights));

    // Apply
    work.forwardPassTesting();

    // Checks
    memory_manager.createTensor("hidden_golden", batch_size, 1, 1, hidden_size);
    memory_manager.createTensor("cell_golden", batch_size, 1, 1, hidden_size);

    ASSERT_TRUE(loadTFData(root / "vanilla_lstm_h.data", memory_manager["hidden_golden"]));
    ASSERT_TRUE(loadTFData(root / "vanilla_lstm_c.data", memory_manager["cell_golden"]));

    const auto& hidden_input_tensor = memory_manager["hidden"];
    const auto& cell_input_tensor = memory_manager["cell"];

    const auto& hidden_new_tensor = memory_manager["new_hidden"];
    const auto& cell_new_tensor = memory_manager["new_cell"];

    EXPECT_EQ(hidden_input_tensor.size(), hidden_new_tensor.size());
    EXPECT_EQ(cell_input_tensor.size(), cell_new_tensor.size());

    const auto& hidden_golden = memory_manager["hidden_golden"];
    const auto& cell_golden = memory_manager["cell_golden"];

    for (size_t i = 0; i < cell_new_tensor.size(); ++i)
    {
        const auto cell_val = cell_new_tensor[i];
        const auto golden_cell_val = cell_golden[i];
        EXPECT_NEAR(cell_val, golden_cell_val, eps);
    }

    for (size_t i = 0; i < hidden_input_tensor.size(); ++i)
    {
        const auto hidden_val = hidden_new_tensor[i];
        const auto golden_hidden_val = hidden_golden[i];
        EXPECT_NEAR(hidden_val, golden_hidden_val, eps);
    }
}

#ifdef ANDROID
TEST(TestLSTMCell, SimpleForwardFP16Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto batch_size = 2U;

    // Initialization
    raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16 };
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", DataParams{ { "in" }, 1, 1, input_size });
    work.add<raul::DataLayer>("data2", DataParams{ { "hidden" }, 1, 1, hidden_size });
    work.add<raul::DataLayer>("data3", DataParams{ { "cell" }, 1, 1, hidden_size });

    // Network
    const auto params = raul::LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {} };
    raul::LSTMCellLayer("lstm_cell", params, networkParameters);
    TENSORS_CREATE(batch_size)

    EXPECT_NO_THROW(work.forwardPassTesting());
}
#endif // ANDROID

#ifdef ANDROID
TEST(TestLSTMCell, SimpleBackwardFP16Unit)
{
    PROFILE_TEST
    using namespace raul;

    // Test parameters
    const auto eps_rel = 1e-1_hf;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto batch_size = 2U;

    // Initialization
    raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16 };
    auto& memory_manager = work.getMemoryManager<raul::MemoryManagerFP16>();
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", DataParams{ { "in" }, 1, 1, input_size });
    work.add<raul::DataLayer>("data2", DataParams{ { "hidden" }, 1, 1, hidden_size });
    work.add<raul::DataLayer>("data3", DataParams{ { "cell" }, 1, 1, hidden_size });

    // Network
    const auto params = LSTMCellParams{ "in", "hidden", "cell", "new_hidden", "new_cell", {} };
    LSTMCellLayer("lstm_cell", params, networkParameters);
    TENSORS_CREATE(batch_size)

    /// @todo(ck): do we really need 2 steps to assign values?
    const Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    const Tensor hidden_init{ 1.892910e+00_dt, 3.111044e+00_dt, -4.583958e-01_dt, -3.359881e-01_dt, -1.569986e+00_dt, 1.231500e+00_dt };
    const Tensor cell_init{ 1.394632e+00_dt, 1.171102e+00_dt, 4.335119e-01_dt, -1.734250e+00_dt, -1.336049e+00_dt, 8.870960e-01_dt };

    ///@todo(ck): implement in-tensor intertype conversion
    {
        auto& tensor = memory_manager["in"];
        for (size_t i = 0; i < input_init.size(); ++i)
        {
            tensor[i] = TOHTYPE(input_init[i]);
        }
    }

    {
        auto& tensor = memory_manager["hidden"];
        for (size_t i = 0; i < hidden_init.size(); ++i)
        {
            tensor[i] = TOHTYPE(hidden_init[i]);
        }
    }

    {
        auto& tensor = memory_manager["cell"];
        for (size_t i = 0; i < cell_init.size(); ++i)
        {
            tensor[i] = TOHTYPE(cell_init[i]);
        }
    }

    for (auto& [param, grad] : work.getTrainableParameters<MemoryManagerFP16>())
    {
        param = 1.0_hf;
    }

    // Apply
    work.forwardPassTraining();

    memory_manager[Name("new_hidden").grad()] = 1.0_hf;
    memory_manager[Name("new_cell").grad()] = 1.0_hf;

    work.backwardPassTraining();

    // Checks
    const TensorFP16 inputs_grad_golden{ 2.458567e-01_hf, 2.458567e-01_hf, 2.458567e-01_hf, 2.458567e-01_hf, 1.941207e-01_hf, 1.941207e-01_hf, 1.941207e-01_hf, 1.941207e-01_hf };

    const auto& inputs_grad = memory_manager[Name("in").grad()];
    const auto& inputs = memory_manager["in"];

    EXPECT_EQ(inputs.size(), inputs_grad.size());

    for (size_t i = 0; i < inputs_grad.size(); ++i)
    {
        const auto val = inputs_grad[i];
        const auto golden_val = inputs_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << TODTYPE(golden_val) << ", got: " << TODTYPE(val);
    }
}
#endif // ANDROID

} // UT namespace
