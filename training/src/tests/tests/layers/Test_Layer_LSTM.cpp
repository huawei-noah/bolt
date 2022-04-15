// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/GTestExtensions.h>
#include <tests/tools/TestTools.h>

#include <training/compiler/Layers.h>
#include <training/compiler/Compiler.h>
#include <training/base/optimizers/SGD.h>
#include <training/base/layers/composite/rnn/LSTMLayer.h>

namespace
{

auto buildLSTMOnPrimitives(const size_t batch_size, const size_t sequence_length, const size_t input_size, const size_t hidden_size)
{
    size_t parts = 4;
    size_t cnt = 0;
    auto work = std::make_unique<raul::WorkflowEager>();
    raul::Names input_names(sequence_length);
    raul::Names output_names(sequence_length);
    std::generate_n(input_names.begin(), sequence_length, [cnt]() mutable { return "in[" + Conversions::toString(cnt++) + "]"; });
    std::generate_n(output_names.begin(), sequence_length, [cnt]() mutable { return "hidden_state[" + Conversions::toString(cnt++) + "]"; });
    work->add<raul::DataLayer>("fake_data_in", raul::DataParams{ { "in", "labels" }, sequence_length, 1, input_size, 0 });
    work->add<raul::DataLayer>("fake_data_state", raul::DataParams{ { "hidden", "cell" }, 1, 1, hidden_size, hidden_size });
    work->add<raul::TensorLayer>("weight_ih", raul::TensorParams{ { "weights_ih" }, 1, 1, hidden_size * parts, input_size });
    work->add<raul::TensorLayer>("weight_hh",
                                 raul::TensorParams{
                                     { "weights_hh" },
                                     1,
                                     1,
                                     hidden_size * parts,
                                     hidden_size,
                                 });
    work->add<raul::TensorLayer>("biases_ih", raul::TensorParams{ { "biases_hh" }, 1, 1, 1, hidden_size * parts });
    work->add<raul::TensorLayer>("biases_hh", raul::TensorParams{ { "biases_ih" }, 1, 1, 1, hidden_size * parts });
    work->add<raul::SlicerLayer>("slice", raul::SlicingParams("in", input_names, "depth"));

    std::string name_hidden_in = "hidden";
    std::string name_cell_in = "cell";
    for (size_t i = 0; i < sequence_length; ++i)
    {
        const auto idx = Conversions::toString(i);
        std::string name_hidden_out = "hidden_state[" + idx + "]";
        std::string name_cell_out = "cell_state[" + idx + "]";
        raul::Name cellName = i == 0 ? "cell" : "unrolled_cell" + idx;
        raul::LSTMCellLayer(cellName, raul::LSTMCellParams(input_names[i], name_hidden_in, name_cell_in, name_hidden_out, name_cell_out, {}), work->getNetworkParameters());
        name_hidden_in = name_hidden_out;
        name_cell_in = name_cell_out;
    }

    work->add<raul::ConcatenationLayer>("concat", raul::BasicParamsWithDim(output_names, { "out" }, "depth"));

    work->preparePipelines();
    work->setBatchSize(batch_size);
    work->prepareMemoryForTraining();

    return work;
}

}

namespace UT
{

using namespace std;

TEST(TestLSTM, PrimitiveBlocksBuildUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto sequence_length = 1U;
    const auto batch_size = 2U;

    // Network
    auto work = buildLSTMOnPrimitives(batch_size, sequence_length, input_size, hidden_size);
    //    work->printInfo(std::cout);
}

TEST(TestLSTM, PrimitiveBlocksForwardSeq1Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto sequence_length = 1U;
    const auto batch_size = 2U;

    // Network
    auto work = buildLSTMOnPrimitives(batch_size, sequence_length, input_size, hidden_size);

    // Initialization
    auto& memory_manager = work->getMemoryManager();

    for (auto& [param, grad] : work->getTrainableParameters())
    {
        param = 1.0_dt;
    }

    const raul::Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    memory_manager["in"] = TORANGE(input_init);

    // Apply
    work->forwardPassTesting();

    // Checks
    const raul::Tensor output_golden{ -5.799451e-02_dt, -5.799451e-02_dt, -5.799451e-02_dt, 7.003791e-01_dt, 7.003791e-01_dt, 7.003791e-01_dt };
    const auto& outputTensor = memory_manager["out"];

    EXPECT_EQ(outputTensor.size(), output_golden.size());

    for (size_t i = 0; i < outputTensor.size(); ++i)
    {
        const auto val = outputTensor[i];
        const auto golden_val = output_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

TEST(TestLSTM, PrimitiveBlocksForwardSeq2Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto sequence_length = 2U;
    const auto batch_size = 2U;

    // Network
    auto work = buildLSTMOnPrimitives(batch_size, sequence_length, input_size, hidden_size);

    // Initialization
    auto& memory_manager = work->getMemoryManager();

    for (auto& [param, grad] : work->getTrainableParameters())
    {
        param = 1.0_dt;
    }

    const raul::Tensor input_init{ -8.024929e-01_dt, -1.295186e+00_dt, -7.501815e-01_dt, -1.311966e+00_dt, -2.188337e-01_dt, -2.435065e+00_dt, -7.291476e-02_dt, -3.398641e-02_dt,
                                   7.968872e-01_dt,  -1.848416e-01_dt, -3.701473e-01_dt, -1.210281e+00_dt, -6.226985e-01_dt, -4.637222e-01_dt, 1.921782e+00_dt,  -4.025455e-01_dt };
    memory_manager["in"] = TORANGE(input_init);

    // Apply
    work->forwardPassTesting();

    // Checks
    const raul::Tensor output_golden{ -1.037908e-02_dt, -1.037908e-02_dt, -1.037908e-02_dt, -7.253138e-02_dt, -7.253138e-02_dt, -7.253138e-02_dt,
                                      3.804927e-01_dt,  3.804927e-01_dt,  3.804927e-01_dt,  8.850382e-01_dt,  8.850382e-01_dt,  8.850382e-01_dt };
    const auto& outputTensor = memory_manager["out"];

    EXPECT_EQ(outputTensor.size(), output_golden.size());

    for (size_t i = 0; i < outputTensor.size(); ++i)
    {
        const auto val = outputTensor[i];
        const auto golden_val = output_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

TEST(TestLSTM, PrimitiveBlocksForwardSeq5Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto sequence_length = 5U;
    const auto batch_size = 2U;

    // Network
    auto work = buildLSTMOnPrimitives(batch_size, sequence_length, input_size, hidden_size);

    // Initialization
    auto& memory_manager = work->getMemoryManager();

    for (auto& [param, grad] : work->getTrainableParameters())
    {
        param = 1.0_dt;
    }

    const raul::Tensor input_init{ -8.024929e-01_dt, -1.295186e+00_dt, -7.501815e-01_dt, -1.311966e+00_dt, -2.188337e-01_dt, -2.435065e+00_dt, -7.291476e-02_dt, -3.398641e-02_dt,
                                   7.968872e-01_dt,  -1.848416e-01_dt, -3.701473e-01_dt, -1.210281e+00_dt, -6.226985e-01_dt, -4.637222e-01_dt, 1.921782e+00_dt,  -4.025455e-01_dt,
                                   9.295023e-02_dt,  -6.660997e-01_dt, 6.080472e-01_dt,  -7.300199e-01_dt, -8.833758e-01_dt, -4.189135e-01_dt, -8.048265e-01_dt, 5.656096e-01_dt,
                                   2.885762e-01_dt,  3.865978e-01_dt,  -2.010639e-01_dt, -1.179270e-01_dt, -8.293669e-01_dt, -1.407257e+00_dt, 1.626847e+00_dt,  1.722732e-01_dt,
                                   -7.042940e-01_dt, 3.147210e-01_dt,  1.573929e-01_dt,  3.853627e-01_dt,  5.736546e-01_dt,  9.979313e-01_dt,  5.436094e-01_dt,  7.880439e-02_dt };
    memory_manager["in"] = TORANGE(input_init);

    // Apply
    work->forwardPassTesting();

    // Checks
    const raul::Tensor output_golden{ -1.037908e-02_dt, -1.037908e-02_dt, -1.037908e-02_dt, -7.253138e-02_dt, -7.253138e-02_dt, -7.253138e-02_dt, 2.026980e-01_dt, 2.026980e-01_dt,
                                      2.026980e-01_dt,  8.062391e-01_dt,  8.062391e-01_dt,  8.062391e-01_dt,  9.519626e-01_dt,  9.519626e-01_dt,  9.519626e-01_dt, 1.573658e-01_dt,
                                      1.573658e-01_dt,  1.573658e-01_dt,  7.829522e-01_dt,  7.829522e-01_dt,  7.829522e-01_dt,  9.537140e-01_dt,  9.537140e-01_dt, 9.537140e-01_dt,
                                      9.895445e-01_dt,  9.895445e-01_dt,  9.895445e-01_dt,  9.986963e-01_dt,  9.986963e-01_dt,  9.986963e-01_dt };
    const auto& outputTensor = memory_manager["out"];

    EXPECT_EQ(outputTensor.size(), output_golden.size());

    for (size_t i = 0; i < outputTensor.size(); ++i)
    {
        const auto val = outputTensor[i];
        const auto golden_val = output_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

TEST(TestLSTM, PrimitiveBlocksBackwardSeq1Unit)
{
    PROFILE_TEST
    using namespace raul;

    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto sequence_length = 1U;
    const auto batch_size = 2U;

    // Network
    auto work = buildLSTMOnPrimitives(batch_size, sequence_length, input_size, hidden_size);

    // Initialization
    auto& memory_manager = work->getMemoryManager();

    for (auto& [param, grad] : work->getTrainableParameters())
    {
        param = 1.0_dt;
    }

    const Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    memory_manager["in"] = TORANGE(input_init);

    memory_manager[Name("out").grad()] = 1.0_dt;

    // Apply
    work->forwardPassTraining();
    work->backwardPassTraining();

    // Checks
    const Tensor inputs_grad_golden{ -1.359324e-01_dt, -1.359324e-01_dt, -1.359324e-01_dt, -1.359324e-01_dt, 1.805458e-01_dt, 1.805458e-01_dt, 1.805458e-01_dt, 1.805458e-01_dt };

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

TEST(TestLSTM, PrimitiveBlocksBackwardSeq2Unit)
{
    PROFILE_TEST
    using namespace raul;

    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto sequence_length = 2U;
    const auto batch_size = 2U;

    // Network
    auto work = buildLSTMOnPrimitives(batch_size, sequence_length, input_size, hidden_size);

    // Initialization
    auto& memory_manager = work->getMemoryManager();

    for (auto& [param, grad] : work->getTrainableParameters())
    {
        param = 1.0_dt;
    }

    const Tensor input_init{ -8.024929e-01_dt, -1.295186e+00_dt, -7.501815e-01_dt, -1.311966e+00_dt, -2.188337e-01_dt, -2.435065e+00_dt, -7.291476e-02_dt, -3.398641e-02_dt,
                             7.968872e-01_dt,  -1.848416e-01_dt, -3.701473e-01_dt, -1.210281e+00_dt, -6.226985e-01_dt, -4.637222e-01_dt, 1.921782e+00_dt,  -4.025455e-01_dt };
    memory_manager["in"] = TORANGE(input_init);

    memory_manager[Name("out").grad()] = 1.0_dt;

    // Apply
    work->forwardPassTraining();
    work->backwardPassTraining();

    // Checks
    const Tensor inputs_grad_golden{ -6.995806e-02_dt, -6.995806e-02_dt, -6.995806e-02_dt, -6.995806e-02_dt, -1.382187e-01_dt, -1.382187e-01_dt, -1.382187e-01_dt, -1.382187e-01_dt,
                                     1.336384e+00_dt,  1.336384e+00_dt,  1.336384e+00_dt,  1.336384e+00_dt,  9.485833e-02_dt,  9.485833e-02_dt,  9.485833e-02_dt,  9.485833e-02_dt };

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

TEST(TestLSTM, PrimitiveBlocksBackwardSeq5Unit)
{
    PROFILE_TEST
    using namespace raul;

    // Test parameters
    const auto eps_rel = 1e-5_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto sequence_length = 5U;
    const auto batch_size = 2U;

    // Network
    auto work = buildLSTMOnPrimitives(batch_size, sequence_length, input_size, hidden_size);

    // Initialization
    auto& memory_manager = work->getMemoryManager();

    for (auto& [param, grad] : work->getTrainableParameters())
    {
        param = 1.0_dt;
    }

    const Tensor input_init{ -8.024929e-01_dt, -1.295186e+00_dt, -7.501815e-01_dt, -1.311966e+00_dt, -2.188337e-01_dt, -2.435065e+00_dt, -7.291476e-02_dt, -3.398641e-02_dt,
                             7.968872e-01_dt,  -1.848416e-01_dt, -3.701473e-01_dt, -1.210281e+00_dt, -6.226985e-01_dt, -4.637222e-01_dt, 1.921782e+00_dt,  -4.025455e-01_dt,
                             9.295023e-02_dt,  -6.660997e-01_dt, 6.080472e-01_dt,  -7.300199e-01_dt, -8.833758e-01_dt, -4.189135e-01_dt, -8.048265e-01_dt, 5.656096e-01_dt,
                             2.885762e-01_dt,  3.865978e-01_dt,  -2.010639e-01_dt, -1.179270e-01_dt, -8.293669e-01_dt, -1.407257e+00_dt, 1.626847e+00_dt,  1.722732e-01_dt,
                             -7.042940e-01_dt, 3.147210e-01_dt,  1.573929e-01_dt,  3.853627e-01_dt,  5.736546e-01_dt,  9.979313e-01_dt,  5.436094e-01_dt,  7.880439e-02_dt };
    memory_manager["in"] = TORANGE(input_init);

    memory_manager[Name("out").grad()] = 1.0_dt;

    // Apply
    work->forwardPassTraining();
    work->backwardPassTraining();

    // Checks
    const Tensor inputs_grad_golden{ -1.586803e-01_dt, -1.586803e-01_dt, -1.586803e-01_dt, -1.586803e-01_dt, -3.506274e-01_dt, -3.506274e-01_dt, -3.506274e-01_dt, -3.506274e-01_dt,
                                     1.747969e+00_dt,  1.747969e+00_dt,  1.747969e+00_dt,  1.747969e+00_dt,  1.852361e-01_dt,  1.852361e-01_dt,  1.852361e-01_dt,  1.852361e-01_dt,
                                     7.512633e-02_dt,  7.512633e-02_dt,  7.512633e-02_dt,  7.512633e-02_dt,  2.134141e+00_dt,  2.134141e+00_dt,  2.134141e+00_dt,  2.134141e+00_dt,
                                     2.289679e-01_dt,  2.289679e-01_dt,  2.289679e-01_dt,  2.289679e-01_dt,  6.543004e-02_dt,  6.543004e-02_dt,  6.543004e-02_dt,  6.543004e-02_dt,
                                     2.018033e-02_dt,  2.018033e-02_dt,  2.018033e-02_dt,  2.018033e-02_dt,  2.330255e-03_dt,  2.330255e-03_dt,  2.330255e-03_dt,  2.330255e-03_dt };

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

TEST(TestLSTM, SimpleForwardSeq1Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-4_dt;
    const size_t input_size = 4U;
    const size_t hidden_size = 3U;
    const size_t sequence_length = 1U;
    const size_t batch_size = 2U;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, sequence_length, 1, input_size });

    // Network
    const auto params = raul::LSTMParams{ { "in" }, { "out" }, hidden_size, false };
    raul::LSTMLayer("lstm", params, networkParameters);
    TENSORS_CREATE(batch_size)

    const raul::Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    memory_manager["in"] = TORANGE(input_init);

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Apply
    work.forwardPassTesting();

    // Checks
    const raul::Tensor output_golden{ -5.799451e-02_dt, -5.799451e-02_dt, -5.799451e-02_dt, 7.003791e-01_dt, 7.003791e-01_dt, 7.003791e-01_dt };
    const raul::Tensor cell_golden{ -2.068135e-01_dt, -2.068135e-01_dt, -2.068135e-01_dt, 9.446084e-01_dt, 9.446084e-01_dt, 9.446084e-01_dt };
    const auto& outputTensor = memory_manager["out"];
    const auto& cellTensor = memory_manager["lstm::cell_state[" + Conversions::toString(sequence_length - 1) + "]"];

    EXPECT_EQ(outputTensor.size(), batch_size * hidden_size * sequence_length);
    EXPECT_EQ(cellTensor.size(), batch_size * hidden_size);

    for (size_t i = 0; i < outputTensor.size(); ++i)
    {
        const auto val = outputTensor[i];
        const auto golden_val = output_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < cellTensor.size(); ++i)
    {
        const auto val = cellTensor[i];
        const auto golden_val = cell_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

TEST(TestLSTM, SimpleForwardSeq2Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-4_dt;
    const size_t input_size = 4U;
    const size_t hidden_size = 3U;
    const size_t sequence_length = 2U;
    const size_t batch_size = 2U;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, sequence_length, 1, input_size });

    // Network
    const auto params = raul::LSTMParams{ { "in" }, { "out" }, hidden_size, false };
    raul::LSTMLayer("lstm", params, networkParameters);
    TENSORS_CREATE(batch_size)

    const raul::Tensor input_init{ -8.024929e-01_dt, -1.295186e+00_dt, -7.501815e-01_dt, -1.311966e+00_dt, -2.188337e-01_dt, -2.435065e+00_dt, -7.291476e-02_dt, -3.398641e-02_dt,
                                   7.968872e-01_dt,  -1.848416e-01_dt, -3.701473e-01_dt, -1.210281e+00_dt, -6.226985e-01_dt, -4.637222e-01_dt, 1.921782e+00_dt,  -4.025455e-01_dt };
    memory_manager["in"] = TORANGE(input_init);

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Apply
    work.forwardPassTesting();

    // Checks
    const raul::Tensor output_golden{ -1.037908e-02_dt, -1.037908e-02_dt, -1.037908e-02_dt, -7.253138e-02_dt, -7.253138e-02_dt, -7.253138e-02_dt,
                                      3.804927e-01_dt,  3.804927e-01_dt,  3.804927e-01_dt,  8.850382e-01_dt,  8.850382e-01_dt,  8.850382e-01_dt };
    const raul::Tensor cell_golden{ -2.369964e-01_dt, -2.369964e-01_dt, -2.369964e-01_dt, 1.526655e+00_dt, 1.526655e+00_dt, 1.526655e+00_dt };
    const auto& outputTensor = memory_manager["out"];
    const auto& cellTensor = memory_manager["lstm::cell_state[" + Conversions::toString(sequence_length - 1) + "]"];

    EXPECT_EQ(outputTensor.size(), batch_size * hidden_size * sequence_length);
    EXPECT_EQ(cellTensor.size(), batch_size * hidden_size);

    for (size_t i = 0; i < outputTensor.size(); ++i)
    {
        const auto val = outputTensor[i];
        const auto golden_val = output_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < cellTensor.size(); ++i)
    {
        const auto val = cellTensor[i];
        const auto golden_val = cell_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

TEST(TestLSTM, SimpleForwardSeq5Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-4_dt;
    const size_t input_size = 4U;
    const size_t hidden_size = 3U;
    const size_t sequence_length = 5U;
    const size_t batch_size = 2U;

    // Use fusion or not
    for (size_t q = 0; q < 2; ++q)
    {
        // Initialization
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, sequence_length, 1, input_size });

        // Network
        bool useFusion = (q == 1 ? true : false);
        const auto params = raul::LSTMParams{ { "in" }, { "out" }, hidden_size, false, true, false, false, 0.0_dt, false, 0.0_dt, useFusion };
        raul::LSTMLayer("lstm", params, networkParameters);
        TENSORS_CREATE(batch_size)

        const raul::Tensor input_init{ -8.024929e-01_dt, -1.295186e+00_dt, -7.501815e-01_dt, -1.311966e+00_dt, -2.188337e-01_dt, -2.435065e+00_dt, -7.291476e-02_dt, -3.398641e-02_dt,
                                    7.968872e-01_dt,  -1.848416e-01_dt, -3.701473e-01_dt, -1.210281e+00_dt, -6.226985e-01_dt, -4.637222e-01_dt, 1.921782e+00_dt,  -4.025455e-01_dt,
                                    9.295023e-02_dt,  -6.660997e-01_dt, 6.080472e-01_dt,  -7.300199e-01_dt, -8.833758e-01_dt, -4.189135e-01_dt, -8.048265e-01_dt, 5.656096e-01_dt,
                                    2.885762e-01_dt,  3.865978e-01_dt,  -2.010639e-01_dt, -1.179270e-01_dt, -8.293669e-01_dt, -1.407257e+00_dt, 1.626847e+00_dt,  1.722732e-01_dt,
                                    -7.042940e-01_dt, 3.147210e-01_dt,  1.573929e-01_dt,  3.853627e-01_dt,  5.736546e-01_dt,  9.979313e-01_dt,  5.436094e-01_dt,  7.880439e-02_dt };
        memory_manager["in"] = TORANGE(input_init);

        for (auto& [param, grad] : work.getTrainableParameters())
        {
            param = 1.0_dt;
        }

        // Apply
        work.forwardPassTesting();

        // Checks
        const raul::Tensor output_golden{ -1.037908e-02_dt, -1.037908e-02_dt, -1.037908e-02_dt, -7.253138e-02_dt, -7.253138e-02_dt, -7.253138e-02_dt, 2.026980e-01_dt, 2.026980e-01_dt,
                                        2.026980e-01_dt,  8.062391e-01_dt,  8.062391e-01_dt,  8.062391e-01_dt,  9.519626e-01_dt,  9.519626e-01_dt,  9.519626e-01_dt, 1.573658e-01_dt,
                                        1.573658e-01_dt,  1.573658e-01_dt,  7.829522e-01_dt,  7.829522e-01_dt,  7.829522e-01_dt,  9.537140e-01_dt,  9.537140e-01_dt, 9.537140e-01_dt,
                                        9.895445e-01_dt,  9.895445e-01_dt,  9.895445e-01_dt,  9.986963e-01_dt,  9.986963e-01_dt,  9.986963e-01_dt };
        const raul::Tensor cell_golden{ 2.183707e+00_dt, 2.183707e+00_dt, 2.183707e+00_dt, 4.118004e+00_dt, 4.118004e+00_dt, 4.118004e+00_dt };
        const auto& outputTensor = memory_manager["out"];
        const auto& cellTensor = memory_manager["lstm::cell_state[" + Conversions::toString(sequence_length - 1) + "]"];

        EXPECT_EQ(outputTensor.size(), batch_size * hidden_size * sequence_length);
        EXPECT_EQ(cellTensor.size(), batch_size * hidden_size);

        for (size_t i = 0; i < outputTensor.size(); ++i)
        {
            const auto val = outputTensor[i];
            const auto golden_val = output_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        for (size_t i = 0; i < cellTensor.size(); ++i)
        {
            const auto val = cellTensor[i];
            const auto golden_val = cell_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }
    }
}

TEST(TestLSTM, SimpleBackwardSeq1Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-4_dt;
    const size_t input_size = 4U;
    const size_t hidden_size = 3U;
    const size_t sequence_length = 1U;
    const size_t batch_size = 2U;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, sequence_length, 1, input_size });

    // Network
    const auto params = raul::LSTMParams{ { "in" }, { "out" }, hidden_size, false };
    raul::LSTMLayer("lstm", params, networkParameters);
    TENSORS_CREATE(batch_size)

    const raul::Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    memory_manager["in"] = TORANGE(input_init);

    memory_manager[raul::Name("out").grad()] = 1.0_dt;

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Apply
    work.forwardPassTraining();

    work.backwardPassTraining();

    // Checks
    const raul::Tensor inputs_grad_golden{ -1.359324e-01_dt, -1.359324e-01_dt, -1.359324e-01_dt, -1.359324e-01_dt, 1.805458e-01_dt, 1.805458e-01_dt, 1.805458e-01_dt, 1.805458e-01_dt };

    const auto& inputs_grad = memory_manager[raul::Name("in").grad()];
    const auto& inputs = memory_manager["in"];

    EXPECT_EQ(inputs.size(), inputs_grad.size());

    for (size_t i = 0; i < inputs_grad.size(); ++i)
    {
        const auto val = inputs_grad[i];
        const auto golden_val = inputs_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    // Check shared parameters grads
    const raul::Tensor weight_ih_grad_golden{ 2.008261e-02_dt, 3.833952e-02_dt,  5.488532e-02_dt,  2.392589e-02_dt,  2.008261e-02_dt, 3.833952e-02_dt,  5.488532e-02_dt,  2.392589e-02_dt,
                                              2.008261e-02_dt, 3.833952e-02_dt,  5.488532e-02_dt,  2.392589e-02_dt,  0.000000e+00_dt, 0.000000e+00_dt,  0.000000e+00_dt,  0.000000e+00_dt,
                                              0.000000e+00_dt, 0.000000e+00_dt,  0.000000e+00_dt,  0.000000e+00_dt,  0.000000e+00_dt, 0.000000e+00_dt,  0.000000e+00_dt,  0.000000e+00_dt,
                                              1.243622e-02_dt, -3.385932e-02_dt, -5.629471e-02_dt, -2.475417e-02_dt, 1.243622e-02_dt, -3.385932e-02_dt, -5.629471e-02_dt, -2.475417e-02_dt,
                                              1.243622e-02_dt, -3.385932e-02_dt, -5.629471e-02_dt, -2.475417e-02_dt, 3.917178e-02_dt, 3.996138e-02_dt,  5.247791e-02_dt,  2.274714e-02_dt,
                                              3.917178e-02_dt, 3.996138e-02_dt,  5.247791e-02_dt,  2.274714e-02_dt,  3.917178e-02_dt, 3.996138e-02_dt,  5.247791e-02_dt,  2.274714e-02_dt };
    const raul::Tensor bias_grad_golden{ -1.981921e-02_dt, -1.981921e-02_dt, -1.981921e-02_dt, 0.000000e+00_dt,  0.000000e+00_dt,  0.000000e+00_dt,
                                         4.108956e-02_dt,  4.108956e-02_dt,  4.108956e-02_dt,  -6.399199e-03_dt, -6.399199e-03_dt, -6.399199e-03_dt };

    const auto paramGrad = work.getTrainableParameters();
    EXPECT_EQ(paramGrad.size(), 4u);

    const auto gradBiasesIH = paramGrad[0].Gradient;
    const auto gradWeightsIH = paramGrad[1].Gradient;
    const auto gradBiasesHH = paramGrad[2].Gradient;
    const auto gradWeightsHH = paramGrad[3].Gradient;

    const auto parts = 4;
    EXPECT_EQ(gradWeightsIH.size(), weight_ih_grad_golden.size());
    EXPECT_EQ(gradWeightsHH.size(), hidden_size * hidden_size * parts);
    EXPECT_EQ(gradBiasesIH.size(), bias_grad_golden.size());
    EXPECT_EQ(gradBiasesHH.size(), bias_grad_golden.size());

    for (size_t i = 0; i < gradWeightsIH.size(); ++i)
    {
        const auto val = gradWeightsIH[i];
        const auto golden_val = weight_ih_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < gradWeightsHH.size(); ++i)
    {
        const auto val = gradWeightsHH[i];
        const auto golden_val = 0.0_dt;
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < gradBiasesIH.size(); ++i)
    {
        const auto val = gradBiasesIH[i];
        const auto golden_val = bias_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < gradBiasesHH.size(); ++i)
    {
        const auto val = gradBiasesHH[i];
        const auto golden_val = bias_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

TEST(TestLSTM, SimpleBackwardSeq2Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-4_dt;
    const size_t input_size = 4U;
    const size_t hidden_size = 3U;
    const size_t sequence_length = 2U;
    const size_t batch_size = 2U;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, sequence_length, 1, input_size });

    // Network
    const auto params = raul::LSTMParams{ { "in" }, { "out" }, hidden_size, false };
    raul::LSTMLayer("lstm", params, networkParameters);
    TENSORS_CREATE(batch_size)

    const raul::Tensor input_init{ -8.024929e-01_dt, -1.295186e+00_dt, -7.501815e-01_dt, -1.311966e+00_dt, -2.188337e-01_dt, -2.435065e+00_dt, -7.291476e-02_dt, -3.398641e-02_dt,
                                   7.968872e-01_dt,  -1.848416e-01_dt, -3.701473e-01_dt, -1.210281e+00_dt, -6.226985e-01_dt, -4.637222e-01_dt, 1.921782e+00_dt,  -4.025455e-01_dt };
    memory_manager["in"] = TORANGE(input_init);

    memory_manager[raul::Name("out").grad()] = 1.0_dt;

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Apply
    work.forwardPassTraining();

    work.backwardPassTraining();

    // Checks
    const raul::Tensor inputs_grad_golden{ -6.995806e-02_dt, -6.995806e-02_dt, -6.995806e-02_dt, -6.995806e-02_dt, -1.382187e-01_dt, -1.382187e-01_dt, -1.382187e-01_dt, -1.382187e-01_dt,
                                           1.336384e+00_dt,  1.336384e+00_dt,  1.336384e+00_dt,  1.336384e+00_dt,  9.485833e-02_dt,  9.485833e-02_dt,  9.485833e-02_dt,  9.485833e-02_dt };

    const auto& inputs_grad = memory_manager[raul::Name("in").grad()];
    const auto& inputs = memory_manager["in"];

    EXPECT_EQ(inputs.size(), inputs_grad.size());

    for (size_t i = 0; i < inputs_grad.size(); ++i)
    {
        const auto val = inputs_grad[i];
        const auto golden_val = inputs_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    // Check shared parameters grads
    const raul::Tensor weight_ih_grad_golden{ 1.097069e-01_dt,  9.967550e-02_dt,  -1.816753e-02_dt, -1.161487e-01_dt, 1.097069e-01_dt,  9.967550e-02_dt,  -1.816753e-02_dt, -1.161487e-01_dt,
                                              1.097069e-01_dt,  9.967550e-02_dt,  -1.816753e-02_dt, -1.161487e-01_dt, -1.854407e-04_dt, 1.433821e-02_dt,  5.339870e-03_dt,  -8.046657e-04_dt,
                                              -1.854407e-04_dt, 1.433821e-02_dt,  5.339870e-03_dt,  -8.046657e-04_dt, -1.854407e-04_dt, 1.433821e-02_dt,  5.339870e-03_dt,  -8.046657e-04_dt,
                                              1.650045e-01_dt,  -1.691406e-01_dt, -8.596025e-02_dt, -2.727654e-01_dt, 1.650045e-01_dt,  -1.691406e-01_dt, -8.596025e-02_dt, -2.727654e-01_dt,
                                              1.650045e-01_dt,  -1.691406e-01_dt, -8.596025e-02_dt, -2.727654e-01_dt, 8.956295e-02_dt,  1.005179e-01_dt,  1.552046e-02_dt,  -1.299831e-01_dt,
                                              8.956295e-02_dt,  1.005179e-01_dt,  1.552046e-02_dt,  -1.299831e-01_dt, 8.956295e-02_dt,  1.005179e-01_dt,  1.552046e-02_dt,  -1.299831e-01_dt };
    const raul::Tensor weight_hh_grad_golden{ 2.120828e-03_dt,  2.120828e-03_dt,  2.120828e-03_dt,  2.120828e-03_dt,  2.120828e-03_dt,  2.120828e-03_dt,  2.120828e-03_dt,  2.120828e-03_dt,
                                              2.120828e-03_dt,  1.031388e-03_dt,  1.031388e-03_dt,  1.031388e-03_dt,  1.031388e-03_dt,  1.031388e-03_dt,  1.031388e-03_dt,  1.031388e-03_dt,
                                              1.031388e-03_dt,  1.031388e-03_dt,  -3.445673e-04_dt, -3.445673e-04_dt, -3.445673e-04_dt, -3.445673e-04_dt, -3.445673e-04_dt, -3.445673e-04_dt,
                                              -3.445673e-04_dt, -3.445673e-04_dt, -3.445673e-04_dt, 9.701515e-03_dt,  9.701515e-03_dt,  9.701515e-03_dt,  9.701515e-03_dt,  9.701515e-03_dt,
                                              9.701515e-03_dt,  9.701515e-03_dt,  9.701515e-03_dt,  9.701515e-03_dt };
    const raul::Tensor bias_grad_golden{ 5.974016e-02_dt, 5.974016e-02_dt, 5.974016e-02_dt, -3.834467e-03_dt, -3.834467e-03_dt, -3.834467e-03_dt,
                                         2.761197e-01_dt, 2.761197e-01_dt, 2.761197e-01_dt, 7.566305e-02_dt,  7.566305e-02_dt,  7.566305e-02_dt };

    const auto paramGrad = work.getTrainableParameters();
    const auto gradBiasesIH = paramGrad[0].Gradient;
    const auto gradWeightsIH = paramGrad[1].Gradient;
    const auto gradBiasesHH = paramGrad[2].Gradient;
    const auto gradWeightsHH = paramGrad[3].Gradient;

    EXPECT_EQ(gradWeightsIH.size(), weight_ih_grad_golden.size());
    EXPECT_EQ(gradWeightsHH.size(), weight_hh_grad_golden.size());
    EXPECT_EQ(gradBiasesIH.size(), bias_grad_golden.size());
    EXPECT_EQ(gradBiasesHH.size(), bias_grad_golden.size());

    for (size_t i = 0; i < gradWeightsIH.size(); ++i)
    {
        const auto val = gradWeightsIH[i];
        const auto golden_val = weight_ih_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < gradWeightsHH.size(); ++i)
    {
        const auto val = gradWeightsHH[i];
        const auto golden_val = weight_hh_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < gradBiasesIH.size(); ++i)
    {
        const auto val = gradBiasesIH[i];
        const auto golden_val = bias_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }

    for (size_t i = 0; i < gradBiasesHH.size(); ++i)
    {
        const auto val = gradBiasesHH[i];
        const auto golden_val = bias_grad_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
    }
}

TEST(TestLSTM, SimpleBackwardSeq5Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-4_dt;
    const size_t input_size = 4U;
    const size_t hidden_size = 3U;
    const size_t sequence_length = 5U;
    const size_t batch_size = 2U;

    // Use fusion or not
    for (size_t q = 0; q < 2; ++q)
    {
        // Initialization
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, sequence_length, 1, input_size });

        // Network
        bool useFusion = (q == 1 ? true : false);
        const auto params = raul::LSTMParams{ { "in" }, { "out" }, hidden_size, false, true, false, false, 0.0_dt, false, 0.0_dt, useFusion };
        raul::LSTMLayer("lstm", params, networkParameters);
        TENSORS_CREATE(batch_size)

        const raul::Tensor input_init{ -8.024929e-01_dt, -1.295186e+00_dt, -7.501815e-01_dt, -1.311966e+00_dt, -2.188337e-01_dt, -2.435065e+00_dt, -7.291476e-02_dt, -3.398641e-02_dt,
                                    7.968872e-01_dt,  -1.848416e-01_dt, -3.701473e-01_dt, -1.210281e+00_dt, -6.226985e-01_dt, -4.637222e-01_dt, 1.921782e+00_dt,  -4.025455e-01_dt,
                                    9.295023e-02_dt,  -6.660997e-01_dt, 6.080472e-01_dt,  -7.300199e-01_dt, -8.833758e-01_dt, -4.189135e-01_dt, -8.048265e-01_dt, 5.656096e-01_dt,
                                    2.885762e-01_dt,  3.865978e-01_dt,  -2.010639e-01_dt, -1.179270e-01_dt, -8.293669e-01_dt, -1.407257e+00_dt, 1.626847e+00_dt,  1.722732e-01_dt,
                                    -7.042940e-01_dt, 3.147210e-01_dt,  1.573929e-01_dt,  3.853627e-01_dt,  5.736546e-01_dt,  9.979313e-01_dt,  5.436094e-01_dt,  7.880439e-02_dt };
        memory_manager["in"] = TORANGE(input_init);

        memory_manager[raul::Name("out").grad()] = 1.0_dt;

        for (auto& [param, grad] : work.getTrainableParameters())
        {
            param = 1.0_dt;
        }

        // Apply
        work.forwardPassTraining();

        work.backwardPassTraining();

        // Checks
        const raul::Tensor inputs_grad_golden{ -1.586803e-01_dt, -1.586803e-01_dt, -1.586803e-01_dt, -1.586803e-01_dt, -3.506274e-01_dt, -3.506274e-01_dt, -3.506274e-01_dt, -3.506274e-01_dt,
                                            1.747969e+00_dt,  1.747969e+00_dt,  1.747969e+00_dt,  1.747969e+00_dt,  1.852361e-01_dt,  1.852361e-01_dt,  1.852361e-01_dt,  1.852361e-01_dt,
                                            7.512633e-02_dt,  7.512633e-02_dt,  7.512633e-02_dt,  7.512633e-02_dt,  2.134141e+00_dt,  2.134141e+00_dt,  2.134141e+00_dt,  2.134141e+00_dt,
                                            2.289679e-01_dt,  2.289679e-01_dt,  2.289679e-01_dt,  2.289679e-01_dt,  6.543004e-02_dt,  6.543004e-02_dt,  6.543004e-02_dt,  6.543004e-02_dt,
                                            2.018033e-02_dt,  2.018033e-02_dt,  2.018033e-02_dt,  2.018033e-02_dt,  2.330255e-03_dt,  2.330255e-03_dt,  2.330255e-03_dt,  2.330255e-03_dt };

        const auto& inputs_grad = memory_manager[raul::Name("in").grad()];
        const auto& inputs = memory_manager["in"];

        EXPECT_EQ(inputs.size(), inputs_grad.size());

        for (size_t i = 0; i < inputs_grad.size(); ++i)
        {
            const auto val = inputs_grad[i];
            const auto golden_val = inputs_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        // Check shared parameters grads
        const raul::Tensor weight_ih_grad_golden{ 1.112274e-01_dt,  5.255696e-01_dt,  -6.308367e-02_dt, -6.135947e-02_dt, 1.112274e-01_dt,  5.255696e-01_dt,  -6.308367e-02_dt, -6.135947e-02_dt,
                                                1.112274e-01_dt,  5.255696e-01_dt,  -6.308367e-02_dt, -6.135947e-02_dt, -3.832054e-02_dt, 8.897363e-02_dt,  3.326690e-02_dt,  6.371387e-02_dt,
                                                -3.832054e-02_dt, 8.897363e-02_dt,  3.326690e-02_dt,  6.371387e-02_dt,  -3.832054e-02_dt, 8.897363e-02_dt,  3.326690e-02_dt,  6.371387e-02_dt,
                                                -2.038447e-01_dt, -9.682778e-01_dt, -5.943990e-01_dt, -2.112956e-01_dt, -2.038447e-01_dt, -9.682778e-01_dt, -5.943990e-01_dt, -2.112956e-01_dt,
                                                -2.038447e-01_dt, -9.682778e-01_dt, -5.943990e-01_dt, -2.112956e-01_dt, -1.620699e-03_dt, 2.575284e-01_dt,  3.971567e-02_dt,  -6.623324e-02_dt,
                                                -1.620699e-03_dt, 2.575284e-01_dt,  3.971567e-02_dt,  -6.623324e-02_dt, -1.620699e-03_dt, 2.575284e-01_dt,  3.971567e-02_dt,  -6.623324e-02_dt };
        const raul::Tensor weight_hh_grad_golden{ -8.980469e-04_dt, -8.980469e-04_dt, -8.980469e-04_dt, -8.980469e-04_dt, -8.980469e-04_dt, -8.980469e-04_dt, -8.980469e-04_dt, -8.980469e-04_dt,
                                                -8.980469e-04_dt, 8.354402e-03_dt,  8.354402e-03_dt,  8.354402e-03_dt,  8.354402e-03_dt,  8.354402e-03_dt,  8.354402e-03_dt,  8.354402e-03_dt,
                                                8.354402e-03_dt,  8.354402e-03_dt,  -3.099198e-02_dt, -3.099198e-02_dt, -3.099198e-02_dt, -3.099198e-02_dt, -3.099198e-02_dt, -3.099198e-02_dt,
                                                -3.099198e-02_dt, -3.099198e-02_dt, -3.099198e-02_dt, 5.146423e-02_dt,  5.146423e-02_dt,  5.146423e-02_dt,  5.146423e-02_dt,  5.146423e-02_dt,
                                                5.146423e-02_dt,  5.146423e-02_dt,  5.146423e-02_dt,  5.146423e-02_dt };
        const raul::Tensor bias_grad_golden{ 2.827185e-02_dt, 2.827185e-02_dt, 2.827185e-02_dt, -7.531291e-02_dt, -7.531291e-02_dt, -7.531291e-02_dt,
                                            1.223329e+00_dt, 1.223329e+00_dt, 1.223329e+00_dt, 1.404038e-01_dt,  1.404038e-01_dt,  1.404038e-01_dt };

        const auto paramGrad = work.getTrainableParameters();
        const auto gradBiasesIH = paramGrad[(useFusion ? 2 : 0)].Gradient;
        const auto gradWeightsIH = paramGrad[(useFusion ? 3 : 1)].Gradient;
        const auto gradBiasesHH = paramGrad[(useFusion ? 0 : 2)].Gradient;
        const auto gradWeightsHH = paramGrad[(useFusion ? 1 : 3)].Gradient;

        EXPECT_EQ(gradWeightsIH.size(), weight_ih_grad_golden.size());
        EXPECT_EQ(gradWeightsHH.size(), weight_hh_grad_golden.size());
        EXPECT_EQ(gradBiasesIH.size(), bias_grad_golden.size());
        EXPECT_EQ(gradBiasesHH.size(), bias_grad_golden.size());

        for (size_t i = 0; i < gradWeightsIH.size(); ++i)
        {
            const auto val = gradWeightsIH[i];
            const auto golden_val = weight_ih_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        for (size_t i = 0; i < gradWeightsHH.size(); ++i)
        {
            const auto val = gradWeightsHH[i];
            const auto golden_val = weight_hh_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        for (size_t i = 0; i < gradBiasesIH.size(); ++i)
        {
            const auto val = gradBiasesIH[i];
            const auto golden_val = bias_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        for (size_t i = 0; i < gradBiasesHH.size(); ++i)
        {
            const auto val = gradBiasesHH[i];
            const auto golden_val = bias_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }
    }
}

TEST(TestLSTM, SimpleForwardSeq5ExtStateUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-4_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto sequence_length = 5U;
    const auto batch_size = 2U;

    // Use fusion or not
    for (size_t q = 0; q < 2; ++q)
    {
        // Initialization
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, sequence_length, 1, input_size });
        work.add<raul::DataLayer>("data2", raul::DataParams{ { "hidden_in" }, 1, 1, hidden_size });
        work.add<raul::DataLayer>("data3", raul::DataParams{ { "cell_in" }, 1, 1, hidden_size });

        // Network
        bool useFusion = (q == 1 ? true : false);
        const auto params = raul::LSTMParams{ "in", "hidden_in", "cell_in", "out", "hidden_out", "cell_out", false, true, false, false, 0.0_dt, false, 0.0_dt, useFusion };
        raul::LSTMLayer("lstm", params, networkParameters);
        TENSORS_CREATE(batch_size)

        const raul::Tensor input_init{ -8.024929e-01_dt, -1.295186e+00_dt, -7.501815e-01_dt, -1.311966e+00_dt, -2.188337e-01_dt, -2.435065e+00_dt, -7.291476e-02_dt, -3.398641e-02_dt,
                                    7.968872e-01_dt,  -1.848416e-01_dt, -3.701473e-01_dt, -1.210281e+00_dt, -6.226985e-01_dt, -4.637222e-01_dt, 1.921782e+00_dt,  -4.025455e-01_dt,
                                    9.295023e-02_dt,  -6.660997e-01_dt, 6.080472e-01_dt,  -7.300199e-01_dt, -8.833758e-01_dt, -4.189135e-01_dt, -8.048265e-01_dt, 5.656096e-01_dt,
                                    2.885762e-01_dt,  3.865978e-01_dt,  -2.010639e-01_dt, -1.179270e-01_dt, -8.293669e-01_dt, -1.407257e+00_dt, 1.626847e+00_dt,  1.722732e-01_dt,
                                    -7.042940e-01_dt, 3.147210e-01_dt,  1.573929e-01_dt,  3.853627e-01_dt,  5.736546e-01_dt,  9.979313e-01_dt,  5.436094e-01_dt,  7.880439e-02_dt };
        const raul::Tensor hidden_init{ -4.468389e-01_dt, 4.520225e-01_dt, -9.759244e-01_dt, 7.112372e-01_dt, -7.582265e-01_dt, -6.435831e-01_dt };
        const raul::Tensor cell_init{ -6.461524e-01_dt, -1.590926e-01_dt, -1.778664e+00_dt, 8.476512e-01_dt, 2.459428e-01_dt, -1.311679e-01_dt };

        memory_manager["in"] = TORANGE(input_init);
        memory_manager["hidden_in"] = TORANGE(hidden_init);
        memory_manager["cell_in"] = TORANGE(cell_init);

        for (auto& [param, grad] : work.getTrainableParameters())
        {
            param = 1.0_dt;
        }

        // Apply
        work.forwardPassTesting();

        // Checks
        const raul::Tensor output_golden{ -2.873812e-03_dt, -2.023149e-03_dt, -4.841400e-03_dt, -7.045983e-02_dt, -6.851754e-02_dt, -7.495427e-02_dt, 2.086206e-01_dt, 2.114406e-01_dt,
                                        2.020342e-01_dt,  8.093282e-01_dt,  8.104742e-01_dt,  8.066313e-01_dt,  9.525990e-01_dt,  9.527962e-01_dt,  9.521345e-01_dt, 1.182437e-01_dt,
                                        3.509183e-03_dt,  -6.965960e-02_dt, 7.515067e-01_dt,  6.616085e-01_dt,  5.865334e-01_dt,  9.432809e-01_dt,  9.260048e-01_dt, 9.104356e-01_dt,
                                        9.885837e-01_dt,  9.860258e-01_dt,  9.836924e-01_dt,  9.986322e-01_dt,  9.982802e-01_dt,  9.979585e-01_dt };
        const raul::Tensor hidden_golden{ 9.525990e-01_dt, 9.527962e-01_dt, 9.521345e-01_dt, 9.986322e-01_dt, 9.982802e-01_dt, 9.979585e-01_dt };
        const raul::Tensor cell_golden{ 2.193398e+00_dt, 2.197572e+00_dt, 2.183694e+00_dt, 4.067711e+00_dt, 3.832196e+00_dt, 3.684592e+00_dt };
        const auto& outputTensor = memory_manager["out"];
        const auto& hiddenTensor = memory_manager["hidden_out"];
        const auto& cellTensor = memory_manager["cell_out"];

        EXPECT_EQ(outputTensor.size(), batch_size * hidden_size * sequence_length);
        EXPECT_EQ(hiddenTensor.size(), batch_size * hidden_size);
        EXPECT_EQ(cellTensor.size(), batch_size * hidden_size);

        for (size_t i = 0; i < outputTensor.size(); ++i)
        {
            const auto val = outputTensor[i];
            const auto golden_val = output_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        for (size_t i = 0; i < hiddenTensor.size(); ++i)
        {
            const auto val = hiddenTensor[i];
            const auto golden_val = hidden_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        for (size_t i = 0; i < cellTensor.size(); ++i)
        {
            const auto val = cellTensor[i];
            const auto golden_val = cell_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }
    }
}

TEST(TestLSTM, SimpleBackwardSeq5ExtStateUnit)
{
    PROFILE_TEST
    using namespace raul;

    // Test parameters
    const auto eps_rel = 1e-4_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto sequence_length = 5U;
    const auto batch_size = 2U;

    // Use fusion or not
    for (size_t q = 0; q < 2; ++q)
    {
        // Initialization
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, sequence_length, 1, input_size });
        work.add<raul::DataLayer>("data2", raul::DataParams{ { "hidden_in" }, 1, 1, hidden_size });
        work.add<raul::DataLayer>("data3", raul::DataParams{ { "cell_in" }, 1, 1, hidden_size });

        // Network
        bool useFusion = (q == 1 ? true : false);
        const auto params = LSTMParams{ "in", "hidden_in", "cell_in", "out", "hidden_out", "cell_out", false, true, false, false, 0.0_dt, false, 0.0_dt, useFusion };
        LSTMLayer("lstm", params, networkParameters);
        TENSORS_CREATE(batch_size)

        const Tensor input_init{ -8.024929e-01_dt, -1.295186e+00_dt, -7.501815e-01_dt, -1.311966e+00_dt, -2.188337e-01_dt, -2.435065e+00_dt, -7.291476e-02_dt, -3.398641e-02_dt,
                                7.968872e-01_dt,  -1.848416e-01_dt, -3.701473e-01_dt, -1.210281e+00_dt, -6.226985e-01_dt, -4.637222e-01_dt, 1.921782e+00_dt,  -4.025455e-01_dt,
                                9.295023e-02_dt,  -6.660997e-01_dt, 6.080472e-01_dt,  -7.300199e-01_dt, -8.833758e-01_dt, -4.189135e-01_dt, -8.048265e-01_dt, 5.656096e-01_dt,
                                2.885762e-01_dt,  3.865978e-01_dt,  -2.010639e-01_dt, -1.179270e-01_dt, -8.293669e-01_dt, -1.407257e+00_dt, 1.626847e+00_dt,  1.722732e-01_dt,
                                -7.042940e-01_dt, 3.147210e-01_dt,  1.573929e-01_dt,  3.853627e-01_dt,  5.736546e-01_dt,  9.979313e-01_dt,  5.436094e-01_dt,  7.880439e-02_dt };
        const Tensor hidden_init{ -4.468389e-01_dt, 4.520225e-01_dt, -9.759244e-01_dt, 7.112372e-01_dt, -7.582265e-01_dt, -6.435831e-01_dt };
        const Tensor cell_init{ -6.461524e-01_dt, -1.590926e-01_dt, -1.778664e+00_dt, 8.476512e-01_dt, 2.459428e-01_dt, -1.311679e-01_dt };

        memory_manager["in"] = TORANGE(input_init);
        memory_manager["hidden_in"] = TORANGE(hidden_init);
        memory_manager["cell_in"] = TORANGE(cell_init);

        for (auto& [param, grad] : work.getTrainableParameters())
        {
            param = 1.0_dt;
        }

        // Apply
        work.forwardPassTraining();

        memory_manager[Name("out").grad()] = 1.0_dt;
        memory_manager[Name("hidden_out").grad()] = 1.0_dt;
        memory_manager[Name("cell_out").grad()] = 1.0_dt;

        work.backwardPassTraining();

        // Checks
        const Tensor inputs_grad_golden{ -2.154902e-01_dt, -2.154902e-01_dt, -2.154902e-01_dt, -2.154902e-01_dt, -4.182333e-01_dt, -4.182333e-01_dt, -4.182333e-01_dt, -4.182333e-01_dt,
                                        3.471492e+00_dt,  3.471492e+00_dt,  3.471492e+00_dt,  3.471492e+00_dt,  4.183314e-01_dt,  4.183314e-01_dt,  4.183314e-01_dt,  4.183314e-01_dt,
                                        3.095903e-01_dt,  3.095903e-01_dt,  3.095903e-01_dt,  3.095903e-01_dt,  2.948041e+00_dt,  2.948041e+00_dt,  2.948041e+00_dt,  2.948041e+00_dt,
                                        7.211524e-01_dt,  7.211524e-01_dt,  7.211524e-01_dt,  7.211524e-01_dt,  2.657774e-01_dt,  2.657774e-01_dt,  2.657774e-01_dt,  2.657774e-01_dt,
                                        8.456340e-02_dt,  8.456340e-02_dt,  8.456340e-02_dt,  8.456340e-02_dt,  1.379239e-02_dt,  1.379239e-02_dt,  1.379239e-02_dt,  1.379239e-02_dt };

        const auto& inputs_grad = memory_manager[Name("in").grad()];
        const auto& inputs = memory_manager["in"];

        EXPECT_EQ(inputs.size(), inputs_grad.size());

        for (size_t i = 0; i < inputs_grad.size(); ++i)
        {
            const auto val = inputs_grad[i];
            const auto golden_val = inputs_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        // Check shared parameters grads
        const Tensor weight_ih_grad_golden{ 4.427108e-01_dt,  9.863892e-01_dt,  1.883101e-01_dt,  -4.382888e-01_dt, 4.584172e-01_dt,  9.972150e-01_dt,  1.983972e-01_dt,  -4.469928e-01_dt,
                                            4.695551e-01_dt,  1.002892e+00_dt,  2.035596e-01_dt,  -4.559649e-01_dt, -4.650742e-01_dt, -7.634363e-02_dt, -1.693685e-01_dt, 3.799740e-01_dt,
                                            -2.327396e-01_dt, -1.716787e-02_dt, 3.827258e-02_dt,  1.929823e-01_dt,  4.139580e-03_dt,  2.546748e-01_dt,  2.607774e-01_dt,  1.691507e-01_dt,
                                            -1.789881e-01_dt, -1.807874e+00_dt, -1.010744e+00_dt, -5.859998e-01_dt, -2.726763e-01_dt, -1.849642e+00_dt, -1.096480e+00_dt, -5.241356e-01_dt,
                                            -3.152885e-01_dt, -1.872623e+00_dt, -1.144722e+00_dt, -5.015457e-01_dt, -1.010975e-02_dt, 4.048809e-01_dt,  4.290407e-02_dt,  -8.930679e-02_dt,
                                            8.417503e-02_dt,  4.327759e-01_dt,  1.314885e-01_dt,  -1.529343e-01_dt, 1.468077e-01_dt,  5.106470e-01_dt,  1.925082e-01_dt,  -1.840367e-01_dt };
        const Tensor weight_hh_grad_golden{ -1.060414e-02_dt, 1.070702e-01_dt,  1.364040e-01_dt,  -1.790520e-02_dt, 1.187281e-01_dt,  1.451716e-01_dt,  -2.102653e-02_dt, 1.251480e-01_dt,
                                            1.497073e-01_dt,  3.912693e-01_dt,  -2.543108e-01_dt, -1.755999e-01_dt, 1.723360e-01_dt,  -3.753334e-02_dt, -1.506338e-02_dt, 4.103640e-02_dt,
                                            9.432380e-02_dt,  1.818821e-01_dt,  5.489463e-01_dt,  -6.905279e-01_dt, -6.048693e-01_dt, 6.254129e-01_dt,  -7.707407e-01_dt, -6.733758e-01_dt,
                                            6.654203e-01_dt,  -8.135428e-01_dt, -7.101704e-01_dt, 1.605264e-01_dt,  -1.921275e-02_dt, -1.248798e-02_dt, 8.037686e-02_dt,  6.404880e-02_dt,
                                            5.817027e-02_dt,  3.009853e-02_dt,  1.164924e-01_dt,  1.054975e-01_dt };

        const Tensor bias_grad_golden{ -7.130340e-03_dt, -4.987031e-03_dt, 4.158601e-03_dt, 3.736566e-01_dt, 8.978634e-02_dt, -2.685242e-01_dt,
                                    2.289106e+00_dt,  2.399543e+00_dt,  2.467558e+00_dt, 1.949094e-01_dt, 8.254965e-02_dt, -2.160972e-02_dt };

        const auto paramGrad = work.getTrainableParameters();
        const auto gradBiasesIH = paramGrad[(useFusion ? 2 : 0)].Gradient;
        const auto gradWeightsIH = paramGrad[(useFusion ? 3 : 1)].Gradient;
        const auto gradBiasesHH = paramGrad[(useFusion ? 0 : 2)].Gradient;
        const auto gradWeightsHH = paramGrad[(useFusion ? 1 : 3)].Gradient;

        EXPECT_EQ(gradWeightsIH.size(), weight_ih_grad_golden.size());
        EXPECT_EQ(gradWeightsHH.size(), weight_hh_grad_golden.size());
        EXPECT_EQ(gradBiasesIH.size(), bias_grad_golden.size());
        EXPECT_EQ(gradBiasesHH.size(), bias_grad_golden.size());

        for (size_t i = 0; i < gradWeightsIH.size(); ++i)
        {
            const auto val = gradWeightsIH[i];
            const auto golden_val = weight_ih_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        for (size_t i = 0; i < gradWeightsHH.size(); ++i)
        {
            const auto val = gradWeightsHH[i];
            const auto golden_val = weight_hh_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        for (size_t i = 0; i < gradBiasesIH.size(); ++i)
        {
            const auto val = gradBiasesIH[i];
            const auto golden_val = bias_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }

        for (size_t i = 0; i < gradBiasesHH.size(); ++i)
        {
            const auto val = gradBiasesHH[i];
            const auto golden_val = bias_grad_golden[i];
            ASSERT_TRUE(tools::expect_near_relative(val, golden_val, eps_rel)) << "at " << i << ", expected: " << golden_val << ", got: " << val;
        }
    }
}

#ifdef ANDROID
TEST(TestLSTM, SimpleForwardSeq1FP16Unit)
{
    PROFILE_TEST
    // Test parameters
    const size_t input_size = 4U;
    const size_t hidden_size = 3U;
    const size_t sequence_length = 1U;
    const size_t batch_size = 2U;

    // Initialization
    raul::WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16 };
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data1", raul::DataParams{ { "in" }, sequence_length, 1, input_size });

    // Network
    const auto params = raul::LSTMParams{ { "in" }, { "out" }, hidden_size, false };
    raul::LSTMLayer("lstm", params, networkParameters);
    TENSORS_CREATE(batch_size)

    // Apply
    EXPECT_NO_THROW(work.forwardPassTesting());
}
#endif // ANDROID

} // UT namespace