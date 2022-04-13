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
#include <training/compiler/Layers.h>
#include <training/base/optimizers/SGD.h>

#include <training/base/layers/composite/rnn/BidirectionalLSTMFunc.h>
#include <training/base/layers/composite/rnn/LSTMLayer.h>

namespace UT
{
using namespace std;
using namespace raul;

TEST(TestBiLSTM, SimpleForwardSeq1Unit)
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

    work.add<DataLayer>("data1", DataParams{ { "in" }, sequence_length, 1, input_size });

    // Network
    const auto params = LSTMParams{ { "in" }, { "out" }, hidden_size, false };
    BidirectionalLSTMFunc("lstm", params, networkParameters, BidirectionalMergeType::ConcatDepth);
    TENSORS_CREATE(batch_size)

    const Tensor input_init{ 1.752833e-01_dt, -9.315211e-01_dt, -1.505490e+00_dt, -6.609825e-01_dt, 1.323202e+00_dt, 3.711430e-02_dt, -2.849093e-01_dt, -1.334417e-01_dt };
    memory_manager["in"] = TORANGE(input_init);

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Apply
    work.forwardPassTesting();

    // Checks
    const Tensor output_golden{ -5.799451e-02_dt, -5.799451e-02_dt, -5.799451e-02_dt, 7.003791e-01_dt, 7.003791e-01_dt, 7.003791e-01_dt };

    const auto& outputTensor = memory_manager["out"];
    const auto& outputTensor_1 = memory_manager["out::direct"];
    const auto& outputTensor_2 = memory_manager["out::reversed"];

    EXPECT_EQ(outputTensor.size(), 2 * batch_size * hidden_size * sequence_length);

    for (size_t i = 0; i < output_golden.size(); ++i)
    {
        const auto val_1 = outputTensor_1[i];
        const auto val_2 = outputTensor_2[i];
        const auto val_g = output_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val_1, val_g, eps_rel)) << "at " << i << ", expected: " << val_g << ", got: " << val_1;
        ASSERT_TRUE(tools::expect_near_relative(val_2, val_g, eps_rel)) << "at " << i << ", expected: " << val_g << ", got: " << val_2;
    }
}

TEST(TestBiLSTM, SimpleForwardSeq2Unit)
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

    work.add<DataLayer>("data1", DataParams{ { "in" }, sequence_length, 1, input_size });

    // Network
    const auto params = LSTMParams{ { "in" }, { "out" }, hidden_size, false };
    BidirectionalLSTMFunc("lstm", params, networkParameters, BidirectionalMergeType::ConcatDepth);

    TENSORS_CREATE(batch_size)

    const Tensor input_init{ 4.386492e-01_dt, -1.070405e-02_dt, 1.338354e+00_dt, -2.794050e-01_dt, -5.518340e-01_dt, -2.889061e+00_dt, -1.509981e+00_dt, 1.024115e+00_dt,
                                   1.953929e-01_dt, -7.371095e-01_dt, 1.700101e+00_dt, 3.462155e-01_dt,  9.711247e-01_dt,  1.450250e+00_dt,  -5.190918e-02_dt, -6.284308e-01_dt };
    memory_manager["in"] = TORANGE(input_init);

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    work.printInfo(std::cout);

    // Apply
    work.forwardPassTesting();

    // Checks
    const Tensor outputTensor_1_golden{ 7.258359e-01_dt, 7.258359e-01_dt, 7.258359e-01_dt, 3.336587e-01_dt, 3.336587e-01_dt, 3.336587e-01_dt,
                                              7.264570e-01_dt, 7.264570e-01_dt, 7.264570e-01_dt, 9.588038e-01_dt, 9.588038e-01_dt, 9.588038e-01_dt };
    const Tensor outputTensor_2_golden{ 6.690636e-01_dt, 6.690636e-01_dt, 6.690636e-01_dt, -1.540969e-02_dt, -1.540969e-02_dt, -1.540969e-02_dt,
                                              9.585935e-01_dt, 9.585935e-01_dt, 9.585935e-01_dt, 7.337949e-01_dt,  7.337949e-01_dt,  7.337949e-01_dt };

    const auto& outputTensor = memory_manager["out"];
    const auto& outputTensor_1 = memory_manager["out::direct"];
    const auto& outputTensor_2 = memory_manager["out::reversed"];

    EXPECT_EQ(outputTensor.size(), 2 * batch_size * hidden_size * sequence_length);

    for (size_t i = 0; i < outputTensor_1.size(); ++i)
    {
        const auto val_1 = outputTensor_1[i];
        const auto val_2 = outputTensor_2[i];
        const auto val_1_g = outputTensor_1_golden[i];
        const auto val_2_g = outputTensor_2_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val_1, val_1_g, eps_rel)) << "at " << i << ", expected: " << val_1_g << ", got: " << val_1;
        ASSERT_TRUE(tools::expect_near_relative(val_2, val_2_g, eps_rel)) << "at " << i << ", expected: " << val_2_g << ", got: " << val_2;
    }
}

TEST(TestBiLSTM, SimpleForwardSeq5Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-4_dt;
    const size_t input_size = 4U;
    const size_t hidden_size = 3U;
    const size_t sequence_length = 5U;
    const size_t batch_size = 2U;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data1", DataParams{ { "in" }, sequence_length, 1, input_size });

    // Network
    const auto params = LSTMParams{ { "in" }, { "out" }, hidden_size, false };
    BidirectionalLSTMFunc("lstm", params, networkParameters, BidirectionalMergeType::ConcatDepth);
    TENSORS_CREATE(batch_size)

    const Tensor input_init{ 4.386492e-01_dt,  -1.070405e-02_dt, 1.338354e+00_dt,  -2.794050e-01_dt, -5.518340e-01_dt, -2.889061e+00_dt, -1.509981e+00_dt, 1.024115e+00_dt,
                                   1.953929e-01_dt,  -7.371095e-01_dt, 1.700101e+00_dt,  3.462155e-01_dt,  9.711247e-01_dt,  1.450250e+00_dt,  -5.190918e-02_dt, -6.284308e-01_dt,
                                   -6.537996e-01_dt, 1.719824e+00_dt,  -9.609554e-01_dt, -6.375025e-01_dt, 7.472499e-02_dt,  5.599695e-01_dt,  5.314036e-01_dt,  1.235090e+00_dt,
                                   -3.937254e-02_dt, -8.014722e-01_dt, -4.955443e-01_dt, -3.615141e-01_dt, 5.851132e-01_dt,  -1.156007e+00_dt, -1.433649e-01_dt, -1.947406e-01_dt,
                                   -8.556341e-02_dt, 1.394520e+00_dt,  5.969000e-01_dt,  -4.828483e-01_dt, -3.660986e-01_dt, -1.327052e+00_dt, 1.695280e+00_dt,  2.065500e+00_dt };
    memory_manager["in"] = TORANGE(input_init);

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Apply
    work.forwardPassTesting();

    // Checks
    const Tensor outputTensor_1_golden{ 7.258359e-01_dt, 7.258359e-01_dt, 7.258359e-01_dt, 3.336587e-01_dt, 3.336587e-01_dt, 3.336587e-01_dt, 9.206030e-01_dt, 9.206030e-01_dt,
                                              9.206030e-01_dt, 9.887827e-01_dt, 9.887827e-01_dt, 9.887827e-01_dt, 9.868459e-01_dt, 9.868459e-01_dt, 9.868459e-01_dt, 7.471700e-01_dt,
                                              7.471700e-01_dt, 7.471700e-01_dt, 8.807512e-01_dt, 8.807512e-01_dt, 8.807512e-01_dt, 9.689146e-01_dt, 9.689146e-01_dt, 9.689146e-01_dt,
                                              9.971335e-01_dt, 9.971335e-01_dt, 9.971335e-01_dt, 9.989925e-01_dt, 9.989925e-01_dt, 9.989925e-01_dt };
    const Tensor outputTensor_2_golden{ 9.950213e-01_dt, 9.950213e-01_dt, 9.950213e-01_dt, 7.308027e-01_dt, 7.308027e-01_dt, 7.308027e-01_dt, 9.894929e-01_dt, 9.894929e-01_dt,
                                              9.894929e-01_dt, 9.332256e-01_dt, 9.332256e-01_dt, 9.332256e-01_dt, 5.067036e-01_dt, 5.067036e-01_dt, 5.067036e-01_dt, 9.991685e-01_dt,
                                              9.991685e-01_dt, 9.991685e-01_dt, 9.608952e-01_dt, 9.608952e-01_dt, 9.608952e-01_dt, 9.757363e-01_dt, 9.757363e-01_dt, 9.757363e-01_dt,
                                              9.588814e-01_dt, 9.588814e-01_dt, 9.588814e-01_dt, 7.414938e-01_dt, 7.414938e-01_dt, 7.414938e-01_dt };

    const auto& outputTensor = memory_manager["out"];
    const auto& outputTensor_1 = memory_manager["out::direct"];
    const auto& outputTensor_2 = memory_manager["out::reversed"];

    EXPECT_EQ(outputTensor.size(), 2 * batch_size * hidden_size * sequence_length);

    for (size_t i = 0; i < outputTensor_1.size(); ++i)
    {
        const auto val_1 = outputTensor_1[i];
        const auto val_2 = outputTensor_2[i];
        const auto val_1_g = outputTensor_1_golden[i];
        const auto val_2_g = outputTensor_2_golden[i];
        ASSERT_TRUE(tools::expect_near_relative(val_1, val_1_g, eps_rel)) << "at " << i << ", expected: " << val_1_g << ", got: " << val_1;
        ASSERT_TRUE(tools::expect_near_relative(val_2, val_2_g, eps_rel)) << "at " << i << ", expected: " << val_2_g << ", got: " << val_2;
    }
}

TEST(TestBiLSTM, SimpleBackwardSeq1Unit)
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

    work.add<DataLayer>("data1", DataParams{ { "in" }, sequence_length, 1, input_size });

    // Network
    const auto params = LSTMParams{ { "in" }, { "out" }, hidden_size, false };
    BidirectionalLSTMFunc("lstm", params, networkParameters, BidirectionalMergeType::ConcatDepth);
    TENSORS_CREATE(batch_size)

    const Tensor input_init{ 4.987862e-01_dt, -5.233232e-01_dt, -2.514795e-01_dt, -1.055532e+00_dt, -5.592613e-01_dt, -1.197086e-01_dt, -1.635457e-01_dt, -2.504632e-01_dt };
    memory_manager["in"] = TORANGE(input_init);

    memory_manager[Name("out").grad()] = 1.0_dt;

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Apply
    work.forwardPassTraining();

    work.backwardPassTraining();

    // Checks
    const Tensor inputs_grad_golden{ 2.437663e+00_dt, 2.437663e+00_dt, 2.437663e+00_dt, 2.437663e+00_dt, 2.210873e+00_dt, 2.210873e+00_dt, 2.210873e+00_dt, 2.210873e+00_dt };

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

TEST(TestBiLSTM, SimpleBackwardSeq2Unit)
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

    work.add<DataLayer>("data1", DataParams{ { "in" }, sequence_length, 1, input_size });

    // Network
    const auto params = LSTMParams{ { "in" }, { "out" }, hidden_size, false };
    BidirectionalLSTMFunc("lstm", params, networkParameters, BidirectionalMergeType::ConcatDepth);
    TENSORS_CREATE(batch_size)

    const Tensor input_init{ 4.386492e-01_dt, -1.070405e-02_dt, 1.338354e+00_dt, -2.794050e-01_dt, -5.518340e-01_dt, -2.889061e+00_dt, -1.509981e+00_dt, 1.024115e+00_dt,
                                   1.953929e-01_dt, -7.371095e-01_dt, 1.700101e+00_dt, 3.462155e-01_dt,  9.711247e-01_dt,  1.450250e+00_dt,  -5.190918e-02_dt, -6.284308e-01_dt };
    memory_manager["in"] = TORANGE(input_init);

    memory_manager[Name("out").grad()] = 1.0_dt;

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Apply
    work.forwardPassTraining();

    work.backwardPassTraining();

    // Checks
    const Tensor inputs_grad_golden{ 3.770120e-01_dt, 3.770120e-01_dt, 3.770120e-01_dt, 3.770120e-01_dt, 1.116199e+00_dt, 1.116199e+00_dt, 1.116199e+00_dt, 1.116199e+00_dt,
                                           1.234487e-01_dt, 1.234487e-01_dt, 1.234487e-01_dt, 1.234487e-01_dt, 9.806208e-02_dt, 9.806208e-02_dt, 9.806208e-02_dt, 9.806208e-02_dt };

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

TEST(TestBiLSTM, SimpleBackwardSeq5Unit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps_rel = 1e-4_dt;
    const size_t input_size = 4U;
    const size_t hidden_size = 3U;
    const size_t sequence_length = 5U;
    const size_t batch_size = 2U;

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data1", DataParams{ { "in" }, sequence_length, 1, input_size });

    // Network
    const auto params = LSTMParams{ { "in" }, { "out" }, hidden_size, false };
    BidirectionalLSTMFunc("lstm", params, networkParameters, BidirectionalMergeType::ConcatDepth);
    TENSORS_CREATE(batch_size)

    const Tensor input_init{ 4.386492e-01_dt,  -1.070405e-02_dt, 1.338354e+00_dt,  -2.794050e-01_dt, -5.518340e-01_dt, -2.889061e+00_dt, -1.509981e+00_dt, 1.024115e+00_dt,
                                   1.953929e-01_dt,  -7.371095e-01_dt, 1.700101e+00_dt,  3.462155e-01_dt,  9.711247e-01_dt,  1.450250e+00_dt,  -5.190918e-02_dt, -6.284308e-01_dt,
                                   -6.537996e-01_dt, 1.719824e+00_dt,  -9.609554e-01_dt, -6.375025e-01_dt, 7.472499e-02_dt,  5.599695e-01_dt,  5.314036e-01_dt,  1.235090e+00_dt,
                                   -3.937254e-02_dt, -8.014722e-01_dt, -4.955443e-01_dt, -3.615141e-01_dt, 5.851132e-01_dt,  -1.156007e+00_dt, -1.433649e-01_dt, -1.947406e-01_dt,
                                   -8.556341e-02_dt, 1.394520e+00_dt,  5.969000e-01_dt,  -4.828483e-01_dt, -3.660986e-01_dt, -1.327052e+00_dt, 1.695280e+00_dt,  2.065500e+00_dt };
    memory_manager["in"] = TORANGE(input_init);

    memory_manager[Name("out").grad()] = 1.0_dt;

    for (auto& [param, grad] : work.getTrainableParameters())
    {
        param = 1.0_dt;
    }

    // Apply
    work.forwardPassTraining();

    work.backwardPassTraining();

    // Checks
    const Tensor inputs_grad_golden{ 3.341508e-01_dt, 3.341508e-01_dt, 3.341508e-01_dt, 3.341508e-01_dt, 2.412369e+00_dt, 2.412369e+00_dt, 2.412369e+00_dt, 2.412369e+00_dt,
                                           4.854282e-02_dt, 4.854282e-02_dt, 4.854282e-02_dt, 4.854282e-02_dt, 2.387276e-02_dt, 2.387276e-02_dt, 2.387276e-02_dt, 2.387276e-02_dt,
                                           9.123158e-01_dt, 9.123158e-01_dt, 9.123158e-01_dt, 9.123158e-01_dt, 6.033855e-02_dt, 6.033855e-02_dt, 6.033855e-02_dt, 6.033855e-02_dt,
                                           3.709853e-01_dt, 3.709853e-01_dt, 3.709853e-01_dt, 3.709853e-01_dt, 1.347198e-01_dt, 1.347198e-01_dt, 1.347198e-01_dt, 1.347198e-01_dt,
                                           1.808115e-02_dt, 1.808115e-02_dt, 1.808115e-02_dt, 1.808115e-02_dt, 6.819304e-02_dt, 6.819304e-02_dt, 6.819304e-02_dt, 6.819304e-02_dt };

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

} // UT namespace