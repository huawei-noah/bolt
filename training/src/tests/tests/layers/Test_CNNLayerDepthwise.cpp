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
#include <training/base/common/MemoryManager.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/trainable/ConvolutionDepthwiseLayer.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestCNNDepthwiseLayer, BiasesUnit)
{
    PROFILE_TEST
    using namespace raul;

    const size_t KERNEL_SIZE = 3;
    const size_t FILTERS = 1;
    const size_t STRIDE = 1;
    const size_t PADDING = 0;

    const dtype EPSILON = TODTYPE(1e-6);

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    Tensor input = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

    work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, 1, 3, 3 });
    ConvolutionDepthwiseLayer cnnLayer("cnn1", Convolution2DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE, FILTERS, STRIDE, PADDING, true }, networkParameters);
    TENSORS_CREATE(1);
    memory_manager["in"] = TORANGE(input);

    memory_manager["cnn1::Weights"] = TORANGE((Tensor{ 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }));
    memory_manager["cnn1::Biases"] = TORANGE((Tensor{ 2.0f }));
    ASSERT_NO_THROW(memory_manager["cnn1::Biases"]);
    ASSERT_NO_THROW(memory_manager["cnn1::BiasesGradient"]);

    ASSERT_NO_THROW(cnnLayer.forwardCompute(NetworkMode::Train));

    EXPECT_EQ(memory_manager["cnn1"].size(), static_cast<size_t>(FILTERS));
    CHECK_NEAR(memory_manager["cnn1"][0], 11.0f, EPSILON);

    memory_manager[Name("cnn1").grad()] = TORANGE((Tensor{ 1.0f }));
    auto& tt = memory_manager[Name("in").grad()];
    ASSERT_NO_THROW(cnnLayer.backwardCompute());
    for (auto& t : tt)
    {
        cout << t << " ";
    }
    cout << endl;
}

TEST(TestCNNDepthwiseLayer, Biases3Unit)
{
    PROFILE_TEST
    using namespace raul;

    const size_t KERNEL_SIZE = 3;
    const size_t FILTERS = 2;
    const size_t STRIDE = 1;
    const size_t PADDING = 1;

    const dtype EPSILON = TODTYPE(1e-6);

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    Tensor input = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f };

    Tensor realOutput(1, 2, 3, 2, { 11.0f, 11.0f, 22.0f, 22.0f, 19.0f, 19.0f, -18.0f, -18.0f, -40.0f, -40.0f, -34.0f, -34.0f });
    Tensor realInputGrad(1, 2, 3, 2, { 4.0f, 4.0f, 6.0f, 6.0f, 4.0f, 4.0f, 8.0f, 8.0f, 12.0f, 12.0f, 8.0f, 8.0f });

    work.add<raul::DataLayer>("data", raul::DataParams{ { "in" }, 2, 3, 2 });
    ConvolutionDepthwiseLayer cnnLayer("cnn1", Convolution2DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE, FILTERS, STRIDE, PADDING, true }, networkParameters);
    TENSORS_CREATE(1);
    memory_manager["in"] = TORANGE(input);

    memory_manager["cnn1::Weights"] = TORANGE((Tensor{ 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f }));
    memory_manager["cnn1::Biases"] = TORANGE((Tensor{ 1.0f, 2.0f }));

    Tensor realWeightsGrad(memory_manager["cnn1::Weights"].getShape(), { 4.0f, 10.0f, 6.0f, 9.0f, 21.0f, 12.0f, 8.0f, 18.0f, 10.0f, -4.0f, -10.0f, -6.0f, -9.0f, -21.0f, -12.0f, -8.0f, -18.0f, -10.0f });
    Tensor realBiasesGrad(memory_manager["cnn1::Biases"].getShape(), { 6.0f, 6.0f });

    ASSERT_NO_THROW(memory_manager["cnn1::Biases"]);
    ASSERT_NO_THROW(memory_manager["cnn1::BiasesGradient"]);

    ASSERT_NO_THROW(cnnLayer.forwardCompute(NetworkMode::Train));

    EXPECT_EQ(memory_manager["cnn1"].getShape(), realOutput.getShape());
    for (size_t i = 0; i < realOutput.size(); ++i)
    {
        CHECK_NEAR(memory_manager["cnn1"][i], realOutput[i], EPSILON);
    }

    memory_manager[Name("cnn1").grad()] = 1_dt;
    
    ASSERT_NO_THROW(cnnLayer.backwardCompute());
    EXPECT_EQ(memory_manager[Name("in").grad()].getShape(), realInputGrad.getShape());
    for (size_t i = 0; i < realInputGrad.size(); ++i)
    {
        CHECK_NEAR(memory_manager[Name("in").grad()][i], realInputGrad[i], EPSILON);
    }

    EXPECT_EQ(memory_manager[Name("cnn1::Weights").grad()].getShape(), realWeightsGrad.getShape());
    for (size_t i = 0; i < realWeightsGrad.size(); ++i)
    {
        CHECK_NEAR(memory_manager[Name("cnn1::Weights").grad()][i], realWeightsGrad[i], EPSILON);
    }

    EXPECT_EQ(memory_manager[Name("cnn1::Biases").grad()].getShape(), realBiasesGrad.getShape());
    for (size_t i = 0; i < realBiasesGrad.size(); ++i)
    {
        CHECK_NEAR(memory_manager[Name("cnn1::Biases").grad()][i], realBiasesGrad[i], EPSILON);
    }
}

TEST(TestCNNDepthwiseLayer, Biases2Unit)
{
    PROFILE_TEST
    using namespace raul;

    const size_t KERNEL_SIZE = 3;
    const size_t FILTERS = 1;
    const size_t STRIDE = 1;
    const size_t PADDING = 0;

    const raul::dtype EPSILON = TODTYPE(1e-6);

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    Tensor input = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

    work.add<raul::DataLayer>("data", raul::DataParams{ { "in", "labels" }, 1, 3, 3, 1 });
    ConvolutionDepthwiseLayer cnnLayer("cnn1", Convolution2DParams{ { "in" }, { "cnn1" }, KERNEL_SIZE, FILTERS, STRIDE, PADDING, false }, networkParameters);
    TENSORS_CREATE(1);
    memory_manager["in"] = TORANGE(input);

    memory_manager["cnn1::Weights"] = TORANGE((Tensor{ 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }));
    ASSERT_THROW(memory_manager["cnn1::Biases"], raul::Exception);
    ASSERT_THROW(memory_manager["cnn1::BiasesGradient"], raul::Exception);

    ASSERT_NO_THROW(cnnLayer.forwardCompute(NetworkMode::Train));

    EXPECT_EQ(memory_manager["cnn1"].size(), static_cast<size_t>(FILTERS));
    CHECK_NEAR(memory_manager["cnn1"][0], 9.0f, EPSILON);

    memory_manager[Name("cnn1").grad()] = TORANGE((Tensor{ 1.0f }));

    auto& tt = memory_manager[Name("in").grad()];
    ASSERT_NO_THROW(cnnLayer.backwardCompute());
    for (auto& t : tt)
    {
        cout << t << " ";
    }
    cout << endl;
}

} // UT namespace