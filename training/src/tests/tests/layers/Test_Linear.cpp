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
#include <training/compiler/Workflow.h>
#include <training/base/optimizers/SGD.h>

namespace UT
{

using namespace raul;
using namespace std;

struct TestLinear : public testing::Test
{
    using il = initializer_list<dtype>;

    static constexpr dtype eps = 2e-4_dt;

    size_t MODEL_SIZE = 5;
    size_t BATCH_SIZE = 2;
    size_t DEPTH = 1;
    size_t HEIGHT = 2;
    size_t LIN_SIZE = 3;

    unique_ptr<Tensor> Input;
    unique_ptr<Tensor> Weights;
    unique_ptr<Tensor> Bias;
    unique_ptr<Tensor> RealOut;
    unique_ptr<Tensor> RealInGrad;
    unique_ptr<Tensor> RealWeightGrad;
    unique_ptr<Tensor> RealBiasGrad;
    unique_ptr<Tensor> OutNabla;

    void SetUp() final
    {
        Input = make_unique<Tensor>(BATCH_SIZE, DEPTH, HEIGHT, MODEL_SIZE, il{ 1., 1., 2., 0., 5., -1., 2., 2., 0., 5., -1., 4., 1., 2., 1., -3., 4., 5., 2., 1. });
        Weights = make_unique<Tensor>(il{ -0.2381f, 0.1714f, -0.0612f, -0.1329f, -0.3701f, 0.0283f, -0.2147f, -0.0502f, 0.209f, 0.4333f, -0.12f, 0.1664f, -0.3021f, -0.225f, 0.3329f });
        Bias = make_unique<Tensor>(il{ 0.3548f, 0.2879f, 0.0343f });

        RealOut = make_unique<Tensor>(BATCH_SIZE, DEPTH, HEIGHT, LIN_SIZE, il{ -1.6848f, 2.1676f, 1.141f, -1.0372f, 1.8963f, 1.5474f, 0.5814f, 0.2019f, 0.4007f, 0.8128f, -0.0555f, -0.5677f });

        RealInGrad = make_unique<Tensor>(BATCH_SIZE, DEPTH, HEIGHT, MODEL_SIZE, il{ -0.0615f,  -0.4244f, 0.1405f,  0.5101f, 0.1636f, -0.2107f, 0.0374f, -0.3829f, -0.0824f, 0.581145f,
                                                                                    -0.06925f, -1.0361f, -0.6339f, 0.9626f, 2.7476f, -0.6845f, 0.8071f, -0.5253f, -0.8123f, -0.67415f });

        RealWeightGrad = make_unique<Tensor>(1, 1, LIN_SIZE, MODEL_SIZE, il{ -6.f, 12.f, 13.5f, 5.f, 10.f, -2.f, 24.f, 7.f, 10.f, 20.f, -7.5f, 11.f, 8.5f, 5.f, 2.5f });

        RealBiasGrad = make_unique<Tensor>(1, 1, 1, LIN_SIZE, il{ 4.f, 8.f, 2.5f });

        OutNabla = make_unique<Tensor>(BATCH_SIZE, DEPTH, HEIGHT, LIN_SIZE, il{ 1.f, 2.f, -1.f, 0.5f, 1.f, 1.f, 0.5f, 6.f, 1.f, 2.f, -1.f, 1.5f });
    }
};

// corresponds to linear.py test
TEST_F(TestLinear, LinearUnit)
{
    PROFILE_TEST
    using namespace raul;

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD);

    auto& memory_manager = work.getMemoryManager();
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<raul::DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, MODEL_SIZE });
    work.add<LinearLayer>("l", LinearParams{ { "in" }, { "out" }, LIN_SIZE });

    TENSORS_CREATE(BATCH_SIZE)
    memory_manager["in"] = TORANGE(*Input);
    memory_manager["l::Weights"] = TORANGE(*Weights);
    memory_manager["l::Biases"] = TORANGE(*Bias);
    memory_manager[Name("out").grad()] = TORANGE(*OutNabla);

    work.forwardPassTraining();
    work.backwardPassTraining();

    ASSERT_FLOAT_TENSORS_EQ(memory_manager["out"], (*RealOut), eps);
    ASSERT_FLOAT_TENSORS_EQ(memory_manager[Name("in").grad()], (*RealInGrad), eps);
    ASSERT_FLOAT_TENSORS_EQ(memory_manager["l::WeightsGradient"], (*RealWeightGrad), eps);
    ASSERT_FLOAT_TENSORS_EQ(memory_manager["l::BiasesGradient"], (*RealBiasGrad), eps);

    printf(" - Linear backward is Ok.\n");
}

// corresponds to linear.py test
TEST_F(TestLinear, ShouldAccumulateGradientsDuringBackwardPassUnit)
{
    PROFILE_TEST

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD);

    auto& memory_manager = work.getMemoryManager();
    NETWORK_PARAMS_DEFINE(networkParameters);

    transform(RealWeightGrad->begin(), RealWeightGrad->end(), RealWeightGrad->begin(), [](dtype v) { return v * 2; });
    transform(RealBiasGrad->begin(), RealBiasGrad->end(), RealBiasGrad->begin(), [](dtype v) { return v * 2; });
    transform(RealInGrad->begin(), RealInGrad->end(), RealInGrad->begin(), [](dtype v) { return v * 2; });
    work.add<raul::DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, MODEL_SIZE });
    work.add<LinearLayer> ("l", LinearParams{ { "in" }, { "out" }, LIN_SIZE });

    TENSORS_CREATE(BATCH_SIZE)
    memory_manager["in"] = TORANGE(*Input);
    memory_manager["l::Weights"] = TORANGE(*Weights);
    memory_manager["l::Biases"] = TORANGE(*Bias);
    memory_manager[Name("out").grad()] = TORANGE(*OutNabla);

    work.forwardPassTraining();
    work.backwardPassTraining();
    work.backwardPassTraining();

    ASSERT_FLOAT_TENSORS_EQ(memory_manager["out"], (*RealOut), eps);
    ASSERT_FLOAT_TENSORS_EQ(memory_manager[Name("in").grad()], (*RealInGrad), eps);
    ASSERT_FLOAT_TENSORS_EQ(memory_manager["l::WeightsGradient"], (*RealWeightGrad), eps);
    ASSERT_FLOAT_TENSORS_EQ(memory_manager["l::BiasesGradient"], (*RealBiasGrad), eps);
}

// linear_sharing.py
TEST_F(TestLinear, LinearSharingUnit)
{
    PROFILE_TEST
    using namespace raul;
    using namespace std;

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD);

    auto& memory_manager = work.getMemoryManager();
    NETWORK_PARAMS_DEFINE(networkParameters);

    LIN_SIZE = MODEL_SIZE;
    size_t LAYERS_COUNT = 3;

    Tensor in = Tensor({ 1.0, 1.0, 2.0, 0.0, 5.0, -1.0, 2.0, 2.0, 0.0, 5.0, -1.0, 4.0, 1.0, 2.0, 1.0, -3.0, 4.0, 5.0, 2.0, 1.0 });

    work.add<raul::DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, MODEL_SIZE });

    string layerName = "l";
    string inName = "in";
    string outName = "out0";
    for (size_t i = 1; i < LAYERS_COUNT; ++i)
    {
        auto suffix = "_" + to_string(i);

        work.add<LinearLayer>(layerName + suffix, LinearParams{ { { inName }, { outName }, { layerName + "::Weights", layerName + "::Biases" } }, LIN_SIZE });

        inName = outName;
        outName = i == LAYERS_COUNT - 1 ? "out" : "out" + to_string(i);
    }
    work.add<LinearLayer>(layerName, LinearParams{ inName, outName, LIN_SIZE });

    Tensor weights = { -0.2381_dt, 0.1714_dt, -0.0612_dt, -0.1329_dt, -0.3701_dt, 0.0283_dt,  -0.2147_dt, -0.0502_dt, 0.2090_dt, 0.4333_dt, -0.1200_dt, 0.1664_dt, -0.3021_dt,
                       -0.2250_dt, 0.3329_dt, -0.1200_dt, 0.1664_dt,  -0.3021_dt, -0.2250_dt, 0.3329_dt,  0.1200_dt,  0.1664_dt, 0.3021_dt, 0.2250_dt,  0.3329_dt };

    Tensor bias = { 0.3548_dt, 0.2879_dt, 0.0343_dt, 0.1269_dt, 0.2234_dt };

    Tensor realOut(BATCH_SIZE, DEPTH, HEIGHT, LIN_SIZE, { -0.3120_dt, 1.0290_dt, 0.3937_dt, 0.4863_dt, 1.5408_dt, -0.2452_dt, 1.0373_dt, 0.7068_dt, 0.7994_dt, 1.4122_dt,
                                                          0.1546_dt,  0.6165_dt, 0.4569_dt, 0.5495_dt, 0.9656_dt, 0.2022_dt,  0.5715_dt, 0.0744_dt, 0.1670_dt, 1.3219_dt });

    Tensor realInGrad(BATCH_SIZE, DEPTH, HEIGHT, MODEL_SIZE, { -0.0438_dt, 0.1801_dt, -0.0390_dt, -0.0028_dt, 0.3830_dt, -0.0828_dt, 0.2114_dt, -0.1093_dt, -0.0880_dt, 0.3210_dt,
                                                               -0.0687_dt, 0.3145_dt, -0.1089_dt, 0.1102_dt,  1.1691_dt, -0.1112_dt, 0.2029_dt, -0.1073_dt, -0.2367_dt, -0.2216_dt });

    Tensor realWeightGrad(1, 1, LIN_SIZE, MODEL_SIZE, { -3.0951_dt, 7.2950_dt,  5.5701_dt,  4.8715_dt,  5.2898_dt, -2.1553_dt, 11.0552_dt, 1.7749_dt, 4.7404_dt,
                                                        15.4022_dt, -3.0406_dt, 7.4858_dt,  3.8192_dt,  3.2663_dt, 4.5374_dt,  -2.3601_dt, 8.4283_dt, 6.2212_dt,
                                                        4.9974_dt,  10.4397_dt, -4.0017_dt, 16.0471_dt, 9.8539_dt, 10.2852_dt, 22.5297_dt });
    TENSORS_CREATE(BATCH_SIZE)
    memory_manager["in"] = TORANGE(in);
    memory_manager[Name("out").grad()] =
        TORANGE(Tensor({ 1.0_dt, 2.0_dt, -1.0_dt, 2.0_dt, 1.0_dt, 0.5_dt, 1.0_dt, 1.0_dt, 0.4_dt, 0.8_dt, 0.5_dt, 6.0_dt, 1.0_dt, 1.0_dt, 2.0_dt, 2.0_dt, -1.0_dt, 1.5_dt, -0.5_dt, 0.1_dt }));

    Tensor realBiasGrad(1, 1, 1, LIN_SIZE, { 3.9697_dt, 9.0458_dt, 3.0549_dt, 5.1412_dt, 11.1337_dt });

    memory_manager[layerName + "::Weights"] = TORANGE(weights);
    memory_manager[layerName + "::Biases"] = TORANGE(bias);

    work.forwardPassTraining();

    const Tensor& out = memory_manager["out"];
    EXPECT_EQ(out.getShape(), realOut.getShape());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], realOut[i], eps);
    }
    printf(" - Linear forward is Ok.\n");

    work.backwardPassTraining();

    auto& grad = memory_manager[Name("in").grad()];
    EXPECT_EQ(grad.getShape(), realInGrad.getShape());
    for (size_t i = 0; i < grad.size(); ++i)
    {
        EXPECT_NEAR(grad[i], realInGrad[i], eps);
    }

    auto& gradW = memory_manager[layerName + "::WeightsGradient"];

    EXPECT_EQ(gradW.getShape(), realWeightGrad.getShape());
    for (size_t i = 0; i < gradW.size(); ++i)
    {
        EXPECT_NEAR(gradW[i], realWeightGrad[i], eps);
    }

    auto& gradB = memory_manager[layerName + "::BiasesGradient"];
    EXPECT_EQ(gradB.getShape(), realBiasGrad.getShape());
    for (size_t i = 0; i < gradB.size(); ++i)
    {
        EXPECT_NEAR(gradB[i], realBiasGrad[i], eps);
    }

    printf(" - Linear backward is Ok.\n");
}

TEST_F(TestLinear, BiasesUnit)
{
    PROFILE_TEST
    using namespace raul;

    const size_t INPUT_SIZE = 2;
    const size_t OUTPUT_SIZE = 2;
    const dtype EPSILON = TODTYPE(1e-6);

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD);

    auto& memory_manager = work.getMemoryManager();
    NETWORK_PARAMS_DEFINE(networkParameters);

    Tensor input = { 10.0, 20.0 };

    work.add<raul::DataLayer>("data", DataParams{ { "in" }, 1, 1, INPUT_SIZE });
    work.add<LinearLayer>("fc1", LinearParams{ { "in" }, { "fc1" }, OUTPUT_SIZE, true });
    TENSORS_CREATE(1);
    memory_manager["in"] = TORANGE(input);

    memory_manager["fc1::Weights"] = TORANGE((Tensor{ 1.0f, 1.0f, 1.0f, 1.0f }));
    memory_manager["fc1::Biases"] = TORANGE((Tensor{ 1.0f, 1.0f }));
    ASSERT_NO_THROW(memory_manager["fc1::Biases"]);
    ASSERT_NO_THROW(memory_manager["fc1::BiasesGradient"]);

    ASSERT_NO_THROW(work.forwardPassTraining());

    EXPECT_EQ(memory_manager["fc1"].size(), static_cast<size_t>(OUTPUT_SIZE));
    CHECK_NEAR(memory_manager["fc1"][0], 31.0f, EPSILON);
    CHECK_NEAR(memory_manager["fc1"][1], 31.0f, EPSILON);

    memory_manager[Name("fc1").grad()] = TORANGE((Tensor{ 1.0f, 1.0f }));

    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST_F(TestLinear, Biases2Unit)
{
    PROFILE_TEST
    using namespace raul;

    const size_t INPUT_SIZE = 2;
    const size_t OUTPUT_SIZE = 2;
    const dtype EPSILON = TODTYPE(1e-6);

    WorkflowEager work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD);

    auto& memory_manager = work.getMemoryManager();
    NETWORK_PARAMS_DEFINE(networkParameters);

    Tensor input = { 10.0, 20.0 };

    work.add<raul::DataLayer>("data", DataParams{ { "in" }, 1, 1, INPUT_SIZE });
    work.add<LinearLayer>("fc1", LinearParams{ { "in" }, { "fc1" }, OUTPUT_SIZE, false });
    TENSORS_CREATE(1);
    memory_manager["in"] = TORANGE(input);

    memory_manager["fc1::Weights"] = TORANGE((Tensor{ 1.0f, 1.0f, 1.0f, 1.0f }));
    ASSERT_THROW(memory_manager["fc1::Biases"], raul::Exception);
    ASSERT_THROW(memory_manager["fc1::BiasesGradient"], raul::Exception);

    ASSERT_NO_THROW(work.forwardPassTraining());

    EXPECT_EQ(memory_manager["fc1"].size(), static_cast<size_t>(OUTPUT_SIZE));
    CHECK_NEAR(memory_manager["fc1"][0], 30.0f, EPSILON);
    CHECK_NEAR(memory_manager["fc1"][1], 30.0f, EPSILON);

    memory_manager[Name("fc1").grad()] = TORANGE((Tensor{ 1.0f, 1.0f }));

    ASSERT_NO_THROW(work.backwardPassTraining());
}

}