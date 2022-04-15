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

#include <training/base/common/MemoryManager.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/TransposeLayer.h>
#include <training/base/layers/basic/trainable/LinearLayer.h>
#include <training/compiler/Compiler.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerTrainableParamsConvert, CompilerImplicittFailToWrapNonTrainableLayerUnit)
{
    PROFILE_TEST

    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16, true);
    auto& compiler = work.getCompiler();
    work.add<raul::DataLayer>("in", raul::DataParams{ { "in" }, 1u, 1u, 1u });
    work.add<raul::TransposeLayer>("t", raul::TransposingParams{ "in", "out", raul::Dimension::Width, raul::Dimension::Height });
    compiler.setConstraint(raul::Constraint("t", raul::ConstraintImpl::CPUFP16FP32MasterWeights));
    EXPECT_THROW(work.preparePipelines(), raul::Exception);
}

TEST(TestLayerTrainableParamsConvert, CompilerImplicitWrapTrainableLayerUnit)
{
    PROFILE_TEST

    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16, true);
    auto& compiler = work.getCompiler();

    work.add<raul::DataLayer>("in", raul::DataParams{ { "in" }, 2u, 2u, 2u });
    work.add<raul::LinearLayer>("l", raul::LinearParams{ { "in" }, { "out" }, 1 });

    compiler.setConstraint(raul::Constraint("l", raul::ConstraintImpl::CPUFP16FP32MasterWeights));

    EXPECT_NO_THROW(work.preparePipelines());
    EXPECT_NO_THROW(work.prepareMemoryForTraining());
    EXPECT_NO_THROW(work.setBatchSize(1));

    // Check that needed weights exists
    const auto& memory_managerFP16 = work.getMemoryManager<MemoryManagerFP16>();
    const auto& memory_managerFP32 = work.getMemoryManager<MemoryManager>();

    // Initial
    EXPECT_TRUE(memory_managerFP16.tensorExists("l::Weights"));
    EXPECT_TRUE(memory_managerFP16.tensorExists("l::Biases"));

    // Initial_copies
    EXPECT_TRUE(memory_managerFP32.tensorExists("l::Weights_fp32"));
    EXPECT_TRUE(memory_managerFP32.tensorExists("l::Biases_fp32"));

    EXPECT_EQ(work.getTrainableParameterNames().size(), 4);

    ASSERT_NO_THROW(work.forwardPassTraining());
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST(TestLayerTrainableParamsConvert, CompilerImplicitForwardBackwardWithSharingUnit)
{
    PROFILE_TEST

    constexpr size_t MODEL_SIZE = 5;
    constexpr size_t BATCH_SIZE = 2;
    constexpr size_t DEPTH = 1;
    constexpr size_t HEIGHT = 2;
    constexpr raul::dtype EPS = 2e-2_dt;

    raul::TensorFP16 in{ 1.0_hf, 1.0_hf, 2.0_hf, 0.0_hf, 5.0_hf, -1.0_hf, 2.0_hf, 2.0_hf, 0.0_hf, 5.0_hf, -1.0_hf, 4.0_hf, 1.0_hf, 2.0_hf, 1.0_hf, -3.0_hf, 4.0_hf, 5.0_hf, 2.0_hf, 1.0_hf };
    raul::TensorFP16 weights{ -0.2381_hf, 0.1714_hf, -0.0612_hf, -0.1329_hf, -0.3701_hf, 0.0283_hf,  -0.2147_hf, -0.0502_hf, 0.2090_hf, 0.4333_hf, -0.1200_hf, 0.1664_hf, -0.3021_hf,
                              -0.2250_hf, 0.3329_hf, -0.1200_hf, 0.1664_hf,  -0.3021_hf, -0.2250_hf, 0.3329_hf,  0.1200_hf,  0.1664_hf, 0.3021_hf, 0.2250_hf,  0.3329_hf };
    raul::TensorFP16 biases{ 0.3548_hf, 0.2879_hf, 0.0343_hf, 0.1269_hf, 0.2234_hf };

    raul::TensorFP16 weightsGrad{ -0.0078125_hf, 0.585938_hf, -0.772461_hf, 1.5752_hf,   6.19922_hf, -3.15234_hf, 9.22656_hf,   7.08594_hf, 4.71875_hf,
                                  15.6328_hf,    0.325928_hf, -0.335938_hf, -1.60938_hf, 1.24219_hf, 5.19531_hf,  -0.730469_hf, 2.57227_hf, 1.03223_hf,
                                  2.29688_hf,    8.36719_hf,  -5.57812_hf,  15.8984_hf,  13.1484_hf, 7.14453_hf,  22.9062_hf };
    raul::TensorFP16 biasesGrad{ 2.67969_hf, 5.82812_hf, 2.34375_hf, 3.40625_hf, 8.25_hf };

    raul::Workflow work(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::CPUFP16, true);
    auto& compiler = work.getCompiler();

    work.add<raul::DataLayer>("in", raul::DataParams{ { "in" }, DEPTH, HEIGHT, MODEL_SIZE });
    // First
    work.add<raul::LinearLayer>("l1", raul::LinearParams{ { "in" }, { "out1" }, MODEL_SIZE });
    // Second
    work.add<raul::LinearLayer>("l2", raul::LinearParams{ { "out1" }, { "out2" }, "l1", MODEL_SIZE });

    compiler.setConstraint(raul::Constraint("l1", raul::ConstraintImpl::CPUFP16FP32MasterWeights));
    compiler.setConstraint(raul::Constraint("l2", raul::ConstraintImpl::CPUFP16FP32MasterWeights));

    EXPECT_NO_THROW(work.preparePipelines());
    EXPECT_NO_THROW(work.prepareMemoryForTraining());
    EXPECT_NO_THROW(work.setBatchSize(BATCH_SIZE));

    // Check that needed weights exists
    auto& memory_managerFP16 = work.getMemoryManager<MemoryManagerFP16>();
    auto& memory_managerFP32 = work.getMemoryManager<MemoryManager>();

    // Initial
    EXPECT_TRUE(memory_managerFP16.tensorExists("l1::Weights"));
    EXPECT_TRUE(memory_managerFP16.tensorExists("l1::Biases"));
    EXPECT_TRUE(!memory_managerFP16.tensorExists("l2::Weights"));
    EXPECT_TRUE(!memory_managerFP16.tensorExists("l2::Biases"));

    // Initial_copies
    EXPECT_TRUE(memory_managerFP32.tensorExists("l1::Weights_fp32"));
    EXPECT_TRUE(memory_managerFP32.tensorExists("l1::Biases_fp32"));
    EXPECT_TRUE(!memory_managerFP32.tensorExists("l2::Weights_fp32"));
    EXPECT_TRUE(!memory_managerFP32.tensorExists("l2::Biases_fp32"));

    EXPECT_EQ(work.getTrainableParameterNames().size(), 4);

    memory_managerFP16["l1::Weights"] = TORANGE_FP16(weights);
    memory_managerFP16["l1::Biases"] = TORANGE_FP16(biases);
    memory_managerFP16["in"] = TORANGE_FP16(in);

    EXPECT_NO_THROW(work.forwardPassTraining());

    // Check
    const auto& weightsFP32 = memory_managerFP32["l1::Weights_fp32"];
    EXPECT_EQ(weights.size(), weightsFP32.size());
    for (size_t i = 0; i < weightsFP32.size(); ++i)
    {
        EXPECT_NEAR(weights[i], weightsFP32[i], EPS);
    }

    const auto& biasesFP32 = memory_managerFP32["l1::Biases_fp32"];
    EXPECT_EQ(biases.size(), biasesFP32.size());
    for (size_t i = 0; i < biasesFP32.size(); ++i)
    {
        EXPECT_NEAR(biases[i], biasesFP32[i], EPS);
    }

    memory_managerFP16[Name("out2").grad()].memAllocate(nullptr);
    memory_managerFP16[Name("out2").grad()] = 1.0_hf;

    EXPECT_NO_THROW(work.backwardPassTraining());

    // Check
    const auto& weightsGradFP32 = memory_managerFP32[Name("l1::Weights_fp32").grad()];
    EXPECT_EQ(weightsGrad.size(), weightsGradFP32.size());
    for (size_t i = 0; i < weightsGradFP32.size(); ++i)
    {
        EXPECT_NEAR(weightsGrad[i], weightsGradFP32[i], EPS);
    }

    const auto& biasesGradFP32 = memory_managerFP32[Name("l1::Biases_fp32").grad()];
    EXPECT_EQ(biasesGrad.size(), biasesGradFP32.size());
    for (size_t i = 0; i < biasesGradFP32.size(); ++i)
    {
        EXPECT_NEAR(biasesGrad[i], biasesGradFP32[i], EPS);
    }
}

}