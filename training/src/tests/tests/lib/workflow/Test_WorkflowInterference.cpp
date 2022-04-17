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
#include <training/compiler/Workflow.h>

#include <tests/tools/callbacks/TensorChecker.h>

namespace UT
{

struct TestWorkflowInterference : public testing::Test
{
    using il = std::initializer_list<raul::dtype>;

    const size_t BATCH = 1U;
    const size_t DEPTH = 2U;
    const size_t HEIGHT = 1U;
    const size_t WIDTH = 4U;
    const size_t LINSIZE = 2U;
    const raul::dtype SFACTOR = 3.0_dt;
    const raul::dtype eps = 1.0e-6_dt;

    std::unique_ptr<raul::Tensor> Input;
    std::unique_ptr<raul::Tensor> Weights1;
    std::unique_ptr<raul::Tensor> Weights2;
    std::unique_ptr<raul::Tensor> RealOut;
    std::unique_ptr<raul::Tensor> RealInGrad;

    void SetUp() final
    {
        Input = std::make_unique<raul::Tensor>(BATCH, DEPTH, HEIGHT, WIDTH, il{ 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 6.0_dt, 7.0_dt, 8.0_dt });
        Weights1 = std::make_unique<raul::Tensor>(1U, 1U, WIDTH, LINSIZE, il{ 0.0_dt, 0.5_dt, 1.5_dt, 2.0_dt, 2.5_dt, 3.5_dt, 4.5_dt, 5.0_dt });
        Weights2 = std::make_unique<raul::Tensor>(1U, 1U, LINSIZE, LINSIZE, il{ 4.0_dt, 3.0_dt, 2.0_dt, 1.0_dt });
        RealOut = std::make_unique<raul::Tensor>(1U, 1U, 1U, 1U, il{ 637.5_dt });
        RealInGrad = std::make_unique<raul::Tensor>(BATCH, DEPTH, HEIGHT, WIDTH, il{ 7.5_dt, 12.75_dt, 20.25_dt, 24.0_dt, 7.5_dt, 12.75_dt, 20.25_dt, 24.0_dt });
    }
};

TEST_F(TestWorkflowInterference, DirectCalculationUnit)
{
    raul::Workflow work;

    work.add<raul::DataLayer>("data", raul::DataParams{ { "in", "realInGradient" }, DEPTH, HEIGHT, WIDTH });
    work.add<raul::LinearLayer>("linear1", raul::LinearParams{ { "in" }, { "linear1_out" }, LINSIZE });
    work.add<raul::ScaleLayer>("scale", raul::ScaleParams{ { "linear1_out" }, { "scale_out" }, SFACTOR });
    work.add<raul::LinearLayer>("linear2", raul::LinearParams{ { "scale_out" }, { "linear2_out" }, LINSIZE });
    work.add<raul::ReduceMeanLayer>("mean", raul::BasicParamsWithDim{ { "linear2_out" }, { "out" } });
    work.add<raul::DataLayer>("dataOut", raul::DataParams{ { "realOut" }, 1U, 1U, 1U });

    work.preparePipelines();
    work.setBatchSize(BATCH);
    work.prepareMemoryForTraining();

    auto& memory_manager = work.getMemoryManager();
    memory_manager["in"] = TORANGE(*Input);
    memory_manager["linear1::Weights"] = TORANGE(*Weights1);
    memory_manager["linear2::Weights"] = TORANGE(*Weights2);
    memory_manager["outGradient"].memAllocate(nullptr);
    memory_manager["outGradient"] = 1.0_dt;
    memory_manager["realOut"] = TORANGE(*RealOut);
    memory_manager["realInGradient"] = TORANGE(*RealInGrad);

    tools::callbacks::TensorChecker checker{ { { "out", "realOut" } }, { { "inGradient", "realInGradient" } }, eps };
    auto& networkParameters = work.getNetworkParameters();
    networkParameters.mCallback = checker;

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());
    // Backward
    ASSERT_NO_THROW(work.backwardPassTraining());
}

TEST_F(TestWorkflowInterference, ExternalCalculationThroughCallbackUnit)
{
    raul::Workflow work;

    work.add<raul::DataLayer>("data", raul::DataParams{ { "in", "realInGradient" }, DEPTH, HEIGHT, WIDTH });
    work.add<raul::DataLayer>("dataOut", raul::DataParams{ { "realOut" }, 1U, 1U, 1U });
    work.add<raul::LinearLayer>("linear1", raul::LinearParams{ { "in" }, { "linear1_out" }, LINSIZE });
    work.add<raul::DataLayer>("external_data", raul::DataParams{ { "external_out" }, DEPTH, HEIGHT, LINSIZE });
    work.add<raul::LinearLayer>("linear2", raul::LinearParams{ { "external_out" }, { "linear2_out" }, LINSIZE });
    work.add<raul::ReduceMeanLayer>("mean", raul::BasicParamsWithDim{ { "linear2_out" }, { "out" } });

    work.preparePipelines();
    work.setBatchSize(BATCH);
    work.prepareMemoryForTraining();

    auto& memory_manager = work.getMemoryManager();
    memory_manager["in"] = TORANGE(*Input);
    memory_manager["linear1::Weights"] = TORANGE(*Weights1);
    memory_manager["linear2::Weights"] = TORANGE(*Weights2);
    memory_manager["outGradient"].memAllocate(nullptr);
    memory_manager["outGradient"] = 1.0_dt;
    memory_manager["realOut"] = TORANGE(*RealOut);
    memory_manager["realInGradient"] = TORANGE(*RealInGrad);

    auto afterForward = [](raul::BasicLayer* layer, raul::MemoryManager& mem) {
        if (layer->getName() == "linear1")
        {
            // Scale forward
            mem["external_out"] = TORANGE(mem[layer->getOutputs()[0]]);
            mem["external_out"] *= 3.0_dt;
        }
        else if (layer->getName() == "mean")
        {
            EXPECT_EQ(mem["out"].size(), mem["realOut"].size());
            EXPECT_EQ(mem["out"][0], mem["realOut"][0]);
        }
    };

    auto afterBackward = [](raul::BasicLayer* layer, raul::MemoryManager& mem) {
        if (layer->getName() == "linear2")
        {
            // Scale backward
            mem["linear1_outGradient"].memAllocate(nullptr);
            mem["linear1_outGradient"] = TORANGE(mem[layer->getInputs()[0].grad()]);
            mem["linear1_outGradient"] *= 3.0_dt;
        }
        else if (layer->getName() == "linear1")
        {
            EXPECT_EQ(mem["inGradient"].size(), mem["realInGradient"].size());
            for (size_t i = 0; i < mem["realInGradient"].size(); ++i)
            {
                EXPECT_EQ(mem["inGradient"][i], mem["realInGradient"][i]);
            }
        }
    };

    auto& networkParameters = work.getNetworkParameters();
    networkParameters.mCallback = raul::CallbackHelper(std::nullopt, afterForward, std::nullopt, afterBackward);

    // Forward
    ASSERT_NO_THROW(work.forwardPassTraining());
    // Backward
    ASSERT_NO_THROW(work.backwardPassTraining());
}

}