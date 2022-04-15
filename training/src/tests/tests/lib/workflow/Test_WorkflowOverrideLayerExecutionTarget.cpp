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
#include <training/base/common/Conversions.h>
#include <training/base/common/MemoryManager.h>

#include <training/compiler/Workflow.h>

#include <training/compiler/Layers.h>
#include <training/base/layers/basic/ConvertPrecisionLayer.h>

#include <training/base/optimizers/SGD.h>

namespace
{

class TestLayerOverride : public raul::BasicLayer
{
  public:
    TestLayerOverride(const raul::Name& name, const raul::BasicParams& params, std::function<void()> onForward, std::function<void()> onBackward, raul::NetworkParameters& networkParameters)
        : BasicLayer(name, "test", params, networkParameters, { false, false }),
        mOnForward(onForward),
        mOnBackward(onBackward)
    {

        if (mInputs.size() != 1)
        {
            THROW(mTypeName, mName, "wrong number of input names");
        }
        if (mOutputs.size() != 1)
        {
            THROW(mTypeName, mName, "wrong number of output names");
        }

        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mInputs[0], mInputs[0].grad(), DEC_BACK_WRIT_ZERO);

        mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputs[0], WShape{ BS(), 1u, 1u, 1u}, DEC_FORW_WRIT_COMP);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);

        mLayerExecutionTarget = mNetworkParams.mWorkflow.getOverrideLayerExecutionTarget();
    }

    void forwardComputeImpl(raul::NetworkMode) override
    {
        mOnForward();

        auto executionTarget = mNetworkParams.mWorkflow.getExecutionTarget();

        if (mLayerExecutionTarget != raul::LayerExecutionTarget::Default)
        {
            executionTarget = static_cast<raul::ExecutionTarget>(mLayerExecutionTarget);
        }

        if (executionTarget == raul::ExecutionTarget::CPU)
        {
            mNetworkParams.mWorkflow.getMemoryManager<MemoryManager>()[mOutputs[0]] = TORANGE(mNetworkParams.mWorkflow.getMemoryManager<MemoryManager>()[mInputs[0]]);
        }
        else if (executionTarget == raul::ExecutionTarget::CPUFP16)
        {
            mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()[mOutputs[0]] = TORANGE_FP16(mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()[mInputs[0]]);
        }
        else
        {
            THROW(mTypeName, mName, "Target not supported");
        }
    }

    void backwardComputeImpl() override
    {
        mOnBackward();

        auto executionTarget = mNetworkParams.mWorkflow.getExecutionTarget();

        if (mLayerExecutionTarget != raul::LayerExecutionTarget::Default)
        {
            executionTarget = static_cast<raul::ExecutionTarget>(mLayerExecutionTarget);
        }

        if (executionTarget == raul::ExecutionTarget::CPU)
        {
            mNetworkParams.mWorkflow.getMemoryManager<MemoryManager>()[mInputs[0].grad()] = TORANGE(mNetworkParams.mWorkflow.getMemoryManager<MemoryManager>()[mOutputs[0].grad()]);
        }
        else if (executionTarget == raul::ExecutionTarget::CPUFP16)
        {
            mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()[mInputs[0].grad()] = TORANGE_FP16(mNetworkParams.mWorkflow.getMemoryManager<MemoryManagerFP16>()[mOutputs[0].grad()]);
        }
        else
        {
            THROW(mTypeName, mName, "Target not supported");
        }
    }

  private:

      std::function<void()> mOnForward;
      std::function<void()> mOnBackward;

      raul::LayerExecutionTarget mLayerExecutionTarget;

};

} //anonymous

namespace UT
{

TEST(TestWorkflowOverrideLayerExecutionTarget, OverrideUnit)
{
    PROFILE_TEST

    {
        Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::CPU);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "data"}, 1u, 1u, 1u });
        work.add<TestLayerOverride>("o1", raul::BasicParams{ { "data"}, {"o1"} }, [](){}, [&]()
            {
                ASSERT_EQ(work.getMemoryManager<MemoryManager>()[raul::Name("o1").grad()][0], 33_dt);
            });

        EXPECT_THROW(work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::Default), raul::Exception);

        work.add<raul::ConvertPrecisionLayer>("converter", raul::ConvertPrecisionParams{ { "o1"}, { "o1_fp16" }, raul::LayerExecutionTarget::Default, raul::LayerExecutionTarget::CPUFP16 });

        work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::CPUFP16);
        EXPECT_THROW(work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::Default), raul::Exception);
        EXPECT_THROW(work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::CPU), raul::Exception);
        ASSERT_TRUE(work.getOverrideLayerExecutionTarget() == raul::LayerExecutionTarget::CPUFP16);

        work.add<TestLayerOverride>("o2", raul::BasicParams{ { "o1_fp16"}, {"o2_fp16"} }, [](){}, [](){});

        //reset before pipelines preparation
        EXPECT_THROW(work.preparePipelines(), raul::Exception);

        work.resetLayerExecutionTargetOverride();
        ASSERT_TRUE(work.getOverrideLayerExecutionTarget() == raul::LayerExecutionTarget::Default);

        EXPECT_THROW(work.add<raul::ConvertPrecisionLayer>("converter2", raul::ConvertPrecisionParams{ { "o2_fp16"}, { "o2"}, raul::LayerExecutionTarget::Default, raul::LayerExecutionTarget::Default }), raul::Exception);
        work.add<raul::ConvertPrecisionLayer>("converter2", raul::ConvertPrecisionParams{ { "o2_fp16"}, { "o2"}, raul::LayerExecutionTarget::CPUFP16, raul::LayerExecutionTarget::Default });

        work.add<TestLayerOverride>("o3", raul::BasicParams{ { "o2"}, {"o3"} }, [](){}, [&]()
            {
                work.getMemoryManager<MemoryManager>()[raul::Name("o3").grad()][0] = 33_dt;
                ASSERT_EQ(work.getMemoryManager<MemoryManager>()["o2"][0], 12_dt);
            });

        work.preparePipelines();
        work.setBatchSize(1u);
        work.prepareMemoryForTraining();

        work.getMemoryManager<MemoryManager>()["data"][0] = 12_dt;

        ASSERT_TRUE(work.getMemoryManager<MemoryManager>().tensorExists("data"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManager>().tensorExists("o1"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManagerFP16>().tensorExists("o1_fp16"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManagerFP16>().tensorExists("o2_fp16"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManager>().tensorExists("o2"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManager>().tensorExists("o3"));

        work.forwardPassTraining();
        work.backwardPassTraining();
    }

    {
        Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::STANDARD, ExecutionTarget::CPUFP16);

        work.add<raul::DataLayer>("data", raul::DataParams{ { "data"}, 1u, 1u, 1u });
        work.add<TestLayerOverride>("o1", raul::BasicParams{ { "data"}, {"o1"} }, [](){}, [&]()
            {
                ASSERT_EQ(work.getMemoryManager<MemoryManagerFP16>()[raul::Name("o1").grad()][0], 33_hf);
            });

        work.add<raul::ConvertPrecisionLayer>("converter", raul::ConvertPrecisionParams{ { "o1"}, { "o1_fp32"}, raul::LayerExecutionTarget::Default, raul::LayerExecutionTarget::CPU });

        work.overrideLayerExecutionTarget(raul::LayerExecutionTarget::CPU);
        ASSERT_TRUE(work.getOverrideLayerExecutionTarget() == raul::LayerExecutionTarget::CPU);

        work.add<TestLayerOverride>("o2", raul::BasicParams{ { "o1_fp32"}, {"o2_fp32"} }, [](){}, [](){});

        work.resetLayerExecutionTargetOverride();
        ASSERT_TRUE(work.getOverrideLayerExecutionTarget() == raul::LayerExecutionTarget::Default);

        work.add<raul::ConvertPrecisionLayer>("converter2", raul::ConvertPrecisionParams{ { "o2_fp32"}, { "o2"}, raul::LayerExecutionTarget::CPU, raul::LayerExecutionTarget::Default });

        work.add<TestLayerOverride>("o3", raul::BasicParams{ { "o2"}, {"o3"} }, [](){}, [&]()
            {
                work.getMemoryManager<MemoryManagerFP16>()[raul::Name("o3").grad()][0] = 33_hf;
                ASSERT_EQ(work.getMemoryManager<MemoryManagerFP16>()["o2"][0], 12_hf);
            });

        work.preparePipelines();
        work.setBatchSize(1u);
        work.prepareMemoryForTraining();

        work.getMemoryManager<MemoryManagerFP16>()["data"][0] = 12_hf;

        ASSERT_TRUE(work.getMemoryManager<MemoryManagerFP16>().tensorExists("data"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManagerFP16>().tensorExists("o1"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManager>().tensorExists("o1_fp32"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManager>().tensorExists("o2_fp32"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManagerFP16>().tensorExists("o2"));
        ASSERT_TRUE(work.getMemoryManager<MemoryManagerFP16>().tensorExists("o3"));

        work.forwardPassTraining();
        work.backwardPassTraining();
    }
}

} // UT namespace