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
#include <training/base/layers/BasicLayer.h>

#include <training/compiler/Workflow.h>
#include <training/compiler/WorkflowActions.h>

#include "Test_WorkflowTools.h"

namespace
{

class TestLayerCompress : public raul::BasicLayer
{
  public:
    TestLayerCompress(const raul::Name& name, const raul::BasicParams& params, bool isCompress, raul::NetworkParameters& networkParameters)
        : BasicLayer(name, "test", params, networkParameters, { false, false })
    {
        for (auto& input : params.getInputs())
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, input, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
        }

        for (auto& output : params.getOutputs())
        {
            mNetworkParams.mWorkflow.tensorNeeded(
                name, output, raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Write, true, true, false, false, isCompress);
        }
    }

    void forwardComputeImpl(raul::NetworkMode) override {}
    void backwardComputeImpl() override {}
};

} // anonymous namespace

namespace UT
{

TEST(TestWorkflowCompression, SimpleTopologyCompressUnit)
{
    PROFILE_TEST

    {
        raul::Workflow w;

        w.add<TestLayerCompress>("f1", raul::BasicParams{ {}, { "x1" } }, true);
        w.add<TestLayerCompress>("f2", raul::BasicParams{ { "x1" }, { "x2" } }, false);
        w.add<TestLayerCompress>("f3", raul::BasicParams{ { "x1" }, { "x3" } }, false);
        w.add<TestLayerCompress>("f4", raul::BasicParams{ { "x2", "x3" }, {} }, false);

        w.preparePipelines();

        const raul::Workflow::Pipeline& pipeForwardTrain = w.getPipeline(raul::Workflow::Pipelines::ForwardTrain);
        ASSERT_EQ(pipeForwardTrain.size(), 7u);

        const raul::Workflow::Pipeline& pipeBackwardTrain = w.getPipeline(raul::Workflow::Pipelines::BackwardTrain);
        ASSERT_EQ(pipeBackwardTrain.size(), 7u);
    }

    {
        raul::Workflow w(raul::CompressionMode::FP16);

        w.add<TestLayerCompress>("f1", raul::BasicParams{ {}, { "x1" } }, true);
        w.add<TestLayerCompress>("f2", raul::BasicParams{ { "x1" }, { "x2" } }, false);
        w.add<TestLayerCompress>("f3", raul::BasicParams{ { "x1" }, { "x3" } }, false);
        w.add<TestLayerCompress>("f4", raul::BasicParams{ { "x2", "x3" }, {} }, false);

        w.preparePipelines();

        // forward train pipe
        {
            const raul::Workflow::Pipeline& pipeForwardTrain = w.getPipeline(raul::Workflow::Pipelines::ForwardTrain);

            ASSERT_EQ(pipeForwardTrain.size(), 8u);

            ASSERT_EQ(pipeForwardTrain[0]->type(), "Allocate"); // x1
            ASSERT_EQ(pipeForwardTrain[1]->type(), "Forward");  // f1
            ASSERT_EQ(pipeForwardTrain[2]->type(), "Allocate"); // x2
            ASSERT_EQ(pipeForwardTrain[3]->type(), "Forward");  // f2
            ASSERT_EQ(pipeForwardTrain[4]->type(), "Allocate"); // x3
            ASSERT_EQ(pipeForwardTrain[5]->type(), "Forward");  // f3
            ASSERT_EQ(pipeForwardTrain[6]->type(), "Compress"); // x1
            ASSERT_EQ(pipeForwardTrain[7]->type(), "Forward");  // f4

            ASSERT_TRUE(checkName(pipeForwardTrain[0].get(), "x1"));
            ASSERT_TRUE(checkName(pipeForwardTrain[1].get(), "f1"));
            ASSERT_TRUE(checkName(pipeForwardTrain[2].get(), "x2"));
            ASSERT_TRUE(checkName(pipeForwardTrain[3].get(), "f2"));
            ASSERT_TRUE(checkName(pipeForwardTrain[4].get(), "x3"));
            ASSERT_TRUE(checkName(pipeForwardTrain[5].get(), "f3"));
            ASSERT_TRUE(checkName(pipeForwardTrain[6].get(), "x1"));
            ASSERT_TRUE(checkName(pipeForwardTrain[7].get(), "f4"));
        }

        // backward train pipe
        {
            const raul::Workflow::Pipeline& pipeBackwardTrain = w.getPipeline(raul::Workflow::Pipelines::BackwardTrain);

            ASSERT_EQ(pipeBackwardTrain.size(), 8u);

            ASSERT_EQ(pipeBackwardTrain[0]->type(), "Backward");   // f4
            ASSERT_EQ(pipeBackwardTrain[1]->type(), "Deallocate"); // x2
            ASSERT_EQ(pipeBackwardTrain[2]->type(), "Deallocate"); // x3
            ASSERT_EQ(pipeBackwardTrain[3]->type(), "Decompress"); // x1
            ASSERT_EQ(pipeBackwardTrain[4]->type(), "Backward");   // f3
            ASSERT_EQ(pipeBackwardTrain[5]->type(), "Backward");   // f2
            ASSERT_EQ(pipeBackwardTrain[6]->type(), "Deallocate"); // x1
            ASSERT_EQ(pipeBackwardTrain[7]->type(), "Backward");   // f1

            ASSERT_TRUE(checkName(pipeBackwardTrain[0].get(), "f4"));
            ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[1].get()), getName(pipeBackwardTrain[2].get()) }, { "x2", "x3" }));
            ASSERT_TRUE(checkName(pipeBackwardTrain[3].get(), "x1"));
            ASSERT_TRUE(checkName(pipeBackwardTrain[4].get(), "f3"));
            ASSERT_TRUE(checkName(pipeBackwardTrain[5].get(), "f2"));
            ASSERT_TRUE(checkName(pipeBackwardTrain[6].get(), "x1"));
            ASSERT_TRUE(checkName(pipeBackwardTrain[7].get(), "f1"));
        }
    }
}

TEST(TestWorkflowCompression, ComplexTopologyCompressUnit)
{
    PROFILE_TEST

    raul::Workflow w(raul::CompressionMode::FP16);

    w.add<TestLayerCompress>("f1", raul::BasicParams{ {}, { "x1" } }, true);
    w.add<TestLayerCompress>("f2", raul::BasicParams{ { "x1" }, { "x2" } }, true);
    w.add<TestLayerCompress>("f3", raul::BasicParams{ { "x1" }, { "x3" } }, true);
    w.add<TestLayerCompress>("f4", raul::BasicParams{ { "x1" }, { "x4" } }, true);
    w.add<TestLayerCompress>("f5", raul::BasicParams{ { "x2", "x3" }, { "x5" } }, false);
    w.add<TestLayerCompress>("f6", raul::BasicParams{ { "x2" }, { "x6" } }, true);
    w.add<TestLayerCompress>("f7", raul::BasicParams{ { "x4", "x5", "x6" }, { "x7" } }, true);
    w.add<TestLayerCompress>("f8",
                             raul::BasicParams{
                                 { "x7" },
                                 {},
                             },
                             false);

    w.preparePipelines();

    // forward train pipe
    {
        const raul::Workflow::Pipeline& pipeForwardTrain = w.getPipeline(raul::Workflow::Pipelines::ForwardTrain);

        ASSERT_EQ(pipeForwardTrain.size(), 21u);

        ASSERT_EQ(pipeForwardTrain[0]->type(), "Allocate");  // x1
        ASSERT_EQ(pipeForwardTrain[1]->type(), "Forward");   // f1
        ASSERT_EQ(pipeForwardTrain[2]->type(), "Allocate");  // x2
        ASSERT_EQ(pipeForwardTrain[3]->type(), "Forward");   // f2
        ASSERT_EQ(pipeForwardTrain[4]->type(), "Allocate");  // x3
        ASSERT_EQ(pipeForwardTrain[5]->type(), "Forward");   // f3
        ASSERT_EQ(pipeForwardTrain[6]->type(), "Allocate");  // x4
        ASSERT_EQ(pipeForwardTrain[7]->type(), "Forward");   // f4
        ASSERT_EQ(pipeForwardTrain[8]->type(), "Compress");  // x1
        ASSERT_EQ(pipeForwardTrain[9]->type(), "Allocate");  // x5
        ASSERT_EQ(pipeForwardTrain[10]->type(), "Forward");  // f5
        ASSERT_EQ(pipeForwardTrain[11]->type(), "Compress"); // x3
        ASSERT_EQ(pipeForwardTrain[12]->type(), "Allocate"); // x6
        ASSERT_EQ(pipeForwardTrain[13]->type(), "Forward");  // f6
        ASSERT_EQ(pipeForwardTrain[14]->type(), "Compress"); // x2
        ASSERT_EQ(pipeForwardTrain[15]->type(), "Allocate"); // x7
        ASSERT_EQ(pipeForwardTrain[16]->type(), "Forward");  // f7
        ASSERT_EQ(pipeForwardTrain[17]->type(), "Compress"); // x4
        ASSERT_EQ(pipeForwardTrain[18]->type(), "Compress"); // x6
        ASSERT_EQ(pipeForwardTrain[19]->type(), "Forward");  // f8
        ASSERT_EQ(pipeForwardTrain[20]->type(), "Compress"); // x7

        ASSERT_TRUE(checkName(pipeForwardTrain[0].get(), "x1"));
        ASSERT_TRUE(checkName(pipeForwardTrain[1].get(), "f1"));
        ASSERT_TRUE(checkName(pipeForwardTrain[2].get(), "x2"));
        ASSERT_TRUE(checkName(pipeForwardTrain[3].get(), "f2"));
        ASSERT_TRUE(checkName(pipeForwardTrain[4].get(), "x3"));
        ASSERT_TRUE(checkName(pipeForwardTrain[5].get(), "f3"));
        ASSERT_TRUE(checkName(pipeForwardTrain[6].get(), "x4"));
        ASSERT_TRUE(checkName(pipeForwardTrain[7].get(), "f4"));
        ASSERT_TRUE(checkName(pipeForwardTrain[8].get(), "x1"));
        ASSERT_TRUE(checkName(pipeForwardTrain[9].get(), "x5"));
        ASSERT_TRUE(checkName(pipeForwardTrain[10].get(), "f5"));
        ASSERT_TRUE(checkName(pipeForwardTrain[11].get(), "x3"));
        ASSERT_TRUE(checkName(pipeForwardTrain[12].get(), "x6"));
        ASSERT_TRUE(checkName(pipeForwardTrain[13].get(), "f6"));
        ASSERT_TRUE(checkName(pipeForwardTrain[14].get(), "x2"));
        ASSERT_TRUE(checkName(pipeForwardTrain[15].get(), "x7"));
        ASSERT_TRUE(checkName(pipeForwardTrain[16].get(), "f7"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeForwardTrain[17].get()), getName(pipeForwardTrain[18].get()) }, { "x4", "x6" }));
        ASSERT_TRUE(checkName(pipeForwardTrain[19].get(), "f8"));
        ASSERT_TRUE(checkName(pipeForwardTrain[20].get(), "x7"));
    }

    // backward train pipe
    {
        const raul::Workflow::Pipeline& pipeBackwardTrain = w.getPipeline(raul::Workflow::Pipelines::BackwardTrain);

        ASSERT_EQ(pipeBackwardTrain.size(), 21u);

        ASSERT_EQ(pipeBackwardTrain[0]->type(), "Decompress");  // x7
        ASSERT_EQ(pipeBackwardTrain[1]->type(), "Backward");    // f8
        ASSERT_EQ(pipeBackwardTrain[2]->type(), "Deallocate");  // x7
        ASSERT_EQ(pipeBackwardTrain[3]->type(), "Decompress");  // x4
        ASSERT_EQ(pipeBackwardTrain[4]->type(), "Decompress");  // x6
        ASSERT_EQ(pipeBackwardTrain[5]->type(), "Backward");    // f7
        ASSERT_EQ(pipeBackwardTrain[6]->type(), "Deallocate");  // x4
        ASSERT_EQ(pipeBackwardTrain[7]->type(), "Deallocate");  // x5
        ASSERT_EQ(pipeBackwardTrain[8]->type(), "Deallocate");  // x6
        ASSERT_EQ(pipeBackwardTrain[9]->type(), "Decompress");  // x2
        ASSERT_EQ(pipeBackwardTrain[10]->type(), "Backward");   // f6
        ASSERT_EQ(pipeBackwardTrain[11]->type(), "Decompress"); // x3
        ASSERT_EQ(pipeBackwardTrain[12]->type(), "Backward");   // f5
        ASSERT_EQ(pipeBackwardTrain[13]->type(), "Deallocate"); // x3
        ASSERT_EQ(pipeBackwardTrain[14]->type(), "Deallocate"); // x2
        ASSERT_EQ(pipeBackwardTrain[15]->type(), "Decompress"); // x1
        ASSERT_EQ(pipeBackwardTrain[16]->type(), "Backward");   // f4
        ASSERT_EQ(pipeBackwardTrain[17]->type(), "Backward");   // f3
        ASSERT_EQ(pipeBackwardTrain[18]->type(), "Backward");   // f2
        ASSERT_EQ(pipeBackwardTrain[19]->type(), "Deallocate"); // x1
        ASSERT_EQ(pipeBackwardTrain[20]->type(), "Backward");   // f1

        ASSERT_TRUE(checkName(pipeBackwardTrain[0].get(), "x7"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[1].get(), "f8"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[2].get(), "x7"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[3].get()), getName(pipeBackwardTrain[4].get()) }, { "x4", "x6" }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[5].get(), "f7"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[6].get()), getName(pipeBackwardTrain[7].get()), getName(pipeBackwardTrain[8].get()) }, { "x4", "x5", "x6" }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[9].get(), "x2"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[10].get(), "f6"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[11].get(), "x3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[12].get(), "f5"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[13].get()), getName(pipeBackwardTrain[14].get()) }, { "x2", "x3" }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[15].get(), "x1"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[16].get()), getName(pipeBackwardTrain[17].get()), getName(pipeBackwardTrain[18].get()) }, { "f4", "f3", "f2" }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[19].get(), "x1"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[20].get(), "f1"));
    }
}

TEST(TestWorkflowCompression, SimpleTopologyCompressCheckpointedUnit)
{
    PROFILE_TEST

    raul::Workflow w(raul::CompressionMode::FP16);

    w.add<TestLayerCompress>("f1", raul::BasicParams{ {}, { "x1" } }, true);
    w.add<TestLayerCompress>("f2", raul::BasicParams{ { "x1" }, { "x2" } }, false);
    w.add<TestLayerCompress>("f3", raul::BasicParams{ { "x1" }, { "x3" } }, false);
    w.add<TestLayerCompress>("f4", raul::BasicParams{ { "x2", "x3" }, {} }, true);

    // compression not possible together with checkpoints
    EXPECT_THROW(w.preparePipelines(raul::Workflow::Execution::Checkpointed), raul::Exception);
}

} // UT namespace