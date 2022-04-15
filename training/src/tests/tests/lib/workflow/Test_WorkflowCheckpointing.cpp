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
#include <training/base/optimizers/SGD.h>

#include "Test_WorkflowTools.h"

namespace UT
{

TEST(TestWorkflowCheckpointing, PreparePipelinesSimpleTopologyCheckpointedUnit)
{
    PROFILE_TEST

    raul::Workflow w;

    w.add<TestLayer>("f0", raul::BasicParams{ {}, { "x1" } }, true);
    w.add<TestLayer>("f1", raul::BasicParams{ { "x1" }, { "x2" } }, true);
    w.add<TestLayer>("f2", raul::BasicParams{ { "x2" }, { "x3" } }, true);
    w.add<TestLayer>("f3", raul::BasicParams{ { "x3" }, {} }, true);

    EXPECT_NO_THROW(w.preparePipelines());

    // forward train pipe
    {
        const raul::Workflow::Pipeline& pipeForwardTrain = w.getPipeline(raul::Workflow::Pipelines::ForwardTrain);

        ASSERT_EQ(pipeForwardTrain.size(), 7u);

        ASSERT_EQ(pipeForwardTrain[0]->type(), "Allocate"); // x1
        ASSERT_EQ(pipeForwardTrain[1]->type(), "Forward");  // f0
        ASSERT_EQ(pipeForwardTrain[2]->type(), "Allocate"); // x2
        ASSERT_EQ(pipeForwardTrain[3]->type(), "Forward");  // f1
        ASSERT_EQ(pipeForwardTrain[4]->type(), "Allocate"); // x3
        ASSERT_EQ(pipeForwardTrain[5]->type(), "Forward");  // f2
        ASSERT_EQ(pipeForwardTrain[6]->type(), "Forward");  // f3
    }

    EXPECT_THROW(w.preparePipelines(raul::Workflow::Execution::Checkpointed), raul::Exception);

    EXPECT_NO_THROW(w.setCheckpoints({}));
    EXPECT_THROW(w.setCheckpoints({ "x1", "x2", "w1" }), raul::Exception);

    EXPECT_THROW(w.setCheckpoints({ "x1", "x1" }), raul::Exception);

    EXPECT_NO_THROW(w.setCheckpoints({ "x1" }));
    EXPECT_NO_THROW(w.preparePipelines(raul::Workflow::Execution::Checkpointed));

    // forward train pipe
    {
        const raul::Workflow::Pipeline& pipeForwardTrain = w.getPipeline(raul::Workflow::Pipelines::ForwardTrain);

        ASSERT_EQ(pipeForwardTrain.size(), 9u);

        ASSERT_EQ(pipeForwardTrain[0]->type(), "Allocate");   // x1
        ASSERT_EQ(pipeForwardTrain[1]->type(), "Forward");    // f0
        ASSERT_EQ(pipeForwardTrain[2]->type(), "Allocate");   // x2
        ASSERT_EQ(pipeForwardTrain[3]->type(), "Forward");    // f1
        ASSERT_EQ(pipeForwardTrain[4]->type(), "Allocate");   // x3
        ASSERT_EQ(pipeForwardTrain[5]->type(), "Forward");    // f2
        ASSERT_EQ(pipeForwardTrain[6]->type(), "Deallocate"); // x2
        ASSERT_EQ(pipeForwardTrain[7]->type(), "Forward");    // f3
        ASSERT_EQ(pipeForwardTrain[8]->type(), "Deallocate"); // x3

        ASSERT_TRUE(checkName(pipeForwardTrain[0].get(), "x1"));
        ASSERT_TRUE(checkName(pipeForwardTrain[1].get(), "f0"));
        ASSERT_TRUE(checkName(pipeForwardTrain[2].get(), "x2"));
        ASSERT_TRUE(checkName(pipeForwardTrain[3].get(), "f1"));
        ASSERT_TRUE(checkName(pipeForwardTrain[4].get(), "x3"));
        ASSERT_TRUE(checkName(pipeForwardTrain[5].get(), "f2"));
        ASSERT_TRUE(checkName(pipeForwardTrain[6].get(), "x2"));
        ASSERT_TRUE(checkName(pipeForwardTrain[7].get(), "f3"));
        ASSERT_TRUE(checkName(pipeForwardTrain[8].get(), "x3"));
    }

    // backward train pipe
    {
        const raul::Workflow::Pipeline& pipeBackwardTrain = w.getPipeline(raul::Workflow::Pipelines::BackwardTrain);

        ASSERT_EQ(pipeBackwardTrain.size(), 23u);

        ASSERT_TRUE(checkBlocksType(pipeBackwardTrain,
                                    0,
                                    6,
                                    { {
                                          "Allocate", // xg3
                                          "Zero"      // xg3
                                      },
                                      {
                                          "Allocate",  // x2
                                          "Forward",   // f1
                                          "Allocate",  // x3
                                          "Forward",   // f2
                                          "Deallocate" // x2
                                      } }));
        ASSERT_EQ(pipeBackwardTrain[7]->type(), "Backward");   // f3
        ASSERT_EQ(pipeBackwardTrain[8]->type(), "Deallocate"); // x3
        ASSERT_TRUE(checkBlocksType(pipeBackwardTrain,
                                    9,
                                    12,
                                    { {
                                          "Allocate", // xg2
                                          "Zero"      // xg2
                                      },
                                      {
                                          "Allocate", // x2
                                          "Forward"   // f1
                                      } }));
        ASSERT_EQ(pipeBackwardTrain[13]->type(), "Backward");   // f2
        ASSERT_EQ(pipeBackwardTrain[14]->type(), "Deallocate"); // x2
        ASSERT_EQ(pipeBackwardTrain[15]->type(), "Deallocate"); // xg3
        ASSERT_EQ(pipeBackwardTrain[16]->type(), "Allocate");   // xg1
        ASSERT_EQ(pipeBackwardTrain[17]->type(), "Zero");       // xg1
        ASSERT_EQ(pipeBackwardTrain[18]->type(), "Backward");   // f1
        ASSERT_EQ(pipeBackwardTrain[19]->type(), "Deallocate"); // xg2
        ASSERT_EQ(pipeBackwardTrain[20]->type(), "Backward");   // f0
        ASSERT_EQ(pipeBackwardTrain[21]->type(), "Deallocate"); // x1
        ASSERT_EQ(pipeBackwardTrain[22]->type(), "Deallocate"); // xg1

        ASSERT_TRUE(checkBlocksName(pipeBackwardTrain, 0, 6, { { getGradName("x3"), getGradName("x3") }, { "x2", "f1", "x3", "f2", "x2" } }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[7].get(), "f3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[8].get(), "x3"));
        ASSERT_TRUE(checkBlocksName(pipeBackwardTrain, 9, 12, { { getGradName("x2"), getGradName("x2") }, { "x2", "f1" } }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[13].get(), "f2"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[14].get()), getName(pipeBackwardTrain[15].get()) }, { "x2", getGradName("x3") }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[16].get(), getGradName("x1")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[17].get(), getGradName("x1")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[18].get(), "f1"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[19].get(), getGradName("x2")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[20].get(), "f0"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[21].get()), getName(pipeBackwardTrain[22].get()) }, { "x1", getGradName("x1") }));
    }

    EXPECT_NO_THROW(w.setCheckpoints({ "x1", "x2", "x3" }));
    EXPECT_NO_THROW(w.preparePipelines(raul::Workflow::Execution::Checkpointed));

    // forward train pipe
    {
        const raul::Workflow::Pipeline& pipeForwardTrain = w.getPipeline(raul::Workflow::Pipelines::ForwardTrain);

        ASSERT_EQ(pipeForwardTrain.size(), 7u);

        ASSERT_EQ(pipeForwardTrain[0]->type(), "Allocate"); // x1
        ASSERT_EQ(pipeForwardTrain[1]->type(), "Forward");  // f0
        ASSERT_EQ(pipeForwardTrain[2]->type(), "Allocate"); // x2
        ASSERT_EQ(pipeForwardTrain[3]->type(), "Forward");  // f1
        ASSERT_EQ(pipeForwardTrain[4]->type(), "Allocate"); // x3
        ASSERT_EQ(pipeForwardTrain[5]->type(), "Forward");  // f2
        ASSERT_EQ(pipeForwardTrain[6]->type(), "Forward");  // f3
    }

    // backward train pipe
    {
        const raul::Workflow::Pipeline& pipeBackwardTrain = w.getPipeline(raul::Workflow::Pipelines::BackwardTrain);

        ASSERT_EQ(pipeBackwardTrain.size(), 16u);

        ASSERT_EQ(pipeBackwardTrain[0]->type(), "Allocate");    // xg3
        ASSERT_EQ(pipeBackwardTrain[1]->type(), "Zero");        // xg3
        ASSERT_EQ(pipeBackwardTrain[2]->type(), "Backward");    // f3
        ASSERT_EQ(pipeBackwardTrain[3]->type(), "Allocate");    // xg2
        ASSERT_EQ(pipeBackwardTrain[4]->type(), "Zero");        // xg2
        ASSERT_EQ(pipeBackwardTrain[5]->type(), "Backward");    // f2
        ASSERT_EQ(pipeBackwardTrain[6]->type(), "Deallocate");  // x3
        ASSERT_EQ(pipeBackwardTrain[7]->type(), "Deallocate");  // xg3
        ASSERT_EQ(pipeBackwardTrain[8]->type(), "Allocate");    // xg1
        ASSERT_EQ(pipeBackwardTrain[9]->type(), "Zero");        // xg1
        ASSERT_EQ(pipeBackwardTrain[10]->type(), "Backward");   // f1
        ASSERT_EQ(pipeBackwardTrain[11]->type(), "Deallocate"); // x2
        ASSERT_EQ(pipeBackwardTrain[12]->type(), "Deallocate"); // xg2
        ASSERT_EQ(pipeBackwardTrain[13]->type(), "Backward");   // f0
        ASSERT_EQ(pipeBackwardTrain[14]->type(), "Deallocate"); // x1
        ASSERT_EQ(pipeBackwardTrain[15]->type(), "Deallocate"); // xg1
    }

    EXPECT_NO_THROW(w.preparePipelines());

    // forward train pipe
    {
        const raul::Workflow::Pipeline& pipeForwardTrain = w.getPipeline(raul::Workflow::Pipelines::ForwardTrain);

        ASSERT_EQ(pipeForwardTrain.size(), 7u);

        ASSERT_EQ(pipeForwardTrain[0]->type(), "Allocate"); // x1
        ASSERT_EQ(pipeForwardTrain[1]->type(), "Forward");  // f0
        ASSERT_EQ(pipeForwardTrain[2]->type(), "Allocate"); // x2
        ASSERT_EQ(pipeForwardTrain[3]->type(), "Forward");  // f1
        ASSERT_EQ(pipeForwardTrain[4]->type(), "Allocate"); // x3
        ASSERT_EQ(pipeForwardTrain[5]->type(), "Forward");  // f2
        ASSERT_EQ(pipeForwardTrain[6]->type(), "Forward");  // f3
    }
}

TEST(TestWorkflowCheckpointing, PreparePipelinesComplexTopologyCheckpointedUnit)
{
    PROFILE_TEST

    raul::Workflow w;

    w.add<TestLayer>("f0", raul::BasicParams{ {}, { "x1" } }, true);
    w.add<TestLayer>("f1", raul::BasicParams{ { "x1" }, { "x2" }, { "w1", "w2" } }, true);
    w.add<TestLayer>("f2", raul::BasicParams{ { "x1" }, { "x3" } }, true);
    w.add<TestLayer>("f3", raul::BasicParams{ { "x1" }, { "x4" }, raul::Names{ "w1" } }, true);
    w.add<TestLayer>("f4", raul::BasicParams{ { "x2", "x3" }, { "x5" } }, true);
    w.add<TestLayer>("f5", raul::BasicParams{ { "x2" }, { "x6" } }, true);
    w.add<TestLayer>("f6", raul::BasicParams{ { "x4", "x5", "x6" }, { "x7" } }, true);
    w.add<TestLayer>("f7", raul::BasicParams{ { "x7" }, {}, raul::Names{ "w2" } }, true);

    EXPECT_NO_THROW(w.setCheckpoints({ "x1" }));

    EXPECT_NO_THROW(w.preparePipelines(raul::Workflow::Execution::Checkpointed));

    // forward train pipe
    {
        const raul::Workflow::Pipeline& pipeForwardTrain = w.getPipeline(raul::Workflow::Pipelines::ForwardTrain);

        ASSERT_EQ(pipeForwardTrain.size(), 21u);

        ASSERT_EQ(pipeForwardTrain[0]->type(), "Allocate");    // x1
        ASSERT_EQ(pipeForwardTrain[1]->type(), "Forward");     // f0
        ASSERT_EQ(pipeForwardTrain[2]->type(), "Allocate");    // x2
        ASSERT_EQ(pipeForwardTrain[3]->type(), "Forward");     // f1
        ASSERT_EQ(pipeForwardTrain[4]->type(), "Allocate");    // x3
        ASSERT_EQ(pipeForwardTrain[5]->type(), "Forward");     // f2
        ASSERT_EQ(pipeForwardTrain[6]->type(), "Allocate");    // x4
        ASSERT_EQ(pipeForwardTrain[7]->type(), "Forward");     // f3
        ASSERT_EQ(pipeForwardTrain[8]->type(), "Allocate");    // x5
        ASSERT_EQ(pipeForwardTrain[9]->type(), "Forward");     // f4
        ASSERT_EQ(pipeForwardTrain[10]->type(), "Deallocate"); // x3
        ASSERT_EQ(pipeForwardTrain[11]->type(), "Allocate");   // x6
        ASSERT_EQ(pipeForwardTrain[12]->type(), "Forward");    // f5
        ASSERT_EQ(pipeForwardTrain[13]->type(), "Deallocate"); // x2
        ASSERT_EQ(pipeForwardTrain[14]->type(), "Allocate");   // x7
        ASSERT_EQ(pipeForwardTrain[15]->type(), "Forward");    // f6
        ASSERT_EQ(pipeForwardTrain[16]->type(), "Deallocate"); // x4
        ASSERT_EQ(pipeForwardTrain[17]->type(), "Deallocate"); // x5
        ASSERT_EQ(pipeForwardTrain[18]->type(), "Deallocate"); // x6
        ASSERT_EQ(pipeForwardTrain[19]->type(), "Forward");    // f7
        ASSERT_EQ(pipeForwardTrain[20]->type(), "Deallocate"); // x7

        ASSERT_TRUE(checkName(pipeForwardTrain[0].get(), "x1"));
        ASSERT_TRUE(checkName(pipeForwardTrain[1].get(), "f0"));
        ASSERT_TRUE(checkName(pipeForwardTrain[2].get(), "x2"));
        ASSERT_TRUE(checkName(pipeForwardTrain[3].get(), "f1"));
        ASSERT_TRUE(checkName(pipeForwardTrain[4].get(), "x3"));
        ASSERT_TRUE(checkName(pipeForwardTrain[5].get(), "f2"));
        ASSERT_TRUE(checkName(pipeForwardTrain[6].get(), "x4"));
        ASSERT_TRUE(checkName(pipeForwardTrain[7].get(), "f3"));
        ASSERT_TRUE(checkName(pipeForwardTrain[8].get(), "x5"));
        ASSERT_TRUE(checkName(pipeForwardTrain[9].get(), "f4"));
        ASSERT_TRUE(checkName(pipeForwardTrain[10].get(), "x3"));
        ASSERT_TRUE(checkName(pipeForwardTrain[11].get(), "x6"));
        ASSERT_TRUE(checkName(pipeForwardTrain[12].get(), "f5"));
        ASSERT_TRUE(checkName(pipeForwardTrain[13].get(), "x2"));
        ASSERT_TRUE(checkName(pipeForwardTrain[14].get(), "x7"));
        ASSERT_TRUE(checkName(pipeForwardTrain[15].get(), "f6"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeForwardTrain[16].get()), getName(pipeForwardTrain[17].get()), getName(pipeForwardTrain[18].get()) }, { "x4", "x5", "x6" }));
        ASSERT_TRUE(checkName(pipeForwardTrain[19].get(), "f7"));
        ASSERT_TRUE(checkName(pipeForwardTrain[20].get(), "x7"));
    }

    // backward train pipe
    {
        const raul::Workflow::Pipeline& pipeBackwardTrain = w.getPipeline(raul::Workflow::Pipelines::BackwardTrain);

        ASSERT_EQ(pipeBackwardTrain.size(), 72u);

        ASSERT_TRUE(checkBlocksType(pipeBackwardTrain,
                                    0,
                                    18,
                                    { {
                                          "Allocate", // xg7
                                          "Zero"      // xg7
                                      },
                                      {
                                          // x7 forward
                                          "Allocate",   // x2
                                          "Forward",    // f1
                                          "Allocate",   // x6
                                          "Forward",    // f5
                                          "Allocate",   // x3
                                          "Forward",    // f2
                                          "Allocate",   // x5
                                          "Forward",    // f4
                                          "Deallocate", // x2
                                          "Deallocate", // x3
                                          "Allocate",   // x4
                                          "Forward",    // f3
                                          "Allocate",   // x7
                                          "Forward",    // f6
                                          "Deallocate", // x4
                                          "Deallocate", // x5
                                          "Deallocate"  // x6
                                      } }));
        ASSERT_EQ(pipeBackwardTrain[19]->type(), "Backward");   // f7
        ASSERT_EQ(pipeBackwardTrain[20]->type(), "Deallocate"); // x7
        ASSERT_TRUE(checkBlocksType(pipeBackwardTrain,
                                    21,
                                    41,
                                    { {
                                          "Allocate", // xg4
                                          "Zero",     // xg4
                                      },
                                      {
                                          "Allocate", // xg5
                                          "Zero",     // xg5
                                      },
                                      {
                                          "Allocate", // xg6
                                          "Zero"      // xg6
                                      },
                                      {
                                          // x4 forward
                                          "Allocate", // x4
                                          "Forward"   // f3
                                      },
                                      {
                                          // x5 forward
                                          "Allocate",   // x3
                                          "Forward",    // f2
                                          "Allocate",   // x2
                                          "Forward",    // f1
                                          "Allocate",   // x5
                                          "Forward",    // f4
                                          "Deallocate", // x2
                                          "Deallocate"  // x3
                                      },
                                      {
                                          // x6 forward
                                          "Allocate",  // x2
                                          "Forward",   // f1
                                          "Allocate",  // x6
                                          "Forward",   // f5
                                          "Deallocate" // x2
                                      } }));
        ASSERT_EQ(pipeBackwardTrain[42]->type(), "Backward");   // f6
        ASSERT_EQ(pipeBackwardTrain[43]->type(), "Deallocate"); // x4
        ASSERT_EQ(pipeBackwardTrain[44]->type(), "Deallocate"); // x5
        ASSERT_EQ(pipeBackwardTrain[45]->type(), "Deallocate"); // x6
        ASSERT_EQ(pipeBackwardTrain[46]->type(), "Deallocate"); // xg7
        ASSERT_TRUE(checkBlocksType(pipeBackwardTrain,
                                    47,
                                    50,
                                    { {
                                          "Allocate", // xg2
                                          "Zero"      // xg2
                                      },
                                      {
                                          // x2 forward
                                          "Allocate", // x2
                                          "Forward",  // f1
                                      } }));
        ASSERT_EQ(pipeBackwardTrain[51]->type(), "Backward");   // f5
        ASSERT_EQ(pipeBackwardTrain[52]->type(), "Deallocate"); // xg6
        ASSERT_TRUE(checkBlocksType(pipeBackwardTrain,
                                    53,
                                    56,
                                    { {
                                          "Allocate", // xg3
                                          "Zero"      // xg3
                                      },
                                      {
                                          // x3 forward
                                          "Allocate", // x3
                                          "Forward",  // f2
                                      } }));
        ASSERT_EQ(pipeBackwardTrain[57]->type(), "Backward");   // f4
        ASSERT_EQ(pipeBackwardTrain[58]->type(), "Deallocate"); // x2
        ASSERT_EQ(pipeBackwardTrain[59]->type(), "Deallocate"); // x3
        ASSERT_EQ(pipeBackwardTrain[60]->type(), "Deallocate"); // xg5
        ASSERT_EQ(pipeBackwardTrain[61]->type(), "Allocate");   // xg1
        ASSERT_EQ(pipeBackwardTrain[62]->type(), "Zero");       // xg1
        ASSERT_EQ(pipeBackwardTrain[63]->type(), "Backward");   // f3
        ASSERT_EQ(pipeBackwardTrain[64]->type(), "Deallocate"); // xg4
        ASSERT_EQ(pipeBackwardTrain[65]->type(), "Backward");   // f2
        ASSERT_EQ(pipeBackwardTrain[66]->type(), "Deallocate"); // xg3
        ASSERT_EQ(pipeBackwardTrain[67]->type(), "Backward");   // f1
        ASSERT_EQ(pipeBackwardTrain[68]->type(), "Deallocate"); // xg2
        ASSERT_EQ(pipeBackwardTrain[69]->type(), "Backward");   // f0
        ASSERT_EQ(pipeBackwardTrain[70]->type(), "Deallocate"); // x1
        ASSERT_EQ(pipeBackwardTrain[71]->type(), "Deallocate"); // xg1

        ASSERT_TRUE(checkBlocksName(pipeBackwardTrain,
                                    0,
                                    18,
                                    { { getGradName("x7"), getGradName("x7") },
                                      { // x7 forward
                                        "x2",
                                        "f1",
                                        "x6",
                                        "f5",
                                        "x3",
                                        "f2",
                                        "x5",
                                        "f4",
                                        "x2",
                                        "x3",
                                        "x4",
                                        "f3",
                                        "x7",
                                        "f6" },
                                      { // x7 forward more
                                        "x4" },
                                      { // x7 forward more
                                        "x5" },
                                      { // x7 forward more
                                        "x6" } }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[19].get(), "f7"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[20].get(), "x7"));
        ASSERT_TRUE(checkBlocksName(pipeBackwardTrain,
                                    21,
                                    41,
                                    { { getGradName("x4"), getGradName("x4") },
                                      { getGradName("x5"), getGradName("x5") },
                                      { getGradName("x6"), getGradName("x6") },
                                      { // x4 forward
                                        "x4",
                                        "f3" },
                                      { // x5 forward
                                        "x3",
                                        "f2",
                                        "x2",
                                        "f1",
                                        "x5",
                                        "f4" },
                                      { // x5 forward more
                                        "x2" },
                                      { // x5 forward more
                                        "x3" },
                                      { // x6 forward
                                        "x2",
                                        "f1",
                                        "x6",
                                        "f5",
                                        "x2" } }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[42].get(), "f6"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[43].get()), getName(pipeBackwardTrain[44].get()), getName(pipeBackwardTrain[45].get()), getName(pipeBackwardTrain[46].get()) },
                                     { "x4", "x5", "x6", getGradName("x7") }));
        ASSERT_TRUE(checkBlocksName(pipeBackwardTrain,
                                    47,
                                    50,
                                    { { getGradName("x2"), getGradName("x2") },
                                      { // x2 forward
                                        "x2",
                                        "f1" } }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[51].get(), "f5"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[52].get(), getGradName("x6")));
        ASSERT_TRUE(checkBlocksName(pipeBackwardTrain,
                                    53,
                                    56,
                                    { { getGradName("x3"), getGradName("x3") },
                                      { // x3 forward
                                        "x3",
                                        "f2" } }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[57].get(), "f4"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[58].get()), getName(pipeBackwardTrain[59].get()), getName(pipeBackwardTrain[60].get()) }, { "x2", "x3", getGradName("x5") }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[61].get(), getGradName("x1")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[62].get(), getGradName("x1")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[63].get(), "f3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[64].get(), getGradName("x4")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[65].get(), "f2"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[66].get(), getGradName("x3")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[67].get(), "f1"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[68].get(), getGradName("x2")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[69].get(), "f0"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[70].get()), getName(pipeBackwardTrain[71].get()) }, { "x1", getGradName("x1") }));
    }

    EXPECT_NO_THROW(w.setCheckpoints({ "x1", "x3" }));

    EXPECT_NO_THROW(w.preparePipelines(raul::Workflow::Execution::Checkpointed));

    // backward train pipe
    {
        const raul::Workflow::Pipeline& pipeBackwardTrain = w.getPipeline(raul::Workflow::Pipelines::BackwardTrain);

        ASSERT_EQ(pipeBackwardTrain.size(), 64u);

        ASSERT_TRUE(checkBlocksType(pipeBackwardTrain,
                                    0,
                                    15,
                                    { {
                                          "Allocate", // xg7
                                          "Zero"      // xg7
                                      },
                                      {
                                          // x7 forward
                                          "Allocate",   // x2
                                          "Forward",    // f1
                                          "Allocate",   // x6
                                          "Forward",    // f5
                                          "Allocate",   // x5
                                          "Forward",    // f4
                                          "Deallocate", // x2
                                          "Allocate",   // x4
                                          "Forward",    // f3
                                          "Allocate",   // x7
                                          "Forward",    // f6
                                          "Deallocate", // x4
                                          "Deallocate", // x5
                                          "Deallocate"  // x6
                                      } }));
        ASSERT_EQ(pipeBackwardTrain[16]->type(), "Backward");   // f7
        ASSERT_EQ(pipeBackwardTrain[17]->type(), "Deallocate"); // x7
        ASSERT_TRUE(checkBlocksType(pipeBackwardTrain,
                                    18,
                                    35,
                                    { {
                                          "Allocate", // xg4
                                          "Zero",     // xg4
                                      },
                                      {
                                          "Allocate", // xg5
                                          "Zero",     // xg5
                                      },
                                      {
                                          "Allocate", // xg6
                                          "Zero"      // xg6
                                      },
                                      {
                                          // x4 forward
                                          "Allocate", // x4
                                          "Forward"   // f3
                                      },
                                      {
                                          // x5 forward
                                          "Allocate",  // x2
                                          "Forward",   // f1
                                          "Allocate",  // x5
                                          "Forward",   // f4
                                          "Deallocate" // x2
                                      },
                                      {
                                          // x6 forward
                                          "Allocate",  // x2
                                          "Forward",   // f1
                                          "Allocate",  // x6
                                          "Forward",   // f5
                                          "Deallocate" // x2
                                      } }));
        ASSERT_EQ(pipeBackwardTrain[36]->type(), "Backward");   // f6
        ASSERT_EQ(pipeBackwardTrain[37]->type(), "Deallocate"); // x4
        ASSERT_EQ(pipeBackwardTrain[38]->type(), "Deallocate"); // x5
        ASSERT_EQ(pipeBackwardTrain[39]->type(), "Deallocate"); // x6
        ASSERT_EQ(pipeBackwardTrain[40]->type(), "Deallocate"); // xg7
        ASSERT_TRUE(checkBlocksType(pipeBackwardTrain,
                                    41,
                                    44,
                                    { {
                                          "Allocate", // xg2
                                          "Zero"      // xg2
                                      },
                                      {
                                          // x2 forward
                                          "Allocate", // x2
                                          "Forward",  // f1
                                      } }));
        ASSERT_EQ(pipeBackwardTrain[45]->type(), "Backward");   // f5
        ASSERT_EQ(pipeBackwardTrain[46]->type(), "Deallocate"); // xg6
        ASSERT_EQ(pipeBackwardTrain[47]->type(), "Allocate");   // xg3
        ASSERT_EQ(pipeBackwardTrain[48]->type(), "Zero");       // xg3
        ASSERT_EQ(pipeBackwardTrain[49]->type(), "Backward");   // f4
        ASSERT_EQ(pipeBackwardTrain[50]->type(), "Deallocate"); // x2
        ASSERT_EQ(pipeBackwardTrain[51]->type(), "Deallocate"); // xg5
        ASSERT_EQ(pipeBackwardTrain[52]->type(), "Allocate");   // xg1
        ASSERT_EQ(pipeBackwardTrain[53]->type(), "Zero");       // xg1
        ASSERT_EQ(pipeBackwardTrain[54]->type(), "Backward");   // f3
        ASSERT_EQ(pipeBackwardTrain[55]->type(), "Deallocate"); // xg4
        ASSERT_EQ(pipeBackwardTrain[56]->type(), "Backward");   // f2
        ASSERT_EQ(pipeBackwardTrain[57]->type(), "Deallocate"); // x3
        ASSERT_EQ(pipeBackwardTrain[58]->type(), "Deallocate"); // xg3
        ASSERT_EQ(pipeBackwardTrain[59]->type(), "Backward");   // f1
        ASSERT_EQ(pipeBackwardTrain[60]->type(), "Deallocate"); // xg2
        ASSERT_EQ(pipeBackwardTrain[61]->type(), "Backward");   // f0
        ASSERT_EQ(pipeBackwardTrain[62]->type(), "Deallocate"); // x1
        ASSERT_EQ(pipeBackwardTrain[63]->type(), "Deallocate"); // xg1

        ASSERT_TRUE(checkBlocksName(pipeBackwardTrain,
                                    0,
                                    15,
                                    { { getGradName("x7"), getGradName("x7") },
                                      { // x7 forward
                                        "x2",
                                        "f1",
                                        "x6",
                                        "f5",
                                        "x5",
                                        "f4",
                                        "x2",
                                        "x4",
                                        "f3",
                                        "x7",
                                        "f6" },
                                      { // x7 forward more
                                        "x4" },
                                      { // x7 forward more
                                        "x5" },
                                      { // x7 forward more
                                        "x6" } }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[16].get(), "f7"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[17].get(), "x7"));
        ASSERT_TRUE(checkBlocksName(pipeBackwardTrain,
                                    18,
                                    35,
                                    { { getGradName("x4"), getGradName("x4") },
                                      { getGradName("x5"), getGradName("x5") },
                                      { getGradName("x6"), getGradName("x6") },
                                      { // x4 forward
                                        "x4",
                                        "f3" },
                                      { // x5 forward
                                        "x2",
                                        "f1",
                                        "x5",
                                        "f4" },
                                      { // x5 forward more
                                        "x2" },
                                      { // x6 forward
                                        "x2",
                                        "f1",
                                        "x6",
                                        "f5",
                                        "x2" } }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[36].get(), "f6"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[37].get()), getName(pipeBackwardTrain[38].get()), getName(pipeBackwardTrain[39].get()), getName(pipeBackwardTrain[40].get()) },
                                     { "x4", "x5", "x6", getGradName("x7") }));
        ASSERT_TRUE(checkBlocksName(pipeBackwardTrain,
                                    41,
                                    44,
                                    { { getGradName("x2"), getGradName("x2") },
                                      { // x2 forward
                                        "x2",
                                        "f1" } }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[45].get(), "f5"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[46].get(), getGradName("x6")));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[47].get()), getName(pipeBackwardTrain[48].get()) }, { getGradName("x3") }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[49].get(), "f4"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[50].get()), getName(pipeBackwardTrain[51].get()) }, { "x2", getGradName("x5") }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[52].get(), getGradName("x1")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[53].get(), getGradName("x1")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[54].get(), "f3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[55].get(), getGradName("x4")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[56].get(), "f2"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[57].get()), getName(pipeBackwardTrain[58].get()) }, { "x3", getGradName("x3") }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[59].get(), "f1"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[60].get(), getGradName("x2")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[61].get(), "f0"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[62].get()), getName(pipeBackwardTrain[63].get()) }, { "x1", getGradName("x1") }));
    }
}

TEST(TestWorkflowCheckpointing, PreparePipelinesForwardBackwardTensorInTopologyCheckpointedUnit)
{
    PROFILE_TEST

    class Test : public raul::BasicLayer
    {
      public:
        Test(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
            : BasicLayer(name, "test", params, networkParameters, { false, false })
        {
            if (!params.getInputs().empty())
            {
                mNetworkParams.mWorkflow.copyDeclaration(name, params.getInputs()[0], raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
            }

            if (!params.getOutputs().empty())
            {
                mNetworkParams.mWorkflow.tensorNeeded(
                    name, params.getOutputs()[0], raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, true, true, false, false, false);
            }

            mNetworkParams.mWorkflow.tensorNeeded(mName, mName / "XHat", raul::WShape{ raul::BS(), 1u, 1u, 1u }, DEC_FORW_WRIT);
            mNetworkParams.mWorkflow.copyDeclaration(mName, mName / "XHat", DEC_BACK_READ);
        }

        void forwardComputeImpl(raul::NetworkMode) override {}
        void backwardComputeImpl() override {}
    };

    raul::Workflow w;

    w.add<Test>("f1", raul::BasicParams{ {}, { "x1" } });
    w.add<Test>("f2", raul::BasicParams{ { "x1" }, { "x2" } });
    w.add<Test>("f3", raul::BasicParams{ { "x2" }, { "x3" } });

    EXPECT_NO_THROW(w.setCheckpoints({ "x1" }));

    EXPECT_NO_THROW(w.preparePipelines(raul::Workflow::Execution::Checkpointed));

    // forward train pipe
    {
        const raul::Workflow::Pipeline& pipeForwardTrain = w.getPipeline(raul::Workflow::Pipelines::ForwardTrain);

        ASSERT_EQ(pipeForwardTrain.size(), 11u);

        ASSERT_EQ(pipeForwardTrain[0]->type(), "Allocate");    // x1
        ASSERT_EQ(pipeForwardTrain[1]->type(), "Allocate");    // xh1
        ASSERT_EQ(pipeForwardTrain[2]->type(), "Forward");     // f1
        ASSERT_EQ(pipeForwardTrain[3]->type(), "Allocate");    // x2
        ASSERT_EQ(pipeForwardTrain[4]->type(), "Allocate");    // xh2
        ASSERT_EQ(pipeForwardTrain[5]->type(), "Forward");     // f2
        ASSERT_EQ(pipeForwardTrain[6]->type(), "Allocate");    // x3
        ASSERT_EQ(pipeForwardTrain[7]->type(), "Allocate");    // xh3
        ASSERT_EQ(pipeForwardTrain[8]->type(), "Forward");     // f3
        ASSERT_EQ(pipeForwardTrain[9]->type(), "Deallocate");  // x2
        ASSERT_EQ(pipeForwardTrain[10]->type(), "Deallocate"); // x3

        ASSERT_TRUE(checkGroupedName({ getName(pipeForwardTrain[0].get()), getName(pipeForwardTrain[1].get()) }, { "x1", "f1::XHat" }));
        ASSERT_TRUE(checkName(pipeForwardTrain[2].get(), "f1"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeForwardTrain[3].get()), getName(pipeForwardTrain[4].get()) }, { "x2", "f2::XHat" }));
        ASSERT_TRUE(checkName(pipeForwardTrain[5].get(), "f2"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeForwardTrain[6].get()), getName(pipeForwardTrain[7].get()) }, { "x3", "f3::XHat" }));
        ASSERT_TRUE(checkName(pipeForwardTrain[8].get(), "f3"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeForwardTrain[9].get()), getName(pipeForwardTrain[10].get()) }, { "x2", "x3" }));
    }

    // backward train pipe
    {
        const raul::Workflow::Pipeline& pipeBackwardTrain = w.getPipeline(raul::Workflow::Pipelines::BackwardTrain);

        ASSERT_EQ(pipeBackwardTrain.size(), 10u);

        ASSERT_EQ(pipeBackwardTrain[0]->type(), "Allocate");   // x2
        ASSERT_EQ(pipeBackwardTrain[1]->type(), "Forward");    // f2
        ASSERT_EQ(pipeBackwardTrain[2]->type(), "Backward");   // f3
        ASSERT_EQ(pipeBackwardTrain[3]->type(), "Deallocate"); // xh3
        ASSERT_EQ(pipeBackwardTrain[4]->type(), "Deallocate"); // x2
        ASSERT_EQ(pipeBackwardTrain[5]->type(), "Backward");   // f2
        ASSERT_EQ(pipeBackwardTrain[6]->type(), "Deallocate"); // xh2
        ASSERT_EQ(pipeBackwardTrain[7]->type(), "Backward");   // f1
        ASSERT_EQ(pipeBackwardTrain[8]->type(), "Deallocate"); // x1
        ASSERT_EQ(pipeBackwardTrain[9]->type(), "Deallocate"); // xh1

        ASSERT_TRUE(checkName(pipeBackwardTrain[0].get(), "x2"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[1].get(), "f2"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[2].get(), "f3"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[3].get()), getName(pipeBackwardTrain[4].get()) }, { "x2", "f3::XHat" }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[5].get(), "f2"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[6].get(), "f2::XHat"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[7].get(), "f1"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[8].get()), getName(pipeBackwardTrain[9].get()) }, { "x1", "f1::XHat" }));
    }
}

TEST(TestWorkflowCheckpointing, PreparePipelinesSkipConnectionCheckpointedUnit)
{
    PROFILE_TEST

    class Test : public raul::BasicLayer
    {
      public:
        Test(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
            : BasicLayer(name, "test", params, networkParameters, { false, false })
        {
            for (auto& input : params.getInputs())
            {
                mNetworkParams.mWorkflow.copyDeclaration(name, input, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
            }

            if (!params.getOutputs().empty())
            {
                mNetworkParams.mWorkflow.tensorNeeded(
                    name, params.getOutputs()[0], raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, true, true, false, false, false);
            }
        }

        void forwardComputeImpl(raul::NetworkMode) override {}
        void backwardComputeImpl() override {}
    };

    raul::Workflow w;

    w.add<Test>("f1", raul::BasicParams{ {}, { "x1" } });
    w.add<Test>("f2", raul::BasicParams{ { "x1" }, { "x2" } });
    w.add<Test>("f3", raul::BasicParams{ { "x2" }, { "x3" } });
    w.add<Test>("f4", raul::BasicParams{ { "x2", "x3" }, {} });

    EXPECT_NO_THROW(w.setCheckpoints({ "x1" }));

    EXPECT_NO_THROW(w.preparePipelines(raul::Workflow::Execution::Checkpointed));

    // forward train pipe
    {
        const raul::Workflow::Pipeline& pipeForwardTrain = w.getPipeline(raul::Workflow::Pipelines::ForwardTrain);

        ASSERT_EQ(pipeForwardTrain.size(), 9u);

        ASSERT_EQ(pipeForwardTrain[0]->type(), "Allocate");   // x1
        ASSERT_EQ(pipeForwardTrain[1]->type(), "Forward");    // f1
        ASSERT_EQ(pipeForwardTrain[2]->type(), "Allocate");   // x2
        ASSERT_EQ(pipeForwardTrain[3]->type(), "Forward");    // f2
        ASSERT_EQ(pipeForwardTrain[4]->type(), "Allocate");   // x3
        ASSERT_EQ(pipeForwardTrain[5]->type(), "Forward");    // f3
        ASSERT_EQ(pipeForwardTrain[6]->type(), "Forward");    // f4
        ASSERT_EQ(pipeForwardTrain[7]->type(), "Deallocate"); // x2
        ASSERT_EQ(pipeForwardTrain[8]->type(), "Deallocate"); // x3

        ASSERT_TRUE(checkName(pipeForwardTrain[0].get(), "x1"));
        ASSERT_TRUE(checkName(pipeForwardTrain[1].get(), "f1"));
        ASSERT_TRUE(checkName(pipeForwardTrain[2].get(), "x2"));
        ASSERT_TRUE(checkName(pipeForwardTrain[3].get(), "f2"));
        ASSERT_TRUE(checkName(pipeForwardTrain[4].get(), "x3"));
        ASSERT_TRUE(checkName(pipeForwardTrain[5].get(), "f3"));
        ASSERT_TRUE(checkName(pipeForwardTrain[6].get(), "f4"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeForwardTrain[7].get()), getName(pipeForwardTrain[8].get()) }, { "x2", "x3" }));
    }

    // backward train pipe
    {
        const raul::Workflow::Pipeline& pipeBackwardTrain = w.getPipeline(raul::Workflow::Pipelines::BackwardTrain);

        ASSERT_EQ(pipeBackwardTrain.size(), 11u);

        ASSERT_EQ(pipeBackwardTrain[0]->type(), "Allocate");    // x2
        ASSERT_EQ(pipeBackwardTrain[1]->type(), "Forward");     // f2
        ASSERT_EQ(pipeBackwardTrain[2]->type(), "Allocate");    // x3
        ASSERT_EQ(pipeBackwardTrain[3]->type(), "Forward");     // f3
        ASSERT_EQ(pipeBackwardTrain[4]->type(), "Backward");    // f4
        ASSERT_EQ(pipeBackwardTrain[5]->type(), "Deallocate");  // x3
        ASSERT_EQ(pipeBackwardTrain[6]->type(), "Backward");    // f3
        ASSERT_EQ(pipeBackwardTrain[7]->type(), "Deallocate");  // x2
        ASSERT_EQ(pipeBackwardTrain[8]->type(), "Backward");    // f2
        ASSERT_EQ(pipeBackwardTrain[9]->type(), "Backward");    // f1
        ASSERT_EQ(pipeBackwardTrain[10]->type(), "Deallocate"); // x1

        ASSERT_TRUE(checkName(pipeBackwardTrain[0].get(), "x2"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[1].get(), "f2"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[2].get(), "x3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[3].get(), "f3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[4].get(), "f4"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[5].get(), "x3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[6].get(), "f3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[7].get(), "x2"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[8].get()), getName(pipeBackwardTrain[9].get()) }, { "f1", "f2" }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[10].get(), "x1"));
    }
}

TEST(TestWorkflowCheckpointing, PreparePipelinesKeepActivationCheckpointedUnit)
{
    PROFILE_TEST

    class Test : public raul::BasicLayer
    {
      public:
        Test(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
            : BasicLayer(name, "test", params, networkParameters, { false, false })
        {
            for (auto& input : params.getInputs())
            {
                mNetworkParams.mWorkflow.copyDeclaration(name, input, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
            }

            if (!params.getOutputs().empty())
            {
                mNetworkParams.mWorkflow.tensorNeeded(
                    name, params.getOutputs()[0], raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, true, true, false, false, false);
            }
        }

        void forwardComputeImpl(raul::NetworkMode) override {}
        void backwardComputeImpl() override {}
    };

    raul::Workflow w;

    w.add<Test>("f1", raul::BasicParams{ {}, { "x1" } });
    w.add<Test>("f2", raul::BasicParams{ { "x1" }, { "x2" } });
    w.add<Test>("f3", raul::BasicParams{ { "x2" }, { "x3" } });
    w.add<Test>("f4", raul::BasicParams{ { "x2", "x3" }, { "x4" } });
    w.add<Test>("f5", raul::BasicParams{ { "x2", "x4" }, {} });

    {
        raul::Names checkpoints = w.getPotentialCheckpoints();
        EXPECT_EQ(checkpoints.size(), 4u);

        ASSERT_TRUE(checkGroupedName(checkpoints, { "x1", "x2", "x3", "x4" }));
    }

    EXPECT_NO_THROW(w.setCheckpoints({ "x1" }));

    EXPECT_NO_THROW(w.preparePipelines(raul::Workflow::Execution::Checkpointed));

    // backward train pipe
    {
        const raul::Workflow::Pipeline& pipeBackwardTrain = w.getPipeline(raul::Workflow::Pipelines::BackwardTrain);

        ASSERT_EQ(pipeBackwardTrain.size(), 18u);

        ASSERT_EQ(pipeBackwardTrain[0]->type(), "Allocate");    // x2
        ASSERT_EQ(pipeBackwardTrain[1]->type(), "Forward");     // f2
        ASSERT_EQ(pipeBackwardTrain[2]->type(), "Allocate");    // x3
        ASSERT_EQ(pipeBackwardTrain[3]->type(), "Forward");     // f3
        ASSERT_EQ(pipeBackwardTrain[4]->type(), "Allocate");    // x4
        ASSERT_EQ(pipeBackwardTrain[5]->type(), "Forward");     // f4
        ASSERT_EQ(pipeBackwardTrain[6]->type(), "Deallocate");  // x3
        ASSERT_EQ(pipeBackwardTrain[7]->type(), "Backward");    // f5
        ASSERT_EQ(pipeBackwardTrain[8]->type(), "Deallocate");  // x4
        ASSERT_EQ(pipeBackwardTrain[9]->type(), "Allocate");    // x3
        ASSERT_EQ(pipeBackwardTrain[10]->type(), "Forward");    // f3
        ASSERT_EQ(pipeBackwardTrain[11]->type(), "Backward");   // f4
        ASSERT_EQ(pipeBackwardTrain[12]->type(), "Deallocate"); // x3
        ASSERT_EQ(pipeBackwardTrain[13]->type(), "Backward");   // f3
        ASSERT_EQ(pipeBackwardTrain[14]->type(), "Deallocate"); // x2
        ASSERT_EQ(pipeBackwardTrain[15]->type(), "Backward");   // f2
        ASSERT_EQ(pipeBackwardTrain[16]->type(), "Backward");   // f1
        ASSERT_EQ(pipeBackwardTrain[17]->type(), "Deallocate"); // x1

        ASSERT_TRUE(checkName(pipeBackwardTrain[0].get(), "x2"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[1].get(), "f2"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[2].get(), "x3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[3].get(), "f3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[4].get(), "x4"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[5].get(), "f4"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[6].get(), "x3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[7].get(), "f5"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[8].get(), "x4"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[9].get(), "x3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[10].get(), "f3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[11].get(), "f4"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[12].get(), "x3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[13].get(), "f3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[14].get(), "x2"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[15].get()), getName(pipeBackwardTrain[16].get()) }, { "f1", "f2" }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[17].get(), "x1"));
    }
}

} // UT namespace