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

namespace UT
{

TEST(TestWorkflowBenchmarks, TensorNeededPerfUnit)
{
    PROFILE_TEST
    raul::Workflow ww;

    auto timeStart = std::chrono::steady_clock::now();
    for (size_t q = 0; q < 500; ++q)
    {
        for (size_t w = 0; w < 100; ++w)
        {
            ww.tensorNeeded("L" + Conversions::toString(q),
                            "t" + Conversions::toString(w),
                            raul::WShape{ raul::BS(), 1u, 1u, 1u },
                            raul::Workflow::Usage::ForwardAndBackward,
                            raul::Workflow::Mode::Read,
                            false,
                            false,
                            false,
                            false,
                            false);
        }
    }
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count() << "ms" << std::endl;

    timeStart = std::chrono::steady_clock::now();
    EXPECT_THROW(ww.tensorNeeded("L499", "t99", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false),
                 raul::Exception);
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count() << "ms" << std::endl;

    ww.preparePipelines();
    EXPECT_THROW(ww.tensorNeeded("L500", "t0", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false),
                 raul::Exception);
}

TEST(TestWorkflowBenchmarks, TensorPipelinesPerfUnit)
{
    PROFILE_TEST

    class TestLayer : public raul::BasicLayer
    {
      public:
        TestLayer(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
            : BasicLayer(name, "test", params, networkParameters, { false, false })
        {
            if (!params.getOutputs().empty())
            {
                mNetworkParams.mWorkflow.tensorNeeded(
                    name, params.getOutputs()[0], raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, true, true, false, false, false);
            }
        }

        void forwardComputeImpl(raul::NetworkMode) override {}
        void backwardComputeImpl() override {}
    };

    raul::Workflow w;

    auto timeStart = std::chrono::steady_clock::now();

#ifdef _DEBUG
    const size_t maxLayers = 500u;
#else
    const size_t maxLayers = 20000u;
#endif

    for (size_t q = 0; q < maxLayers; ++q)
    {
        if (q == 0)
        {
            w.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ {}, { "x0" } });
        }
        else if (q == maxLayers - 1) // last
        {
            w.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ { "x" + Conversions::toString(q - 1) }, {} });
        }
        else
        {
            w.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ { "x" + Conversions::toString(q - 1) }, { "x" + Conversions::toString(q) } });
        }
    }
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count() << "ms" << std::endl;

    timeStart = std::chrono::steady_clock::now();
    EXPECT_NO_THROW(w.preparePipelines());
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count() << "ms" << std::endl;

    std::cout << "Forward test pipeline length: " << w.getPipeline(raul::Workflow::Pipelines::ForwardTest).size() << std::endl;
    std::cout << "Forward train pipeline length: " << w.getPipeline(raul::Workflow::Pipelines::ForwardTrain).size() << std::endl;
    std::cout << "Backward train pipeline length: " << w.getPipeline(raul::Workflow::Pipelines::BackwardTrain).size() << std::endl;

    EXPECT_NO_THROW(w.prepareMemoryForTraining());
    EXPECT_NO_THROW(w.setBatchSize(10));

    EXPECT_NO_THROW(w.forwardPassTesting());
    EXPECT_NO_THROW(w.forwardPassTraining());
    EXPECT_NO_THROW(w.backwardPassTraining());

    // order matters
    EXPECT_THROW(w.backwardPassTraining(), raul::Exception);
    EXPECT_NO_THROW(w.forwardPassTraining());
    EXPECT_THROW(w.forwardPassTraining(), raul::Exception);
    EXPECT_NO_THROW(w.backwardPassTraining());
}

TEST(TestWorkflowBenchmarks, TensorPipelinesCheckpointedPerfUnit)
{
    PROFILE_TEST

    class TestLayer : public raul::BasicLayer
    {
      public:
        TestLayer(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
            : BasicLayer(name, "test", params, networkParameters, { false, false })
        {
            if (!params.getOutputs().empty())
            {
                mNetworkParams.mWorkflow.tensorNeeded(
                    name, params.getOutputs()[0], raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, true, true, false, false, false);
            }
        }

        void forwardComputeImpl(raul::NetworkMode) override {}
        void backwardComputeImpl() override {}
    };

    raul::Workflow w;

    auto timeStart = std::chrono::steady_clock::now();

#ifdef _DEBUG
    const size_t maxLayers = 500u;
    const size_t step = 5u;
#else
    const size_t maxLayers = 20000u;
    const size_t step = 10u;
#endif

    for (size_t q = 0; q < maxLayers; ++q)
    {
        if (q == 0)
        {
            w.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ {}, { "x0" } });
        }
        else if (q == maxLayers - 1) // last
        {
            w.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ { "x" + Conversions::toString(q - 1) }, {} });
        }
        else
        {
            w.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ { "x" + Conversions::toString(q - 1) }, { "x" + Conversions::toString(q) } });
        }
    }
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count() << "ms" << std::endl;

#if 0
    EXPECT_NO_THROW(w.setCheckpoints({ "x0" }));

    timeStart = std::chrono::steady_clock::now();
    EXPECT_NO_THROW(w.preparePipelines(raul::Workflow::Execution::Checkpointed));
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count() << "ms" << std::endl;

    std::cout << "Forward test pipeline length: " << w.getPipeline(raul::Workflow::Pipelines::ForwardTest).size() << std::endl;
    std::cout << "Forward train pipeline length: " << w.getPipeline(raul::Workflow::Pipelines::ForwardTrain).size() << std::endl;
    std::cout << "Backward train pipeline length: " << w.getPipeline(raul::Workflow::Pipelines::BackwardTrain).size() << std::endl;
#endif

    raul::Names checkpoints;

    checkpoints.clear();

#ifdef _DEBUG
    for (size_t q = 0; q < maxLayers - 1; q += step)
#else
    for (size_t q = 0; q < maxLayers - 1; q += step)
#endif
    {
        checkpoints.push_back("x" + Conversions::toString(q));
    }
    EXPECT_NO_THROW(w.setCheckpoints(checkpoints));

    timeStart = std::chrono::steady_clock::now();
    EXPECT_NO_THROW(w.preparePipelines(raul::Workflow::Execution::Checkpointed));
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count() << "ms" << std::endl;

    std::cout << "Forward test pipeline length: " << w.getPipeline(raul::Workflow::Pipelines::ForwardTest).size() << std::endl;
    std::cout << "Forward train pipeline length: " << w.getPipeline(raul::Workflow::Pipelines::ForwardTrain).size() << std::endl;
    std::cout << "Backward train pipeline length: " << w.getPipeline(raul::Workflow::Pipelines::BackwardTrain).size() << std::endl;

    checkpoints.clear();
    for (size_t q = 0; q < maxLayers - 1; ++q)
    {
        checkpoints.push_back("x" + Conversions::toString(q));
    }
    EXPECT_NO_THROW(w.setCheckpoints(checkpoints));

    timeStart = std::chrono::steady_clock::now();
    EXPECT_NO_THROW(w.preparePipelines(raul::Workflow::Execution::Checkpointed));
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count() << "ms" << std::endl;

    std::cout << "Forward test pipeline length: " << w.getPipeline(raul::Workflow::Pipelines::ForwardTest).size() << std::endl;
    std::cout << "Forward train pipeline length: " << w.getPipeline(raul::Workflow::Pipelines::ForwardTrain).size() << std::endl;
    std::cout << "Backward train pipeline length: " << w.getPipeline(raul::Workflow::Pipelines::BackwardTrain).size() << std::endl;
}

TEST(TestWorkflowBenchmarks, CheckpointingMemoryUnit)
{
    PROFILE_TEST

    class TestLayer : public raul::BasicLayer
    {
      public:
        TestLayer(const raul::Name& name, const raul::BasicParams& params, size_t depth, size_t height, size_t width, raul::NetworkParameters& networkParameters)
            : BasicLayer(name, "test", params, networkParameters, { false, false })
        {
            if (!params.getInputs().empty())
            {
                mNetworkParams.mWorkflow.copyDeclaration(mName, params.getInputs()[0], raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
            }

            if (!params.getOutputs().empty())
            {
                mNetworkParams.mWorkflow.tensorNeeded(name,
                                                      params.getOutputs()[0],
                                                      raul::WShape{ raul::BS(), depth, height, width },
                                                      raul::Workflow::Usage::ForwardAndBackward,
                                                      raul::Workflow::Mode::Read,
                                                      true,
                                                      true,
                                                      false,
                                                      false,
                                                      false);
            }
        }

        void forwardComputeImpl(raul::NetworkMode) override {}
        void backwardComputeImpl() override {}
    };

    raul::Workflow w;

    const size_t maxLayers = 50u;
    const size_t tensorN = 64u;
    const size_t tensorC = 10u;
    const size_t tensorHW = 224u;

    for (size_t q = 0; q < maxLayers; ++q)
    {
        const raul::Name lName = "f" + Conversions::toString(q);

        if (q == 0)
        {
            w.add<TestLayer>(lName, raul::BasicParams{ {}, { "x0" } }, tensorC, tensorHW, tensorHW);
        }
        else if (q == maxLayers - 1) // last
        {
            w.add<TestLayer>(lName, raul::BasicParams{ { "x" + Conversions::toString(q - 1) }, {} }, tensorC, tensorHW, tensorHW);
        }
        else
        {
            w.add<TestLayer>(lName, raul::BasicParams{ { "x" + Conversions::toString(q - 1) }, { "x" + Conversions::toString(q) } }, tensorC, tensorHW, tensorHW);
        }
    }

    // w.setCheckpoints({"x0"});
    // w.setCheckpoints({"x0", "x10"});
    // w.setCheckpoints({"x0", "x10", "x20"});
    // w.setCheckpoints({"x0", "x10", "x20", "x30"});
    w.setCheckpoints({ "x0", "x10", "x20", "x30", "x40" });
    EXPECT_NO_THROW(w.preparePipelines(raul::Workflow::Execution::Checkpointed));
    // EXPECT_NO_THROW(w.preparePipelines());
    EXPECT_NO_THROW(w.prepareMemoryForTraining());
    EXPECT_NO_THROW(w.setBatchSize(tensorN));

    std::cout << "Forward test pipeline length: " << w.getPipeline(raul::Workflow::Pipelines::ForwardTest).size() << std::endl;
    std::cout << "Forward train pipeline length: " << w.getPipeline(raul::Workflow::Pipelines::ForwardTrain).size() << std::endl;
    std::cout << "Backward train pipeline length: " << w.getPipeline(raul::Workflow::Pipelines::BackwardTrain).size() << std::endl;
    std::cout << "Each tensor size: " << (sizeof(raul::dtype) * tensorN * tensorC * tensorHW * tensorHW) / (1024 * 1024) << "MB" << std::endl;

    auto timeStart = std::chrono::steady_clock::now();
    EXPECT_NO_THROW(w.forwardPassTraining());
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count() << "ms" << std::endl;
    timeStart = std::chrono::steady_clock::now();
    EXPECT_NO_THROW(w.backwardPassTraining());
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count() << "ms" << std::endl;
}

TEST(TestWorkflowBenchmarks, MemoryPoolPerformanceUnit)
{
    PROFILE_TEST

    class TestLayer : public raul::BasicLayer
    {
      public:
        TestLayer(const raul::Name& name, const raul::BasicParams& params, size_t depth, size_t height, size_t width, raul::NetworkParameters& networkParameters)
            : BasicLayer(name, "test", params, networkParameters, { false, false })
        {
            if (!params.getInputs().empty())
            {
                mNetworkParams.mWorkflow.copyDeclaration(mName, params.getInputs()[0], raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
            }

            if (!params.getOutputs().empty())
            {
                mNetworkParams.mWorkflow.tensorNeeded(name,
                                                      params.getOutputs()[0],
                                                      raul::WShape{ raul::BS(), depth, height, width },
                                                      raul::Workflow::Usage::ForwardAndBackward,
                                                      raul::Workflow::Mode::Read,
                                                      true,
                                                      true,
                                                      false,
                                                      false,
                                                      false);
            }
        }

        void forwardComputeImpl(raul::NetworkMode) override {}
        void backwardComputeImpl() override {}
    };

    const size_t maxLayers = 50u;
    const size_t tensorN = 64u;
    const size_t tensorC = 10u;
    const size_t tensorHW = 224u;

    raul::Workflow w(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD);
    raul::Workflow wPool(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::POOL);

    for (size_t q = 0; q < maxLayers; ++q)
    {
        if (q == 0)
        {
            w.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ {}, { "x0" } }, tensorC, tensorHW, tensorHW);
            wPool.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ {}, { "x0" } }, tensorC, tensorHW, tensorHW);
        }
        else if (q == maxLayers - 1) // last
        {
            w.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ { "x" + Conversions::toString(q - 1) }, {} }, tensorC, tensorHW, tensorHW);
            wPool.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ { "x" + Conversions::toString(q - 1) }, {} }, tensorC, tensorHW, tensorHW);
        }
        else
        {
            w.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ { "x" + Conversions::toString(q - 1) }, { "x" + Conversions::toString(q) } }, tensorC, tensorHW, tensorHW);
            wPool.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ { "x" + Conversions::toString(q - 1) }, { "x" + Conversions::toString(q) } }, tensorC, tensorHW, tensorHW);
        }
    }

    EXPECT_NO_THROW(w.preparePipelines());
    EXPECT_NO_THROW(w.prepareMemoryForTraining());
    EXPECT_NO_THROW(w.setBatchSize(tensorN));

    EXPECT_NO_THROW(wPool.preparePipelines());
    EXPECT_NO_THROW(wPool.prepareMemoryForTraining());
    EXPECT_NO_THROW(wPool.setBatchSize(tensorN));

    for (size_t q = 0; q < 5; ++q)
    {
        auto timeStart = std::chrono::steady_clock::now();
        EXPECT_NO_THROW(w.forwardPassTraining());
        EXPECT_NO_THROW(w.backwardPassTraining());
        std::cout << "Train:" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count() << "ms" << std::endl;
    }

    for (size_t q = 0; q < 5; ++q)
    {
        auto timeStart = std::chrono::steady_clock::now();
        EXPECT_NO_THROW(wPool.forwardPassTraining());
        EXPECT_NO_THROW(wPool.backwardPassTraining());
        std::cout << "Train pool:" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count() << "ms" << std::endl;
    }
}

TEST(TestWorkflowBenchmarks, MemoryPoolPerformance2Unit)
{
    PROFILE_TEST

    class TestLayer : public raul::BasicLayer
    {
      public:
        TestLayer(const raul::Name& name, const raul::BasicParams& params, size_t depth, size_t height, size_t width, raul::NetworkParameters& networkParameters)
            : BasicLayer(name, "test", params, networkParameters, { false, false })
        {
            if (!params.getInputs().empty())
            {
                mNetworkParams.mWorkflow.copyDeclaration(mName, params.getInputs()[0], raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
            }

            if (!params.getOutputs().empty())
            {
                mNetworkParams.mWorkflow.tensorNeeded(name,
                                                      params.getOutputs()[0],
                                                      raul::WShape{ raul::BS(), depth, height, width },
                                                      raul::Workflow::Usage::ForwardAndBackward,
                                                      raul::Workflow::Mode::Read,
                                                      true,
                                                      true,
                                                      false,
                                                      false,
                                                      false);
            }
        }

        void forwardComputeImpl(raul::NetworkMode) override {}
        void backwardComputeImpl() override {}
    };

#ifdef _DEBUG
    const size_t maxLayers = 500u;
#else
    const size_t maxLayers = 20000u;
#endif

    const size_t tensorN = 10u;
    const size_t tensorC = 1u;
    const size_t tensorHW = 1u;

    raul::Workflow w(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD);
    raul::Workflow wPool(raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::POOL);

    for (size_t q = 0; q < maxLayers; ++q)
    {
        if (q == 0)
        {
            w.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ {}, { "x0" } }, tensorC, tensorHW, tensorHW);
            wPool.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ {}, { "x0" } }, tensorC, tensorHW, tensorHW);
        }
        else if (q == maxLayers - 1) // last
        {
            w.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ { "x" + Conversions::toString(q - 1) }, {} }, tensorC, tensorHW, tensorHW);
            wPool.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ { "x" + Conversions::toString(q - 1) }, {} }, tensorC, tensorHW, tensorHW);
        }
        else
        {
            w.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ { "x" + Conversions::toString(q - 1) }, { "x" + Conversions::toString(q) } }, tensorC, tensorHW, tensorHW);
            wPool.add<TestLayer>("f" + Conversions::toString(q), raul::BasicParams{ { "x" + Conversions::toString(q - 1) }, { "x" + Conversions::toString(q) } }, tensorC, tensorHW, tensorHW);
        }
    }

    EXPECT_NO_THROW(w.preparePipelines());
    EXPECT_NO_THROW(w.prepareMemoryForTraining());
    EXPECT_NO_THROW(w.setBatchSize(tensorN));

    EXPECT_NO_THROW(wPool.preparePipelines());
    EXPECT_NO_THROW(wPool.prepareMemoryForTraining());
    EXPECT_NO_THROW(wPool.setBatchSize(tensorN));

    for (size_t q = 0; q < 5; ++q)
    {
        auto timeStart = std::chrono::steady_clock::now();
        EXPECT_NO_THROW(w.forwardPassTraining());
        EXPECT_NO_THROW(w.backwardPassTraining());
        std::cout << "Train:" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count() << "ms" << std::endl;
    }

    for (size_t q = 0; q < 5; ++q)
    {
        auto timeStart = std::chrono::steady_clock::now();
        EXPECT_NO_THROW(wPool.forwardPassTraining());
        EXPECT_NO_THROW(wPool.backwardPassTraining());
        std::cout << "Train pool:" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count() << "ms" << std::endl;
    }
}

} // UT namespace