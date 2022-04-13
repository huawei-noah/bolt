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

#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>
#include <training/compiler/WorkflowActions.h>

#include "Test_WorkflowTools.h"

namespace UT
{

TEST(TestWorkflow, WShapeConstructorUnit)
{
    {
        raul::WShape wShape;

        EXPECT_FALSE(wShape.isBSDependent());

        raul::Workflow w;

        raul::shape shapeVal = wShape.getShape(w);

        EXPECT_EQ(shapeVal[0], 0u);
        EXPECT_EQ(shapeVal[1], 0u);
        EXPECT_EQ(shapeVal[2], 0u);
        EXPECT_EQ(shapeVal[3], 0u);
    }

    {
        raul::WShape wShape(1u, 2u, 3u, 4u);

        EXPECT_FALSE(wShape.isBSDependent());

        raul::Workflow w;

        raul::shape shapeVal = wShape.getShape(w);

        EXPECT_EQ(shapeVal[0], 1u);
        EXPECT_EQ(shapeVal[1], 2u);
        EXPECT_EQ(shapeVal[2], 3u);
        EXPECT_EQ(shapeVal[3], 4u);
    }

    {
        raul::WShape wShape(raul::BS(), 2u, 3u, 4u);

        EXPECT_TRUE(wShape.isBSDependent());

        raul::Workflow w;

        EXPECT_THROW([[maybe_unused]] const auto shape = wShape.getShape(w), raul::Exception);
    }

    {
        raul::WShape wShape(raul::BS(), 2u, 3u, 4u);

        EXPECT_TRUE(wShape.isBSDependent());

        raul::Workflow w;

        w.preparePipelines();
        w.setBatchSize(1u);

        raul::shape shapeVal = wShape.getShape(w);

        EXPECT_EQ(shapeVal[0], 1u);
        EXPECT_EQ(shapeVal[1], 2u);
        EXPECT_EQ(shapeVal[2], 3u);
        EXPECT_EQ(shapeVal[3], 4u);

        w.setBatchSize(10u);

        shapeVal = wShape.getShape(w);

        EXPECT_EQ(shapeVal[0], 10u);
        EXPECT_EQ(shapeVal[1], 2u);
        EXPECT_EQ(shapeVal[2], 3u);
        EXPECT_EQ(shapeVal[3], 4u);
    }

    {
        raul::WShape wShape(1u, raul::BS(), 3u, raul::BS(2u));

        EXPECT_TRUE(wShape.isBSDependent());

        raul::Workflow w;

        w.preparePipelines();
        w.setBatchSize(1u);

        raul::shape shapeVal = wShape.getShape(w);

        EXPECT_EQ(shapeVal[0], 1u);
        EXPECT_EQ(shapeVal[1], 1u);
        EXPECT_EQ(shapeVal[2], 3u);
        EXPECT_EQ(shapeVal[3], 2u);

        w.setBatchSize(10u);

        shapeVal = wShape.getShape(w);

        EXPECT_EQ(shapeVal[0], 1u);
        EXPECT_EQ(shapeVal[1], 10u);
        EXPECT_EQ(shapeVal[2], 3u);
        EXPECT_EQ(shapeVal[3], 20u);
    }
}

TEST(TestWorkflow, WShapeComparisonUnit)
{
    // equal
    {
        raul::WShape wShape;
        raul::WShape wShape2;

        EXPECT_TRUE(wShape == wShape2);
        EXPECT_FALSE(wShape != wShape2);
    }

    {
        raul::WShape wShape(1u, 2u, 3u, 4u);
        raul::WShape wShape2(1u, 2u, 3u, 4u);

        EXPECT_TRUE(wShape == wShape2);
        EXPECT_FALSE(wShape != wShape2);
    }

    {
        raul::WShape wShape(1u, raul::BS(), 3u, 4u);
        raul::WShape wShape2(1u, raul::BS(), 3u, 4u);

        EXPECT_TRUE(wShape == wShape2);
        EXPECT_FALSE(wShape != wShape2);
    }

    {
        raul::WShape wShape(1u, raul::BS(), 3u, 4u);
        raul::WShape wShape2(1u, raul::BS(1u), 3u, 4u);

        EXPECT_TRUE(wShape == wShape2);
        EXPECT_FALSE(wShape != wShape2);
    }

    {
        raul::WShape wShape(1u, raul::BS(), 3u, raul::BS(2u));
        raul::WShape wShape2(1u, raul::BS(), 3u, raul::BS(2u));

        EXPECT_TRUE(wShape == wShape2);
        EXPECT_FALSE(wShape != wShape2);
    }

    // not equal
    {
        raul::WShape wShape(1u, raul::BS(), 3u, raul::BS(2u));
        raul::WShape wShape2(2u, raul::BS(), 3u, raul::BS(2u));

        EXPECT_TRUE(wShape != wShape2);
        EXPECT_FALSE(wShape == wShape2);
    }

    {
        raul::WShape wShape(1u, raul::BS(), 3u, raul::BS(2u));
        raul::WShape wShape2(1u, raul::BS(), 3u, 4u);

        EXPECT_TRUE(wShape != wShape2);
        EXPECT_FALSE(wShape == wShape2);
    }

    {
        raul::WShape wShape(1u, raul::BS(), 3u, raul::BS(2u));
        raul::WShape wShape2(1u, raul::BS(), 3u, raul::BS(3u));

        EXPECT_TRUE(wShape != wShape2);
        EXPECT_FALSE(wShape == wShape2);
    }
}

TEST(TestWorkflow, TestOfCheckBlockUnit)
{
    EXPECT_TRUE(checkBlock({ "1", "2", "3" }, 0, { "1", "2" }));
    EXPECT_TRUE(checkBlock({ "1", "2", "3" }, 1, { "2", "3" }));
    EXPECT_FALSE(checkBlock({ "1", "2", "3" }, 2, { "2", "3" }));
}

TEST(TestWorkflow, TestOfCheckBlocksUnit)
{
    EXPECT_FALSE(checkBlocks({ "1", "2", "3" }, { { "1" }, { "3" } }));
    EXPECT_TRUE(checkBlocks({ "1", "2", "3" }, { { "1", "2" }, { "3" } }));
    EXPECT_TRUE(checkBlocks({ "1", "2", "3" }, { { "3" }, { "1", "2" } }));
    EXPECT_FALSE(checkBlocks({ "1", "2", "3" }, { { "3", "4" }, { "1", "2" } }));
    EXPECT_FALSE(checkBlocks({ "1", "2", "3", "5" }, { { "3", "4" }, { "1", "2" } }));
    EXPECT_TRUE(checkBlocks({ "1", "2", "3", "4", "1", "2", "3" }, { { "1", "2", "3" }, { "1", "2", "3", "4" } }));
    EXPECT_FALSE(checkBlocks({ "1", "2", "3", "1", "2", "4" }, { { "1", "2", "3" }, { "1", "2", "3" } }));
}

TEST(TestWorkflow, TensorNeededUnit)
{
    PROFILE_TEST
    raul::Workflow w;

    EXPECT_FALSE(w.isTensorDeclared("t1"));

    w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
    w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false);
    EXPECT_THROW(w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false), raul::Exception);
    EXPECT_THROW(w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false),
                 raul::Exception);
    EXPECT_THROW(w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, false, false, false, false, false),
                 raul::Exception);

    EXPECT_TRUE(w.isTensorDeclared("t1"));
    EXPECT_TRUE(w.isBatchPlaceholded("t1"));
    EXPECT_FALSE(w.isDepthPlaceholded("t1"));
    EXPECT_FALSE(w.isHeightPlaceholded("t1"));
    EXPECT_FALSE(w.isWidthPlaceholded("t1"));
    EXPECT_THROW(w.getBatch("t1"), raul::Exception);
    EXPECT_EQ(w.getDepth("t1"), 1u);
    EXPECT_EQ(w.getHeight("t1"), 1u);
    EXPECT_EQ(w.getWidth("t1"), 1u);

    // WSHape
    EXPECT_THROW(w.getShape("t2"), raul::Exception);
    raul::WShape sh = w.getShape("t1");
    EXPECT_THROW([[maybe_unused]] const auto shape = sh.getShape(w), raul::Exception);
    EXPECT_TRUE(sh.isBSDependent());

    w.tensorNeeded("L1", "t2", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false);

    w.tensorNeeded("L2", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false);

    w.tensorNeeded("L2", "t2", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, false, false, false, false, false);
    EXPECT_THROW(w.tensorNeeded("L2", "t2", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false), raul::Exception);
    EXPECT_THROW(w.tensorNeeded("L2", "t2", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false),
                 raul::Exception);

    w.tensorNeeded("L3", "t3", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
    EXPECT_THROW(w.tensorNeeded("L3", "t3", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false), raul::Exception);
    EXPECT_THROW(w.tensorNeeded("L3", "t3", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, false, false, false, false, false),
                 raul::Exception);

    w.tensorNeeded("L4", "t4", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false);
    EXPECT_THROW(w.tensorNeeded("L4", "t4", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false),
                 raul::Exception);
    EXPECT_THROW(w.tensorNeeded("L4", "t4", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, false, false, false, false, false),
                 raul::Exception);

    // copy to layer (flags ignored)
    w.copyDec("L5", "ignored", "t4", raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false);
    EXPECT_THROW(w.copyDeclaration("L5", "t4", raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false), raul::Exception);

    // copy to tensor
    w.copyDec("L5", "t4", "t5", raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false);
    EXPECT_THROW(w.copyDeclaration("L5", "t5", raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false), raul::Exception);
}

TEST(TestWorkflow, TensorNeededMaxShapeUnit)
{
    PROFILE_TEST
    raul::Workflow w;

    EXPECT_FALSE(w.isTensorDeclared("t1"));

    w.tensorNeededMaxShape("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
    EXPECT_NO_THROW(w.tensorNeededMaxShape("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 2u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false));

    w.tensorNeeded("L1", "t2", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
    EXPECT_THROW(w.tensorNeeded("L1", "t2", raul::WShape{ 1u, 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false), raul::Exception);

    w.tensorNeededMaxShape("L1", "t3", raul::WShape{ 1u, 2u, 3u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
    w.tensorNeededMaxShape("L1", "t3", raul::WShape{ 10u, 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);

    EXPECT_TRUE(w.isTensorDeclared("t1"));
    EXPECT_TRUE(w.isBatchPlaceholded("t1"));
    EXPECT_FALSE(w.isDepthPlaceholded("t1"));
    EXPECT_FALSE(w.isHeightPlaceholded("t1"));
    EXPECT_FALSE(w.isWidthPlaceholded("t1"));
    EXPECT_THROW(w.getBatch("t1"), raul::Exception);
    EXPECT_EQ(w.getDepth("t1"), 1u);
    EXPECT_EQ(w.getHeight("t1"), 1u);
    EXPECT_EQ(w.getWidth("t1"), 2u);

    EXPECT_EQ(w.getBatch("t3"), 10u);
    EXPECT_EQ(w.getDepth("t3"), 2u);
    EXPECT_EQ(w.getHeight("t3"), 3u);
    EXPECT_EQ(w.getWidth("t3"), 1u);

    // WSHape
    raul::WShape sh = w.getShape("t1");
    EXPECT_THROW([[maybe_unused]] const auto shape = sh.getShape(w), raul::Exception);
    EXPECT_TRUE(sh.isBSDependent());
}

TEST(TestWorkflow, AddOpUnit)
{
    PROFILE_TEST

    raul::Workflow w;

    w.add<TestLayer>("f0", raul::BasicParams{ { "in" }, { "x1" } }, true);

    EXPECT_THROW(w.add<TestLayer>("f0", raul::BasicParams{ { "x1" }, { "x2" } }, true), raul::Exception);

    w.add<TestLayer>("f1", raul::BasicParams{ { "x1" }, { "x2" } }, true);

    ASSERT_EQ(w["f0"]->getName(), "f0");
    ASSERT_EQ(w["f1"]->getName(), "f1");

    ASSERT_EQ(w["f0"]->getInputs().size(), 1u);
    ASSERT_EQ(w["f0"]->getOutputs().size(), 1u);

    ASSERT_EQ(w["f0"]->getInputs()[0], "in");
    ASSERT_EQ(w["f0"]->getOutputs()[0], "x1");

    ASSERT_EQ(w["f1"]->getInputs().size(), 1u);
    ASSERT_EQ(w["f1"]->getOutputs().size(), 1u);

    ASSERT_EQ(w["f1"]->getInputs()[0], "x1");
    ASSERT_EQ(w["f1"]->getOutputs()[0], "x2");
}

TEST(TestWorkflow, GetLayerParameterNamesUnit)
{
    PROFILE_TEST

    using namespace raul;

    Workflow w;

    w.add<raul::DataLayer>("data", DataParams{ { "in" }, 1, 1, 1 });
    w.add<raul::LinearLayer>("fc", LinearParams{ "in", "x1", 1 });
    w.add<raul::BatchNormLayer>("bn", BatchnormParams{ { "x1" }, { "x2" } });

    EXPECT_THROW(w.getLayerParameterNames("f"), raul::Exception);

    std::map<Name, std::set<Name>> correctParams = {
        { "data", {} },
        { "fc", { Name("fc") / "Weights", Name("fc") / "Biases" } },
        { "bn", { Name("bn") / "Weights", Name("bn") / "Biases", Name("bn") / "MeanEval", Name("bn") / "VarianceEval" } },
    };

    for (const auto& [name, idealParams] : correctParams)
    {
        auto params = w.getLayerParameterNames(name);
        std::set<Name> paramsSet(params.begin(), params.end());
        EXPECT_EQ(paramsSet.size(), idealParams.size());
        for (const auto& p : paramsSet)
        {
            EXPECT_TRUE(idealParams.find(p) != idealParams.end());
        }
    }
}

TEST(TestWorkflow, AddLayerUnit)
{
    PROFILE_TEST

    raul::Workflow w;

    w.add<TestLayer>("f0", raul::BasicParams{ { "in" }, { "x1" } }, true);

    EXPECT_THROW(w.add<TestLayer>("f0", raul::BasicParams{ { "x1" }, { "x2" } }, true), raul::Exception);

    w.add<TestLayer>("f1", raul::BasicParams{ { "x1" }, { "x2" } }, true);

    ASSERT_EQ(w["f0"]->getName(), "f0");
    ASSERT_EQ(w["f1"]->getName(), "f1");

    ASSERT_EQ(w["f0"]->getInputs().size(), 1u);
    ASSERT_EQ(w["f0"]->getOutputs().size(), 1u);

    ASSERT_EQ(w["f0"]->getInputs()[0], "in");
    ASSERT_EQ(w["f0"]->getOutputs()[0], "x1");

    ASSERT_EQ(w["f1"]->getInputs().size(), 1u);
    ASSERT_EQ(w["f1"]->getOutputs().size(), 1u);

    ASSERT_EQ(w["f1"]->getInputs()[0], "x1");
    ASSERT_EQ(w["f1"]->getOutputs()[0], "x2");
}

TEST(TestWorkflow, PreparePipelinesSimpleUnit)
{
    PROFILE_TEST

    {
        raul::Workflow w;

        w.add<TestLayer>("f0", raul::BasicParams{ {}, { "x1" } }, true);
        w.add<TestLayer>("f1", raul::BasicParams{ { "" }, { "x2" } }, true);

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.add<TestLayer>("f0", raul::BasicParams{ {}, { "x1" } }, true);
        w.add<TestLayer>("f1", raul::BasicParams{ { "x1" }, { "" } }, true);

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }
}

TEST(TestWorkflow, PreparePipelinesSimple2Unit)
{
    PROFILE_TEST

    raul::Workflow w;

    w.add<TestLayer>("f0", raul::BasicParams{ {}, { "x1" } }, true);

    // empty layer name not possible
    EXPECT_THROW(w.add<TestLayer>("", raul::BasicParams{ {}, { "x2" } }, true), raul::Exception);
    EXPECT_THROW(w.add<TestLayer>("", raul::BasicParams{ { "x2" }, { "x3" } }, true), raul::Exception);

    w.preparePipelines();

    EXPECT_THROW(w.add<TestLayer>("f1", raul::BasicParams{ { "x1" }, { "x2" } }, true), raul::Exception);
    EXPECT_THROW(w.add<TestLayer>("f1", raul::BasicParams{ { "x1" }, { "x2" } }, true), raul::Exception);
}

TEST(TestWorkflow, PreparePipelinesInequalityUnit)
{
    PROFILE_TEST

    class TestLayer2 : public raul::BasicLayer
    {
      public:
        TestLayer2(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
            : BasicLayer(name, "test", params, networkParameters, { false, false })
        {
        }

        void forwardComputeImpl(raul::NetworkMode) override {}
        void backwardComputeImpl() override {}
    };

    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.add<TestLayer2>("L1", raul::BasicParams{ {}, {} });

        EXPECT_NO_THROW(w.preparePipelines());
    }

    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
        EXPECT_THROW(w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, true, false),
                     raul::Exception);
        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.add<TestLayer2>("L1", raul::BasicParams{ {}, {} });

        EXPECT_NO_THROW(w.preparePipelines());
    }

    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, true, false, false);
        w.add<TestLayer2>("L1", raul::BasicParams{ {}, {} });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, true, false, false, false);
        w.add<TestLayer2>("L1", raul::BasicParams{ {}, {} });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, true, false, false, false, false);
        w.add<TestLayer2>("L1", raul::BasicParams{ {}, {} });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.tensorNeeded("L1", "t1", raul::WShape{ 1u, 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.add<TestLayer2>("L1", raul::BasicParams{ {}, {} });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 2u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.add<TestLayer2>("L1", raul::BasicParams{ {}, {} });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ 1u, 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.tensorNeeded("L1", "t1", raul::WShape{ 1u, 1u, 1u, 2u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.add<TestLayer2>("L1", raul::BasicParams{ {}, {} });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.tensorNeeded("L2", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.add<TestLayer2>("L1", raul::BasicParams{ {}, {} });
        w.add<TestLayer2>("L2", raul::BasicParams{ {}, {} });

        EXPECT_NO_THROW(w.preparePipelines());
    }

    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.tensorNeeded("L2", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, false, false, true, false, false);
        w.add<TestLayer2>("L1", raul::BasicParams{ {}, {} });
        w.add<TestLayer2>("L2", raul::BasicParams{ {}, {} });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.tensorNeeded("L2", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, false, false, true, false, false);
        w.add<TestLayer2>("L1", raul::BasicParams{ {}, {} });
        w.add<TestLayer2>("L2", raul::BasicParams{ {}, {} });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.tensorNeeded("L2", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, false, true, false, false, false);
        w.add<TestLayer2>("L1", raul::BasicParams{ {}, {} });
        w.add<TestLayer2>("L2", raul::BasicParams{ {}, {} });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.tensorNeeded("L2", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, true, false, false, false, false);
        w.add<TestLayer2>("L1", raul::BasicParams{ {}, {} });
        w.add<TestLayer2>("L2", raul::BasicParams{ {}, {} });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.tensorNeeded("L2", "t1", raul::WShape{ raul::BS(), 1u, 1u, 2u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.add<TestLayer2>("L1", raul::BasicParams{ {}, {} });
        w.add<TestLayer2>("L2", raul::BasicParams{ {}, {} });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.tensorNeeded("L2", "t1", raul::WShape{ 1u, 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.add<TestLayer2>("L1", raul::BasicParams{ {}, {} });
        w.add<TestLayer2>("L2", raul::BasicParams{ {}, {} });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ 1u, 1u, 1u, 1u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.tensorNeeded("L2", "t1", raul::WShape{ 1u, 1u, 1u, 2u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, false, false, false, false, false);
        w.add<TestLayer2>("L1", raul::BasicParams{ {}, {} });
        w.add<TestLayer2>("L2", raul::BasicParams{ {}, {} });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    // copy tensor with flags to new layer
    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ 1u, 2u, 3u, 4u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);

        EXPECT_NO_THROW(w.copyDeclaration("L1", "t1", raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, true, true, false, false, false));
        EXPECT_THROW(w.copyDeclaration("L2", "t2", raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, true, true, false, false, false), raul::Exception);
        EXPECT_NO_THROW(w.copyDeclaration("L2", "t1", raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, true, true, false, false, false));
        EXPECT_THROW(w.copyDeclaration("L2", "t1", raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, true, true, false, false, false), raul::Exception);

        EXPECT_THROW(w.getBatch("t2"), raul::Exception);
        EXPECT_THROW(w.getDepth("t2"), raul::Exception);
        EXPECT_THROW(w.getWidth("t2"), raul::Exception);
        EXPECT_THROW(w.getHeight("t2"), raul::Exception);

        EXPECT_EQ(w.getBatch("t1"), 1u);
        EXPECT_EQ(w.getDepth("t1"), 2u);
        EXPECT_EQ(w.getHeight("t1"), 3u);
        EXPECT_EQ(w.getWidth("t1"), 4u);

        EXPECT_NO_THROW(w.preparePipelines());
    }

    // copy tensor with flags to new tensor (same or new layer)
    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ 1u, 2u, 3u, 4u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);

        EXPECT_NO_THROW(w.copyDeclaration("L1", "t1", "t2", raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read, true, true, false, false, false));
        EXPECT_THROW(w.copyDeclaration("L2", "t3", "t4", raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, true, true, false, false, false), raul::Exception);

        EXPECT_EQ(w.getBatch("t2"), 1u);
        EXPECT_EQ(w.getDepth("t2"), 2u);
        EXPECT_EQ(w.getHeight("t2"), 3u);
        EXPECT_EQ(w.getWidth("t2"), 4u);

        EXPECT_NO_THROW(w.preparePipelines());
    }

    // copy tensor to new layer
    {
        raul::Workflow w;

        w.tensorNeeded("L1", "t1", raul::WShape{ 1u, 2u, 3u, 4u }, raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read, false, false, false, false, false);

        EXPECT_NO_THROW(w.copyDeclaration("L1", "t1", raul::Workflow::Usage::Backward, raul::Workflow::Mode::Read));
        EXPECT_THROW(w.copyDeclaration("L2", "t2", raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read), raul::Exception);
        EXPECT_NO_THROW(w.copyDeclaration("L2", "t1", raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read));
        EXPECT_THROW(w.copyDeclaration("L2", "t1", raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read), raul::Exception);

        EXPECT_THROW(w.getBatch("t2"), raul::Exception);
        EXPECT_THROW(w.getDepth("t2"), raul::Exception);
        EXPECT_THROW(w.getWidth("t2"), raul::Exception);
        EXPECT_THROW(w.getHeight("t2"), raul::Exception);

        EXPECT_EQ(w.getBatch("t1"), 1u);
        EXPECT_EQ(w.getDepth("t1"), 2u);
        EXPECT_EQ(w.getHeight("t1"), 3u);
        EXPECT_EQ(w.getWidth("t1"), 4u);

        EXPECT_NO_THROW(w.preparePipelines());
    }
}

TEST(TestWorkflow, PreparePipelinesCorrectTopologyUnit)
{
    PROFILE_TEST

    {
        raul::Workflow w;

        w.add<TestLayer>("f0", raul::BasicParams{ {}, { "x1" } }, true);
        w.add<TestLayer>("f1", raul::BasicParams{ { "x3" }, { "x2" } }, true);

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.add<TestLayer>("f0", raul::BasicParams{ {}, { "x1" } }, true);
        w.add<TestLayer>("f1", raul::BasicParams{ {}, { "x1" } }, true);
        w.add<TestLayer>("f2", raul::BasicParams{ { "x1" }, { "x3" } }, true);

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    class Test : public raul::BasicLayer
    {
      public:
        Test(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
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

    {
        raul::Workflow w;

        w.add<Test>("f0", raul::BasicParams{ {}, { "x1", "x1" } });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.add<TestLayer>("f0", raul::BasicParams{ {}, { "x1" } }, true);
        w.add<Test>("f1", raul::BasicParams{ { "x1" }, { "x1" } });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.add<TestLayer>("f0", raul::BasicParams{ {}, { "x1" } }, true);
        w.add<Test>("f1", raul::BasicParams{ { "x1", "x1" }, { "x2" } });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    {
        raul::Workflow w;

        w.add<TestLayer>("f0", raul::BasicParams{ {}, { "x1" } }, true);
        w.add<Test>("f1", raul::BasicParams{ { "x1" }, { "x2" }, { "w1", "w1" } });

        EXPECT_THROW(w.preparePipelines(), raul::Exception);
    }

    class Test2 : public raul::BasicLayer
    {
      public:
        Test2(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
            : BasicLayer(name, "test", params, networkParameters, { false, false })
        {
        }

        void forwardComputeImpl(raul::NetworkMode) override {}
        void backwardComputeImpl() override {}
    };

    // output not declared
    {
        raul::Workflow w;

        w.add<TestLayer>("f0", raul::BasicParams{ {}, { "x1" } }, true);
        EXPECT_THROW(w.add<Test2>("f1", raul::BasicParams{ { "x1" }, { "x2" } }), raul::Exception);
    }
}

TEST(TestWorkflow, PreparePipelinesSimpleTopologyUnit)
{
    PROFILE_TEST

    raul::Workflow w;

    w.add<TestLayer>("f0", raul::BasicParams{ {}, { "x1" } }, true);
    w.add<TestLayer>("f1", raul::BasicParams{ { "x1" }, { "x2" } }, true);
    w.add<TestLayer>("f2", raul::BasicParams{ { "x2" }, { "x3" } }, true);
    w.add<TestLayer>("f3", raul::BasicParams{ { "x3" }, {} }, true);

    static_cast<TestLayer*>(w["f0"])->setExpectGrad(3.0_dt);
    static_cast<TestLayer*>(w["f1"])->setExpectGrad(2.0_dt);
    static_cast<TestLayer*>(w["f2"])->setExpectGrad(1.0_dt);

    auto names = w.getTrainableParameterNames();

    EXPECT_EQ(names.size(), 4u);

    {
        std::unordered_set<raul::Name> tNames(names.begin(), names.end());

        ASSERT_NE(tNames.find(raul::Name("f0") / "Weights"), tNames.end());
        ASSERT_NE(tNames.find(raul::Name("f1") / "Weights"), tNames.end());
        ASSERT_NE(tNames.find(raul::Name("f2") / "Weights"), tNames.end());
        ASSERT_NE(tNames.find(raul::Name("f3") / "Weights"), tNames.end());
    }

    EXPECT_THROW(w.getTrainableParameters(), raul::Exception);

    EXPECT_NO_THROW(w.preparePipelines());

    // forward test pipe
    {
        const raul::Workflow::Pipeline& pipeForwardTest = w.getPipeline(raul::Workflow::Pipelines::ForwardTest);

        ASSERT_EQ(pipeForwardTest.size(), 10u);

        ASSERT_EQ(pipeForwardTest[0]->type(), "Allocate");   // x1
        ASSERT_EQ(pipeForwardTest[1]->type(), "Forward");    // f0
        ASSERT_EQ(pipeForwardTest[2]->type(), "Allocate");   // x2
        ASSERT_EQ(pipeForwardTest[3]->type(), "Forward");    // f1
        ASSERT_EQ(pipeForwardTest[4]->type(), "Deallocate"); // x1
        ASSERT_EQ(pipeForwardTest[5]->type(), "Allocate");   // x3
        ASSERT_EQ(pipeForwardTest[6]->type(), "Forward");    // f2
        ASSERT_EQ(pipeForwardTest[7]->type(), "Deallocate"); // x2
        ASSERT_EQ(pipeForwardTest[8]->type(), "Forward");    // f3
        ASSERT_EQ(pipeForwardTest[9]->type(), "Deallocate"); // x3

        ASSERT_TRUE(checkName(pipeForwardTest[0].get(), "x1"));
        ASSERT_TRUE(checkName(pipeForwardTest[1].get(), "f0"));
        ASSERT_TRUE(checkName(pipeForwardTest[2].get(), "x2"));
        ASSERT_TRUE(checkName(pipeForwardTest[3].get(), "f1"));
        ASSERT_TRUE(checkName(pipeForwardTest[4].get(), "x1"));
        ASSERT_TRUE(checkName(pipeForwardTest[5].get(), "x3"));
        ASSERT_TRUE(checkName(pipeForwardTest[6].get(), "f2"));
        ASSERT_TRUE(checkName(pipeForwardTest[7].get(), "x2"));
        ASSERT_TRUE(checkName(pipeForwardTest[8].get(), "f3"));
        ASSERT_TRUE(checkName(pipeForwardTest[9].get(), "x3"));
    }

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

        ASSERT_TRUE(checkName(pipeForwardTrain[0].get(), "x1"));
        ASSERT_TRUE(checkName(pipeForwardTrain[1].get(), "f0"));
        ASSERT_TRUE(checkName(pipeForwardTrain[2].get(), "x2"));
        ASSERT_TRUE(checkName(pipeForwardTrain[3].get(), "f1"));
        ASSERT_TRUE(checkName(pipeForwardTrain[4].get(), "x3"));
        ASSERT_TRUE(checkName(pipeForwardTrain[5].get(), "f2"));
        ASSERT_TRUE(checkName(pipeForwardTrain[6].get(), "f3"));
    }

    // backward train pipe
    {
        const raul::Workflow::Pipeline& pipeBackwardTrain = w.getPipeline(raul::Workflow::Pipelines::BackwardTrain);

        ASSERT_EQ(pipeBackwardTrain.size(), 16u);

        ASSERT_EQ(pipeBackwardTrain[0]->type(), "Allocate");    // xg3
        ASSERT_EQ(pipeBackwardTrain[1]->type(), "Zero");        // xg3
        ASSERT_EQ(pipeBackwardTrain[2]->type(), "Backward");    // f3
        ASSERT_EQ(pipeBackwardTrain[3]->type(), "Deallocate");  // x3
        ASSERT_EQ(pipeBackwardTrain[4]->type(), "Allocate");    // xg2
        ASSERT_EQ(pipeBackwardTrain[5]->type(), "Zero");        // xg2
        ASSERT_EQ(pipeBackwardTrain[6]->type(), "Backward");    // f2
        ASSERT_EQ(pipeBackwardTrain[7]->type(), "Deallocate");  // x2
        ASSERT_EQ(pipeBackwardTrain[8]->type(), "Deallocate");  // xg3
        ASSERT_EQ(pipeBackwardTrain[9]->type(), "Allocate");    // xg1
        ASSERT_EQ(pipeBackwardTrain[10]->type(), "Zero");       // xg1
        ASSERT_EQ(pipeBackwardTrain[11]->type(), "Backward");   // f1
        ASSERT_EQ(pipeBackwardTrain[12]->type(), "Deallocate"); // x1
        ASSERT_EQ(pipeBackwardTrain[13]->type(), "Deallocate"); // xg2
        ASSERT_EQ(pipeBackwardTrain[14]->type(), "Backward");   // f0
        ASSERT_EQ(pipeBackwardTrain[15]->type(), "Deallocate"); // xg1

        ASSERT_TRUE(checkName(pipeBackwardTrain[0].get(), getGradName("x3")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[1].get(), getGradName("x3")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[2].get(), "f3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[3].get(), "x3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[4].get(), getGradName("x2")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[5].get(), getGradName("x2")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[6].get(), "f2"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[7].get()), getName(pipeBackwardTrain[8].get()) }, { "x2", getGradName("x3") }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[9].get(), getGradName("x1")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[10].get(), getGradName("x1")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[11].get(), "f1"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[12].get()), getName(pipeBackwardTrain[13].get()) }, { "x1", getGradName("x2") }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[14].get(), "f0"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[15].get(), getGradName("x1")));
    }

    EXPECT_NO_THROW(w.prepareMemoryForTraining());
    EXPECT_NO_THROW(w.setBatchSize(10));

    auto trainables = w.getTrainableParameters();
    EXPECT_EQ(trainables.size(), 4u);

    // order matters
    EXPECT_NO_THROW(w.forwardPassTraining());
    EXPECT_THROW(w.preparePipelines(), raul::Exception);

    EXPECT_NO_THROW(w.backwardPassTraining());
    EXPECT_NO_THROW(w.preparePipelines());
}

TEST(TestWorkflow, WorkflowEagerBatchSizeChangedUnit)
{
    PROFILE_TEST

    constexpr size_t BATCH_SIZE = 10;

    raul::WorkflowEager w;
    w.add<TestInitLayer>("f", raul::BasicParams{ {}, {} });

    auto* layer = static_cast<TestInitLayer*>(w["f"]);

    ASSERT_TRUE(layer != nullptr);

    EXPECT_NO_THROW(w.preparePipelines());
    EXPECT_NO_THROW(w.prepareMemoryForTraining());

    ASSERT_NE(layer->getBatchSize(), BATCH_SIZE);
    EXPECT_NO_THROW(w.setBatchSize(BATCH_SIZE));
    ASSERT_EQ(layer->getBatchSize(), BATCH_SIZE);
}

TEST(TestWorkflow, PreparePipelinesSimpleTopologyZeroingUnit)
{
    PROFILE_TEST

    raul::Workflow w;

    w.add<TestLayer>("f0", raul::BasicParams{ {}, { "x1" } }, true);
    w.add<TestLayer>("f1", raul::BasicParams{ { "x1" }, { "x2" } }, true);
    w.add<TestLayer>("f2", raul::BasicParams{ { "x2" }, { "x3" } }, true);
    w.add<TestLayer>("f3", raul::BasicParams{ { "x3" }, {} }, true);

    static_cast<TestLayer*>(w["f0"])->setExpectGrad(3.0_dt);
    static_cast<TestLayer*>(w["f1"])->setExpectGrad(2.0_dt);
    static_cast<TestLayer*>(w["f2"])->setExpectGrad(1.0_dt);

    EXPECT_NO_THROW(w.preparePipelines());
    EXPECT_NO_THROW(w.prepareMemoryForTraining());
    EXPECT_NO_THROW(w.setBatchSize(10));

    EXPECT_NO_THROW(w.forwardPassTraining());
    EXPECT_NO_THROW(w.backwardPassTraining());

    static_cast<TestLayer*>(w["f0"])->setPerformGradWeightsChecks(false);
    static_cast<TestLayer*>(w["f1"])->setPerformGradWeightsChecks(false);
    static_cast<TestLayer*>(w["f2"])->setPerformGradWeightsChecks(false);
    static_cast<TestLayer*>(w["f3"])->setPerformGradWeightsChecks(false);

    EXPECT_NO_THROW(w.forwardPassTraining(false));
    EXPECT_NO_THROW(w.backwardPassTraining());

    raul::MemoryManager& mm = w.getMemoryManager();

    EXPECT_EQ(mm[(raul::Name("f0") / "Weights").grad()][0], 2_dt);
    EXPECT_EQ(mm[(raul::Name("f1") / "Weights").grad()][0], 2_dt);
    EXPECT_EQ(mm[(raul::Name("f2") / "Weights").grad()][0], 2_dt);
    EXPECT_EQ(mm[(raul::Name("f3") / "Weights").grad()][0], 2_dt);

    static_cast<TestLayer*>(w["f0"])->setPerformGradWeightsChecks(true);
    static_cast<TestLayer*>(w["f1"])->setPerformGradWeightsChecks(true);
    static_cast<TestLayer*>(w["f2"])->setPerformGradWeightsChecks(true);
    static_cast<TestLayer*>(w["f3"])->setPerformGradWeightsChecks(true);

    EXPECT_NO_THROW(w.forwardPassTraining(true));
    EXPECT_NO_THROW(w.backwardPassTraining());
}

TEST(TestWorkflow, PreparePipelinesSimpleTopologyWShapeUnit)
{
    PROFILE_TEST

    class Test : public raul::BasicLayer
    {
      public:
        Test(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
            : BasicLayer(name, "test", params, networkParameters, { false, false })
        {
            if (!params.getOutputs().empty())
            {
                mNetworkParams.mWorkflow.tensorNeeded(
                    name, params.getOutputs()[0], raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, true, true, false, false, false);
            }
        }

        void forwardComputeImpl(raul::NetworkMode) override
        {
            auto& mm = mNetworkParams.mWorkflow.getMemoryManager();
            size_t bs = mNetworkParams.mWorkflow.getBatchSize();

            const raul::Tensor& t = mm[mOutputs[0]];

            EXPECT_EQ(t.size(), bs);

            auto wShape = mNetworkParams.mWorkflow.getShape(mOutputs[0]);
            EXPECT_TRUE(wShape.isBSDependent());

            auto shape = wShape.getShape(mNetworkParams.mWorkflow);
            EXPECT_EQ(shape[0], bs);
            EXPECT_EQ(shape[1], 1u);
            EXPECT_EQ(shape[2], 1u);
            EXPECT_EQ(shape[3], 1u);

            EXPECT_EQ(mNetworkParams.mWorkflow.getBatch(mOutputs[0]), bs);
            EXPECT_EQ(mNetworkParams.mWorkflow.getDepth(mOutputs[0]), 1u);
            EXPECT_EQ(mNetworkParams.mWorkflow.getHeight(mOutputs[0]), 1u);
            EXPECT_EQ(mNetworkParams.mWorkflow.getWidth(mOutputs[0]), 1u);
        }
        void backwardComputeImpl() override {}
    };

    class Test2 : public raul::BasicLayer
    {
      public:
        Test2(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
            : BasicLayer(name, "test", params, networkParameters, { false, false })
        {
            if (!params.getInputs().empty())
            {
                mNetworkParams.mWorkflow.copyDeclaration(name, params.getInputs()[0], raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
            }

            if (!params.getOutputs().empty())
            {
                mNetworkParams.mWorkflow.tensorNeeded(
                    name, params.getOutputs()[0], raul::WShape{ 1u, 1u, raul::BS(8u), 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, true, true, false, false, false);
            }
        }

        void forwardComputeImpl(raul::NetworkMode) override
        {
            auto& mm = mNetworkParams.mWorkflow.getMemoryManager();
            size_t bs = mNetworkParams.mWorkflow.getBatchSize();

            {
                const raul::Tensor& t = mm[mInputs[0]];

                EXPECT_EQ(t.size(), bs);

                auto wShape = mNetworkParams.mWorkflow.getShape(mInputs[0]);
                EXPECT_TRUE(wShape.isBSDependent());

                auto shape = wShape.getShape(mNetworkParams.mWorkflow);
                EXPECT_EQ(shape[0], bs);
                EXPECT_EQ(shape[1], 1u);
                EXPECT_EQ(shape[2], 1u);
                EXPECT_EQ(shape[3], 1u);

                EXPECT_EQ(mNetworkParams.mWorkflow.getBatch(mInputs[0]), bs);
                EXPECT_EQ(mNetworkParams.mWorkflow.getDepth(mInputs[0]), 1u);
                EXPECT_EQ(mNetworkParams.mWorkflow.getHeight(mInputs[0]), 1u);
                EXPECT_EQ(mNetworkParams.mWorkflow.getWidth(mInputs[0]), 1u);
            }

            {
                const raul::Tensor& t = mm[mOutputs[0]];

                EXPECT_EQ(t.size(), bs * 8u);

                auto wShape = mNetworkParams.mWorkflow.getShape(mOutputs[0]);
                EXPECT_TRUE(wShape.isBSDependent());

                auto shape = wShape.getShape(mNetworkParams.mWorkflow);
                EXPECT_EQ(shape[0], 1u);
                EXPECT_EQ(shape[1], 1u);
                EXPECT_EQ(shape[2], bs * 8u);
                EXPECT_EQ(shape[3], 1u);

                EXPECT_EQ(mNetworkParams.mWorkflow.getBatch(mOutputs[0]), 1u);
                EXPECT_EQ(mNetworkParams.mWorkflow.getDepth(mOutputs[0]), 1u);
                EXPECT_EQ(mNetworkParams.mWorkflow.getHeight(mOutputs[0]), bs * 8u);
                EXPECT_EQ(mNetworkParams.mWorkflow.getWidth(mOutputs[0]), 1u);
            }
        }
        void backwardComputeImpl() override {}
    };

    raul::Workflow w;

    w.add<Test>("f0", raul::BasicParams{ {}, { "x1" } });
    w.add<Test2>("f1", raul::BasicParams{ { "x1" }, { "x2" } });

    w.preparePipelines();
    w.prepareMemoryForTraining();
    w.setBatchSize(10u);

    w.forwardPassTesting();

    w.setBatchSize(20u);
    w.forwardPassTesting();
}

TEST(TestWorkflow, PreparePipelinesSimpleTopologyEagerUnit)
{
    PROFILE_TEST

    raul::WorkflowEager w;

    w.add<TestLayer>("f0", raul::BasicParams{ {}, { "x1" } }, true);
    w.add<TestLayer>("f1", raul::BasicParams{ { "x1" }, { "x2" } }, true);
    w.add<TestLayer>("f2", raul::BasicParams{ { "x2" }, { "x3" } }, true);
    w.add<TestLayer>("f3", raul::BasicParams{ { "x3" }, {} }, true);

    static_cast<TestLayer*>(w["f0"])->setExpectGrad(3.0_dt);
    static_cast<TestLayer*>(w["f1"])->setExpectGrad(2.0_dt);
    static_cast<TestLayer*>(w["f2"])->setExpectGrad(1.0_dt);

    static_cast<TestLayer*>(w["f0"])->setPerformSizeChecks(false);
    static_cast<TestLayer*>(w["f1"])->setPerformSizeChecks(false);
    static_cast<TestLayer*>(w["f2"])->setPerformSizeChecks(false);
    static_cast<TestLayer*>(w["f3"])->setPerformSizeChecks(false);

    EXPECT_NO_THROW(w.preparePipelines());
    EXPECT_NO_THROW(w.setBatchSize(10));
    EXPECT_NO_THROW(w.prepareMemoryForTraining());

    raul::MemoryManager& mm = w.getMemoryManager();

    EXPECT_EQ(mm["x1"].size(), 10u);
    EXPECT_EQ(mm["x2"].size(), 10u);
    EXPECT_EQ(mm["x3"].size(), 10u);

    EXPECT_EQ(mm[raul::Name("x1").grad()].size(), 10u);
    EXPECT_EQ(mm[raul::Name("x2").grad()].size(), 10u);
    EXPECT_EQ(mm[raul::Name("x3").grad()].size(), 10u);

    EXPECT_NO_THROW(w.setBatchSize(20));

    EXPECT_EQ(mm["x1"].size(), 20u);
    EXPECT_EQ(mm["x2"].size(), 20u);
    EXPECT_EQ(mm["x3"].size(), 20u);

    EXPECT_EQ(mm[raul::Name("x1").grad()].size(), 20u);
    EXPECT_EQ(mm[raul::Name("x2").grad()].size(), 20u);
    EXPECT_EQ(mm[raul::Name("x3").grad()].size(), 20u);

    EXPECT_NO_THROW(w.forwardPassTraining());
    EXPECT_NO_THROW(w.backwardPassTraining());

    EXPECT_NO_THROW(w.forwardPassTraining());
    EXPECT_NO_THROW(w.backwardPassTraining());

    static_cast<TestLayer*>(w["f0"])->setExpectGrad(6.0_dt);
    static_cast<TestLayer*>(w["f1"])->setExpectGrad(5.0_dt);
    static_cast<TestLayer*>(w["f2"])->setExpectGrad(2.0_dt);

    static_cast<TestLayer*>(w["f0"])->setPerformGradWeightsChecks(false);
    static_cast<TestLayer*>(w["f1"])->setPerformGradWeightsChecks(false);
    static_cast<TestLayer*>(w["f2"])->setPerformGradWeightsChecks(false);
    static_cast<TestLayer*>(w["f3"])->setPerformGradWeightsChecks(false);

    EXPECT_NO_THROW(w["f3"]->backwardCompute());
    EXPECT_NO_THROW(w["f2"]->backwardCompute());
    EXPECT_NO_THROW(w["f1"]->backwardCompute());
    EXPECT_NO_THROW(w["f0"]->backwardCompute());
}

TEST(TestWorkflow, PreparePipelinesResidualTopologyUnit)
{
    PROFILE_TEST

    raul::Workflow w;

    w.add<TestLayer>("f0", raul::BasicParams{ {}, { "x1" } }, true);
    w.add<TestLayer>("f1", raul::BasicParams{ { "x1" }, { "x2" } }, true);
    w.add<TestLayer>("f2", raul::BasicParams{ { "x1" }, { "x3" } }, true);
    w.add<TestLayer>("f3", raul::BasicParams{ { "x2", "x3" }, { "x4" } }, true);

    static_cast<TestLayer*>(w["f1"])->setExpectGrad(4.0_dt);
    static_cast<TestLayer*>(w["f2"])->setExpectGrad(2.0_dt);
    static_cast<TestLayer*>(w["f3"])->setExpectGrad(1.0_dt);

    class Loss : public raul::BasicLayer
    {
      public:
        Loss(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
            : BasicLayer(name, "test", params, networkParameters, { false, false })
        {
            for (auto& input : params.getInputs())
            {
                mNetworkParams.mWorkflow.tensorNeeded(
                    name, input.grad(), raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::Backward, raul::Workflow::Mode::Write, true, true, false, true, false);
            }

            // batched not optimized
            mNetworkParams.mWorkflow.tensorNeeded(
                name, "T1", raul::WShape{ raul::BS(), 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, false, false, false, false, false);

            // not batched optimized
            mNetworkParams.mWorkflow.tensorNeeded(name, "T2", raul::WShape{ 50u, 1u, 1u, 1u }, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read, false, true, false, false, false);
        }

        void forwardComputeImpl(raul::NetworkMode) override
        {
            ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists("T1"));
            ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists("T2"));

            ASSERT_EQ(mNetworkParams.mMemoryManager["T1"].size(), 1u * mNetworkParams.mWorkflow.getBatchSize());
            ASSERT_EQ(mNetworkParams.mMemoryManager["T2"].size(), 50u);
        }

        void backwardComputeImpl() override
        {
            ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists("T1"));
            ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists("T2"));

            ASSERT_EQ(mNetworkParams.mMemoryManager["T1"].size(), 1u * mNetworkParams.mWorkflow.getBatchSize());
            ASSERT_EQ(mNetworkParams.mMemoryManager["T2"].size(), 50u);

            for (auto& input : mInputs)
            {
                ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists(input));
                ASSERT_EQ(mNetworkParams.mMemoryManager[input].size(), 0u);

                ASSERT_TRUE(mNetworkParams.mMemoryManager.tensorExists(input.grad()));
                ASSERT_EQ(mNetworkParams.mMemoryManager[input.grad()].size(), 1u * mNetworkParams.mWorkflow.getBatchSize());
            }
        }
    };

    w.add<Loss>("f4", raul::BasicParams{ { "x4" }, {} });

    w.preparePipelines();

    ASSERT_EQ(w["f0"]->getName(), "f0");
    ASSERT_EQ(w["f1"]->getName(), "f1");
    ASSERT_EQ(w["f2"]->getName(), "f2");
    ASSERT_EQ(w["f3"]->getName(), "f3");
    ASSERT_EQ(w["f4"]->getName(), "f4");

    EXPECT_THROW(w.getBatchSize(), raul::Exception);

    // create batched pipe
    {
        const raul::Workflow::Pipeline& pipeCreateBatched = w.getPipeline(raul::Workflow::Pipelines::CreateBatched);

        std::unordered_set<raul::Name> tNames;

        ASSERT_EQ(pipeCreateBatched.size(), 9u + 5u);

        size_t createShapeCount = 0;
        size_t createTensorCount = 0;

        for (size_t q = 0; q < 9; ++q)
        {
            if (pipeCreateBatched[q]->type() == "CreateShape") ++createShapeCount;
            if (pipeCreateBatched[q]->type() == "CreateTensor") ++createTensorCount; // T1
            tNames.insert(static_cast<raul::TensorAction<raul::MemoryManager>*>(pipeCreateBatched[q].get())->mName);
        }

        ASSERT_EQ(createShapeCount, 8u);
        ASSERT_EQ(createTensorCount, 1u); // T1

        ASSERT_NE(tNames.find("x1"), tNames.end());
        ASSERT_NE(tNames.find("x2"), tNames.end());
        ASSERT_NE(tNames.find("x3"), tNames.end());
        ASSERT_NE(tNames.find("x4"), tNames.end());
        ASSERT_NE(tNames.find("T1"), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x1")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x2")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x3")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x4")), tNames.end());
    }

    // delete batched pipe
    {
        const raul::Workflow::Pipeline& pipeDeleteBatched = w.getPipeline(raul::Workflow::Pipelines::DeleteBatched);

        std::unordered_set<raul::Name> tNames;

        ASSERT_EQ(pipeDeleteBatched.size(), 9u);

        for (size_t q = 0; q < 9; ++q)
        {
            ASSERT_EQ(pipeDeleteBatched[q]->type(), "DeleteTensor");
            tNames.insert(static_cast<raul::TensorAction<raul::MemoryManager>*>(pipeDeleteBatched[q].get())->mName);
        }

        ASSERT_NE(tNames.find("x1"), tNames.end());
        ASSERT_NE(tNames.find("x2"), tNames.end());
        ASSERT_NE(tNames.find("x3"), tNames.end());
        ASSERT_NE(tNames.find("x4"), tNames.end());
        ASSERT_NE(tNames.find("T1"), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x1")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x2")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x3")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x4")), tNames.end());
    }

    // create not batched pipe
    {
        const raul::Workflow::Pipeline& pipeCreateNotBatched = w.getPipeline(raul::Workflow::Pipelines::CreateNotBatched);

        std::unordered_set<raul::Name> tNames;

        ASSERT_EQ(pipeCreateNotBatched.size(), 9u + 5u);

        size_t createShapeCount = 0;
        size_t createTensorCount = 0;

        for (size_t q = 0; q < 9; ++q)
        {
            if (pipeCreateNotBatched[q]->type() == "CreateTensor") ++createTensorCount;
            if (pipeCreateNotBatched[q]->type() == "CreateShape") ++createShapeCount; // T2
            tNames.insert(static_cast<raul::TensorAction<raul::MemoryManager>*>(pipeCreateNotBatched[q].get())->mName);
        }

        ASSERT_EQ(createTensorCount, 8u);
        ASSERT_EQ(createShapeCount, 1u); // T2

        ASSERT_NE(tNames.find("T2"), tNames.end());
        ASSERT_NE(tNames.find(raul::Name("f0") / "Weights"), tNames.end());
        ASSERT_NE(tNames.find(raul::Name("f1") / "Weights"), tNames.end());
        ASSERT_NE(tNames.find(raul::Name("f2") / "Weights"), tNames.end());
        ASSERT_NE(tNames.find(raul::Name("f3") / "Weights"), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f0") / "Weights")), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f1") / "Weights")), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f2") / "Weights")), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f3") / "Weights")), tNames.end());
    }

    // zero pipe
    {
        std::unordered_set<raul::Name> tNames;

        const raul::Workflow::Pipeline& pipeZero = w.getPipeline(raul::Workflow::Pipelines::Zero);

        ASSERT_EQ(pipeZero.size(), 4u);

        for (size_t q = 0; q < 4; ++q)
        {
            ASSERT_EQ(pipeZero[q]->type(), "Zero");
            tNames.insert(static_cast<raul::TensorAction<raul::MemoryManager>*>(pipeZero[q].get())->mName);
        }

        ASSERT_NE(tNames.find(getGradName(raul::Name("f0") / "Weights")), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f1") / "Weights")), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f2") / "Weights")), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f3") / "Weights")), tNames.end());
    }

    // forward test pipe
    {
        const raul::Workflow::Pipeline& pipeForwardTest = w.getPipeline(raul::Workflow::Pipelines::ForwardTest);

        ASSERT_EQ(pipeForwardTest.size(), 15u);

        ASSERT_EQ(pipeForwardTest[0]->type(), "Allocate");    // x1
        ASSERT_EQ(pipeForwardTest[1]->type(), "Forward");     // f0
        ASSERT_EQ(pipeForwardTest[2]->type(), "Allocate");    // x2
        ASSERT_EQ(pipeForwardTest[3]->type(), "Forward");     // f1
        ASSERT_EQ(pipeForwardTest[4]->type(), "Allocate");    // x3
        ASSERT_EQ(pipeForwardTest[5]->type(), "Forward");     // f2
        ASSERT_EQ(pipeForwardTest[6]->type(), "Deallocate");  // x1
        ASSERT_EQ(pipeForwardTest[7]->type(), "Allocate");    // x4
        ASSERT_EQ(pipeForwardTest[8]->type(), "Forward");     // f3
        ASSERT_EQ(pipeForwardTest[9]->type(), "Deallocate");  // x2
        ASSERT_EQ(pipeForwardTest[10]->type(), "Deallocate"); // x3
        ASSERT_EQ(pipeForwardTest[11]->type(), "Deallocate"); // x4
        ASSERT_EQ(pipeForwardTest[12]->type(), "Allocate");   // T2
        ASSERT_EQ(pipeForwardTest[13]->type(), "Forward");    // f4
        ASSERT_EQ(pipeForwardTest[14]->type(), "Deallocate"); // T2

        ASSERT_TRUE(checkName(pipeForwardTest[0].get(), "x1"));
        ASSERT_TRUE(checkName(pipeForwardTest[1].get(), "f0"));
        ASSERT_TRUE(checkName(pipeForwardTest[2].get(), "x2"));
        ASSERT_TRUE(checkName(pipeForwardTest[3].get(), "f1"));
        ASSERT_TRUE(checkName(pipeForwardTest[4].get(), "x3"));
        ASSERT_TRUE(checkName(pipeForwardTest[5].get(), "f2"));
        ASSERT_TRUE(checkName(pipeForwardTest[6].get(), "x1"));
        ASSERT_TRUE(checkName(pipeForwardTest[7].get(), "x4"));
        ASSERT_TRUE(checkName(pipeForwardTest[8].get(), "f3"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeForwardTest[9].get()), getName(pipeForwardTest[10].get()), getName(pipeForwardTest[11].get()) }, { "x2", "x3", "x4" }));
        ASSERT_TRUE(checkName(pipeForwardTest[12].get(), "T2"));
        ASSERT_TRUE(checkName(pipeForwardTest[13].get(), "f4"));
        ASSERT_TRUE(checkName(pipeForwardTest[14].get(), "T2"));
    }

    // forward train pipe
    {
        const raul::Workflow::Pipeline& pipeForwardTrain = w.getPipeline(raul::Workflow::Pipelines::ForwardTrain);

        ASSERT_EQ(pipeForwardTrain.size(), 11u);

        ASSERT_EQ(pipeForwardTrain[0]->type(), "Allocate");   // x1
        ASSERT_EQ(pipeForwardTrain[1]->type(), "Forward");    // f0
        ASSERT_EQ(pipeForwardTrain[2]->type(), "Allocate");   // x2
        ASSERT_EQ(pipeForwardTrain[3]->type(), "Forward");    // f1
        ASSERT_EQ(pipeForwardTrain[4]->type(), "Allocate");   // x3
        ASSERT_EQ(pipeForwardTrain[5]->type(), "Forward");    // f2
        ASSERT_EQ(pipeForwardTrain[6]->type(), "Allocate");   // x4
        ASSERT_EQ(pipeForwardTrain[7]->type(), "Forward");    // f3
        ASSERT_EQ(pipeForwardTrain[8]->type(), "Deallocate"); // x4
        ASSERT_EQ(pipeForwardTrain[9]->type(), "Allocate");   // T2
        ASSERT_EQ(pipeForwardTrain[10]->type(), "Forward");   // f4

        ASSERT_TRUE(checkName(pipeForwardTrain[0].get(), "x1"));
        ASSERT_TRUE(checkName(pipeForwardTrain[1].get(), "f0"));
        ASSERT_TRUE(checkName(pipeForwardTrain[2].get(), "x2"));
        ASSERT_TRUE(checkName(pipeForwardTrain[3].get(), "f1"));
        ASSERT_TRUE(checkName(pipeForwardTrain[4].get(), "x3"));
        ASSERT_TRUE(checkName(pipeForwardTrain[5].get(), "f2"));
        ASSERT_TRUE(checkName(pipeForwardTrain[6].get(), "x4"));
        ASSERT_TRUE(checkName(pipeForwardTrain[7].get(), "f3"));
        ASSERT_TRUE(checkName(pipeForwardTrain[8].get(), "x4"));
        ASSERT_TRUE(checkName(pipeForwardTrain[9].get(), "T2"));
        ASSERT_TRUE(checkName(pipeForwardTrain[10].get(), "f4"));
    }

    // backward train pipe
    {
        const raul::Workflow::Pipeline& pipeBackwardTrain = w.getPipeline(raul::Workflow::Pipelines::BackwardTrain);

        ASSERT_EQ(pipeBackwardTrain.size(), 21u);

        ASSERT_EQ(pipeBackwardTrain[0]->type(), "Allocate");    // xg4
        ASSERT_EQ(pipeBackwardTrain[1]->type(), "Zero");        // xg4
        ASSERT_EQ(pipeBackwardTrain[2]->type(), "Backward");    // f4
        ASSERT_EQ(pipeBackwardTrain[3]->type(), "Deallocate");  // T2
        ASSERT_EQ(pipeBackwardTrain[4]->type(), "Allocate");    // xg2
        ASSERT_EQ(pipeBackwardTrain[5]->type(), "Zero");        // xg2
        ASSERT_EQ(pipeBackwardTrain[6]->type(), "Allocate");    // xg3
        ASSERT_EQ(pipeBackwardTrain[7]->type(), "Zero");        // xg3
        ASSERT_EQ(pipeBackwardTrain[8]->type(), "Backward");    // f3
        ASSERT_EQ(pipeBackwardTrain[9]->type(), "Deallocate");  // x2
        ASSERT_EQ(pipeBackwardTrain[10]->type(), "Deallocate"); // x3
        ASSERT_EQ(pipeBackwardTrain[11]->type(), "Deallocate"); // xg4
        ASSERT_EQ(pipeBackwardTrain[12]->type(), "Allocate");   // xg1
        ASSERT_EQ(pipeBackwardTrain[13]->type(), "Zero");       // xg1
        ASSERT_EQ(pipeBackwardTrain[14]->type(), "Backward");   // f2
        ASSERT_EQ(pipeBackwardTrain[15]->type(), "Deallocate"); // x3
        ASSERT_EQ(pipeBackwardTrain[16]->type(), "Backward");   // f1
        ASSERT_EQ(pipeBackwardTrain[17]->type(), "Deallocate"); // xg2
        ASSERT_EQ(pipeBackwardTrain[18]->type(), "Deallocate"); // x1
        ASSERT_EQ(pipeBackwardTrain[19]->type(), "Backward");   // f0
        ASSERT_EQ(pipeBackwardTrain[20]->type(), "Deallocate"); // xg1

        ASSERT_TRUE(checkName(pipeBackwardTrain[0].get(), getGradName("x4")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[1].get(), getGradName("x4")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[2].get(), "f4"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[3].get(), "T2"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[4].get()), getName(pipeBackwardTrain[5].get()), getName(pipeBackwardTrain[6].get()), getName(pipeBackwardTrain[7].get()) },
                                     { getGradName("x2"), getGradName("x3") }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[8].get(), "f3"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[9].get()), getName(pipeBackwardTrain[10].get()), getName(pipeBackwardTrain[11].get()) }, { "x2", "x3", getGradName("x4") }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[12].get(), getGradName("x1")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[13].get(), getGradName("x1")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[14].get(), "f2"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[15].get(), getGradName("x3")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[16].get(), "f1"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[17].get()), getName(pipeBackwardTrain[18].get()) }, { "x1", getGradName("x2") }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[19].get(), "f0"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[20].get(), getGradName("x1")));
    }

    EXPECT_THROW(w.forwardPassTesting(), raul::Exception);
    EXPECT_THROW(w.forwardPassTraining(), raul::Exception);
    EXPECT_THROW(w.backwardPassTraining(), raul::Exception);

    w.setBatchSize(10);
    ASSERT_EQ(w.getBatchSize(), 10u);
    ASSERT_EQ(w.getMemoryManager()["x1"].getBatchSize(), 10u);
    ASSERT_EQ(w.getMemoryManager()["x2"].getBatchSize(), 10u);
    ASSERT_EQ(w.getMemoryManager()["x3"].getBatchSize(), 10u);
    ASSERT_EQ(w.getMemoryManager()["x4"].getBatchSize(), 10u);
    ASSERT_EQ(w.getMemoryManager()["T1"].getBatchSize(), 10u);
    ASSERT_EQ(w.getMemoryManager()[getGradName("x1")].getBatchSize(), 10u);
    ASSERT_EQ(w.getMemoryManager()[getGradName("x2")].getBatchSize(), 10u);
    ASSERT_EQ(w.getMemoryManager()[getGradName("x3")].getBatchSize(), 10u);
    ASSERT_EQ(w.getMemoryManager()[getGradName("x4")].getBatchSize(), 10u);
    ASSERT_EQ(w.getMemoryManager()["x1"].size(), 0u);
    ASSERT_EQ(w.getMemoryManager()["x2"].size(), 0u);
    ASSERT_EQ(w.getMemoryManager()["x3"].size(), 0u);
    ASSERT_EQ(w.getMemoryManager()["x4"].size(), 0u);
    ASSERT_EQ(w.getMemoryManager()["T1"].size(), 10u);
    ASSERT_EQ(w.getMemoryManager()[getGradName("x1")].size(), 0u);
    ASSERT_EQ(w.getMemoryManager()[getGradName("x2")].size(), 0u);
    ASSERT_EQ(w.getMemoryManager()[getGradName("x3")].size(), 0u);
    ASSERT_EQ(w.getMemoryManager()[getGradName("x4")].size(), 0u);

    w.setBatchSize(2);
    ASSERT_EQ(w.getBatchSize(), 2u);
    ASSERT_EQ(w.getMemoryManager()["x1"].getBatchSize(), 2u);
    ASSERT_EQ(w.getMemoryManager()["x2"].getBatchSize(), 2u);
    ASSERT_EQ(w.getMemoryManager()["x3"].getBatchSize(), 2u);
    ASSERT_EQ(w.getMemoryManager()["x4"].getBatchSize(), 2u);
    ASSERT_EQ(w.getMemoryManager()["T1"].getBatchSize(), 2u);
    ASSERT_EQ(w.getMemoryManager()[getGradName("x1")].getBatchSize(), 2u);
    ASSERT_EQ(w.getMemoryManager()[getGradName("x2")].getBatchSize(), 2u);
    ASSERT_EQ(w.getMemoryManager()[getGradName("x3")].getBatchSize(), 2u);
    ASSERT_EQ(w.getMemoryManager()[getGradName("x4")].getBatchSize(), 2u);
    ASSERT_EQ(w.getMemoryManager()["x1"].size(), 0u);
    ASSERT_EQ(w.getMemoryManager()["x2"].size(), 0u);
    ASSERT_EQ(w.getMemoryManager()["x3"].size(), 0u);
    ASSERT_EQ(w.getMemoryManager()["x4"].size(), 0u);
    ASSERT_EQ(w.getMemoryManager()["T1"].size(), 2u);
    ASSERT_EQ(w.getMemoryManager()[getGradName("x1")].size(), 0u);
    ASSERT_EQ(w.getMemoryManager()[getGradName("x2")].size(), 0u);
    ASSERT_EQ(w.getMemoryManager()[getGradName("x3")].size(), 0u);
    ASSERT_EQ(w.getMemoryManager()[getGradName("x4")].size(), 0u);

    ASSERT_FALSE(w.getMemoryManager().tensorExists(raul::Name("f0") / "Weights"));
    ASSERT_FALSE(w.getMemoryManager().tensorExists("T2"));

    EXPECT_THROW(w.forwardPassTesting(), raul::Exception);
    EXPECT_THROW(w.forwardPassTraining(), raul::Exception);
    EXPECT_THROW(w.backwardPassTraining(), raul::Exception);

    w.prepareMemoryForTraining();

    ASSERT_TRUE(w.getMemoryManager().tensorExists(raul::Name("f0") / "Weights"));
    ASSERT_TRUE(w.getMemoryManager().tensorExists("T2"));

    ASSERT_EQ(w.getMemoryManager()[raul::Name("f0") / "Weights"].getBatchSize(), 1u);
    ASSERT_EQ(w.getMemoryManager()[raul::Name("f1") / "Weights"].getBatchSize(), 1u);
    ASSERT_EQ(w.getMemoryManager()[raul::Name("f2") / "Weights"].getBatchSize(), 1u);
    ASSERT_EQ(w.getMemoryManager()[raul::Name("f3") / "Weights"].getBatchSize(), 1u);
    ASSERT_EQ(w.getMemoryManager()[getGradName(raul::Name("f0") / "Weights")].getBatchSize(), 1u);
    ASSERT_EQ(w.getMemoryManager()[getGradName(raul::Name("f1") / "Weights")].getBatchSize(), 1u);
    ASSERT_EQ(w.getMemoryManager()[getGradName(raul::Name("f2") / "Weights")].getBatchSize(), 1u);
    ASSERT_EQ(w.getMemoryManager()[getGradName(raul::Name("f3") / "Weights")].getBatchSize(), 1u);
    ASSERT_EQ(w.getMemoryManager()["T2"].getBatchSize(), 50u);
    ASSERT_EQ(w.getMemoryManager()[raul::Name("f0") / "Weights"].size(), 1u);
    ASSERT_EQ(w.getMemoryManager()[raul::Name("f1") / "Weights"].size(), 1u);
    ASSERT_EQ(w.getMemoryManager()[raul::Name("f2") / "Weights"].size(), 1u);
    ASSERT_EQ(w.getMemoryManager()[raul::Name("f3") / "Weights"].size(), 1u);
    ASSERT_EQ(w.getMemoryManager()[getGradName(raul::Name("f0") / "Weights")].size(), 1u);
    ASSERT_EQ(w.getMemoryManager()[getGradName(raul::Name("f1") / "Weights")].size(), 1u);
    ASSERT_EQ(w.getMemoryManager()[getGradName(raul::Name("f2") / "Weights")].size(), 1u);
    ASSERT_EQ(w.getMemoryManager()[getGradName(raul::Name("f3") / "Weights")].size(), 1u);
    ASSERT_EQ(w.getMemoryManager()["T2"].size(), 0u);

    EXPECT_THROW(w.prepareMemoryForTraining(), raul::Exception);

    EXPECT_NO_THROW(w.forwardPassTesting());
    EXPECT_NO_THROW(w.forwardPassTraining());
    EXPECT_NO_THROW(w.backwardPassTraining());
    EXPECT_NO_THROW(w.forwardPassTesting());

    ASSERT_EQ(static_cast<TestLayer*>(w["f0"])->getForwardCountTest(), 2u);
    ASSERT_EQ(static_cast<TestLayer*>(w["f1"])->getForwardCountTest(), 2u);
    ASSERT_EQ(static_cast<TestLayer*>(w["f2"])->getForwardCountTest(), 2u);
    ASSERT_EQ(static_cast<TestLayer*>(w["f3"])->getForwardCountTest(), 2u);

    ASSERT_EQ(static_cast<TestLayer*>(w["f0"])->getForwardCountTrain(), 1u);
    ASSERT_EQ(static_cast<TestLayer*>(w["f1"])->getForwardCountTrain(), 1u);
    ASSERT_EQ(static_cast<TestLayer*>(w["f2"])->getForwardCountTrain(), 1u);
    ASSERT_EQ(static_cast<TestLayer*>(w["f3"])->getForwardCountTrain(), 1u);

    ASSERT_EQ(static_cast<TestLayer*>(w["f0"])->getBackwardCount(), 1u);
    ASSERT_EQ(static_cast<TestLayer*>(w["f1"])->getBackwardCount(), 1u);
    ASSERT_EQ(static_cast<TestLayer*>(w["f2"])->getBackwardCount(), 1u);
    ASSERT_EQ(static_cast<TestLayer*>(w["f3"])->getBackwardCount(), 1u);

    EXPECT_NO_THROW(w.forwardPassTraining());
    EXPECT_NO_THROW(w.backwardPassTraining());

    ASSERT_EQ(static_cast<TestLayer*>(w["f0"])->getForwardCountTrain(), 2u);
    ASSERT_EQ(static_cast<TestLayer*>(w["f1"])->getForwardCountTrain(), 2u);
    ASSERT_EQ(static_cast<TestLayer*>(w["f2"])->getForwardCountTrain(), 2u);
    ASSERT_EQ(static_cast<TestLayer*>(w["f3"])->getForwardCountTrain(), 2u);

    ASSERT_EQ(static_cast<TestLayer*>(w["f0"])->getBackwardCount(), 2u);
    ASSERT_EQ(static_cast<TestLayer*>(w["f1"])->getBackwardCount(), 2u);
    ASSERT_EQ(static_cast<TestLayer*>(w["f2"])->getBackwardCount(), 2u);
    ASSERT_EQ(static_cast<TestLayer*>(w["f3"])->getBackwardCount(), 2u);
}

TEST(TestWorkflow, PreparePipelinesComplexTopologyUnit)
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

    EXPECT_THROW(w["f8"], raul::Exception);

    static_cast<TestLayer*>(w["f1"])->setExpectGrad(10.0_dt);
    static_cast<TestLayer*>(w["f2"])->setExpectGrad(5.0_dt);
    static_cast<TestLayer*>(w["f3"])->setExpectGrad(2.0_dt);
    static_cast<TestLayer*>(w["f4"])->setExpectGrad(4.0_dt);
    static_cast<TestLayer*>(w["f5"])->setExpectGrad(2.0_dt);
    static_cast<TestLayer*>(w["f6"])->setExpectGrad(1.0_dt);

    EXPECT_NO_THROW(w.preparePipelines());
    EXPECT_NO_THROW(w.prepareMemoryForTraining());
    EXPECT_NO_THROW(w.setBatchSize(10));

    // create batched pipe
    {
        const raul::Workflow::Pipeline& pipeCreateBatched = w.getPipeline(raul::Workflow::Pipelines::CreateBatched);

        std::unordered_set<raul::Name> tNames;

        ASSERT_EQ(pipeCreateBatched.size(), 14u + 8u);

        for (size_t q = 0; q < 14; ++q)
        {
            ASSERT_EQ(pipeCreateBatched[q]->type(), "CreateShape");
            tNames.insert(static_cast<raul::TensorAction<raul::MemoryManager>*>(pipeCreateBatched[q].get())->mName);
        }

        ASSERT_NE(tNames.find("x1"), tNames.end());
        ASSERT_NE(tNames.find("x2"), tNames.end());
        ASSERT_NE(tNames.find("x3"), tNames.end());
        ASSERT_NE(tNames.find("x4"), tNames.end());
        ASSERT_NE(tNames.find("x5"), tNames.end());
        ASSERT_NE(tNames.find("x6"), tNames.end());
        ASSERT_NE(tNames.find("x7"), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x1")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x2")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x3")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x4")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x5")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x6")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x7")), tNames.end());
    }

    // delete batched pipe
    {
        const raul::Workflow::Pipeline& pipeDeleteBatched = w.getPipeline(raul::Workflow::Pipelines::DeleteBatched);

        std::unordered_set<raul::Name> tNames;

        ASSERT_EQ(pipeDeleteBatched.size(), 14u);

        for (size_t q = 0; q < 14; ++q)
        {
            ASSERT_EQ(pipeDeleteBatched[q]->type(), "DeleteTensor");
            tNames.insert(static_cast<raul::TensorAction<raul::MemoryManager>*>(pipeDeleteBatched[q].get())->mName);
        }

        ASSERT_NE(tNames.find("x1"), tNames.end());
        ASSERT_NE(tNames.find("x2"), tNames.end());
        ASSERT_NE(tNames.find("x3"), tNames.end());
        ASSERT_NE(tNames.find("x4"), tNames.end());
        ASSERT_NE(tNames.find("x5"), tNames.end());
        ASSERT_NE(tNames.find("x6"), tNames.end());
        ASSERT_NE(tNames.find("x7"), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x1")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x2")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x3")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x4")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x5")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x6")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("x7")), tNames.end());
    }

    // create not batched pipe
    {
        const raul::Workflow::Pipeline& pipeCreateNotBatched = w.getPipeline(raul::Workflow::Pipelines::CreateNotBatched);

        std::unordered_set<raul::Name> tNames;

        ASSERT_EQ(pipeCreateNotBatched.size(), 14u + 8u);

        for (size_t q = 0; q < 14; ++q)
        {
            ASSERT_EQ(pipeCreateNotBatched[q]->type(), "CreateTensor");
            tNames.insert(static_cast<raul::TensorAction<raul::MemoryManager>*>(pipeCreateNotBatched[q].get())->mName);
        }

        ASSERT_NE(tNames.find("w1"), tNames.end());
        ASSERT_NE(tNames.find("w2"), tNames.end());
        ASSERT_NE(tNames.find(raul::Name("f0") / "Weights"), tNames.end());
        ASSERT_NE(tNames.find(raul::Name("f2") / "Weights"), tNames.end());
        ASSERT_NE(tNames.find(raul::Name("f4") / "Weights"), tNames.end());
        ASSERT_NE(tNames.find(raul::Name("f5") / "Weights"), tNames.end());
        ASSERT_NE(tNames.find(raul::Name("f6") / "Weights"), tNames.end());
        ASSERT_NE(tNames.find(getGradName("w1")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("w2")), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f0") / "Weights")), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f2") / "Weights")), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f4") / "Weights")), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f5") / "Weights")), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f6") / "Weights")), tNames.end());
    }

    // zero pipe
    {
        const raul::Workflow::Pipeline& pipeZero = w.getPipeline(raul::Workflow::Pipelines::Zero);

        std::unordered_set<raul::Name> tNames;

        ASSERT_EQ(pipeZero.size(), 7u);

        for (size_t q = 0; q < 7; ++q)
        {
            ASSERT_EQ(pipeZero[q]->type(), "Zero");
            tNames.insert(static_cast<raul::TensorAction<raul::MemoryManager>*>(pipeZero[q].get())->mName);
        }

        ASSERT_NE(tNames.find(getGradName("w1")), tNames.end());
        ASSERT_NE(tNames.find(getGradName("w2")), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f0") / "Weights")), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f2") / "Weights")), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f4") / "Weights")), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f5") / "Weights")), tNames.end());
        ASSERT_NE(tNames.find(getGradName(raul::Name("f6") / "Weights")), tNames.end());
    }

    // forward test pipe
    {
        const raul::Workflow::Pipeline& pipeForwardTest = w.getPipeline(raul::Workflow::Pipelines::ForwardTest);

        ASSERT_EQ(pipeForwardTest.size(), 22u);

        ASSERT_EQ(pipeForwardTest[0]->type(), "Allocate");    // x1
        ASSERT_EQ(pipeForwardTest[1]->type(), "Forward");     // f0
        ASSERT_EQ(pipeForwardTest[2]->type(), "Allocate");    // x2
        ASSERT_EQ(pipeForwardTest[3]->type(), "Forward");     // f1
        ASSERT_EQ(pipeForwardTest[4]->type(), "Allocate");    // x3
        ASSERT_EQ(pipeForwardTest[5]->type(), "Forward");     // f2
        ASSERT_EQ(pipeForwardTest[6]->type(), "Allocate");    // x4
        ASSERT_EQ(pipeForwardTest[7]->type(), "Forward");     // f3
        ASSERT_EQ(pipeForwardTest[8]->type(), "Deallocate");  // x1
        ASSERT_EQ(pipeForwardTest[9]->type(), "Allocate");    // x5
        ASSERT_EQ(pipeForwardTest[10]->type(), "Forward");    // f4
        ASSERT_EQ(pipeForwardTest[11]->type(), "Deallocate"); // x3
        ASSERT_EQ(pipeForwardTest[12]->type(), "Allocate");   // x6
        ASSERT_EQ(pipeForwardTest[13]->type(), "Forward");    // f5
        ASSERT_EQ(pipeForwardTest[14]->type(), "Deallocate"); // x2
        ASSERT_EQ(pipeForwardTest[15]->type(), "Allocate");   // x7
        ASSERT_EQ(pipeForwardTest[16]->type(), "Forward");    // f6
        ASSERT_EQ(pipeForwardTest[17]->type(), "Deallocate"); // x4
        ASSERT_EQ(pipeForwardTest[18]->type(), "Deallocate"); // x5
        ASSERT_EQ(pipeForwardTest[19]->type(), "Deallocate"); // x6
        ASSERT_EQ(pipeForwardTest[20]->type(), "Forward");    // f7
        ASSERT_EQ(pipeForwardTest[21]->type(), "Deallocate"); // x7

        ASSERT_TRUE(checkName(pipeForwardTest[0].get(), "x1"));
        ASSERT_TRUE(checkName(pipeForwardTest[1].get(), "f0"));
        ASSERT_TRUE(checkName(pipeForwardTest[2].get(), "x2"));
        ASSERT_TRUE(checkName(pipeForwardTest[3].get(), "f1"));
        ASSERT_TRUE(checkName(pipeForwardTest[4].get(), "x3"));
        ASSERT_TRUE(checkName(pipeForwardTest[5].get(), "f2"));
        ASSERT_TRUE(checkName(pipeForwardTest[6].get(), "x4"));
        ASSERT_TRUE(checkName(pipeForwardTest[7].get(), "f3"));
        ASSERT_TRUE(checkName(pipeForwardTest[8].get(), "x1"));
        ASSERT_TRUE(checkName(pipeForwardTest[9].get(), "x5"));
        ASSERT_TRUE(checkName(pipeForwardTest[10].get(), "f4"));
        ASSERT_TRUE(checkName(pipeForwardTest[11].get(), "x3"));
        ASSERT_TRUE(checkName(pipeForwardTest[12].get(), "x6"));
        ASSERT_TRUE(checkName(pipeForwardTest[13].get(), "f5"));
        ASSERT_TRUE(checkName(pipeForwardTest[14].get(), "x2"));
        ASSERT_TRUE(checkName(pipeForwardTest[15].get(), "x7"));
        ASSERT_TRUE(checkName(pipeForwardTest[16].get(), "f6"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeForwardTest[17].get()), getName(pipeForwardTest[18].get()), getName(pipeForwardTest[19].get()) }, { "x4", "x5", "x6" }));
        ASSERT_TRUE(checkName(pipeForwardTest[20].get(), "f7"));
        ASSERT_TRUE(checkName(pipeForwardTest[21].get(), "x7"));
    }

    // forward train pipe
    {
        const raul::Workflow::Pipeline& pipeForwardTrain = w.getPipeline(raul::Workflow::Pipelines::ForwardTrain);

        ASSERT_EQ(pipeForwardTrain.size(), 15u);

        ASSERT_EQ(pipeForwardTrain[0]->type(), "Allocate");  // x1
        ASSERT_EQ(pipeForwardTrain[1]->type(), "Forward");   // f0
        ASSERT_EQ(pipeForwardTrain[2]->type(), "Allocate");  // x2
        ASSERT_EQ(pipeForwardTrain[3]->type(), "Forward");   // f1
        ASSERT_EQ(pipeForwardTrain[4]->type(), "Allocate");  // x3
        ASSERT_EQ(pipeForwardTrain[5]->type(), "Forward");   // f2
        ASSERT_EQ(pipeForwardTrain[6]->type(), "Allocate");  // x4
        ASSERT_EQ(pipeForwardTrain[7]->type(), "Forward");   // f3
        ASSERT_EQ(pipeForwardTrain[8]->type(), "Allocate");  // x5
        ASSERT_EQ(pipeForwardTrain[9]->type(), "Forward");   // f4
        ASSERT_EQ(pipeForwardTrain[10]->type(), "Allocate"); // x6
        ASSERT_EQ(pipeForwardTrain[11]->type(), "Forward");  // f5
        ASSERT_EQ(pipeForwardTrain[12]->type(), "Allocate"); // x7
        ASSERT_EQ(pipeForwardTrain[13]->type(), "Forward");  // f6
        ASSERT_EQ(pipeForwardTrain[14]->type(), "Forward");  // f7

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
        ASSERT_TRUE(checkName(pipeForwardTrain[10].get(), "x6"));
        ASSERT_TRUE(checkName(pipeForwardTrain[11].get(), "f5"));
        ASSERT_TRUE(checkName(pipeForwardTrain[12].get(), "x7"));
        ASSERT_TRUE(checkName(pipeForwardTrain[13].get(), "f6"));
        ASSERT_TRUE(checkName(pipeForwardTrain[14].get(), "f7"));
    }

    // backward train pipe
    {
        const raul::Workflow::Pipeline& pipeBackwardTrain = w.getPipeline(raul::Workflow::Pipelines::BackwardTrain);

        ASSERT_EQ(pipeBackwardTrain.size(), 36u);

        ASSERT_EQ(pipeBackwardTrain[0]->type(), "Allocate");    // xg7
        ASSERT_EQ(pipeBackwardTrain[1]->type(), "Zero");        // xg7
        ASSERT_EQ(pipeBackwardTrain[2]->type(), "Backward");    // f7
        ASSERT_EQ(pipeBackwardTrain[3]->type(), "Deallocate");  // x7
        ASSERT_EQ(pipeBackwardTrain[4]->type(), "Allocate");    // xg4
        ASSERT_EQ(pipeBackwardTrain[5]->type(), "Zero");        // xg4
        ASSERT_EQ(pipeBackwardTrain[6]->type(), "Allocate");    // xg5
        ASSERT_EQ(pipeBackwardTrain[7]->type(), "Zero");        // xg5
        ASSERT_EQ(pipeBackwardTrain[8]->type(), "Allocate");    // xg6
        ASSERT_EQ(pipeBackwardTrain[9]->type(), "Zero");        // xg6
        ASSERT_EQ(pipeBackwardTrain[10]->type(), "Backward");   // f6
        ASSERT_EQ(pipeBackwardTrain[11]->type(), "Deallocate"); // x4
        ASSERT_EQ(pipeBackwardTrain[12]->type(), "Deallocate"); // x5
        ASSERT_EQ(pipeBackwardTrain[13]->type(), "Deallocate"); // x6
        ASSERT_EQ(pipeBackwardTrain[14]->type(), "Deallocate"); // xg7
        ASSERT_EQ(pipeBackwardTrain[15]->type(), "Allocate");   // xg2
        ASSERT_EQ(pipeBackwardTrain[16]->type(), "Zero");       // xg2
        ASSERT_EQ(pipeBackwardTrain[17]->type(), "Backward");   // f5
        ASSERT_EQ(pipeBackwardTrain[18]->type(), "Deallocate"); // xg6
        ASSERT_EQ(pipeBackwardTrain[19]->type(), "Allocate");   // xg3
        ASSERT_EQ(pipeBackwardTrain[20]->type(), "Zero");       // xg3
        ASSERT_EQ(pipeBackwardTrain[21]->type(), "Backward");   // f4
        ASSERT_EQ(pipeBackwardTrain[22]->type(), "Deallocate"); // x2
        ASSERT_EQ(pipeBackwardTrain[23]->type(), "Deallocate"); // x3
        ASSERT_EQ(pipeBackwardTrain[24]->type(), "Deallocate"); // xg5
        ASSERT_EQ(pipeBackwardTrain[25]->type(), "Allocate");   // xg1
        ASSERT_EQ(pipeBackwardTrain[26]->type(), "Zero");       // xg1
        ASSERT_EQ(pipeBackwardTrain[27]->type(), "Backward");   // f3
        ASSERT_EQ(pipeBackwardTrain[28]->type(), "Deallocate"); // xg4
        ASSERT_EQ(pipeBackwardTrain[29]->type(), "Backward");   // f2
        ASSERT_EQ(pipeBackwardTrain[30]->type(), "Deallocate"); // xg3
        ASSERT_EQ(pipeBackwardTrain[31]->type(), "Backward");   // f1
        ASSERT_EQ(pipeBackwardTrain[32]->type(), "Deallocate"); // xg2
        ASSERT_EQ(pipeBackwardTrain[33]->type(), "Deallocate"); // x1
        ASSERT_EQ(pipeBackwardTrain[34]->type(), "Backward");   // f0
        ASSERT_EQ(pipeBackwardTrain[35]->type(), "Deallocate"); // xg1

        ASSERT_TRUE(checkName(pipeBackwardTrain[0].get(), getGradName("x7")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[1].get(), getGradName("x7")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[2].get(), "f7"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[3].get(), "x7"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[4].get()),
                                       getName(pipeBackwardTrain[5].get()),
                                       getName(pipeBackwardTrain[6].get()),
                                       getName(pipeBackwardTrain[7].get()),
                                       getName(pipeBackwardTrain[8].get()),
                                       getName(pipeBackwardTrain[9].get()) },
                                     { getGradName("x4"), getGradName("x5"), getGradName("x6") }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[10].get(), "f6"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[11].get()), getName(pipeBackwardTrain[12].get()), getName(pipeBackwardTrain[13].get()), getName(pipeBackwardTrain[14].get()) },
                                     { "x4", "x5", "x6", getGradName("x7") }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[15].get(), getGradName("x2")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[16].get(), getGradName("x2")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[17].get(), "f5"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[18].get(), getGradName("x6")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[19].get(), getGradName("x3")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[20].get(), getGradName("x3")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[21].get(), "f4"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[22].get()), getName(pipeBackwardTrain[23].get()), getName(pipeBackwardTrain[24].get()) }, { "x2", "x3", getGradName("x5") }));
        ASSERT_TRUE(checkName(pipeBackwardTrain[25].get(), getGradName("x1")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[26].get(), getGradName("x1")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[27].get(), "f3"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[28].get(), getGradName("x4")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[29].get(), "f2"));
        ASSERT_TRUE(checkName(pipeBackwardTrain[30].get(), getGradName("x3")));
        ASSERT_TRUE(checkName(pipeBackwardTrain[31].get(), "f1"));
        ASSERT_TRUE(checkGroupedName({ getName(pipeBackwardTrain[32].get()), getName(pipeBackwardTrain[33].get()) }, { "x1", getGradName("x2") }));
        ASSERT_EQ(static_cast<raul::LayerAction*>(pipeBackwardTrain[34].get())->mLayer->getName(), "f0");
        ASSERT_EQ(static_cast<raul::TensorAction<raul::MemoryManager>*>(pipeBackwardTrain[35].get())->mName, getGradName("x1"));
    }

    EXPECT_NO_THROW(w.forwardPassTesting());

    for (size_t epoch = 1; epoch <= 10; ++epoch)
    {
        EXPECT_NO_THROW(w.forwardPassTraining());
        EXPECT_NO_THROW(w.backwardPassTraining());
    }

    EXPECT_NO_THROW(w.forwardPassTesting());
}

TEST(TestWorkflow, FlushUnit)
{
    PROFILE_TEST

    raul::Workflow work;

    ASSERT_NO_THROW(work.add<TestLayer>("f0", raul::BasicParams{ {}, { "x1" } }, true));
    ASSERT_NO_THROW(work.add<TestLayer>("f1", raul::BasicParams{ { "x1" }, { "x2" }, { "w1", "w2" } }, true));

    // Prepare
    ASSERT_NO_THROW(work.preparePipelines());
    ASSERT_NO_THROW(work.setBatchSize(1));
    ASSERT_NO_THROW(work.prepareMemoryForTraining());

    // Flush
    ASSERT_NO_THROW(work.flush());

    // Checks
    EXPECT_EQ(work.getPipeline(raul::Workflow::Pipelines::ForwardTrain).size(), 0u);
    EXPECT_EQ(work.getPipeline(raul::Workflow::Pipelines::BackwardTrain).size(), 0u);
    EXPECT_EQ(work.getPipeline(raul::Workflow::Pipelines::ForwardTest).size(), 0u);
    EXPECT_EQ(work.getPipeline(raul::Workflow::Pipelines::CreateBatched).size(), 0u);
    EXPECT_EQ(work.getPipeline(raul::Workflow::Pipelines::CreateNotBatched).size(), 0u);
    EXPECT_EQ(work.getPipeline(raul::Workflow::Pipelines::DeleteBatched).size(), 0u);
    EXPECT_EQ(work.getPipeline(raul::Workflow::Pipelines::Zero).size(), 0u);
    EXPECT_EQ(work.getMemoryManager().size(), 0u);

    // Can add new operations
    ASSERT_NO_THROW(work.add<TestLayer>("f2", raul::BasicParams{ { "x1" }, { "x3" } }, true));
    ASSERT_NO_THROW(work.add<TestLayer>("f3", raul::BasicParams{ { "x1" }, { "x4" } }, true));

    // Can't start training after flush
    EXPECT_THROW(work.setBatchSize(1), raul::Exception);
    EXPECT_THROW(work.prepareMemoryForTraining(), raul::Exception);
    EXPECT_THROW(work.forwardPassTraining(), raul::Exception);
    EXPECT_THROW(work.backwardPassTraining(), raul::Exception);
    EXPECT_THROW(work.forwardPassTesting(), raul::Exception);
}

TEST(TestWorkflow, ListenersUnit)
{
    PROFILE_TEST

    raul::Workflow work;

    class TestLayer2 : public raul::BasicLayer
    {
      public:
        TestLayer2(const raul::Name& name, const raul::BasicParams& params, raul::NetworkParameters& networkParameters)
            : BasicLayer(name, "test", params, networkParameters, { false, false })
        {
        }

        void forwardComputeImpl(raul::NetworkMode) override {}
        void backwardComputeImpl() override {}
    };

    ASSERT_NO_THROW(work.add<TestLayer2>("f0", raul::BasicParams{ {}, {} }));
    ASSERT_NO_THROW(work.add<TestLayer2>("f1", raul::BasicParams{ {}, {} }));

    ASSERT_NO_THROW(work.preparePipelines());
    ASSERT_NO_THROW(work.setBatchSize(1));
    ASSERT_NO_THROW(work.prepareMemoryForTraining());

    class ListenerHelper : public raul::WorkflowListener
    {
      public:
        ListenerHelper()
            : beforeF(0)
            , afterF(0)
            , beforeB(0)
            , afterB(0)
        {
        }

        void BeforeForward(raul::Workflow&) override { ++beforeF; }

        void AfterForward(raul::Workflow&) override { ++afterF; }

        void BeforeBackward(raul::Workflow&) override { ++beforeB; }

        void AfterBackward(raul::Workflow&) override { ++afterB; }

        size_t beforeF;
        size_t afterF;
        size_t beforeB;
        size_t afterB;
    };

    ListenerHelper listener;
    ListenerHelper listener2;

    EXPECT_THROW(work.addCallback("", listener), raul::Exception);
    EXPECT_THROW(work.addCallback("f", listener), raul::Exception);
    EXPECT_NO_THROW(work.addCallback("f0", listener));
    EXPECT_NO_THROW(work.addCallback("f1", listener));
    EXPECT_NO_THROW(work.addCallback("f1", listener2));

    EXPECT_EQ(listener.beforeF, 0u);
    EXPECT_EQ(listener.afterF, 0u);
    EXPECT_EQ(listener.beforeB, 0u);
    EXPECT_EQ(listener.afterB, 0u);

    EXPECT_EQ(listener2.beforeF, 0u);
    EXPECT_EQ(listener2.afterF, 0u);
    EXPECT_EQ(listener2.beforeB, 0u);
    EXPECT_EQ(listener2.afterB, 0u);

    work.forwardPassTraining();

    EXPECT_EQ(listener.beforeF, 2u);
    EXPECT_EQ(listener.afterF, 2u);
    EXPECT_EQ(listener.beforeB, 0u);
    EXPECT_EQ(listener.afterB, 0u);

    EXPECT_EQ(listener2.beforeF, 1u);
    EXPECT_EQ(listener2.afterF, 1u);
    EXPECT_EQ(listener2.beforeB, 0u);
    EXPECT_EQ(listener2.afterB, 0u);

    work.backwardPassTraining();

    EXPECT_EQ(listener.beforeF, 2u);
    EXPECT_EQ(listener.afterF, 2u);
    EXPECT_EQ(listener.beforeB, 2u);
    EXPECT_EQ(listener.afterB, 2u);

    EXPECT_EQ(listener2.beforeF, 1u);
    EXPECT_EQ(listener2.afterF, 1u);
    EXPECT_EQ(listener2.beforeB, 1u);
    EXPECT_EQ(listener2.afterB, 1u);
}

} // UT namespace