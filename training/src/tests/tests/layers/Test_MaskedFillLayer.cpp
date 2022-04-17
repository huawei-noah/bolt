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
#include <training/base/layers/basic/MaskedFillLayer.h>
#include <training/base/layers/basic/TensorLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestMaskedFill, PlainUnit)
{
    PROFILE_TEST
    using namespace raul;

    const size_t BATCH_SIZE = 2;
    const size_t WIDTH = 4;
    const size_t HEIGHT = 1;
    const size_t DEPTH = 3;

    const dtype EPSILON = TODTYPE(1e-6);
    const dtype FILL_VALUE = TODTYPE(1e-3);

    // Inputs
    const raul::Tensor in{ 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
    const raul::Tensor mask{ 0.f, 1.f, 0.f, 0.f };

    // Outputs
    const raul::Tensor realOut{ 1.f, 1e-3f, 1.f, 1.f, 1.f, 1e-3f, 1.f, 1.f, 1.f, 1e-3f, 1.f, 1.f, 1.f, 1e-3f, 1.f, 1.f, 1.f, 1e-3f, 1.f, 1.f, 1.f, 1e-3f, 1.f, 1.f };
    const raul::Tensor realGrad{ 2.f, 0.f, 2.f, 2.f, 2.f, 0.f, 2.f, 2.f, 2.f, 0.f, 2.f, 2.f, 2.f, 0.f, 2.f, 2.f, 2.f, 0.f, 2.f, 2.f, 2.f, 0.f, 2.f, 2.f };

    // Initialization
    raul::Workflow work;

    work.add<raul::DataLayer>("data_in", raul::DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<raul::TensorLayer>("data_mask", raul::TensorParams{ { "mask" }, 1u, 1u, HEIGHT, WIDTH });
    work.add<raul::MaskedFillLayer>("mask", raul::MaskedFillParams{ { "in", "mask" }, { "out" }, FILL_VALUE });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    auto& memory_manager = work.getMemoryManager();
    memory_manager["in"] = TORANGE(in);
    memory_manager["mask"] = TORANGE(mask);
    memory_manager[raul::Name("out").grad()].memAllocate(nullptr);
    memory_manager[raul::Name("out").grad()] = 2.0_dt;

    work.forwardPassTraining();

    const raul::Tensor& out = memory_manager["out"];

    EXPECT_EQ(out.getShape(), shape(BATCH_SIZE, DEPTH, HEIGHT, WIDTH));
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], realOut[i], EPSILON);
    }

    printf(" - MaskedFill forward is Ok.\n");

    work.backwardPassTraining();

    const raul::Tensor& inGrad = memory_manager[raul::Name("in").grad()];
    EXPECT_EQ(inGrad.getShape(), shape(BATCH_SIZE, DEPTH, HEIGHT, WIDTH));
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], realGrad[i], EPSILON);
    }

    printf(" - MaskedFill backward is Ok.\n");
}

TEST(TestMaskedFill, ChannelWiseUnit)
{
    PROFILE_TEST
    using namespace raul;

    const size_t BATCH_SIZE = 2;
    const size_t WIDTH = 4;
    const size_t HEIGHT = 1;
    const size_t DEPTH = 3;

    const dtype EPSILON = TODTYPE(1e-6);
    const dtype FILL_VALUE = TODTYPE(1e-3);

    // Inputs
    const raul::Tensor mask{ 0.f, 1.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f };

    // Outputs
    const raul::Tensor realOut = { 1.f, 1e-3f, 1.f, 1.f, 1e-3f, 1e-3f, 1e-3f, 1e-3f, 1.f, 1.f, 1.f, 1.f, 1.f, 1e-3f, 1.f, 1.f, 1e-3f, 1e-3f, 1e-3f, 1e-3f, 1.f, 1.f, 1.f, 1.f };
    const raul::Tensor realGrad = { 2.f, 0.f, 2.f, 2.f, 0.f, 0.f, 0.f, 0.f, 2.f, 2.f, 2.f, 2.f, 2.f, 0.f, 2.f, 2.f, 0.f, 0.f, 0.f, 0.f, 2.f, 2.f, 2.f, 2.f };

    // Initialization
    raul::Workflow work;

    work.add<raul::DataLayer>("data_in", raul::DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<raul::TensorLayer>("data_mask", raul::TensorParams{ { "mask" }, 1u, DEPTH, HEIGHT, WIDTH });
    work.add<raul::MaskedFillLayer>("mask", raul::MaskedFillParams{ { "in", "mask" }, { "out" }, FILL_VALUE });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    auto& memory_manager = work.getMemoryManager();
    memory_manager["in"] = 1.0_dt;
    memory_manager["mask"] = TORANGE(mask);
    memory_manager[raul::Name("out").grad()].memAllocate(nullptr);
    memory_manager[raul::Name("out").grad()] = 2.0_dt;

    work.forwardPassTraining();

    const raul::Tensor& out = memory_manager["out"];

    EXPECT_EQ(out.getShape(), shape(BATCH_SIZE, DEPTH, HEIGHT, WIDTH));
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], realOut[i], EPSILON);
    }

    printf(" - MaskedFill forward is Ok.\n");

    work.backwardPassTraining();

    const raul::Tensor& inGrad = memory_manager[raul::Name("in").grad()];
    EXPECT_EQ(inGrad.getShape(), shape(BATCH_SIZE, DEPTH, HEIGHT, WIDTH));
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], realGrad[i], EPSILON);
    }

    printf(" - MaskedFill backward is Ok.\n");
}

TEST(TestMaskedFill, BroadcastUnit)
{
    PROFILE_TEST
    using namespace raul;

    const size_t BATCH_SIZE = 2;
    const size_t WIDTH = 2;
    const size_t HEIGHT = 2;
    const size_t DEPTH = 3;

    const dtype EPSILON = TODTYPE(1e-6);
    const dtype FILL_VALUE = TODTYPE(1e-3);

    // Inputs
    const raul::Tensor mask{ 0.0_dt, 1.0_dt };

    // Outputs
    const raul::Tensor realOut = {
        1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt,
        1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt,
    };
    const raul::Tensor realGrad = { 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt,
                                    2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt };

    // Initialization
    raul::Workflow work;

    work.add<raul::DataLayer>("data_in", raul::DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<raul::TensorLayer>("data_mask", raul::TensorParams{ { "mask" }, 1u, 1u, 1u, WIDTH });
    work.add<raul::MaskedFillLayer>("mask", raul::MaskedFillParams{ { "in", "mask" }, { "out" }, FILL_VALUE });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    auto& memory_manager = work.getMemoryManager();
    memory_manager["in"] = 1.0_dt;
    memory_manager["mask"] = TORANGE(mask);
    memory_manager[raul::Name("out").grad()].memAllocate(nullptr);
    memory_manager[raul::Name("out").grad()] = 2.0_dt;

    work.forwardPassTraining();

    const raul::Tensor& out = memory_manager["out"];

    EXPECT_EQ(out.getShape(), shape(BATCH_SIZE, DEPTH, HEIGHT, WIDTH));
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], realOut[i], EPSILON);
    }

    printf(" - MaskedFill forward is Ok.\n");

    work.backwardPassTraining();

    const raul::Tensor& inGrad = memory_manager[raul::Name("in").grad()];
    EXPECT_EQ(inGrad.getShape(), shape(BATCH_SIZE, DEPTH, HEIGHT, WIDTH));
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], realGrad[i], EPSILON);
    }

    printf(" - MaskedFill backward is Ok.\n");
}

TEST(TestMaskedFill, Broadcast2Unit)
{
    PROFILE_TEST
    using namespace raul;

    const size_t BATCH_SIZE = 2;
    const size_t WIDTH = 2;
    const size_t HEIGHT = 2;
    const size_t DEPTH = 3;

    const dtype EPSILON = TODTYPE(1e-6);
    const dtype FILL_VALUE = TODTYPE(1e-3);

    // Inputs
    const raul::Tensor mask{ 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt, 0.0_dt, 1.0_dt };

    // Outputs
    const raul::Tensor realOut = {
        1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt,
        1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt, 1.0_dt, 1e-3_dt,
    };
    const raul::Tensor realGrad = { 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt,
                                    2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt, 2.0_dt, 0.0_dt };

    // Initialization
    raul::Workflow work;

    work.add<raul::DataLayer>("data_in", raul::DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<raul::TensorLayer>("data_mask", raul::TensorParams{ { "mask" }, 1u, DEPTH, 1u, WIDTH });
    work.add<raul::MaskedFillLayer>("mask", raul::MaskedFillParams{ { "in", "mask" }, { "out" }, FILL_VALUE });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    auto& memory_manager = work.getMemoryManager();
    memory_manager["in"] = 1.0_dt;
    memory_manager["mask"] = TORANGE(mask);
    memory_manager[raul::Name("out").grad()].memAllocate(nullptr);
    memory_manager[raul::Name("out").grad()] = 2.0_dt;

    work.forwardPassTraining();

    const raul::Tensor& out = memory_manager["out"];

    EXPECT_EQ(out.getShape(), shape(BATCH_SIZE, DEPTH, HEIGHT, WIDTH));
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], realOut[i], EPSILON);
    }

    printf(" - MaskedFill forward is Ok.\n");

    work.backwardPassTraining();

    const raul::Tensor& inGrad = memory_manager[raul::Name("in").grad()];
    EXPECT_EQ(inGrad.getShape(), shape(BATCH_SIZE, DEPTH, HEIGHT, WIDTH));
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], realGrad[i], EPSILON);
    }

    printf(" - MaskedFill backward is Ok.\n");
}

} // UT namespace