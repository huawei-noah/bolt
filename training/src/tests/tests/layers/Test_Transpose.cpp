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
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/TransposeLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestTranspose, TransposeUnit)
{
    PROFILE_TEST
    using namespace raul;

    Tensor data = { 111._dt, 112._dt, 121._dt, 122._dt, 131._dt, 132._dt, 141._dt, 142._dt, 211._dt, 212._dt, 221._dt, 222._dt, 231._dt, 232._dt, 241._dt, 242._dt };
    Tensor data_t_wh = { 111._dt, 121._dt, 131._dt, 141._dt, 112._dt, 122._dt, 132._dt, 142._dt, 211._dt, 221._dt, 231._dt, 241._dt, 212._dt, 222._dt, 232._dt, 242._dt };
    Tensor data_t_dh = { 111._dt, 112._dt, 211._dt, 212._dt, 121._dt, 122._dt, 221._dt, 222._dt, 131._dt, 132._dt, 231._dt, 232._dt, 141._dt, 142._dt, 241._dt, 242._dt };

    Tensor data_t_bd = { 111._dt, 112._dt, 121._dt, 122._dt, 131._dt, 132._dt, 141._dt, 142._dt, 211._dt, 212._dt, 221._dt, 222._dt, 231._dt, 232._dt, 241._dt, 242._dt };

    Tensor data_t_bh = { 111._dt, 112._dt, 211._dt, 212._dt, 121._dt, 122._dt, 221._dt, 222._dt, 131._dt, 132._dt, 231._dt, 232._dt, 141._dt, 142._dt, 241._dt, 242._dt };

    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("in", "in", raul::WShape{ raul::BS(), 1u, 4u, 2u }, DEC_FORW_READ_NOMEMOPT);

        TransposeLayer t("t", { "in", "out", Dimension::Width, Dimension::Height }, networkParameters);
        TENSORS_CREATE(2);

        memory_manager["in"] = TORANGE(data);

        t.forwardCompute(NetworkMode::Train);

        const auto& out = memory_manager["out"];
        const auto& in = memory_manager["in"];

        EXPECT_EQ(in.getBatchSize(), out.getBatchSize());
        EXPECT_EQ(in.getDepth(), out.getDepth());
        EXPECT_EQ(in.getWidth(), out.getHeight());
        EXPECT_EQ(in.getHeight(), out.getWidth());

        for (size_t i = 0; i < out.size(); ++i)
        {
            EXPECT_EQ(data_t_wh[i], out[i]);
        }

        printf(" - TransposeLayer[W <-> H] forward is Ok.\n");

        memory_manager[raul::Name("out").grad()] = TORANGE(data_t_wh);

        t.backwardCompute();

        const auto& in_nabla = memory_manager[raul::Name("in").grad()];
        const auto shape1 = in_nabla.getShape();
        const auto shape2 = in.getShape();

        for (size_t i = 0; i < 4; ++i)
        {
            EXPECT_EQ(shape1[i], shape2[i]);
        }

        for (size_t i = 0; i < in.size(); ++i)
        {
            EXPECT_EQ(in[i], in_nabla[i]);
        }

        printf(" - TransposeLayer[W <-> H] backward is Ok.\n");
    }

    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("in", "in", raul::WShape{ raul::BS(), 2u, 4u, 2u }, DEC_FORW_READ_NOMEMOPT);

        TransposeLayer t("t", { "in", "out", Dimension::Depth, Dimension::Height }, networkParameters);
        TENSORS_CREATE(1);

        memory_manager["in"] = TORANGE(data);

        t.forwardCompute(NetworkMode::Train);

        const auto& out = memory_manager["out"];
        const auto& in = memory_manager["in"];

        EXPECT_EQ(in.getBatchSize(), out.getBatchSize());
        EXPECT_EQ(in.getDepth(), out.getHeight());
        EXPECT_EQ(in.getWidth(), out.getWidth());
        EXPECT_EQ(in.getHeight(), out.getDepth());

        for (size_t i = 0; i < out.size(); ++i)
        {
            EXPECT_EQ(data_t_dh[i], out[i]);
        }

        printf(" - TransposeLayer[C <-> H] forward is Ok.\n");

        memory_manager[raul::Name("out").grad()] = TORANGE(data_t_dh);

        t.backwardCompute();

        const auto& in_nabla = memory_manager[raul::Name("in").grad()];
        const auto shape1 = in_nabla.getShape();
        const auto shape2 = in.getShape();
        for (size_t i = 0; i < 4; ++i)
        {
            EXPECT_EQ(shape1[i], shape2[i]);
        }

        for (size_t i = 0; i < in.size(); ++i)
        {
            EXPECT_EQ(in[i], in_nabla[i]);
        }

        printf(" - TransposeLayer[C <-> H] backward is Ok.\n");
    }

    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("in", "in", raul::WShape{ raul::BS(), 1u, 4u, 2u }, DEC_FORW_READ_NOMEMOPT);

        TransposeLayer t("t", { "in", "out" }, networkParameters);
        TENSORS_CREATE(2);
        memory_manager["in"] = TORANGE(data);

        t.forwardCompute(NetworkMode::Train);

        const auto& out = memory_manager["out"];
        const auto& in = memory_manager["in"];

        EXPECT_EQ(in.getBatchSize(), out.getBatchSize());
        EXPECT_EQ(in.getDepth(), out.getDepth());
        EXPECT_EQ(in.getWidth(), out.getHeight());
        EXPECT_EQ(in.getHeight(), out.getWidth());

        for (size_t i = 0; i < out.size(); ++i)
        {
            EXPECT_EQ(data_t_wh[i], out[i]);
        }

        printf(" - TransposeLayer[Default] forward is Ok.\n");

        memory_manager[raul::Name("out").grad()] = TORANGE(data_t_wh);

        t.backwardCompute();

        const auto& in_nabla = memory_manager[raul::Name("in").grad()];
        const auto shape1 = in_nabla.getShape();
        const auto shape2 = in.getShape();
        for (size_t i = 0; i < 4; ++i)
        {
            EXPECT_EQ(shape1[i], shape2[i]);
        }

        for (size_t i = 0; i < in.size(); ++i)
        {
            EXPECT_EQ(in[i], in_nabla[i]);
        }

        printf(" - TransposeLayer[Default] backward is Ok.\n");
    }

    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("in", "in", raul::WShape{ raul::BS(), 1u, 4u, 2u }, DEC_FORW_READ_NOMEMOPT);

        TransposeLayer t("t", { "in", "out", Dimension::Width, Dimension::Width }, networkParameters);
        TENSORS_CREATE(2);
        memory_manager["in"] = TORANGE(data);

        t.forwardCompute(NetworkMode::Train);

        const auto& out = memory_manager["out"];
        const auto& in = memory_manager["in"];

        EXPECT_EQ(in.getBatchSize(), out.getBatchSize());
        EXPECT_EQ(in.getDepth(), out.getDepth());
        EXPECT_EQ(in.getWidth(), out.getWidth());
        EXPECT_EQ(in.getHeight(), out.getHeight());

        for (size_t i = 0; i < out.size(); ++i)
        {
            EXPECT_EQ(data[i], out[i]);
        }

        printf(" - TransposeLayer[W <-> W] forward is Ok.\n");

        memory_manager[raul::Name("out").grad()] = TORANGE(data);

        t.backwardCompute();

        const auto& in_nabla = memory_manager[raul::Name("in").grad()];
        const auto shape1 = in_nabla.getShape();
        const auto shape2 = in.getShape();
        for (size_t i = 0; i < 4; ++i)
        {
            EXPECT_EQ(shape1[i], shape2[i]);
        }

        for (size_t i = 0; i < in.size(); ++i)
        {
            EXPECT_EQ(in[i], in_nabla[i]);
        }

        printf(" - TransposeLayer[W <-> W] backward is Ok.\n");
    }

    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("in", "in", raul::WShape{ raul::BS(), 1u, 4u, 2u }, DEC_FORW_READ_NOMEMOPT);

        TransposeLayer t("t", { "in", "out", Dimension::Batch, Dimension::Batch }, networkParameters);
        TENSORS_CREATE(2);
        memory_manager["in"] = TORANGE(data);

        t.forwardCompute(NetworkMode::Train);

        const auto& out = memory_manager["out"];
        const auto& in = memory_manager["in"];

        EXPECT_EQ(in.getBatchSize(), out.getBatchSize());
        EXPECT_EQ(in.getDepth(), out.getDepth());
        EXPECT_EQ(in.getWidth(), out.getWidth());
        EXPECT_EQ(in.getHeight(), out.getHeight());

        for (size_t i = 0; i < out.size(); ++i)
        {
            EXPECT_EQ(data[i], out[i]);
        }

        printf(" - TransposeLayer[B <-> B] forward is Ok.\n");

        memory_manager[raul::Name("out").grad()] = TORANGE(data);

        t.backwardCompute();

        const auto& in_nabla = memory_manager[raul::Name("in").grad()];
        const auto shape1 = in_nabla.getShape();
        const auto shape2 = in.getShape();
        for (size_t i = 0; i < 4; ++i)
        {
            EXPECT_EQ(shape1[i], shape2[i]);
        }

        for (size_t i = 0; i < in.size(); ++i)
        {
            EXPECT_EQ(in[i], in_nabla[i]);
        }

        printf(" - TransposeLayer[B <-> B] backward is Ok.\n");
    }

    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("in", "in", raul::WShape{ raul::BS(), 1u, 4u, 2u }, DEC_FORW_READ_NOMEMOPT);

        TransposeLayer t("t", { "in", "out", Dimension::Batch, Dimension::Depth }, networkParameters);
        TENSORS_CREATE(2);
        memory_manager["in"] = TORANGE(data);

        t.forwardCompute(NetworkMode::Train);

        const auto& out = memory_manager["out"];
        const auto& in = memory_manager["in"];

        EXPECT_EQ(in.getBatchSize(), out.getDepth());
        EXPECT_EQ(in.getDepth(), out.getBatchSize());
        EXPECT_EQ(in.getWidth(), out.getWidth());
        EXPECT_EQ(in.getHeight(), out.getHeight());

        for (size_t i = 0; i < out.size(); ++i)
        {
            EXPECT_EQ(data_t_bd[i], out[i]);
        }

        printf(" - TransposeLayer[B <-> D] forward is Ok.\n");

        memory_manager[raul::Name("out").grad()] = TORANGE(data_t_bd);

        t.backwardCompute();

        const auto& in_nabla = memory_manager[raul::Name("in").grad()];
        const auto shape1 = in_nabla.getShape();
        const auto shape2 = in.getShape();
        for (size_t i = 0; i < 4; ++i)
        {
            EXPECT_EQ(shape1[i], shape2[i]);
        }

        for (size_t i = 0; i < in.size(); ++i)
        {
            EXPECT_EQ(in[i], in_nabla[i]);
        }

        printf(" - TransposeLayer[B <-> D] backward is Ok.\n");
    }

    {
        MANAGERS_DEFINE
        NETWORK_PARAMS_DEFINE(networkParameters);

        work.tensorNeeded("in", "in", raul::WShape{ raul::BS(), 1u, 4u, 2u }, DEC_FORW_READ_NOMEMOPT);

        TransposeLayer t("t", { "in", "out", Dimension::Batch, Dimension::Height }, networkParameters);
        TENSORS_CREATE(2);
        memory_manager["in"] = TORANGE(data);

        t.forwardCompute(NetworkMode::Train);

        const auto& out = memory_manager["out"];
        const auto& in = memory_manager["in"];

        EXPECT_EQ(in.getBatchSize(), out.getHeight());
        EXPECT_EQ(in.getDepth(), out.getDepth());
        EXPECT_EQ(in.getWidth(), out.getWidth());
        EXPECT_EQ(in.getHeight(), out.getBatchSize());

        for (size_t i = 0; i < out.size(); ++i)
        {
            EXPECT_EQ(data_t_bh[i], out[i]);
        }

        printf(" - TransposeLayer[B <-> H] forward is Ok.\n");

        memory_manager[raul::Name("out").grad()] = TORANGE(data_t_bh);

        t.backwardCompute();

        const auto& in_nabla = memory_manager[raul::Name("in").grad()];
        const auto shape1 = in_nabla.getShape();
        const auto shape2 = in.getShape();
        for (size_t i = 0; i < 4; ++i)
        {
            EXPECT_EQ(shape1[i], shape2[i]);
        }

        for (size_t i = 0; i < in.size(); ++i)
        {
            EXPECT_EQ(in[i], in_nabla[i]);
        }

        printf(" - TransposeLayer[B <-> H] backward is Ok.\n");
    }
}

} // UT namespace