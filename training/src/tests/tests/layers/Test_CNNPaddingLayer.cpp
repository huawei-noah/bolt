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

#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/PaddingLayer.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>
#include <training/base/optimizers/SGD.h>

using namespace raul;

namespace UT
{

struct TestPaddingLayer : public testing::Test
{
    std::unique_ptr<Tensor> symmetricPaddingResult;
    std::unique_ptr<Tensor> asymmetricPaddingResult;
    std::unique_ptr<Tensor> symmetricReflectionPaddingResult;
    std::unique_ptr<Tensor> asymmetricReflectionPaddingResult;
    std::unique_ptr<Tensor> reflectionPaddingBackwardPassResult;
    std::unique_ptr<Tensor> symmetricReplicationPaddingResult;
    std::unique_ptr<Tensor> asymmetricReplicationPaddingResult;
    std::unique_ptr<Tensor> replicationPaddingBackwardPassResult;

    void SetUp() final;

    void ExpectEqual(const Tensor& t1, const Tensor& t2, dtype epsilon)
    {
        ASSERT_EQ(t1.size(), t2.size());
        for (size_t i = 0; i < t1.size(); ++i)
        {
            ASSERT_NEAR(t1[i], t2[i], epsilon);
        }
    }
};

TEST_F(TestPaddingLayer, SymmetricConstantPaddingUnit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    uint32_t commonPadding = 3;
    work.add<DataLayer>("data", DataParams{ { "in" }, 3u, 2u, 3u });
    work.add<PaddingLayer>("pad", PaddingLayerParams{ { "in" }, { "out" }, commonPadding, 5._dt });

    TENSORS_CREATE(2);
    memory_manager["in"] = 1._dt;
    work.forwardPassTraining();

    const Tensor& out = memory_manager["out"];
    const Tensor& in = memory_manager["in"];
    ASSERT_EQ(out.getBatchSize(), in.getBatchSize());
    ASSERT_EQ(out.getDepth(), in.getDepth());
    ASSERT_EQ(out.getHeight(), commonPadding + in.getHeight() + commonPadding);
    ASSERT_EQ(out.getWidth(), commonPadding + in.getWidth() + commonPadding);
    ExpectEqual(out, *symmetricPaddingResult, 1e-4_dt);
}

TEST_F(TestPaddingLayer, AsymmetricConstantPaddingUnit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    uint32_t topPadding = 1;
    uint32_t bottomPadding = 2;
    uint32_t leftPadding = 3;
    uint32_t rightPadding = 4;
    work.add<DataLayer>("data", DataParams{ { "in" }, 3u, 2u, 3u });
    work.add<PaddingLayer>("pad", PaddingLayerParams{ { "in" }, { "out" }, topPadding, bottomPadding, leftPadding, rightPadding, 5._dt });

    TENSORS_CREATE(2);
    memory_manager["in"] = 1._dt;
    work.forwardPassTraining();

    const Tensor& out = memory_manager["out"];
    const Tensor& in = memory_manager["in"];
    ASSERT_EQ(out.getBatchSize(), in.getBatchSize());
    ASSERT_EQ(out.getDepth(), in.getDepth());
    ASSERT_EQ(out.getHeight(), topPadding + in.getHeight() + bottomPadding);
    ASSERT_EQ(out.getWidth(), leftPadding + in.getWidth() + rightPadding);
    ExpectEqual(out, *asymmetricPaddingResult, 1e-4_dt);
}

TEST_F(TestPaddingLayer, NoConstantPaddingUnit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    uint32_t noPadding = 0;
    work.add<DataLayer>("data", DataParams{ { "in" }, 3u, 2u, 3u });
    work.add<PaddingLayer>("pad", PaddingLayerParams{ { "in" }, { "out" }, noPadding, 5._dt });

    TENSORS_CREATE(2);
    memory_manager["in"] = 1._dt;
    work.forwardPassTraining();

    const Tensor& out = memory_manager["out"];
    const Tensor& in = memory_manager["in"];
    ASSERT_EQ(out.getBatchSize(), in.getBatchSize());
    ASSERT_EQ(out.getDepth(), in.getDepth());
    ASSERT_EQ(out.getHeight(), in.getHeight());
    ASSERT_EQ(out.getWidth(), in.getWidth());
    ExpectEqual(out, in, 1e-4_dt);
}

TEST_F(TestPaddingLayer, ConstantPaddingBackwardPassUnit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    uint32_t pad = 2;
    work.add<DataLayer>("data", DataParams{ { "in" }, 3u, 2u, 3u });
    work.add<PaddingLayer>("pad", PaddingLayerParams{ { "in" }, { "out" }, pad, 5._dt });

    TENSORS_CREATE(2);
    memory_manager["in"] = 1._dt;
    tools::init_rand_tensor(raul::Name("out").grad(), { -1.f, 1.f }, memory_manager);
    work.forwardPassTraining();
    work.backwardPassTraining();

    const auto& in = memory_manager[raul::Name("in")];
    const auto& in_nabla = memory_manager[raul::Name("in").grad()];
    const auto& out_nabla = memory_manager[raul::Name("out").grad()];
    ASSERT_EQ(in_nabla.size(), in.size());
    auto in_nabla_4d_view = in_nabla.get4DView();
    auto out_nabla_4d_view = out_nabla.get4DView();
    dtype epsilon = 1e-4_dt;
    for (size_t b = 0; b < out_nabla.getBatchSize(); ++b)
    {
        for (size_t d = 0; d < out_nabla.getDepth(); ++d)
        {
            for (size_t h = 0; h < out_nabla.getHeight(); ++h)
            {
                for (size_t w = 0; w < out_nabla.getWidth(); ++w)
                {
                    if ((h < pad || h >= pad + in.getHeight()) || (w < pad || w >= pad + in.getWidth()))
                    {
                        continue;
                    }
                    else
                    {
                        ASSERT_NEAR(out_nabla_4d_view.at(b, d, h, w), in_nabla_4d_view.at(b, d, h - pad, w - pad), epsilon);
                    }
                }
            }
        }
    }
}

TEST_F(TestPaddingLayer, SymmetricReflectionPaddingUnit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    uint32_t commonPadding = 2;
    work.add<DataLayer>("data", DataParams{ { "in" }, 3u, 3u, 4u });
    work.add<PaddingLayer>("pad", PaddingLayerParams{ { "in" }, { "out" }, commonPadding, PaddingLayerParams::FillingMode::REFLECTION });

    TENSORS_CREATE(2);
    Tensor& in = memory_manager["in"];
    std::generate(in.begin(), in.end(), [n = 0._dt]() mutable {
        dtype result = n;
        n += 1._dt;
        return result;
    });
    work.forwardPassTraining();

    const Tensor& out = memory_manager["out"];
    ASSERT_EQ(out.getBatchSize(), in.getBatchSize());
    ASSERT_EQ(out.getDepth(), in.getDepth());
    ASSERT_EQ(out.getHeight(), commonPadding + in.getHeight() + commonPadding);
    ASSERT_EQ(out.getWidth(), commonPadding + in.getWidth() + commonPadding);
    ExpectEqual(out, *symmetricReflectionPaddingResult, 1e-4_dt);
}

TEST_F(TestPaddingLayer, AsymmetricReflectionPaddingUnit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    uint32_t topPadding = 1;
    uint32_t bottomPadding = 0;
    uint32_t leftPadding = 3;
    uint32_t rightPadding = 2;
    work.add<DataLayer>("data", DataParams{ { "in" }, 3u, 3u, 4u });
    work.add<PaddingLayer>("pad", PaddingLayerParams{ { "in" }, { "out" }, topPadding, bottomPadding, leftPadding, rightPadding, PaddingLayerParams::FillingMode::REFLECTION });

    TENSORS_CREATE(2);
    Tensor& in = memory_manager["in"];
    std::generate(in.begin(), in.end(), [n = 0._dt]() mutable {
        dtype result = n;
        n += 1._dt;
        return result;
    });
    work.forwardPassTraining();

    const Tensor& out = memory_manager["out"];
    ASSERT_EQ(out.getBatchSize(), in.getBatchSize());
    ASSERT_EQ(out.getDepth(), in.getDepth());
    ASSERT_EQ(out.getHeight(), topPadding + in.getHeight() + bottomPadding);
    ASSERT_EQ(out.getWidth(), leftPadding + in.getWidth() + rightPadding);
    ExpectEqual(out, *asymmetricReflectionPaddingResult, 1e-4_dt);
}

TEST_F(TestPaddingLayer, NoReflectionPaddingUnit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data", DataParams{ { "in" }, 3u, 3u, 4u });
    work.add<PaddingLayer>("pad", PaddingLayerParams{ { "in" }, { "out" }, 0, PaddingLayerParams::FillingMode::REFLECTION });

    TENSORS_CREATE(2);
    Tensor& in = memory_manager["in"];
    std::generate(in.begin(), in.end(), [n = 0._dt]() mutable {
        dtype result = n;
        n += 1._dt;
        return result;
    });
    work.forwardPassTraining();

    const Tensor& out = memory_manager["out"];
    ASSERT_EQ(out.getBatchSize(), in.getBatchSize());
    ASSERT_EQ(out.getDepth(), in.getDepth());
    ASSERT_EQ(out.getHeight(), in.getHeight());
    ASSERT_EQ(out.getWidth(), in.getWidth());
    ExpectEqual(out, in, 1e-4_dt);
}

TEST_F(TestPaddingLayer, ThrowExceptionIfReflectionPaddingHasInaccesibleValueUnit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data", DataParams{ { "in1", "in2", "in3", "in4", "in5" }, 3u, 3u, 4u });

    PaddingLayerParams badParams1{ { "in1" }, { "out1" }, 3, 2, 3, 3, PaddingLayerParams::FillingMode::REFLECTION };
    ASSERT_THROW(work.add<PaddingLayer>("pad", badParams1), raul::Exception);

    PaddingLayerParams badParams2{ { "in2" }, { "out2" }, 2, 3, 3, 3, PaddingLayerParams::FillingMode::REFLECTION };
    ASSERT_THROW(work.add<PaddingLayer>("pad", badParams2), raul::Exception);

    PaddingLayerParams badParams3{ { "in3" }, { "out3" }, 2, 2, 4, 3, PaddingLayerParams::FillingMode::REFLECTION };
    ASSERT_THROW(work.add<PaddingLayer>("pad", badParams3), raul::Exception);

    PaddingLayerParams badParams4{ { "in4" }, { "out4" }, 2, 2, 3, 4, PaddingLayerParams::FillingMode::REFLECTION };
    ASSERT_THROW(work.add<PaddingLayer>("pad", badParams4), raul::Exception);

    PaddingLayerParams badParams5{ { "in5" }, { "out5" }, 3, PaddingLayerParams::FillingMode::REFLECTION };
    ASSERT_THROW(work.add<PaddingLayer>("pad", badParams5), raul::Exception);
}

TEST_F(TestPaddingLayer, ReflectionPaddingBackwardPassUnit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    uint32_t pad = 2;
    work.add<DataLayer>("data", DataParams{ { "in" }, 3u, 3u, 4u });
    work.add<PaddingLayer>("pad", PaddingLayerParams{ { "in" }, { "out" }, pad, PaddingLayerParams::FillingMode::REFLECTION });

    TENSORS_CREATE(2);
    Tensor& in = memory_manager["in"];
    std::generate(in.begin(), in.end(), [n = 0._dt]() mutable {
        dtype result = n;
        n += 1._dt;
        return result;
    });
    memory_manager[raul::Name("out").grad()] = 1.0_dt;
    work.forwardPassTraining();
    work.backwardPassTraining();

    const auto& in_nabla = memory_manager[raul::Name("in").grad()];
    ASSERT_EQ(in_nabla.size(), in.size());
    ExpectEqual(in_nabla, *reflectionPaddingBackwardPassResult, 1e-5_dt);
}

TEST_F(TestPaddingLayer, SymmetricReplicationPaddingUnit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    uint32_t commonPadding = 2;
    work.add<DataLayer>("data", DataParams{ { "in" }, 3u, 3u, 4u });
    work.add<PaddingLayer>("pad", PaddingLayerParams{ { "in" }, { "out" }, commonPadding, PaddingLayerParams::FillingMode::REPLICATION });

    TENSORS_CREATE(2);
    Tensor& in = memory_manager["in"];
    std::generate(in.begin(), in.end(), [n = 0._dt]() mutable {
        dtype result = n;
        n += 1._dt;
        return result;
    });
    work.forwardPassTraining();

    const Tensor& out = memory_manager["out"];
    ASSERT_EQ(out.getBatchSize(), in.getBatchSize());
    ASSERT_EQ(out.getDepth(), in.getDepth());
    ASSERT_EQ(out.getHeight(), commonPadding + in.getHeight() + commonPadding);
    ASSERT_EQ(out.getWidth(), commonPadding + in.getWidth() + commonPadding);
    ExpectEqual(out, *symmetricReplicationPaddingResult, 1e-4_dt);
}

TEST_F(TestPaddingLayer, AsymmetricReplicationPaddingUnit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    uint32_t topPadding = 1;
    uint32_t bottomPadding = 0;
    uint32_t leftPadding = 3;
    uint32_t rightPadding = 2;
    work.add<DataLayer>("data", DataParams{ { "in" }, 3u, 3u, 4u });
    work.add<PaddingLayer>("pad", PaddingLayerParams{ { "in" }, { "out" }, topPadding, bottomPadding, leftPadding, rightPadding, PaddingLayerParams::FillingMode::REPLICATION });

    TENSORS_CREATE(2);
    Tensor& in = memory_manager["in"];
    std::generate(in.begin(), in.end(), [n = 0._dt]() mutable {
        dtype result = n;
        n += 1._dt;
        return result;
    });
    work.forwardPassTraining();

    const Tensor& out = memory_manager["out"];
    ASSERT_EQ(out.getBatchSize(), in.getBatchSize());
    ASSERT_EQ(out.getDepth(), in.getDepth());
    ASSERT_EQ(out.getHeight(), topPadding + in.getHeight() + bottomPadding);
    ASSERT_EQ(out.getWidth(), leftPadding + in.getWidth() + rightPadding);
    ExpectEqual(out, *asymmetricReplicationPaddingResult, 1e-4_dt);
}

TEST_F(TestPaddingLayer, NoReplicationPaddingUnit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data", DataParams{ { "in" }, 3u, 3u, 4u });
    work.add<PaddingLayer>("pad", PaddingLayerParams{ { "in" }, { "out" }, 0, PaddingLayerParams::FillingMode::REPLICATION });

    TENSORS_CREATE(2);
    Tensor& in = memory_manager["in"];
    std::generate(in.begin(), in.end(), [n = 0._dt]() mutable {
        dtype result = n;
        n += 1._dt;
        return result;
    });
    work.forwardPassTraining();

    const Tensor& out = memory_manager["out"];
    ASSERT_EQ(out.getBatchSize(), in.getBatchSize());
    ASSERT_EQ(out.getDepth(), in.getDepth());
    ASSERT_EQ(out.getHeight(), in.getHeight());
    ASSERT_EQ(out.getWidth(), in.getWidth());
    ExpectEqual(out, in, 1e-4_dt);
}

TEST_F(TestPaddingLayer, ReplicationPaddingBackwardPassUnit)
{
    PROFILE_TEST
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    uint32_t pad = 2;
    work.add<DataLayer>("data", DataParams{ { "in" }, 3u, 3u, 4u });
    work.add<PaddingLayer>("pad", PaddingLayerParams{ { "in" }, { "out" }, pad, PaddingLayerParams::FillingMode::REPLICATION });

    TENSORS_CREATE(2);
    Tensor& in = memory_manager["in"];
    std::generate(in.begin(), in.end(), [n = 0._dt]() mutable {
        dtype result = n;
        n += 1._dt;
        return result;
    });
    memory_manager[Name("out").grad()] = 1.0_dt;
    work.forwardPassTraining();
    work.backwardPassTraining();

    const auto& in_nabla = memory_manager[raul::Name("in").grad()];
    ASSERT_EQ(in_nabla.size(), in.size());
    ExpectEqual(in_nabla, *replicationPaddingBackwardPassResult, 1e-5_dt);
}

void TestPaddingLayer::SetUp()
{
    symmetricPaddingResult = std::make_unique<Tensor>(std::initializer_list<dtype>{
        5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f });

    asymmetricPaddingResult = std::make_unique<Tensor>(std::initializer_list<dtype>{
        5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f,
        5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f,
        1.f, 1.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f,
        5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f,
        5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f });

    symmetricReflectionPaddingResult = std::make_unique<Tensor>(std::initializer_list<dtype>{
        10.f, 9.f,  8.f,  9.f,  10.f, 11.f, 10.f, 9.f,  6.f,  5.f,  4.f,  5.f,  6.f,  7.f,  6.f,  5.f,  2.f,  1.f,  0.f,  1.f,  2.f,  3.f,  2.f,  1.f,  6.f,  5.f,  4.f,  5.f,  6.f,  7.f,  6.f,
        5.f,  10.f, 9.f,  8.f,  9.f,  10.f, 11.f, 10.f, 9.f,  6.f,  5.f,  4.f,  5.f,  6.f,  7.f,  6.f,  5.f,  2.f,  1.f,  0.f,  1.f,  2.f,  3.f,  2.f,  1.f,  22.f, 21.f, 20.f, 21.f, 22.f, 23.f,
        22.f, 21.f, 18.f, 17.f, 16.f, 17.f, 18.f, 19.f, 18.f, 17.f, 14.f, 13.f, 12.f, 13.f, 14.f, 15.f, 14.f, 13.f, 18.f, 17.f, 16.f, 17.f, 18.f, 19.f, 18.f, 17.f, 22.f, 21.f, 20.f, 21.f, 22.f,
        23.f, 22.f, 21.f, 18.f, 17.f, 16.f, 17.f, 18.f, 19.f, 18.f, 17.f, 14.f, 13.f, 12.f, 13.f, 14.f, 15.f, 14.f, 13.f, 34.f, 33.f, 32.f, 33.f, 34.f, 35.f, 34.f, 33.f, 30.f, 29.f, 28.f, 29.f,
        30.f, 31.f, 30.f, 29.f, 26.f, 25.f, 24.f, 25.f, 26.f, 27.f, 26.f, 25.f, 30.f, 29.f, 28.f, 29.f, 30.f, 31.f, 30.f, 29.f, 34.f, 33.f, 32.f, 33.f, 34.f, 35.f, 34.f, 33.f, 30.f, 29.f, 28.f,
        29.f, 30.f, 31.f, 30.f, 29.f, 26.f, 25.f, 24.f, 25.f, 26.f, 27.f, 26.f, 25.f, 46.f, 45.f, 44.f, 45.f, 46.f, 47.f, 46.f, 45.f, 42.f, 41.f, 40.f, 41.f, 42.f, 43.f, 42.f, 41.f, 38.f, 37.f,
        36.f, 37.f, 38.f, 39.f, 38.f, 37.f, 42.f, 41.f, 40.f, 41.f, 42.f, 43.f, 42.f, 41.f, 46.f, 45.f, 44.f, 45.f, 46.f, 47.f, 46.f, 45.f, 42.f, 41.f, 40.f, 41.f, 42.f, 43.f, 42.f, 41.f, 38.f,
        37.f, 36.f, 37.f, 38.f, 39.f, 38.f, 37.f, 58.f, 57.f, 56.f, 57.f, 58.f, 59.f, 58.f, 57.f, 54.f, 53.f, 52.f, 53.f, 54.f, 55.f, 54.f, 53.f, 50.f, 49.f, 48.f, 49.f, 50.f, 51.f, 50.f, 49.f,
        54.f, 53.f, 52.f, 53.f, 54.f, 55.f, 54.f, 53.f, 58.f, 57.f, 56.f, 57.f, 58.f, 59.f, 58.f, 57.f, 54.f, 53.f, 52.f, 53.f, 54.f, 55.f, 54.f, 53.f, 50.f, 49.f, 48.f, 49.f, 50.f, 51.f, 50.f,
        49.f, 70.f, 69.f, 68.f, 69.f, 70.f, 71.f, 70.f, 69.f, 66.f, 65.f, 64.f, 65.f, 66.f, 67.f, 66.f, 65.f, 62.f, 61.f, 60.f, 61.f, 62.f, 63.f, 62.f, 61.f, 66.f, 65.f, 64.f, 65.f, 66.f, 67.f,
        66.f, 65.f, 70.f, 69.f, 68.f, 69.f, 70.f, 71.f, 70.f, 69.f, 66.f, 65.f, 64.f, 65.f, 66.f, 67.f, 66.f, 65.f, 62.f, 61.f, 60.f, 61.f, 62.f, 63.f, 62.f, 61.f });

    asymmetricReflectionPaddingResult = std::make_unique<Tensor>(std::initializer_list<dtype>{
        7.f,  6.f,  5.f,  4.f,  5.f,  6.f,  7.f,  6.f,  5.f,  3.f,  2.f,  1.f,  0.f,  1.f,  2.f,  3.f,  2.f,  1.f,  7.f,  6.f,  5.f,  4.f,  5.f,  6.f,  7.f,  6.f,  5.f,  11.f, 10.f, 9.f,  8.f,
        9.f,  10.f, 11.f, 10.f, 9.f,  19.f, 18.f, 17.f, 16.f, 17.f, 18.f, 19.f, 18.f, 17.f, 15.f, 14.f, 13.f, 12.f, 13.f, 14.f, 15.f, 14.f, 13.f, 19.f, 18.f, 17.f, 16.f, 17.f, 18.f, 19.f, 18.f,
        17.f, 23.f, 22.f, 21.f, 20.f, 21.f, 22.f, 23.f, 22.f, 21.f, 31.f, 30.f, 29.f, 28.f, 29.f, 30.f, 31.f, 30.f, 29.f, 27.f, 26.f, 25.f, 24.f, 25.f, 26.f, 27.f, 26.f, 25.f, 31.f, 30.f, 29.f,
        28.f, 29.f, 30.f, 31.f, 30.f, 29.f, 35.f, 34.f, 33.f, 32.f, 33.f, 34.f, 35.f, 34.f, 33.f, 43.f, 42.f, 41.f, 40.f, 41.f, 42.f, 43.f, 42.f, 41.f, 39.f, 38.f, 37.f, 36.f, 37.f, 38.f, 39.f,
        38.f, 37.f, 43.f, 42.f, 41.f, 40.f, 41.f, 42.f, 43.f, 42.f, 41.f, 47.f, 46.f, 45.f, 44.f, 45.f, 46.f, 47.f, 46.f, 45.f, 55.f, 54.f, 53.f, 52.f, 53.f, 54.f, 55.f, 54.f, 53.f, 51.f, 50.f,
        49.f, 48.f, 49.f, 50.f, 51.f, 50.f, 49.f, 55.f, 54.f, 53.f, 52.f, 53.f, 54.f, 55.f, 54.f, 53.f, 59.f, 58.f, 57.f, 56.f, 57.f, 58.f, 59.f, 58.f, 57.f, 67.f, 66.f, 65.f, 64.f, 65.f, 66.f,
        67.f, 66.f, 65.f, 63.f, 62.f, 61.f, 60.f, 61.f, 62.f, 63.f, 62.f, 61.f, 67.f, 66.f, 65.f, 64.f, 65.f, 66.f, 67.f, 66.f, 65.f, 71.f, 70.f, 69.f, 68.f, 69.f, 70.f, 71.f, 70.f, 69.f });

    reflectionPaddingBackwardPassResult = std::make_unique<Tensor>(std::initializer_list<dtype>{
        2.f, 6.f, 6.f, 2.f, 3.f, 9.f, 9.f, 3.f, 2.f, 6.f, 6.f, 2.f, 2.f, 6.f, 6.f, 2.f, 3.f, 9.f, 9.f, 3.f, 2.f, 6.f, 6.f, 2.f, 2.f, 6.f, 6.f, 2.f, 3.f, 9.f, 9.f, 3.f, 2.f, 6.f, 6.f, 2.f,
        2.f, 6.f, 6.f, 2.f, 3.f, 9.f, 9.f, 3.f, 2.f, 6.f, 6.f, 2.f, 2.f, 6.f, 6.f, 2.f, 3.f, 9.f, 9.f, 3.f, 2.f, 6.f, 6.f, 2.f, 2.f, 6.f, 6.f, 2.f, 3.f, 9.f, 9.f, 3.f, 2.f, 6.f, 6.f, 2.f });

    symmetricReplicationPaddingResult = std::make_unique<Tensor>(std::initializer_list<dtype>{
        0.f,  0.f,  0.f,  1.f,  2.f,  3.f,  3.f,  3.f,  0.f,  0.f,  0.f,  1.f,  2.f,  3.f,  3.f,  3.f,  0.f,  0.f,  0.f,  1.f,  2.f,  3.f,  3.f,  3.f,  4.f,  4.f,  4.f,  5.f,  6.f,  7.f,  7.f,
        7.f,  8.f,  8.f,  8.f,  9.f,  10.f, 11.f, 11.f, 11.f, 8.f,  8.f,  8.f,  9.f,  10.f, 11.f, 11.f, 11.f, 8.f,  8.f,  8.f,  9.f,  10.f, 11.f, 11.f, 11.f, 12.f, 12.f, 12.f, 13.f, 14.f, 15.f,
        15.f, 15.f, 12.f, 12.f, 12.f, 13.f, 14.f, 15.f, 15.f, 15.f, 12.f, 12.f, 12.f, 13.f, 14.f, 15.f, 15.f, 15.f, 16.f, 16.f, 16.f, 17.f, 18.f, 19.f, 19.f, 19.f, 20.f, 20.f, 20.f, 21.f, 22.f,
        23.f, 23.f, 23.f, 20.f, 20.f, 20.f, 21.f, 22.f, 23.f, 23.f, 23.f, 20.f, 20.f, 20.f, 21.f, 22.f, 23.f, 23.f, 23.f, 24.f, 24.f, 24.f, 25.f, 26.f, 27.f, 27.f, 27.f, 24.f, 24.f, 24.f, 25.f,
        26.f, 27.f, 27.f, 27.f, 24.f, 24.f, 24.f, 25.f, 26.f, 27.f, 27.f, 27.f, 28.f, 28.f, 28.f, 29.f, 30.f, 31.f, 31.f, 31.f, 32.f, 32.f, 32.f, 33.f, 34.f, 35.f, 35.f, 35.f, 32.f, 32.f, 32.f,
        33.f, 34.f, 35.f, 35.f, 35.f, 32.f, 32.f, 32.f, 33.f, 34.f, 35.f, 35.f, 35.f, 36.f, 36.f, 36.f, 37.f, 38.f, 39.f, 39.f, 39.f, 36.f, 36.f, 36.f, 37.f, 38.f, 39.f, 39.f, 39.f, 36.f, 36.f,
        36.f, 37.f, 38.f, 39.f, 39.f, 39.f, 40.f, 40.f, 40.f, 41.f, 42.f, 43.f, 43.f, 43.f, 44.f, 44.f, 44.f, 45.f, 46.f, 47.f, 47.f, 47.f, 44.f, 44.f, 44.f, 45.f, 46.f, 47.f, 47.f, 47.f, 44.f,
        44.f, 44.f, 45.f, 46.f, 47.f, 47.f, 47.f, 48.f, 48.f, 48.f, 49.f, 50.f, 51.f, 51.f, 51.f, 48.f, 48.f, 48.f, 49.f, 50.f, 51.f, 51.f, 51.f, 48.f, 48.f, 48.f, 49.f, 50.f, 51.f, 51.f, 51.f,
        52.f, 52.f, 52.f, 53.f, 54.f, 55.f, 55.f, 55.f, 56.f, 56.f, 56.f, 57.f, 58.f, 59.f, 59.f, 59.f, 56.f, 56.f, 56.f, 57.f, 58.f, 59.f, 59.f, 59.f, 56.f, 56.f, 56.f, 57.f, 58.f, 59.f, 59.f,
        59.f, 60.f, 60.f, 60.f, 61.f, 62.f, 63.f, 63.f, 63.f, 60.f, 60.f, 60.f, 61.f, 62.f, 63.f, 63.f, 63.f, 60.f, 60.f, 60.f, 61.f, 62.f, 63.f, 63.f, 63.f, 64.f, 64.f, 64.f, 65.f, 66.f, 67.f,
        67.f, 67.f, 68.f, 68.f, 68.f, 69.f, 70.f, 71.f, 71.f, 71.f, 68.f, 68.f, 68.f, 69.f, 70.f, 71.f, 71.f, 71.f, 68.f, 68.f, 68.f, 69.f, 70.f, 71.f, 71.f, 71.f });

    asymmetricReplicationPaddingResult = std::make_unique<Tensor>(std::initializer_list<dtype>{
        0.,  0.,  0.,  0.,  1.,  2.,  3.,  3.,  3.,  0.,  0.,  0.,  0.,  1.,  2.,  3.,  3.,  3.,  4.,  4.,  4.,  4.,  5.,  6.,  7.,  7.,  7.,  8.,  8.,  8.,  8.,  9.,  10., 11., 11., 11.,
        12., 12., 12., 12., 13., 14., 15., 15., 15., 12., 12., 12., 12., 13., 14., 15., 15., 15., 16., 16., 16., 16., 17., 18., 19., 19., 19., 20., 20., 20., 20., 21., 22., 23., 23., 23.,
        24., 24., 24., 24., 25., 26., 27., 27., 27., 24., 24., 24., 24., 25., 26., 27., 27., 27., 28., 28., 28., 28., 29., 30., 31., 31., 31., 32., 32., 32., 32., 33., 34., 35., 35., 35.,
        36., 36., 36., 36., 37., 38., 39., 39., 39., 36., 36., 36., 36., 37., 38., 39., 39., 39., 40., 40., 40., 40., 41., 42., 43., 43., 43., 44., 44., 44., 44., 45., 46., 47., 47., 47.,
        48., 48., 48., 48., 49., 50., 51., 51., 51., 48., 48., 48., 48., 49., 50., 51., 51., 51., 52., 52., 52., 52., 53., 54., 55., 55., 55., 56., 56., 56., 56., 57., 58., 59., 59., 59.,
        60., 60., 60., 60., 61., 62., 63., 63., 63., 60., 60., 60., 60., 61., 62., 63., 63., 63., 64., 64., 64., 64., 65., 66., 67., 67., 67., 68., 68., 68., 68., 69., 70., 71., 71., 71. });

    replicationPaddingBackwardPassResult = std::make_unique<Tensor>(std::initializer_list<dtype>{ 9., 3., 3., 9., 3., 1., 1., 3., 9., 3., 3., 9., 9., 3., 3., 9., 3., 1., 1., 3., 9., 3., 3., 9.,
                                                                                                  9., 3., 3., 9., 3., 1., 1., 3., 9., 3., 3., 9., 9., 3., 3., 9., 3., 1., 1., 3., 9., 3., 3., 9.,
                                                                                                  9., 3., 3., 9., 3., 1., 1., 3., 9., 3., 3., 9., 9., 3., 3., 9., 3., 1., 1., 3., 9., 3., 3., 9. });
}

} // namespace UT