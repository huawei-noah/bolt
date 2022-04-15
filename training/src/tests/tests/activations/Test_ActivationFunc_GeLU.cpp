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
#include <training/base/layers/activations/GeLUActivation.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestActivationFuncGeLU, GeLUErfUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    constexpr auto eps = 1e-4_dt;

    size_t BATCH_SIZE = 2;
    size_t WIDTH = 2;
    size_t HEIGHT = 2;
    size_t DEPTH = 3;

    Tensor realOut = { 0.8413_dt, 0.8413_dt, 1.9545_dt, 1.9545_dt, 0.8413_dt, 0.8413_dt, 1.9545_dt, 1.9545_dt, 0.8413_dt, 0.8413_dt, 1.9545_dt, 1.9545_dt,
                       0.8413_dt, 0.8413_dt, 2.9960_dt, 2.9960_dt, 3.9999_dt, 3.9999_dt, 2.9960_dt, 2.9960_dt, 1.9545_dt, 0.8413_dt, 2.9960_dt, 7.0000_dt };

    Tensor realGrad = { 1.0833_dt, 1.0833_dt, 2.1705_dt, 2.1705_dt, 1.0833_dt, 1.0833_dt, 2.1705_dt, 2.1705_dt, 1.0833_dt, 1.0833_dt, 2.1705_dt, 2.1705_dt,
                        1.0833_dt, 1.0833_dt, 3.0358_dt, 3.0358_dt, 4.0020_dt, 4.0020_dt, 3.0358_dt, 3.0358_dt, 2.1705_dt, 1.0833_dt, 3.0358_dt, 7.0000_dt };

    Tensor raw = { 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 3._dt, 3._dt, 4._dt, 4._dt, 3._dt, 3._dt, 2._dt, 1._dt, 3._dt, 7._dt };

    work.add<DataLayer>("data2", DataParams{ { "in", "target", "weights" }, DEPTH, HEIGHT, WIDTH });
    GeLUErf gelu{ "gelu", { { "in" }, { "out" } }, networkParameters };

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = TORANGE(raw);
    gelu.forwardCompute(NetworkMode::Train);

    memory_manager[Name("out").grad()].memAllocate(nullptr);
    memory_manager[Name("out").grad()] = TORANGE(raw);

    const raul::Tensor& out = memory_manager["out"];

    EXPECT_EQ(out.size(), realOut.size());
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], realOut[i], eps);

    std::cout << " - GeLU_Erf forward is Ok.\n";

    gelu.backwardCompute();

    const raul::Tensor& inGrad = memory_manager[raul::Name("in").grad()];

    EXPECT_EQ(inGrad.size(), realGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
        EXPECT_NEAR(inGrad[i], realGrad[i], eps);

    std::cout << " - GeLU_Erf backward is Ok.\n";
}

TEST(TestActivationFuncGeLU, GeLUATanhUnit)
{
    PROFILE_TEST
    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    constexpr auto eps = 2e-3_dt;

    size_t BATCH_SIZE = 2;
    size_t WIDTH = 2;
    size_t HEIGHT = 2;
    size_t DEPTH = 3;

    Tensor realOut = { 0.8412_dt, 0.8412_dt, 1.9546_dt, 1.9546_dt, 0.8412_dt, 0.8412_dt, 1.9546_dt, 1.9546_dt, 0.8412_dt, 0.8412_dt, 1.9546_dt, 1.9546_dt,
                       0.8412_dt, 0.8412_dt, 2.9964_dt, 2.9964_dt, 3.9999_dt, 3.9999_dt, 2.9964_dt, 2.9964_dt, 1.9546_dt, 0.8412_dt, 2.9964_dt, 7.0000_dt };

    Tensor realGrad = { 1.0830_dt, 1.0830_dt, 2.1722_dt, 2.1722_dt, 1.0830_dt, 1.0830_dt, 2.1722_dt, 2.1722_dt, 1.0830_dt, 1.0830_dt, 2.1722_dt, 2.1722_dt,
                        1.0830_dt, 1.0830_dt, 3.0348_dt, 3.0348_dt, 4.0013_dt, 4.0013_dt, 3.0348_dt, 3.0348_dt, 2.1722_dt, 1.0830_dt, 3.0348_dt, 7.0000_dt };

    Tensor raw = { 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 2._dt, 2._dt, 1._dt, 1._dt, 3._dt, 3._dt, 4._dt, 4._dt, 3._dt, 3._dt, 2._dt, 1._dt, 3._dt, 7._dt };

    work.add<DataLayer>("data2", DataParams{ { "in", "target", "weights" }, DEPTH, HEIGHT, WIDTH });
    GeLUTanh gelu{ "gelu", { { "in" }, { "out" } }, networkParameters };

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in"] = TORANGE(raw);
    gelu.forwardCompute(NetworkMode::Train);

    memory_manager[Name("out").grad()].memAllocate(nullptr);
    memory_manager[Name("out").grad()] = TORANGE(raw);

    const raul::Tensor& out = memory_manager["out"];

    EXPECT_EQ(out.size(), realOut.size());
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], realOut[i], eps);

    std::cout << " - GeLU_Tanh forward is Ok.\n";

    gelu.backwardCompute();

    const raul::Tensor& inGrad = memory_manager[raul::Name("in").grad()];

    EXPECT_EQ(inGrad.size(), realGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
        EXPECT_NEAR(inGrad[i], realGrad[i], eps);

    std::cout << " - GeLU_Tanh backward is Ok.\n";
}

} // UT namespace