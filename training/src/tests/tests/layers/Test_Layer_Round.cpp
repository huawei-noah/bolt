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

#include <training/base/common/MemoryManager.h>
#include <training/base/layers/basic/RoundLayer.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestLayerRound, RoundCPUUnit)
{
    PROFILE_TEST

    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    Tensor t({ 0.9_dt, 2.5_dt, 2.3_dt, 1.5_dt, -4.5_dt, 0.45_dt, 0.51_dt });
    Tensor golden({ 1_dt, 2_dt, 2_dt, 2_dt, -4_dt, 0_dt, 1.0_dt });
    const auto eps_rel = TODTYPE(1e-6);

    work.add<DataLayer>("data", DataParams{ { "x" }, 1, 1, t.size() });

    // Apply function
    RoundLayer l("exp", BasicParams{ { "x" }, { "out" } }, networkParameters);
    TENSORS_CREATE(1);
    work.getMemoryManager()["x"] = TORANGE(t);
    l.forwardCompute(NetworkMode::Train);

    // Checks

    const auto& out_tensor = work.getMemoryManager()["out"];

    EXPECT_EQ(out_tensor.size(), golden.size());

    for (size_t i = 0; i < out_tensor.size(); ++i)
    {
        ASSERT_TRUE(tools::expect_near_relative(out_tensor[i], golden[i], eps_rel));
    }
}

}