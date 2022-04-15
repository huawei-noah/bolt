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
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/ReverseLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

TEST(TestReverseLayer, HeightUnit)
{
    PROFILE_TEST

    using namespace raul;

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    constexpr size_t BATCH_SIZE = 3u;
    constexpr size_t DEPTH = 1u;
    constexpr size_t HEIGHT = 5u;
    constexpr size_t WIDTH = 3u;

    const Tensor input{ 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 0.0_dt, 3.0_dt, 6.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,  0.0_dt, 0.0_dt,  0.0_dt,
                        1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 2.0_dt, 5.0_dt, 6.0_dt, 1.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,  0.0_dt, 0.0_dt,  0.0_dt,
                        1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 0.0_dt, 5.0_dt, 6.0_dt, 9.0_dt, 7.0_dt, 8.0_dt, 10.0_dt, 9.0_dt, 10.0_dt, 11.0_dt };
    const Tensor length{ 0.0_dt, 3.0_dt, 5.0_dt };

    const Tensor realOutput{ 1.0_dt, 2.0_dt,  3.0_dt,  4.0_dt, 5.0_dt, 0.0_dt,  3.0_dt, 6.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                             5.0_dt, 6.0_dt,  1.0_dt,  4.0_dt, 5.0_dt, 2.0_dt,  1.0_dt, 2.0_dt, 3.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt, 0.0_dt,
                             9.0_dt, 10.0_dt, 11.0_dt, 7.0_dt, 8.0_dt, 10.0_dt, 5.0_dt, 6.0_dt, 9.0_dt, 4.0_dt, 5.0_dt, 0.0_dt, 1.0_dt, 2.0_dt, 3.0_dt };

    // Input
    work.add<DataLayer>("data_input", DataParams{ { "input" }, DEPTH, HEIGHT, WIDTH });
    // Length
    work.add<DataLayer>("data_length", DataParams{ { "length" }, 1u, 1u, 1u });

    work.add<ReverseLayer>("reverse", BasicParams{ { "input", "length" }, { "output" } });

    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["input"] = TORANGE(input);
    memory_manager["length"] = TORANGE(length);
    memory_manager["outputGradient"] = TORANGE(input);

    ASSERT_NO_THROW(work.forwardPassTraining());

    // Forward checks
    const auto& output = memory_manager["output"];
    EXPECT_EQ(output.size(), realOutput.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
        EXPECT_EQ(output[i], realOutput[i]);
    }

    ASSERT_NO_THROW(work.backwardPassTraining());

    const auto& inGrad = memory_manager["inputGradient"];
    EXPECT_EQ(inGrad.size(), output.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_EQ(inGrad[i], realOutput[i]);
    }
}

}