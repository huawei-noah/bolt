// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <training/common/MemoryManager.h>
#include <training/layers/composite/tacotron/SequenceMaskLayer.h>
#include <training/network/Layers.h>
#include <training/network/Workflow.h>

namespace UT
{

using namespace raul;
using namespace raul::tacotron;

struct TestLayerSequenceMask : public testing::TestWithParam<tuple<size_t, size_t, size_t, size_t, dtype, dtype>>
{
    const size_t batch = get<0>(GetParam());
    const size_t depth = get<1>(GetParam());
    const size_t height = get<2>(GetParam());
    const size_t width = get<3>(GetParam());
    const std::pair<dtype, dtype> range = std::make_pair(get<4>(GetParam()), get<5>(GetParam()));

    WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU };

    void SetUp() final
    {
        if (!Common::hasOpenCL())
        {
            return;
        }

        work.add<DataLayer>("lengths", DataParams{ { "lengths" }, 1u, 1u, 1u });
        work.add<DataLayer>("input", DataParams{ { "input" }, depth, height, width });
        work.add<SequenceMaskLayer>("seq_mask", BasicParams{ { "input", "lengths" }, { "mask" } }, 1u);
        TENSORS_CREATE(batch);
    }
};

TEST_P(TestLayerSequenceMask, GpuUnit)
{
    PROFILE_TEST

    GPU_ONLY_TEST

    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();

    tools::init_rand_tensor("lengths", range, memory_manager);

    ASSERT_NO_THROW(work.forwardPassTraining());

    const Tensor& lengths = memory_manager["lengths"];
    const Tensor& mask = memory_manager["mask"];
    auto mask2d = mask.reshape(yato::dims(lengths.size(), mask.size() / lengths.size()));
    for (size_t i = 0; i < lengths.size(); ++i)
    {
        for (size_t j = 0; j < mask.size() / lengths.size(); ++j)
        {
            EXPECT_EQ(mask2d[i][j], (j < static_cast<size_t>(lengths[i]) * width) ? 1.0_dt : 0.0_dt);
        }
    }

    ASSERT_NO_THROW(work.backwardPassTraining());
}

INSTANTIATE_TEST_SUITE_P(TestGpu,
                         TestLayerSequenceMask,
                         testing::Values(make_tuple(2, 3, 4, 5, 1.0_dt, 3.0_dt),
                                         make_tuple(5, 4, 3, 2, 0.0_dt, 2.0_dt),
                                         make_tuple(101, 1, 7, 1, 0.0_dt, 6.0_dt),
                                         make_tuple(1, 93, 19, 1, 11.0_dt, 15.0_dt),
                                         make_tuple(21, 1, 101, 1, 21.0_dt, 99.0_dt),
                                         make_tuple(2, 1, 2, 105, 1.0_dt, 1.0_dt),
                                         make_tuple(1, 1, 1, 1, 0.0_dt, 0.0_dt),
                                         make_tuple(93, 94, 95, 96, 11.0_dt, 73.0_dt),
                                         make_tuple(2, 1, 984, 20, 759.0_dt, 984.0_dt),
                                         make_tuple(2, 1, 982, 1, 759.0_dt, 984.0_dt)));
} // UT namespace