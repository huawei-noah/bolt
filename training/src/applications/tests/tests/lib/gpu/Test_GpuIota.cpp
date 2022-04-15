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

#include <cstdio>
#include <tests/tools/TestTools.h>

#include <training/layers/basic/DataLayer.h>
#include <training/opencl/GPUCommon.h>

namespace UT
{

using namespace raul;

struct GpuIotaTest : public testing::TestWithParam<tuple<size_t, size_t, size_t, size_t, dtype>>
{
    const size_t batch = get<0>(GetParam());
    const size_t depth = get<1>(GetParam());
    const size_t height = get<2>(GetParam());
    const size_t width = get<3>(GetParam());
    const dtype startPoint = get<4>(GetParam());
};

TEST_P(GpuIotaTest, IotaGpuUnit)
{
    using namespace raul;

    PROFILE_TEST

    GPU_ONLY_TEST

    WorkflowEager work{ raul::CompressionMode::NONE, raul::CalculationMode::DETERMINISTIC, raul::AllocationMode::STANDARD, raul::ExecutionTarget::GPU };
    work.add<DataLayer>("data", DataParams{ { "in" }, depth, height, width });
    TENSORS_CREATE(batch);

    auto& memory_manager = work.getMemoryManager<MemoryManagerGPU>();
    gpu::iota(work.getKernelManager(), "", startPoint, batch * depth * height * width, memory_manager("in").getBuffer());

    const Tensor& in = memory_manager["in"];
    for (size_t i = 0; i < in.size(); ++i)
    {
        EXPECT_EQ(in[i], static_cast<dtype>(i) + startPoint);
    }
}

INSTANTIATE_TEST_SUITE_P(TestGpu,
                         GpuIotaTest,
                         testing::Values(make_tuple(2, 3, 4, 5, 0.0_dt),
                                         make_tuple(5, 4, 3, 2, 1.0_dt),
                                         make_tuple(101, 1, 1, 1, 5.0_dt),
                                         make_tuple(1, 93, 1, 1, 3.0_dt),
                                         make_tuple(2, 1, 101, 1, 101.0_dt),
                                         make_tuple(2, 1, 1, 105, 4.0_dt),
                                         make_tuple(1, 1, 1, 1, 0.0_dt),
                                         make_tuple(93, 94, 95, 96, 0.0_dt)));
} // UT namespace