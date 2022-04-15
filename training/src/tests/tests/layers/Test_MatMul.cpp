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
#include <training/base/layers/basic/MatMulLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

using namespace raul;

TEST(TestMatMul, MatMulUnit)
{
    PROFILE_TEST

    constexpr auto eps = 1e-6_dt;

    constexpr size_t BATCH_SIZE = 2;
    constexpr size_t MODEL_SIZE = 5;

    const Tensor realOut = { 10._dt, 16._dt, 15._dt, 16._dt, 26._dt, 24._dt, 20._dt, 32._dt, 30._dt, 32._dt, 52._dt, 48._dt };

    const Tensor realGrad1 = { -4.0000, 9.0000,  6.0000,  5.0000,  2.0000, -20.5000, 24.0000, 33.5000, 9.0000,  7.5000,
                               -8.0000, 18.0000, 12.0000, 10.0000, 4.0000, -41.0000, 48.0000, 67.0000, 18.0000, 15.0000 };

    const Tensor realGrad2 = { 0.0000, 1.5000, -7.0000, 3.0000, 3.0000, 11.0000, 4.0000, 5.0000, 10.0000, 0.0000, 0.0000, 0.0000, 10.0000, 12.5000, 25.0000,
                               0.0000, 1.5000, -7.0000, 3.0000, 3.0000, 11.0000, 4.0000, 5.0000, 10.0000, 0.0000, 0.0000, 0.0000, 10.0000, 12.5000, 25.0000 };

    const Tensor raw1 = { 1._dt, 1._dt, 2._dt, 0._dt, 5._dt, -1._dt, 2._dt, 2._dt, 0._dt, 5._dt, 1._dt, 1._dt, 2._dt, 0._dt, 5._dt, -1._dt, 2._dt, 2._dt, 0._dt, 5._dt };

    const Tensor raw2 = { -1., -3., -3., 4., 4., 3., 1., 5., 5., 2., 2., 1., 1., 1., 1., -2., -6., -6., 8., 8., 6., 2., 10., 10., 4., 4., 2., 2., 2., 2. };

    const Tensor deltas = { 1._dt, 2._dt, -1._dt, 1._dt, 0.5_dt, 6._dt, 1._dt, 2._dt, -1._dt, 1._dt, 0.5_dt, 6._dt };

    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data_in1", DataParams{ { "in1" }, 1u, raw1.size() / BATCH_SIZE / MODEL_SIZE, MODEL_SIZE });
    work.add<DataLayer>("data_in2", DataParams{ { "in2" }, 1u, MODEL_SIZE, raw2.size() / BATCH_SIZE / MODEL_SIZE });

    MatMulLayer mm("mm", { { "in1", "in2" }, "out", 1._dt }, networkParameters);
    TENSORS_CREATE(BATCH_SIZE);
    memory_manager["in1"] = TORANGE(raw1);
    memory_manager["in2"] = TORANGE(raw2);
    memory_manager[Name("out").grad()] = TORANGE(deltas);

    mm.forwardCompute(NetworkMode::Train);
    const Tensor& out = memory_manager["out"];

    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], realOut[i], eps);
    }

    printf(" - MatMul forward is Ok.\n");

    mm.backwardCompute();

    const Tensor& in1Grad = memory_manager[Name("in1").grad()];
    const Tensor& in2Grad = memory_manager[Name("in2").grad()];

    for (size_t i = 0; i < in1Grad.size(); ++i)
    {
        EXPECT_NEAR(in1Grad[i], realGrad1[i], eps);
    }

    for (size_t i = 0; i < in2Grad.size(); ++i)
    {
        EXPECT_NEAR(in2Grad[i], realGrad2[i], eps);
    }

    printf(" - MatMul backward is Ok.\n");
}

}