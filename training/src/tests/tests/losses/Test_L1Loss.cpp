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
#include <tests/tools/callbacks/TensorChecker.h>

#include <training/base/common/Common.h>
#include <training/base/common/MemoryManager.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/TensorLayer.h>
#include <training/base/loss/L1Loss.h>
#include <training/compiler/Workflow.h>

namespace UT
{

using namespace raul;
using namespace UT::tools::callbacks;
using namespace std;

TEST(TestLoss, L1LossUnit)
{
    PROFILE_TEST

    dtype eps = TODTYPE(1e-4);

    Tensor inputs = { 1.3_dt, 1.2_dt, 0.1_dt, -4.0_dt, -0.3_dt, -10.0_dt, 1.0_dt, -1.0_dt, 2.0_dt, -2.3_dt };
    Tensor targets = { 0.1_dt, 1.2_dt, 1.0_dt, 0.1_dt, 7.7_dt, 0.2_dt, 0.2_dt, 0.2_dt, -1.3_dt, -2.3_dt };
    Tensor weights = { 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt, 1_dt };

    size_t batch = 2;

    std::string reduction[] = { "none", "sum", "mean", "sum_over_nonzero_weights" };

    Tensor realLoss[] = { { 1.1999999285_dt, 0.0_dt, 0.8999999762_dt, 4.0999999046_dt, 8.0_dt, 10.1999998093_dt, 0.8000000119_dt, 1.2000000477_dt, 3.2999999523_dt, 0.0_dt },
                          { 29.6999988556_dt },
                          { 2.9699997902_dt },
                          { 2.9699997902_dt } };

    Tensor realInGrad[] = { { 1.0_dt, 0.0_dt, -1.0_dt, -1.0_dt, -1.0_dt, -1.0_dt, 1.0_dt, -1.0_dt, 1.0_dt, 0.0_dt },
                            { 1.0_dt, 0.0_dt, -1.0_dt, -1.0_dt, -1.0_dt, -1.0_dt, 1.0_dt, -1.0_dt, 1.0_dt, 0.0_dt },
                            { 0.1000000015_dt, 0.0_dt, -0.1000000015_dt, -0.1000000015_dt, -0.1000000015_dt, -0.1000000015_dt, 0.1000000015_dt, -0.1000000015_dt, 0.1000000015_dt, 0.0_dt },
                            { 0.1000000015_dt, 0.0_dt, -0.1000000015_dt, -0.1000000015_dt, -0.1000000015_dt, -0.1000000015_dt, 0.1000000015_dt, -0.1000000015_dt, 0.1000000015_dt, 0.0_dt } };

    for (size_t iter = 0; iter < std::size(reduction); ++iter)
    {
        Workflow work;

        work.add<DataLayer>("data", DataParams{ { "in", "target", "weights" }, 1, 1, inputs.size() / batch });
        if (iter != 3)
        {
            work.add<L1Loss>("loss", LossParams{ { "in", "target" }, { "loss" }, reduction[iter].c_str() });
        }
        else
        {
            work.add<L1Loss>("loss", LossParams{ { "in", "target", "weights" }, { "loss" }, reduction[iter].c_str() });
        }

        if (iter == 0)
        {
            work.add<TensorLayer>("grad",
                                  TensorParams{ { Name("loss").grad() }, WShape{ BS(), 1u, 1u, inputs.size() / batch }, 1_dt, Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read, true, true });
        }

        TENSORS_CREATE(batch);
        auto& memory_manager = work.getMemoryManager();
        memory_manager["in"] = TORANGE(inputs);
        memory_manager["target"] = TORANGE(targets);
        memory_manager["weights"] = TORANGE(weights);
        memory_manager.createTensor("real_grad", batch, 1, 1, inputs.size() / batch, TORANGE(realInGrad[iter]));

        TensorChecker checker({}, { { Name("in").grad(), "real_grad" } }, eps);

        work.getNetworkParameters().mCallback = checker;

        work.forwardPassTraining();
        const Tensor& loss = memory_manager["loss"];

        EXPECT_EQ(loss.size(), realLoss[iter].size());
        for (size_t i = 0; i < loss.size(); ++i)
        {
            EXPECT_NEAR(loss[i], realLoss[iter][i], eps);
        }
        printf(" - L1Loss[reduction=%s] forward is Ok.\n", reduction[iter].c_str());

        work.backwardPassTraining();

        printf(" - L1Loss[reduction=%s] backward is Ok.\n", reduction[iter].c_str());

        memory_manager.clear();
    }
}

}