// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/GTestExtensions.h>
#include <tests/tools/TestTools.h>

#include <training/base/optimizers/Optimizer.h>

using namespace raul;

namespace UT
{

class OptimizerStub : public optimizers::Optimizer
{
  private:
    void optimize(MemoryManager&, Tensor&, const Tensor&) final {}
    std::ostream& as_ostream(std::ostream& out) const final { return out; }
};

struct TestOptimizer : public testing::Test
{
    MemoryManager memoryManager;
    std::unique_ptr<Tensor> zeros;
    std::unique_ptr<Tensor> testGradients;
    std::unique_ptr<Tensor> testParameter;

    void SetUp() final
    {
        testGradients = std::make_unique<Tensor>("test", 10, 10, 10, 10, 5_dt);
        testParameter = std::make_unique<Tensor>("test", 10, 10, 10, 10, 7_dt);
    }

    void TearDown() final {}
};

TEST_F(TestOptimizer, ShouldNotResetGradientAfterParameterOptimizationUnit)
{
    PROFILE_TEST
    auto optimizer = std::make_unique<OptimizerStub>();

    Tensor grad(testGradients->getShape(), TORANGE(*testGradients));

    optimizer->operator()(memoryManager, *testParameter, *testGradients);

    ASSERT_FLOAT_TENSORS_EQ((*testGradients), grad, 1e-6_dt);
}

} // namespace UT