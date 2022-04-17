// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <gtest/gtest.h>
#include <tests/tools/TestTools.h>

#include <training/base/layers/activations/LeakyReLUActivation.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/compiler/Workflow.h>

namespace UT
{

namespace
{

raul::dtype golden_lrelu(const raul::dtype x, const raul::dtype negSlope)
{
    return std::max(0.0_dt, x) + negSlope * std::min(0.0_dt, x);
}

raul::dtype golden_lrelu_grad(const raul::dtype out, const raul::dtype grad, const raul::dtype negSlope)
{
    return (out > 0.0_dt) ? grad : negSlope * grad;
}

}

TEST(TestActivationFuncLeakyReLU, DeterministicUnit)
{
    PROFILE_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 2;
    constexpr size_t WIDTH = 3;
    constexpr size_t HEIGHT = 4;
    constexpr size_t DEPTH = 5;
    constexpr dtype negSlope = 0.05_dt;
    constexpr dtype eps = 1.0e-6_dt;

    const Tensor in{ -1.12583983_dt, -1.15236020_dt, -2.50578582e-01_dt, -4.33878809e-01_dt, 8.48710358e-01_dt, 6.92009151e-01_dt,
        -3.16012770e-01_dt, -2.11521935_dt, 3.22274923e-01_dt, -1.26333475_dt, 3.49983186e-01_dt, 3.08133930e-01_dt, 1.19841509e-01_dt,
        1.23765790_dt, 1.11677718_dt, -2.47278154e-01_dt, -1.35265374_dt, -1.69593120_dt, 5.66650629e-01_dt, 7.93508351e-01_dt,
        5.98839462e-01_dt, -1.55509508_dt, -3.41360390e-01_dt, 1.85300612_dt, 7.50189483e-01_dt, -5.85497558e-01_dt, -1.73396751e-01_dt,
        1.83477938e-01_dt, 1.38936615_dt, 1.58633423_dt, 9.46298361e-01_dt, -8.43676746e-01_dt, -6.13583088e-01_dt, 3.15927416e-02_dt,
        -4.92676973e-01_dt, 2.48414755e-01_dt, 4.39695835e-01_dt, 1.12411186e-01_dt, 6.40792370e-01_dt, 4.41156268e-01_dt,
        -1.02309652e-01_dt, 7.92443991e-01_dt, -2.89667696e-01_dt, 5.25074862e-02_dt, 5.22860467e-01_dt, 2.30220532_dt, -1.46889389_dt,
        -1.58668876_dt, -6.73089921e-01_dt, 8.72831225e-01_dt, 1.05535746_dt, 1.77843720e-01_dt, -2.30335474e-01_dt, -3.91754389e-01_dt,
        5.43294728e-01_dt, -3.95157546e-01_dt, -4.46217179e-01_dt, 7.44020700e-01_dt, 1.52097952_dt, 3.41050267_dt, -1.53118432_dt,
        -1.23413503_dt, 1.81972528_dt, -5.51528692e-01_dt, -5.69248080e-01_dt, 9.19971406e-01_dt, 1.11081612_dt, 1.28987408_dt,
        -1.47817397_dt, 2.56723285_dt, -4.73119795e-01_dt, 3.35550755e-01_dt, -1.62932599_dt, -5.49743652e-01_dt, -4.79834259e-01_dt,
        -4.99681532e-01_dt, -1.06698036_dt, 1.11493957_dt, -1.40671432e-01_dt, 8.05753589e-01_dt, -9.33482349e-02_dt, 6.87050223e-01_dt,
        -8.38315368e-01_dt, 8.91821750e-04_dt, 8.41894090e-01_dt, -4.00034159e-01_dt, 1.03946197_dt, 3.58153105e-01_dt, -2.46000946e-01_dt,
        2.30251646_dt, -1.88168919_dt, -4.97270226e-02_dt, -1.04497862_dt, -9.56500828e-01_dt, 3.35318595e-02_dt, 7.10086584e-01_dt,
        1.64586699_dt, -1.36016893_dt, 3.44565421e-01_dt, 5.19867718e-01_dt, -2.61332250_dt, -1.69647443_dt, -2.28241786e-01_dt,
        2.79955000e-01_dt, -7.01523602e-01_dt, 1.03668678_dt, -6.03670120e-01_dt, -1.27876520_dt, 9.29502323e-02_dt, -6.66099727e-01_dt,
        6.08047187e-01_dt, -7.30019867e-01_dt, 1.37503791_dt, 6.59631073e-01_dt, 4.76557106e-01_dt, -1.01630747_dt, 1.80366978e-01_dt,
        1.08331867e-01_dt, -7.54823267e-01_dt, 2.44318530e-01_dt };
    
    const Tensor realOut{ -5.62919937e-02_dt, -5.76180108e-02_dt, -1.25289289e-02_dt, -2.16939412e-02_dt, 8.48710358e-01_dt,
        6.92009151e-01_dt, -1.58006381e-02_dt, -1.05760969e-01_dt, 3.22274923e-01_dt, -6.31667376e-02_dt, 3.49983186e-01_dt,
        3.08133930e-01_dt, 1.19841509e-01_dt, 1.23765790_dt, 1.11677718_dt, -1.23639079e-02_dt, -6.76326901e-02_dt, -8.47965628e-02_dt,
        5.66650629e-01_dt, 7.93508351e-01_dt, 5.98839462e-01_dt, -7.77547583e-02_dt, -1.70680191e-02_dt, 1.85300612_dt, 7.50189483e-01_dt,
        -2.92748790e-02_dt, -8.66983738e-03_dt, 1.83477938e-01_dt, 1.38936615_dt, 1.58633423_dt, 9.46298361e-01_dt, -4.21838388e-02_dt,
        -3.06791551e-02_dt, 3.15927416e-02_dt, -2.46338490e-02_dt, 2.48414755e-01_dt, 4.39695835e-01_dt, 1.12411186e-01_dt,
        6.40792370e-01_dt, 4.41156268e-01_dt, -5.11548249e-03_dt, 7.92443991e-01_dt, -1.44833848e-02_dt, 5.25074862e-02_dt,
        5.22860467e-01_dt, 2.30220532_dt, -7.34446943e-02_dt, -7.93344378e-02_dt, -3.36544961e-02_dt, 8.72831225e-01_dt,
        1.05535746_dt, 1.77843720e-01_dt, -1.15167741e-02_dt, -1.95877198e-02_dt, 5.43294728e-01_dt, -1.97578780e-02_dt,
        -2.23108586e-02_dt, 7.44020700e-01_dt, 1.52097952_dt, 3.41050267_dt, -7.65592158e-02_dt, -6.17067516e-02_dt, 1.81972528_dt,
        -2.75764354e-02_dt, -2.84624044e-02_dt, 9.19971406e-01_dt, 1.11081612_dt, 1.28987408_dt, -7.39087015e-02_dt, 2.56723285_dt,
        -2.36559901e-02_dt, 3.35550755e-01_dt, -8.14663023e-02_dt, -2.74871830e-02_dt, -2.39917133e-02_dt, -2.49840766e-02_dt,
        -5.33490181e-02_dt, 1.11493957_dt, -7.03357160e-03_dt, 8.05753589e-01_dt, -4.66741202e-03_dt, 6.87050223e-01_dt,
        -4.19157706e-02_dt, 8.91821750e-04_dt, 8.41894090e-01_dt, -2.00017076e-02_dt, 1.03946197_dt, 3.58153105e-01_dt,
        -1.23000471e-02_dt, 2.30251646_dt, -9.40844640e-02_dt, -2.48635118e-03_dt, -5.22489324e-02_dt, -4.78250422e-02_dt,
        3.35318595e-02_dt, 7.10086584e-01_dt, 1.64586699_dt, -6.80084452e-02_dt, 3.44565421e-01_dt, 5.19867718e-01_dt,
        -1.30666122e-01_dt, -8.48237202e-02_dt, -1.14120897e-02_dt, 2.79955000e-01_dt, -3.50761823e-02_dt, 1.03668678_dt,
        -3.01835071e-02_dt, -6.39382601e-02_dt, 9.29502323e-02_dt, -3.33049856e-02_dt, 6.08047187e-01_dt, -3.65009941e-02_dt,
        1.37503791_dt, 6.59631073e-01_dt, 4.76557106e-01_dt, -5.08153737e-02_dt, 1.80366978e-01_dt, 1.08331867e-01_dt, -3.77411656e-02_dt,
        2.44318530e-01_dt };

    const Tensor realInGrad{ 0.05_dt, 0.05_dt, 0.05_dt, 0.05_dt, 1.00_dt, 1.00_dt, 0.05_dt, 0.05_dt, 1.00_dt, 0.05_dt, 1.00_dt, 1.00_dt,
        1.00_dt, 1.00_dt, 1.00_dt, 0.05_dt, 0.05_dt, 0.05_dt, 1.00_dt, 1.00_dt, 1.00_dt, 0.05_dt, 0.05_dt, 1.00_dt, 1.00_dt, 0.05_dt,
        0.05_dt, 1.00_dt, 1.00_dt, 1.00_dt, 1.00_dt, 0.05_dt, 0.05_dt, 1.00_dt, 0.05_dt, 1.00_dt, 1.00_dt, 1.00_dt, 1.00_dt, 1.00_dt,
        0.05_dt, 1.00_dt, 0.05_dt, 1.00_dt, 1.00_dt, 1.00_dt, 0.05_dt, 0.05_dt, 0.05_dt, 1.00_dt, 1.00_dt, 1.00_dt, 0.05_dt, 0.05_dt,
        1.00_dt, 0.05_dt, 0.05_dt, 1.00_dt, 1.00_dt, 1.00_dt, 0.05_dt, 0.05_dt, 1.00_dt, 0.05_dt, 0.05_dt, 1.00_dt, 1.00_dt, 1.00_dt,
        0.05_dt, 1.00_dt, 0.05_dt, 1.00_dt, 0.05_dt, 0.05_dt, 0.05_dt, 0.05_dt, 0.05_dt, 1.00_dt, 0.05_dt, 1.00_dt, 0.05_dt, 1.00_dt,
        0.05_dt, 1.00_dt, 1.00_dt, 0.05_dt, 1.00_dt, 1.00_dt, 0.05_dt, 1.00_dt, 0.05_dt, 0.05_dt, 0.05_dt, 0.05_dt, 1.00_dt, 1.00_dt,
        1.00_dt, 0.05_dt, 1.00_dt, 1.00_dt, 0.05_dt, 0.05_dt, 0.05_dt, 1.00_dt, 0.05_dt, 1.00_dt, 0.05_dt, 0.05_dt, 1.00_dt, 0.05_dt,
        1.00_dt, 0.05_dt, 1.00_dt, 1.00_dt, 1.00_dt, 0.05_dt, 1.00_dt, 1.00_dt, 0.05_dt, 1.00_dt };

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<LeakyReLUActivation>("lrelu", LeakyReLUParams{ { "in" }, { "out" }, negSlope });

    TENSORS_CREATE(BATCH_SIZE);

    memory_manager["in"] = TORANGE(in);

    ASSERT_NO_THROW(work.forwardPassTraining());
    const Tensor& out = memory_manager["out"];
    EXPECT_EQ(realOut.size(), out.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], realOut[i], eps);
    }

    memory_manager[Name("out").grad()] = 1.0_dt;

    ASSERT_NO_THROW(work.backwardPassTraining());
    const Tensor& inGrad = memory_manager[Name("in").grad()];
    EXPECT_EQ(realInGrad.size(), inGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], realInGrad[i], eps);
    }
}

TEST(TestActivationFuncLeakyReLU, RandomUnit)
{
    PROFILE_TEST

    using namespace raul;

    constexpr size_t BATCH_SIZE = 5;
    constexpr size_t WIDTH = 23;
    constexpr size_t HEIGHT = 11;
    constexpr size_t DEPTH = 7;
    constexpr dtype negSlope = 0.1_dt;
    constexpr dtype eps = 1.0e-6_dt;
    constexpr auto range = std::make_pair(-1.0_dt, 1.0_dt);

    // Initialization
    MANAGERS_DEFINE
    NETWORK_PARAMS_DEFINE(networkParameters);

    work.add<DataLayer>("data", DataParams{ { "in" }, DEPTH, HEIGHT, WIDTH });
    work.add<LeakyReLUActivation>("lrelu", LeakyReLUParams{ { "in" }, { "out" }, negSlope });

    TENSORS_CREATE(BATCH_SIZE);

    tools::init_rand_tensor("in", range, memory_manager);
    tools::init_rand_tensor("outGradient", range, memory_manager);

    ASSERT_NO_THROW(work.forwardPassTraining());
    const Tensor& in = memory_manager["in"];
    const Tensor& out = memory_manager["out"];
    EXPECT_EQ(in.size(), out.size());
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_NEAR(out[i], golden_lrelu(in[i], negSlope), eps);
    }

    ASSERT_NO_THROW(work.backwardPassTraining());
    const Tensor& inGrad = memory_manager[Name("in").grad()];
    const Tensor& outGrad = memory_manager[Name("out").grad()];
    EXPECT_EQ(in.size(), inGrad.size());
    for (size_t i = 0; i < inGrad.size(); ++i)
    {
        EXPECT_NEAR(inGrad[i], golden_lrelu_grad(in[i], outGrad[i], negSlope), eps);
    }
}

}