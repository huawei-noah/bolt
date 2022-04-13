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

#include <fstream>

#include <training/base/tools/DataTransformations.h>

using namespace raul;

struct TestDataTransformationNormalization : public testing::Test
{
    dtype normalizationCoefficient = 255_dt;
    std::unique_ptr<Tensor> testData;
    std::unique_ptr<Tensor> refData;

    void SetUp() final
    {
        testData = std::make_unique<Tensor>(2, 1, 2, 3);
        std::generate(testData->begin(), testData->end(), [n = 0]() mutable { return static_cast<dtype>(n++); });
        refData = std::make_unique<Tensor>(2, 1, 2, 3);
        std::transform(testData->begin(), testData->end(), refData->begin(), [nc = normalizationCoefficient](dtype v) { return v / nc; });
    }
};

TEST_F(TestDataTransformationNormalization, CanReturnTensorWithNormalizedDataUnit)
{
    PROFILE_TEST
    Normalize normalize(normalizationCoefficient);
    auto normalizedData = normalize(*testData);
    ASSERT_FLOAT_TENSORS_EQ((*normalizedData), (*refData), 1e-6_dt);
}

struct TestDataTransformationBuildOneHotVector : public testing::Test
{
    std::unique_ptr<Tensor> testData;
    std::unique_ptr<Tensor> refData;

    void SetUp() final
    {
        testData = std::make_unique<Tensor>(3, 1, 1, 1);
        std::generate(testData->begin(), testData->end(), [n = 1]() mutable { return static_cast<dtype>(n++); });

        refData = std::make_unique<Tensor>(3, 1, 1, 10);
        (*refData)[1] = 1_dt;
        (*refData)[12] = 1_dt;
        (*refData)[23] = 1_dt;
    }
};

TEST_F(TestDataTransformationBuildOneHotVector, CheckThatRationalValueCannotHaveFractionalPart)
{
    PROFILE_TEST
    {
        dtype v = 9_dt;
        ASSERT_TRUE(v == static_cast<dtype>(static_cast<int32_t>(v)));
    }

    {
        dtype v = -3_dt;
        ASSERT_TRUE(v == static_cast<dtype>(static_cast<int32_t>(v)));
    }

    {
        dtype v = 9.7_dt;
        ASSERT_FALSE(v == static_cast<dtype>(static_cast<int32_t>(v)));
    }

    {
        dtype v = -3.2_dt;
        ASSERT_FALSE(v == static_cast<dtype>(static_cast<int32_t>(v)));
    }

    {
        dtype v = -0.999999_dt;
        ASSERT_FALSE(v == static_cast<dtype>(static_cast<int32_t>(v)));
    }

    {
        dtype v = 0.999999_dt;
        ASSERT_FALSE(v == static_cast<dtype>(static_cast<int32_t>(v)));
    }

    {
        dtype v = 0._dt;
        ASSERT_TRUE(v == static_cast<dtype>(static_cast<int32_t>(v)));
        ASSERT_TRUE(static_cast<int32_t>(v) == 0);
    }

    {
        dtype v = -0._dt;
        ASSERT_TRUE(v == static_cast<dtype>(static_cast<int32_t>(v)));
        ASSERT_TRUE(static_cast<int32_t>(v) == 0);
    }

    {
        dtype v = 9.01_dt;
        ASSERT_FALSE(static_cast<dtype>(static_cast<int32_t>(v)) == v);
    }

    {
        dtype v = 9.001_dt;
        ASSERT_FALSE(static_cast<dtype>(static_cast<int32_t>(v)) == v);
    }

    {
        dtype v = 9.0001_dt;
        ASSERT_FALSE(static_cast<dtype>(static_cast<int32_t>(v)) == v);
    }

    {
        dtype v = 9.00001_dt;
        ASSERT_FALSE(static_cast<dtype>(static_cast<int32_t>(v)) == v);
    }

    {
        dtype v = 9.000001_dt;
        ASSERT_FALSE(static_cast<dtype>(static_cast<int32_t>(v)) == v);
    }

    {
        dtype v = 9.0000001_dt;
        ASSERT_TRUE(static_cast<dtype>(static_cast<int32_t>(v)) == v);
    }

    {
        dtype v = -3.01_dt;
        ASSERT_FALSE(static_cast<dtype>(static_cast<int32_t>(v)) == v);
    }

    {
        dtype v = -3.001_dt;
        ASSERT_FALSE(static_cast<dtype>(static_cast<int32_t>(v)) == v);
    }

    {
        dtype v = -3.0001_dt;
        ASSERT_FALSE(static_cast<dtype>(static_cast<int32_t>(v)) == v);
    }

    {
        dtype v = -3.00001_dt;
        ASSERT_FALSE(static_cast<dtype>(static_cast<int32_t>(v)) == v);
    }

    {
        dtype v = -3.000001_dt;
        ASSERT_FALSE(static_cast<dtype>(static_cast<int32_t>(v)) == v);
    }

    {
        dtype v = -3.0000001_dt;
        ASSERT_TRUE(static_cast<dtype>(static_cast<int32_t>(v)) == v);
    }
}

TEST_F(TestDataTransformationBuildOneHotVector, CanReturnTensorWithBinaryVectorsCreatedFromDataWithLevelsUnit)
{
    PROFILE_TEST
    BuildOneHotVector buildOneHotVector(10);
    auto binaryVectors = buildOneHotVector(*testData);
    ASSERT_FLOAT_TENSORS_EQ((*binaryVectors), (*refData), 1e-6_dt);
}

TEST_F(TestDataTransformationBuildOneHotVector, ThrowAnExceptionIfIncomingLevelValueIsIncorrect)
{
    PROFILE_TEST
    BuildOneHotVector buildOneHotVector(10);

    (*testData)[0] = 5.6_dt;
    ASSERT_THROW(buildOneHotVector(*testData), raul::Exception);

    (*testData)[0] = -2_dt;
    ASSERT_THROW(buildOneHotVector(*testData), raul::Exception);

    (*testData)[0] = 10_dt;
    ASSERT_THROW(buildOneHotVector(*testData), raul::Exception);
}

struct TestDataTransformationResize : public testing::Test
{
    Tensor testData =
        Tensor("", 2u, 2u, 3u, 4u, { 1_dt, 2_dt, 3_dt, 4_dt, 5_dt, 6_dt, 7_dt, 8_dt, 9_dt, 10_dt, 11_dt, 12_dt, 1_dt, 2_dt, 3_dt, 4_dt, 5_dt, 6_dt, 7_dt, 8_dt, 9_dt, 10_dt, 11_dt, 12_dt,
                                     1_dt, 2_dt, 3_dt, 4_dt, 5_dt, 6_dt, 7_dt, 8_dt, 9_dt, 10_dt, 11_dt, 12_dt, 1_dt, 2_dt, 3_dt, 4_dt, 5_dt, 6_dt, 7_dt, 8_dt, 9_dt, 10_dt, 11_dt, 12_dt });

    Tensor refData =
        Tensor("", 2u, 2u, 6u, 8u, { 1_dt, 1_dt, 2_dt, 2_dt, 3_dt, 3_dt, 4_dt, 4_dt, 1_dt, 1_dt, 2_dt,  2_dt,  3_dt,  3_dt,  4_dt,  4_dt,  5_dt, 5_dt, 6_dt,  6_dt,  7_dt,  7_dt,  8_dt,  8_dt,
                                     5_dt, 5_dt, 6_dt, 6_dt, 7_dt, 7_dt, 8_dt, 8_dt, 9_dt, 9_dt, 10_dt, 10_dt, 11_dt, 11_dt, 12_dt, 12_dt, 9_dt, 9_dt, 10_dt, 10_dt, 11_dt, 11_dt, 12_dt, 12_dt,
                                     1_dt, 1_dt, 2_dt, 2_dt, 3_dt, 3_dt, 4_dt, 4_dt, 1_dt, 1_dt, 2_dt,  2_dt,  3_dt,  3_dt,  4_dt,  4_dt,  5_dt, 5_dt, 6_dt,  6_dt,  7_dt,  7_dt,  8_dt,  8_dt,
                                     5_dt, 5_dt, 6_dt, 6_dt, 7_dt, 7_dt, 8_dt, 8_dt, 9_dt, 9_dt, 10_dt, 10_dt, 11_dt, 11_dt, 12_dt, 12_dt, 9_dt, 9_dt, 10_dt, 10_dt, 11_dt, 11_dt, 12_dt, 12_dt,
                                     1_dt, 1_dt, 2_dt, 2_dt, 3_dt, 3_dt, 4_dt, 4_dt, 1_dt, 1_dt, 2_dt,  2_dt,  3_dt,  3_dt,  4_dt,  4_dt,  5_dt, 5_dt, 6_dt,  6_dt,  7_dt,  7_dt,  8_dt,  8_dt,
                                     5_dt, 5_dt, 6_dt, 6_dt, 7_dt, 7_dt, 8_dt, 8_dt, 9_dt, 9_dt, 10_dt, 10_dt, 11_dt, 11_dt, 12_dt, 12_dt, 9_dt, 9_dt, 10_dt, 10_dt, 11_dt, 11_dt, 12_dt, 12_dt,
                                     1_dt, 1_dt, 2_dt, 2_dt, 3_dt, 3_dt, 4_dt, 4_dt, 1_dt, 1_dt, 2_dt,  2_dt,  3_dt,  3_dt,  4_dt,  4_dt,  5_dt, 5_dt, 6_dt,  6_dt,  7_dt,  7_dt,  8_dt,  8_dt,
                                     5_dt, 5_dt, 6_dt, 6_dt, 7_dt, 7_dt, 8_dt, 8_dt, 9_dt, 9_dt, 10_dt, 10_dt, 11_dt, 11_dt, 12_dt, 12_dt, 9_dt, 9_dt, 10_dt, 10_dt, 11_dt, 11_dt, 12_dt, 12_dt });
};

TEST_F(TestDataTransformationResize, ShouldCorrectlyDoTransformation)
{
    Resize resize(6, 8);
    auto transformationResult = resize(testData);

    ASSERT_FLOAT_TENSORS_EQ((*transformationResult), refData, 1e-6_dt);
}