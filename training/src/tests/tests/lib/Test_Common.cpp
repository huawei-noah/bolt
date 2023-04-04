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
#include <training/base/common/Tensor.h>

namespace UT
{

const raul::dtype EPSILON_ACCURACY = TODTYPE(1e-6);

TEST(TestCommon, CheckGemmUnit)
{
    PROFILE_TEST

    const raul::Tensor matA = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    const raul::Tensor matB = { 10.0f, 20.0f, 40.0f, 50.0f, 70.0f, 80.0f };
    const raul::Tensor matCG = { 300.0f, 360.0f, 660.0f, 810.0f, 1020.0f, 1260.0f };
    raul::Tensor matC(matCG.size());

    raul::Common::gemm(CblasNoTrans, CblasNoTrans, 3, 2, 3, 1.0_dt, matA.data(), matB.data(), 0.0_dt, matC.data());
    for (size_t q = 0; q < matC.size(); ++q)
        CHECK_NEAR(matC[q], matCG[q], EPSILON_ACCURACY);

    raul::Common::gemm(CblasNoTrans, CblasNoTrans, 3, 2, 3, 1.0_dt, matA.data(), matB.data(), 1.0_dt, matC.data());
    for (size_t q = 0; q < matC.size(); ++q)
        CHECK_NEAR(matC[q], 2.0f * matCG[q], EPSILON_ACCURACY);
}

TEST(TestCommon, CheckGemmFP16Unit)
{
    PROFILE_TEST

    const raul::TensorFP16 matA = { 1.0_hf, 2.0_hf, 3.0_hf, 4.0_hf, 5.0_hf, 6.0_hf, 7.0_hf, 8.0_hf, 9.0_hf };
    const raul::TensorFP16 matB = { 10.0_hf, 20.0_hf, 40.0_hf, 50.0_hf, 70.0_hf, 80.0_hf };
    const raul::TensorFP16 matCG = { 300.0_hf, 360.0_hf, 660.0_hf, 810.0_hf, 1020.0_hf, 1260.0_hf };
    raul::TensorFP16 matC(matCG.size());

    raul::Common::gemm(CblasNoTrans, CblasNoTrans, 3, 2, 3, 1.0_dt, matA.data(), matB.data(), 0.0_dt, matC.data());
    for (size_t q = 0; q < matC.size(); ++q)
        CHECK_NEAR(matC[q], matCG[q], EPSILON_ACCURACY);

    raul::Common::gemm(CblasNoTrans, CblasNoTrans, 3, 2, 3, 1.0_dt, matA.data(), matB.data(), 1.0_dt, matC.data());
    for (size_t q = 0; q < matC.size(); ++q)
        CHECK_NEAR(raul::toFloat32(matC[q]), 2.0f * raul::toFloat32(matCG[q]), EPSILON_ACCURACY);
}

TEST(TestCommon, Arange1Unit)
{
    PROFILE_TEST
    raul::Tensor t(2, 1, 2, 2);
    raul::Common::arange(t.begin(), t.end());
    raul::Tensor realT({ 0.0_dt, 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 6.0_dt, 7.0_dt });
    for (size_t q = 0; q < t.size(); ++q)
        EXPECT_EQ(t[q], realT[q]);
}

TEST(TestCommon, Arange2Unit)
{
    PROFILE_TEST
    raul::Tensor t(2, 1, 2, 2);
    raul::Common::arange(t.begin(), t.end(), 1._dt, 2.0_dt);
    raul::Tensor realT({ 1.0_dt, 3.0_dt, 5.0_dt, 7.0_dt, 9.0_dt, 11.0_dt, 13.0_dt, 15.0_dt });
    for (size_t q = 0; q < t.size(); ++q)
        EXPECT_EQ(t[q], realT[q]);
}

TEST(TestCommon, Arange3Unit)
{
    PROFILE_TEST
    raul::Tensor t(2, 1, 2, 2);
    raul::Common::arange(t.begin(), t.end(), 1._dt);
    raul::Tensor realT({ 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 6.0_dt, 7.0_dt, 8.0_dt });
    for (size_t q = 0; q < t.size(); ++q)
        EXPECT_EQ(t[q], realT[q]);
}

TEST(TestCommon, Arange4Unit)
{
    PROFILE_TEST
    raul::Tensor t(2, 1, 2, 2);
    raul::Common::arange(t.begin(), t.end(), static_cast<unsigned char>(1));
    raul::Tensor realT({ 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 6.0_dt, 7.0_dt, 8.0_dt });
    for (size_t q = 0; q < t.size(); ++q)
        EXPECT_EQ(t[q], realT[q]);
}

TEST(TestCommon, Arange5Unit)
{
    PROFILE_TEST
    raul::Tensor t(2, 1, 2, 2);
    raul::Common::arange(t, static_cast<unsigned char>(1));
    raul::Tensor realT({ 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt, 6.0_dt, 7.0_dt, 8.0_dt });
    for (size_t q = 0; q < t.size(); ++q)
        EXPECT_EQ(t[q], realT[q]);
}

TEST(TestCommon, CheckTriuUnit)
{
    PROFILE_TEST
    const raul::Tensor res_m1 = { 1, 2, 3, 4, 5, 6, 0, 8, 9, 0, 0, 12 };
    const raul::Tensor res_p1 = { 0, 2, 3, 0, 0, 6, 0, 0, 0, 0, 0, 0 };
    const raul::Tensor res_0 = { 1, 2, 3, 0, 5, 6, 0, 0, 9, 0, 0, 0 };

    {
        raul::Tensor m = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        raul::Common::triu(m.data(), 4, 3, -1);
        for (size_t q = 0; q < m.size(); ++q)
            EXPECT_EQ(m[q], res_m1[q]);
    }
    {
        raul::Tensor m = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        raul::Common::triu(m.data(), 4, 3, 1);
        for (size_t q = 0; q < m.size(); ++q)
            EXPECT_EQ(m[q], res_p1[q]);
    }
    {
        raul::Tensor m = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        raul::Common::triu(m.data(), 4, 3);
        for (size_t q = 0; q < m.size(); ++q)
            EXPECT_EQ(m[q], res_0[q]);
    }
}

TEST(TestCommon, CheckGemmMoreUnit)
{
    PROFILE_TEST
    const raul::Tensor matA = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    const raul::Tensor matB = { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f };
    const raul::Tensor matCG = { 300.0f, 360.0f, 420.0f, 660.0f, 810.0f, 960.0f, 1020.0f, 1260.0f, 1500.0f };
    raul::Tensor matC(matCG.size());

    raul::Common::gemm(CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0_dt, matA.data(), matB.data(), 0.0_dt, matC.data());
    for (size_t q = 0; q < matC.size(); ++q)
        CHECK_NEAR(matC[q], matCG[q], EPSILON_ACCURACY);

    raul::Common::gemm(CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0_dt, matA.data(), matB.data(), 1.0_dt, matC.data());
    for (size_t q = 0; q < matC.size(); ++q)
        CHECK_NEAR(matC[q], 2.0f * matCG[q], EPSILON_ACCURACY);

    const raul::Tensor matCG_AT = { 660.0f, 780.0f, 900.0f, 780.0f, 930.0f, 1080.0f, 900.0f, 1080.0f, 1260.0f };

    raul::Common::gemm(CblasTrans, CblasNoTrans, 3, 3, 3, 1.0_dt, matA.data(), matB.data(), 0.0_dt, matC.data());
    for (size_t q = 0; q < matC.size(); ++q)
        CHECK_NEAR(matC[q], matCG_AT[q], EPSILON_ACCURACY);

    const raul::Tensor matCG_BT = { 140.0f, 320.0f, 500.0f, 320.0f, 770.0f, 1220.0f, 500.0f, 1220.0f, 1940.0f };

    raul::Common::gemm(CblasNoTrans, CblasTrans, 3, 3, 3, 1.0_dt, matA.data(), matB.data(), 0.0_dt, matC.data());
    for (size_t q = 0; q < matC.size(); ++q)
        CHECK_NEAR(matC[q], matCG_BT[q], EPSILON_ACCURACY);

    const raul::Tensor matCG_ATBT = { 300.0f, 660.0f, 1020.0f, 360.0f, 810.0f, 1260.0f, 420.0f, 960.0f, 1500.0f };

    raul::Common::gemm(CblasTrans, CblasTrans, 3, 3, 3, 1.0_dt, matA.data(), matB.data(), 0.0_dt, matC.data());
    for (size_t q = 0; q < matC.size(); ++q)
        CHECK_NEAR(matC[q], matCG_ATBT[q], EPSILON_ACCURACY);
}

TEST(TestCommon, CheckGemmFP16MoreUnit)
{
    PROFILE_TEST
    const raul::TensorFP16 matA = { 1.0_hf, 2.0_hf, 3.0_hf, 4.0_hf, 5.0_hf, 6.0_hf, 7.0_hf, 8.0_hf, 9.0_hf };
    const raul::TensorFP16 matB = { 10.0_hf, 20.0_hf, 30.0_hf, 40.0_hf, 50.0_hf, 60.0_hf, 70.0_hf, 80.0_hf, 90.0_hf };
    const raul::TensorFP16 matCG = { 300.0_hf, 360.0_hf, 420.0_hf, 660.0_hf, 810.0_hf, 960.0_hf, 1020.0_hf, 1260.0_hf, 1500.0_hf };
    raul::TensorFP16 matC(matCG.size());

    raul::Common::gemm(CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0_dt, matA.data(), matB.data(), 0.0_dt, matC.data());
    for (size_t q = 0; q < matC.size(); ++q)
        CHECK_NEAR(matC[q], matCG[q], EPSILON_ACCURACY);

    raul::Common::gemm(CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0_dt, matA.data(), matB.data(), 1.0_dt, matC.data());
    for (size_t q = 0; q < matC.size(); ++q)
        CHECK_NEAR(raul::toFloat32(matC[q]), 2.0f * raul::toFloat32(matCG[q]), EPSILON_ACCURACY);

    const raul::TensorFP16 matCG_AT = { 660.0_hf, 780.0_hf, 900.0_hf, 780.0_hf, 930.0_hf, 1080.0_hf, 900.0_hf, 1080.0_hf, 1260.0_hf };

    raul::Common::gemm(CblasTrans, CblasNoTrans, 3, 3, 3, 1.0_dt, matA.data(), matB.data(), 0.0_dt, matC.data());
    for (size_t q = 0; q < matC.size(); ++q)
        CHECK_NEAR(matC[q], matCG_AT[q], EPSILON_ACCURACY);

    const raul::TensorFP16 matCG_BT = { 140.0_hf, 320.0_hf, 500.0_hf, 320.0_hf, 770.0_hf, 1220.0_hf, 500.0_hf, 1220.0_hf, 1940.0_hf };

    raul::Common::gemm(CblasNoTrans, CblasTrans, 3, 3, 3, 1.0_dt, matA.data(), matB.data(), 0.0_dt, matC.data());
    for (size_t q = 0; q < matC.size(); ++q)
        CHECK_NEAR(matC[q], matCG_BT[q], EPSILON_ACCURACY);

    const raul::TensorFP16 matCG_ATBT = { 300.0_hf, 660.0_hf, 1020.0_hf, 360.0_hf, 810.0_hf, 1260.0_hf, 420.0_hf, 960.0_hf, 1500.0_hf };

    raul::Common::gemm(CblasTrans, CblasTrans, 3, 3, 3, 1.0_dt, matA.data(), matB.data(), 0.0_dt, matC.data());
    for (size_t q = 0; q < matC.size(); ++q)
        CHECK_NEAR(matC[q], matCG_ATBT[q], EPSILON_ACCURACY);
}

#if !defined(_BLAS_ENHANCE)
TEST(TestCommon, CheckAXPYUnit)
{
    PROFILE_TEST
    const raul::Tensor vecA = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    const raul::Tensor vecBOriginal = { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f };

    {
        raul::Tensor vecB(TORANGE(vecBOriginal));
        raul::Common::axpy(vecA.size(), 1.0_dt, vecA.data(), 1, vecB.data(), 1);
        for (size_t q = 0; q < vecB.size(); ++q)
            CHECK_NEAR(vecB[q], vecA[q] + vecBOriginal[q], EPSILON_ACCURACY);

        vecB = TORANGE(vecBOriginal);
        raul::Common::axpy(vecA.size(), 2.0_dt, vecA.data(), 1, vecB.data(), 1);
        for (size_t q = 0; q < vecB.size(); ++q)
            CHECK_NEAR(vecB[q], 2.0f * vecA[q] + vecBOriginal[q], EPSILON_ACCURACY);

        vecB = TORANGE(vecBOriginal);
        raul::Common::axpy(3, 1.0_dt, vecA.data(), 1, vecB.data(), 1, 3);
        for (size_t q = 0; q < 3; ++q)
            CHECK_NEAR(vecB[q], vecA[q + 3] + vecBOriginal[q], EPSILON_ACCURACY);

        vecB = TORANGE(vecBOriginal);
        raul::Common::axpy(3, 1.0_dt, vecA.data(), 1, vecB.data(), 1, 0, 3);
        for (size_t q = 0; q < 3; ++q)
            CHECK_NEAR(vecB[q + 3], vecA[q] + vecBOriginal[q + 3], EPSILON_ACCURACY);

        vecB = TORANGE(vecBOriginal);
        raul::Common::axpy(3, 1.0_dt, vecA.data(), 1, vecB.data(), 1, 3, 3);
        for (size_t q = 0; q < 3; ++q)
            CHECK_NEAR(vecB[q + 3], vecA[q + 3] + vecBOriginal[q + 3], EPSILON_ACCURACY);
    }

    const raul::Tensor vecASparse = { 1.0f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 1.0f, 6.0f, 1.0f };
    const raul::Tensor vecBOriginalSparse = { 10.0f, 1.0f, 20.0f, 1.0f, 30.0f, 1.0f, 40.0f, 1.0f, 50.0f, 1.0f, 60.0f, 1.0f };

    {
        raul::Tensor vecB(TORANGE(vecBOriginal));
        raul::Common::axpy(vecA.size(), 1.0_dt, vecASparse.data(), 2, vecB.data(), 1);
        for (size_t q = 0; q < vecB.size(); ++q)
            CHECK_NEAR(vecB[q], vecA[q] + vecBOriginal[q], EPSILON_ACCURACY);
    }

    {
        raul::Tensor vecB(TORANGE(vecBOriginalSparse));

        vecB = TORANGE(vecBOriginalSparse);
        raul::Common::axpy(vecA.size(), 1.0_dt, vecA.data(), 1, vecB.data(), 2);
        for (size_t q = 0; q < vecB.size(); q += 2)
            CHECK_NEAR(vecB[q], vecA[q / 2] + vecBOriginalSparse[q], EPSILON_ACCURACY);

        vecB = TORANGE(vecBOriginalSparse);
        raul::Common::axpy(vecA.size(), 1.0_dt, vecASparse.data(), 2, vecB.data(), 2);
        for (size_t q = 0; q < vecB.size(); q += 2)
            CHECK_NEAR(vecB[q], vecASparse[q] + vecBOriginalSparse[q], EPSILON_ACCURACY);
    }
}

TEST(TestCommon, CheckAXPYFP16Unit)
{
    PROFILE_TEST
    const raul::TensorFP16 vecA = { 1.0_hf, 2.0_hf, 3.0_hf, 4.0_hf, 5.0_hf, 6.0_hf };
    const raul::TensorFP16 vecBOriginal = { 10.0_hf, 20.0_hf, 30.0_hf, 40.0_hf, 50.0_hf, 60.0_hf };

    {
        raul::TensorFP16 vecB(TORANGE_FP16(vecBOriginal));
        raul::Common::axpy(vecA.size(), 1.0_dt, vecA.data(), 1, vecB.data(), 1);
        for (size_t q = 0; q < vecB.size(); ++q)
            CHECK_NEAR(raul::toFloat32(vecB[q]), raul::toFloat32(vecA[q]) + raul::toFloat32(vecBOriginal[q]), EPSILON_ACCURACY);

        vecB = TORANGE_FP16(vecBOriginal);
        raul::Common::axpy(vecA.size(), 2.0_dt, vecA.data(), 1, vecB.data(), 1);
        for (size_t q = 0; q < vecB.size(); ++q)
            CHECK_NEAR(raul::toFloat32(vecB[q]), 2.0f * raul::toFloat32(vecA[q]) + raul::toFloat32(vecBOriginal[q]), EPSILON_ACCURACY);

        vecB = TORANGE_FP16(vecBOriginal);
        raul::Common::axpy(3, 1.0_dt, vecA.data(), 1, vecB.data(), 1, 3);
        for (size_t q = 0; q < 3; ++q)
            CHECK_NEAR(raul::toFloat32(vecB[q]), raul::toFloat32(vecA[q + 3]) + raul::toFloat32(vecBOriginal[q]), EPSILON_ACCURACY);

        vecB = TORANGE_FP16(vecBOriginal);
        raul::Common::axpy(3, 1.0_dt, vecA.data(), 1, vecB.data(), 1, 0, 3);
        for (size_t q = 0; q < 3; ++q)
            CHECK_NEAR(raul::toFloat32(vecB[q + 3]), raul::toFloat32(vecA[q]) + raul::toFloat32(vecBOriginal[q + 3]), EPSILON_ACCURACY);

        vecB = TORANGE_FP16(vecBOriginal);
        raul::Common::axpy(3, 1.0_dt, vecA.data(), 1, vecB.data(), 1, 3, 3);
        for (size_t q = 0; q < 3; ++q)
            CHECK_NEAR(raul::toFloat32(vecB[q + 3]), raul::toFloat32(vecA[q + 3]) + raul::toFloat32(vecBOriginal[q + 3]), EPSILON_ACCURACY);
    }

    const raul::TensorFP16 vecASparse = { 1.0_hf, 1.0_hf, 2.0_hf, 1.0_hf, 3.0_hf, 1.0_hf, 4.0_hf, 1.0_hf, 5.0_hf, 1.0_hf, 6.0_hf, 1.0_hf };
    const raul::TensorFP16 vecBOriginalSparse = { 10.0_hf, 1.0_hf, 20.0_hf, 1.0_hf, 30.0_hf, 1.0_hf, 40.0_hf, 1.0_hf, 50.0_hf, 1.0_hf, 60.0_hf, 1.0_hf };

    {
        raul::TensorFP16 vecB(TORANGE_FP16(vecBOriginal));
        raul::Common::axpy(vecA.size(), 1.0_dt, vecASparse.data(), 2, vecB.data(), 1);
        for (size_t q = 0; q < vecB.size(); ++q)
            CHECK_NEAR(raul::toFloat32(vecB[q]), raul::toFloat32(vecA[q]) + raul::toFloat32(vecBOriginal[q]), EPSILON_ACCURACY);
    }

    {
        raul::TensorFP16 vecB(TORANGE_FP16(vecBOriginalSparse));

        vecB = TORANGE_FP16(vecBOriginalSparse);
        raul::Common::axpy(vecA.size(), 1.0_dt, vecA.data(), 1, vecB.data(), 2);
        for (size_t q = 0; q < vecB.size(); q += 2)
            CHECK_NEAR(raul::toFloat32(vecB[q]), raul::toFloat32(vecA[q / 2]) + raul::toFloat32(vecBOriginalSparse[q]), EPSILON_ACCURACY);

        vecB = TORANGE_FP16(vecBOriginalSparse);
        raul::Common::axpy(vecA.size(), 1.0_dt, vecASparse.data(), 2, vecB.data(), 2);
        for (size_t q = 0; q < vecB.size(); q += 2)
            CHECK_NEAR(raul::toFloat32(vecB[q]), raul::toFloat32(vecASparse[q]) + raul::toFloat32(vecBOriginalSparse[q]), EPSILON_ACCURACY);
    }
}
#endif 

TEST(TestCommon, CheckDotUnit)
{
    PROFILE_TEST
    const raul::Tensor vecA = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    const raul::Tensor vecASparse = { 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f, 5.0f, 0.0f, 6.0f, 0.0f };

    const raul::Tensor vecB = { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f };
    const raul::Tensor vecBSparse = { 10.0f, 0.0f, 20.0f, 0.0f, 30.0f, 0.0f, 40.0f, 0.0f, 50.0f, 0.0f, 60.0f, 0.0f };

    const float dotRes = 910.0f;

    CHECK_NEAR(raul::Common::dot(vecA.size(), vecA.data(), 1, vecB.data(), 1), dotRes, EPSILON_ACCURACY);
    CHECK_NEAR(raul::Common::dot(vecA.size(), vecA.data(), 1, vecBSparse.data(), 2), dotRes, EPSILON_ACCURACY);
    CHECK_NEAR(raul::Common::dot(vecA.size(), vecASparse.data(), 2, vecB.data(), 1), dotRes, EPSILON_ACCURACY);
    CHECK_NEAR(raul::Common::dot(vecA.size(), vecASparse.data(), 2, vecBSparse.data(), 2), dotRes, EPSILON_ACCURACY);
}

TEST(TestCommon, CheckScaleUnit)
{
    PROFILE_TEST
    const raul::Tensor vecAoriginal = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    const raul::Tensor vecAoriginalSparse = { 1.0f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 1.0f, 6.0f, 1.0f };

    {
        raul::Tensor vecA(TORANGE(vecAoriginal));
        raul::Common::scal(vecA.size(), 10.0f, vecA.data(), 1);
        for (size_t q = 0; q < vecA.size(); ++q)
            CHECK_NEAR(vecA[q], vecAoriginal[q] * 10.0f, EPSILON_ACCURACY);
    }

    {
        raul::Tensor vecA(TORANGE(vecAoriginalSparse));
        raul::Common::scal(vecAoriginal.size(), 10.0f, vecA.data(), 2);
        for (size_t q = 0; q < vecA.size(); q += 2)
            CHECK_NEAR(vecA[q], vecAoriginalSparse[q] * 10.0f, EPSILON_ACCURACY);
    }
}

TEST(TestCommon, CheckTransposeUnit)
{
    PROFILE_TEST
    raul::Tensor vecA = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    raul::Common::transpose(vecA, 3);
    CHECK_NEAR(vecA[0], 1.0f, EPSILON_ACCURACY);
    CHECK_NEAR(vecA[1], 4.0f, EPSILON_ACCURACY);
    CHECK_NEAR(vecA[2], 2.0f, EPSILON_ACCURACY);
    CHECK_NEAR(vecA[3], 5.0f, EPSILON_ACCURACY);
    CHECK_NEAR(vecA[4], 3.0f, EPSILON_ACCURACY);
    CHECK_NEAR(vecA[5], 6.0f, EPSILON_ACCURACY);

    raul::Tensor vecB = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    raul::Common::transpose(vecB, 3);
    CHECK_NEAR(vecB[0], 1.0f, EPSILON_ACCURACY);
    CHECK_NEAR(vecB[1], 4.0f, EPSILON_ACCURACY);
    CHECK_NEAR(vecB[2], 7.0f, EPSILON_ACCURACY);
    CHECK_NEAR(vecB[3], 2.0f, EPSILON_ACCURACY);
    CHECK_NEAR(vecB[4], 5.0f, EPSILON_ACCURACY);
    CHECK_NEAR(vecB[5], 8.0f, EPSILON_ACCURACY);
    CHECK_NEAR(vecB[6], 3.0f, EPSILON_ACCURACY);
    CHECK_NEAR(vecB[7], 6.0f, EPSILON_ACCURACY);
    CHECK_NEAR(vecB[8], 9.0f, EPSILON_ACCURACY);
}

TEST(TestCommon, CheckAddPadding2DUnit)
{
    PROFILE_TEST
    {
        raul::Tensor src = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };

        raul::Tensor dst(5 * 4);
        raul::Common::addPadding2D(src.data(), dst.data(), 1, 3, 2, 5, 4);

        raul::Tensor dstGold = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 4.0f, 5.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        ASSERT_EQ(dst.size(), static_cast<size_t>(5 * 4));

        for (size_t i = 0; i < 5 * 4; ++i)
        {
            ASSERT_EQ(dstGold[i], dst[i]);
        }
    }

    {
        raul::Tensor src = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };

        raul::Tensor dst(6 * 5);
        raul::Common::addPadding2D(src.data(), dst.data(), 1, 3, 2, 6, 5);

        raul::Tensor dstGold = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f, 4.0f, 5.0f,
                                 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        ASSERT_EQ(dst.size(), static_cast<size_t>(6 * 5));

        for (size_t i = 0; i < 6 * 5; ++i)
        {
            ASSERT_EQ(dstGold[i], dst[i]);
        }
    }

    {
        raul::Tensor src = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                             4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };

        raul::Tensor dst(2 * 5 * 4);
        raul::Common::addPadding2D(src.data(), dst.data(), 2, 3, 2, 5, 4);

        raul::Tensor dstGold = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 4.0f, 5.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

                                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 5.0f, 6.0f, 0.0f, 0.0f, 7.0f, 8.0f, 9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        ASSERT_EQ(dst.size(), static_cast<size_t>(2 * 5 * 4));

        for (size_t i = 0; i < 5 * 4; ++i)
        {
            ASSERT_EQ(dstGold[i], dst[i]);
            ASSERT_EQ(dstGold[5 * 4 + i], dst[5 * 4 + i]);
        }
    }
}

TEST(TestCommon, CheckRemovePadding2DUnit)
{
    PROFILE_TEST
    {
        raul::Tensor src = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 4.0f, 5.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        raul::Tensor dst(3 * 2);
        raul::Common::removePadding2D(src.data(), dst.data(), 1, 5, 4, 3, 2);

        raul::Tensor dstGold = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };

        ASSERT_EQ(dst.size(), static_cast<size_t>(3 * 2));

        for (size_t i = 0; i < 3 * 2; ++i)
        {
            ASSERT_EQ(dstGold[i], dst[i]);
        }
    }

    {

        raul::Tensor src = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f, 4.0f, 5.0f,
                             6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        raul::Tensor dst(3 * 2);
        raul::Common::removePadding2D(src.data(), dst.data(), 1, 6, 5, 3, 2);

        raul::Tensor dstGold = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        ASSERT_EQ(dst.size(), static_cast<size_t>(3 * 2));

        for (size_t i = 0; i < 3 * 2; ++i)
        {
            ASSERT_EQ(dstGold[i], dst[i]);
        }
    }

    {
        raul::Tensor src = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 4.0f, 5.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 5.0f, 6.0f, 0.0f, 0.0f, 7.0f, 8.0f, 9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        raul::Tensor dst(2 * 3 * 2);
        raul::Common::removePadding2D(src.data(), dst.data(), 2, 5, 4, 3, 2);

        raul::Tensor dstGold = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                                 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };

        ASSERT_EQ(dst.size(), static_cast<size_t>(2 * 3 * 2));

        for (size_t i = 0; i < 3 * 2; ++i)
        {
            ASSERT_EQ(dstGold[i], dst[i]);
            ASSERT_EQ(dstGold[2 * 3 + i], dst[2 * 3 + i]);
        }
    }
}

TEST(TestCommon, CheckAddPadding1DUnit)
{
    PROFILE_TEST
    {
        raul::Tensor src = { 1_dt, 2_dt, 3_dt };

        raul::Tensor dst(5);
        raul::Common::addPadding1D(src.data(), dst.data(), 1, 3, 5);

        raul::Tensor dstGold = { 0_dt, 1_dt, 2_dt, 3_dt, 0_dt };

        for (size_t i = 0; i < dst.size(); ++i)
        {
            ASSERT_EQ(dstGold[i], dst[i]);
        }
    }

    {
        raul::Tensor src = { 1_dt, 2_dt, 3_dt };

        raul::Tensor dst(9);
        raul::Common::addPadding1D(src.data(), dst.data(), 3, 1, 3);

        raul::Tensor dstGold = { 0_dt, 1_dt, 0_dt, 0_dt, 2_dt, 0_dt, 0_dt, 3_dt, 0_dt };

        for (size_t i = 0; i < dst.size(); ++i)
        {
            ASSERT_EQ(dstGold[i], dst[i]);
        }
    }

    {
        raul::Tensor src = { 1_dt, 2_dt, 3_dt, 4_dt, 5_dt, 6_dt };

        raul::Tensor dst(12);
        raul::Common::addPadding1D(src.data(), dst.data(), 2, 3, 6);

        raul::Tensor dstGold = { 0_dt, 1_dt, 2_dt, 3_dt, 0_dt, 0_dt, 0_dt, 4_dt, 5_dt, 6_dt, 0_dt, 0_dt };

        for (size_t i = 0; i < dst.size(); ++i)
        {
            ASSERT_EQ(dstGold[i], dst[i]);
        }
    }
}

TEST(TestCommon, CheckRemovePadding1DUnit)
{
    PROFILE_TEST
    {
        raul::Tensor src = { 0_dt, 1_dt, 2_dt, 3_dt, 0_dt };

        raul::Tensor dst(3);
        raul::Common::removePadding1D(src.data(), dst.data(), 1, 5, 3);

        raul::Tensor dstGold = { 1_dt, 2_dt, 3_dt };

        for (size_t i = 0; i < dst.size(); ++i)
        {
            ASSERT_EQ(dstGold[i], dst[i]);
        }
    }

    {
        raul::Tensor src = { 0_dt, 1_dt, 0_dt, 0_dt, 2_dt, 0_dt, 0_dt, 3_dt, 0_dt };

        raul::Tensor dst(3);
        raul::Common::removePadding1D(src.data(), dst.data(), 3, 3, 1);

        raul::Tensor dstGold = { 1_dt, 2_dt, 3_dt };

        for (size_t i = 0; i < dst.size(); ++i)
        {
            ASSERT_EQ(dstGold[i], dst[i]);
        }
    }

    {
        raul::Tensor src = { 0_dt, 1_dt, 2_dt, 3_dt, 0_dt, 0_dt, 0_dt, 4_dt, 5_dt, 6_dt, 0_dt, 0_dt };

        raul::Tensor dst(6);
        raul::Common::removePadding1D(src.data(), dst.data(), 2, 6, 3);

        raul::Tensor dstGold = { 1_dt, 2_dt, 3_dt, 4_dt, 5_dt, 6_dt };

        for (size_t i = 0; i < dst.size(); ++i)
        {
            ASSERT_EQ(dstGold[i], dst[i]);
        }
    }
}

/*
TEST(TestCommon, Reshape3DTo2DUnit)
{
    PROFILE_TEST

    const std::vector<std::vector<raul::Tensor>> input = { {
                                                              // batch0
                                                              { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 },           // ch0
                                                              { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 }, // ch1
                                                              { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 }  // ch2
                                                          },
                                                          {
                                                              // batch1
                                                              { 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 }, // ch0
                                                              { 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 }, // ch1
                                                              { 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 }  // ch2
                                                          } };

    const std::vector<raul::Tensor> goldOutput = { { // batch0
                                                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 },
                                                  { // batch1
                                                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 } };

    std::vector<raul::Tensor> output = raul::Common::reshape3DTo2D(input, 3, 2);

    ASSERT_EQ(output.size(), static_cast<size_t>(2));
    ASSERT_EQ(output[0].size(), static_cast<size_t>(3 * 10));
    ASSERT_EQ(output[1].size(), static_cast<size_t>(3 * 10));

    for (size_t i = 0; i < goldOutput.size(); ++i)
    {
        for (size_t w = 0; w < goldOutput[i].size(); ++w)
        {
            ASSERT_EQ(goldOutput[i][w], output[i][w]);
        }
    }
}

TEST(TestCommon, Reshape1Dto2DUnit)
{
    PROFILE_TEST

    const raul::Tensor input = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };

    const std::vector<raul::Tensor> goldOutput = { { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 } };

    {
        std::vector<raul::Tensor> output = raul::Common::reshape1DTo2D(input, 10, 2);

        ASSERT_EQ(output.size(), static_cast<size_t>(2));
        ASSERT_EQ(output[0].size(), static_cast<size_t>(10));

        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 10; ++j)
            {
                ASSERT_EQ(goldOutput[i][j], output[i][j]);
            }
    }

    {
        std::vector<raul::Tensor> output = raul::Common::reshape1DTo2D(input, 10, 2, 0);

        ASSERT_EQ(output.size(), static_cast<size_t>(2));
        ASSERT_EQ(output[0].size(), static_cast<size_t>(10));

        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 10; ++j)
            {
                ASSERT_EQ(goldOutput[i][j], output[i][j]);
            }
    }

    const std::vector<raul::Tensor> goldOutputOffset = { { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 } };

    {
        std::vector<raul::Tensor> output = raul::Common::reshape1DTo2D(input, 10, 1, 1);

        ASSERT_EQ(output.size(), static_cast<size_t>(1));
        ASSERT_EQ(output[0].size(), static_cast<size_t>(10));

        for (size_t i = 0; i < 1; ++i)
            for (size_t j = 0; j < 10; ++j)
            {
                ASSERT_EQ(goldOutputOffset[i][j], output[i][j]);
            }
    }
}*/

TEST(TestCommon, Im2ColUnit)
{
    PROFILE_TEST

    // basic
    {
        raul::Tensor image = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };

        raul::Tensor matrix(4 * 4);
        raul::Common::im2col(image.data(), 3, 3, 1, 2, 2, 1, 1, 0, 0, matrix.data());

        ASSERT_EQ(matrix.size(), static_cast<size_t>(4 * 4));

        raul::Tensor matrixGold = { 1.0f, 4.0f, 2.0f, 5.0f, 4.0f, 7.0f, 5.0f, 8.0f, 2.0f, 5.0f, 3.0f, 6.0f, 5.0f, 8.0f, 6.0f, 9.0f };

        for (size_t q = 0; q < 4 * 4; ++q)
        {
            ASSERT_EQ(matrix[q], matrixGold[q]);
        }
    }

    // not symmetric stride
    {
        raul::Tensor image = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };

        raul::Tensor matrix(2 * 4);
        raul::Common::im2col(image.data(), 3, 3, 1, 2, 2, 2, 1, 0, 0, matrix.data());

        ASSERT_EQ(matrix.size(), static_cast<size_t>(2 * 4));

        raul::Tensor matrixGold = { 1.0f, 2.0f, 4.0f, 5.0f, 2.0f, 3.0f, 5.0f, 6.0f };

        for (size_t q = 0; q < 2 * 4; ++q)
        {
            ASSERT_EQ(matrix[q], matrixGold[q]);
        }
    }

    // not symmetric stride
    {
        raul::Tensor image = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };

        raul::Tensor matrix(2 * 4);
        raul::Common::im2col(image.data(), 3, 3, 1, 2, 2, 1, 2, 0, 0, matrix.data());

        ASSERT_EQ(matrix.size(), static_cast<size_t>(2 * 4));

        raul::Tensor matrixGold = {
            1.0f, 4.0f, 4.0f, 7.0f, 2.0f, 5.0f, 5.0f, 8.0f,
        };

        for (size_t q = 0; q < 2 * 4; ++q)
        {
            ASSERT_EQ(matrix[q], matrixGold[q]);
        }
    }

    // not square kernel
    {
        raul::Tensor image = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };

        raul::Tensor matrix(2 * 6);
        raul::Common::im2col(image.data(), 3, 3, 1, 3, 2, 1, 1, 0, 0, matrix.data());

        ASSERT_EQ(matrix.size(), static_cast<size_t>(2 * 6));

        raul::Tensor matrixGold = { 1.0f, 2.0f, 4.0f, 5.0f, 7.0f, 8.0f, 2.0f, 3.0f, 5.0f, 6.0f, 8.0f, 9.0f };

        for (size_t q = 0; q < 2 * 6; ++q)
        {
            ASSERT_EQ(matrix[q], matrixGold[q]);
        }
    }

    // not square kernel
    {
        raul::Tensor image = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };

        raul::Tensor matrix(2 * 6);
        raul::Common::im2col(image.data(), 3, 3, 1, 2, 3, 1, 1, 0, 0, matrix.data());

        ASSERT_EQ(matrix.size(), static_cast<size_t>(2 * 6));

        raul::Tensor matrixGold = { 1.0f, 4.0f, 4.0f, 7.0f, 2.0f, 5.0f, 5.0f, 8.0f, 3.0f, 6.0f, 6.0f, 9.0f };

        for (size_t q = 0; q < 2 * 6; ++q)
        {
            ASSERT_EQ(matrix[q], matrixGold[q]);
        }
    }

    // not symmetric stride & not square kernel
    {
        raul::Tensor image = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };

        raul::Tensor matrix(1 * 6);
        raul::Common::im2col(image.data(), 3, 3, 1, 3, 2, 1, 2, 0, 0, matrix.data());

        ASSERT_EQ(matrix.size(), static_cast<size_t>(1 * 6));

        raul::Tensor matrixGold = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f };

        for (size_t q = 0; q < 1 * 6; ++q)
        {
            ASSERT_EQ(matrix[q], matrixGold[q]);
        }
    }

    // not symmetric stride & not square kernel
    {
        raul::Tensor image = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };

        raul::Tensor matrix(1 * 6);
        raul::Common::im2col(image.data(), 3, 3, 1, 2, 3, 2, 1, 0, 0, matrix.data());

        ASSERT_EQ(matrix.size(), static_cast<size_t>(1 * 6));

        raul::Tensor matrixGold = { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f };

        for (size_t q = 0; q < 1 * 6; ++q)
        {
            ASSERT_EQ(matrix[q], matrixGold[q]);
        }
    }

    // symmetric padding
    {
        raul::Tensor image = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };

        raul::Tensor matrix(16 * 4);
        raul::Common::im2col(image.data(), 3, 3, 1, 2, 2, 1, 1, 1, 1, matrix.data());

        ASSERT_EQ(matrix.size(), static_cast<size_t>(16 * 4));

        raul::Tensor matrixGold = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 4.0f,
                                    7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 0.0f, 0.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        for (size_t q = 0; q < 16 * 4; ++q)
        {
            ASSERT_EQ(matrix[q], matrixGold[q]) << q;
        }
    }

    // not symmetric padding
    {
        raul::Tensor image = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };

        raul::Tensor matrix(8 * 4);
        raul::Common::im2col(image.data(), 3, 3, 1, 2, 2, 1, 1, 1, 0, matrix.data());

        ASSERT_EQ(matrix.size(), static_cast<size_t>(8 * 4));

        raul::Tensor matrixGold = { 0.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 0.0f,
                                    0.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 0.0f };

        for (size_t q = 0; q < 8 * 4; ++q)
        {
            ASSERT_EQ(matrix[q], matrixGold[q]) << q;
        }
    }

    // not symmetric padding
    {
        raul::Tensor image = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };

        raul::Tensor matrix(8 * 4);
        raul::Common::im2col(image.data(), 3, 3, 1, 2, 2, 1, 1, 0, 1, matrix.data());

        ASSERT_EQ(matrix.size(), static_cast<size_t>(8 * 4));

        raul::Tensor matrixGold = { 0.0f, 0.0f, 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, 0.0f, 0.0f, 4.0f, 7.0f, 5.0f, 8.0f, 6.0f, 9.0f,
                                    1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, 0.0f, 0.0f, 4.0f, 7.0f, 5.0f, 8.0f, 6.0f, 9.0f, 0.0f, 0.0f };

        for (size_t q = 0; q < 8 * 4; ++q)
        {
            ASSERT_EQ(matrix[q], matrixGold[q]) << q;
        }
    }

    // larger
    {
        raul::Tensor image = { 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,

                               17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,

                               33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f };

        raul::Tensor matrix(3 * 2 * 2 * 3 * 3);
        raul::Common::im2col(image.data(), 4, 4, 3, 2, 2, 1, 1, 0, 0, matrix.data());

        const size_t mSize = 3 * 2 * 2 * 3 * 3; // 3 channels, 2x2 filter, 3x3 output result after convolution
        ASSERT_EQ(matrix.size(), mSize);

        raul::Tensor matrixGold = {
            1.0f,  2.0f,  3.0f,  5.0f,  6.0f,  7.0f,  9.0f,  10.0f, 11.0f, 2.0f,  3.0f,  4.0f,  6.0f,  7.0f,  8.0f,  10.0f, 11.0f, 12.0f,
            5.0f,  6.0f,  7.0f,  9.0f,  10.0f, 11.0f, 13.0f, 14.0f, 15.0f, 6.0f,  7.0f,  8.0f,  10.0f, 11.0f, 12.0f, 14.0f, 15.0f, 16.0f,

            17.0f, 18.0f, 19.0f, 21.0f, 22.0f, 23.0f, 25.0f, 26.0f, 27.0f, 18.0f, 19.0f, 20.0f, 22.0f, 23.0f, 24.0f, 26.0f, 27.0f, 28.0f,
            21.0f, 22.0f, 23.0f, 25.0f, 26.0f, 27.0f, 29.0f, 30.0f, 31.0f, 22.0f, 23.0f, 24.0f, 26.0f, 27.0f, 28.0f, 30.0f, 31.0f, 32.0f,

            33.0f, 34.0f, 35.0f, 37.0f, 38.0f, 39.0f, 41.0f, 42.0f, 43.0f, 34.0f, 35.0f, 36.0f, 38.0f, 39.0f, 40.0f, 42.0f, 43.0f, 44.0f,
            37.0f, 38.0f, 39.0f, 41.0f, 42.0f, 43.0f, 45.0f, 46.0f, 47.0f, 38.0f, 39.0f, 40.0f, 42.0f, 43.0f, 44.0f, 46.0f, 47.0f, 48.0f,
        };

        for (size_t q = 0; q < mSize; ++q)
        {
            ASSERT_EQ(matrix[q], matrixGold[q]);
        }
    }
}

TEST(TestCommon, Im2ColGEMMUnit)
{
    PROFILE_TEST

    // basic
    {
        raul::Tensor image = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };

        raul::Tensor matrix(4 * 4);
        raul::Common::im2col(image.data(), 3, 3, 1, 2, 2, 1, 1, 0, 0, matrix.data());

        raul::Tensor filter = {
            1.0f,
            3.0f,
            2.0f,
            4.0f,
        };

        raul::Tensor output(2 * 2);

        raul::Common::gemm(CblasNoTrans, CblasNoTrans, 1, 4, 4, 1.0_dt, filter.data(), matrix.data(), 0.0_dt, output.data());

        raul::Tensor outputGold = {
            37.0f,
            67.0f,
            47.0f,
            77.0f,
        };

        for (size_t q = 0; q < 2 * 2; ++q)
        {
            ASSERT_EQ(output[q], outputGold[q]);
        }
    }

    // padding
    {
        raul::Tensor image = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };

        raul::Tensor matrix(16 * 4);
        raul::Common::im2col(image.data(), 3, 3, 1, 2, 2, 1, 1, 1, 1, matrix.data());

        raul::Tensor filter = {
            1.0f,
            3.0f,
            2.0f,
            4.0f,
        };

        raul::Tensor output(4 * 4);

        raul::Common::gemm(CblasNoTrans, CblasNoTrans, 1, 16, 4, 1.0_dt, filter.data(), matrix.data(), 0.0_dt, output.data());

        raul::Tensor outputGold = { 4.0f, 18.0f, 36.0f, 14.0f, 11.0f, 37.0f, 67.0f, 23.0f, 18.0f, 47.0f, 77.0f, 26.0f, 9.0f, 21.0f, 33.0f, 9.0f };

        for (size_t q = 0; q < 4 * 4; ++q)
        {
            ASSERT_EQ(output[q], outputGold[q]);
        }
    }

    // stride & padding
    {
        raul::Tensor image = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };

        raul::Tensor matrix(4 * 4);
        raul::Common::im2col(image.data(), 3, 3, 1, 2, 2, 2, 2, 1, 1, matrix.data());

        raul::Tensor filter = {
            1.0f,
            3.0f,
            2.0f,
            4.0f,
        };

        raul::Tensor output(2 * 2);

        raul::Common::gemm(CblasNoTrans, CblasNoTrans, 1, 4, 4, 1.0_dt, filter.data(), matrix.data(), 0.0_dt, output.data());

        raul::Tensor outputGold = { 4.0f, 36.0f, 18.0f, 77.0f };

        for (size_t q = 0; q < 2 * 2; ++q)
        {
            ASSERT_EQ(output[q], outputGold[q]);
        }
    }

    // not symmetrical stride & symmetrical padding
    {
        raul::Tensor image = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };

        raul::Tensor matrix(8 * 4);
        raul::Common::im2col(image.data(), 3, 3, 1, 2, 2, 2, 1, 1, 1, matrix.data());

        raul::Tensor filter = {
            1.0f,
            3.0f,
            2.0f,
            4.0f,
        };

        raul::Tensor output(2 * 4);

        raul::Common::gemm(CblasNoTrans, CblasNoTrans, 1, 8, 4, 1.0_dt, filter.data(), matrix.data(), 0.0_dt, output.data());

        raul::Tensor outputGold = { 4.0f, 36.0f, 11.0f, 67.0f, 18.0f, 77.0f, 9.0f, 33.0f };

        for (size_t q = 0; q < 2 * 4; ++q)
        {
            ASSERT_EQ(output[q], outputGold[q]);
        }
    }

    // not symmetrical stride & symmetrical padding
    {
        raul::Tensor image = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };

        raul::Tensor matrix(8 * 4);
        raul::Common::im2col(image.data(), 3, 3, 1, 2, 2, 1, 2, 1, 1, matrix.data());

        raul::Tensor filter = {
            1.0f,
            3.0f,
            2.0f,
            4.0f,
        };

        raul::Tensor output(4 * 2);

        raul::Common::gemm(CblasNoTrans, CblasNoTrans, 1, 8, 4, 1.0_dt, filter.data(), matrix.data(), 0.0_dt, output.data());

        raul::Tensor outputGold = { 4.0f, 18.0f, 36.0f, 14.0f, 18.0f, 47.0f, 77.0f, 26.0f };

        for (size_t q = 0; q < 4 * 2; ++q)
        {
            ASSERT_EQ(output[q], outputGold[q]);
        }
    }

    // larger
    {
        raul::Tensor image = { 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,

                               17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,

                               33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f };

        raul::Tensor matrix(9 * 12);
        raul::Common::im2col(image.data(), 4, 4, 3, 2, 2, 1, 1, 0, 0, matrix.data());

        raul::Tensor filter = { 1.0f, 3.0f, 2.0f, 4.0f,

                                1.0f, 3.0f, 2.0f, 4.0f,

                                1.0f, 3.0f, 2.0f, 4.0f };

        raul::Tensor output(3 * 3);

        raul::Common::gemm(CblasNoTrans, CblasNoTrans, 1, 9, 12, 1.0_dt, filter.data(), matrix.data(), 0.0_dt, output.data());

        raul::Tensor outputGold = { 603.0f, 633.0f, 663.0f, 723.0f, 753.0f, 783.0f, 843.0f, 873.0f, 903.0f };

        for (size_t q = 0; q < 3 * 3; ++q)
        {
            ASSERT_EQ(output[q], outputGold[q]);
        }
    }
}

TEST(TestCommon, Col2ImUnit)
{
    PROFILE_TEST

    // basic
    {
        raul::Tensor matrix = { 1.0f, 4.0f, 2.0f, 5.0f, 4.0f, 7.0f, 5.0f, 8.0f, 2.0f, 5.0f, 3.0f, 6.0f, 5.0f, 8.0f, 6.0f, 9.0f };

        raul::Tensor image(3 * 3);
        raul::Common::col2im(matrix.data(), 3, 3, 1, 2, 2, 1, 1, 0, 0, image.data());

        ASSERT_EQ(image.size(), static_cast<size_t>(3 * 3));

        raul::Tensor imageGold = { 1.0f, 8.0f, 7.0f, 4.0f, 20.0f, 16.0f, 3.0f, 12.0f, 9.0f };

        for (size_t q = 0; q < 3 * 3; ++q)
        {
            ASSERT_EQ(image[q], imageGold[q]);
        }
    }

    // not symmetric stride
    {
        raul::Tensor matrix = { 1.0f, 2.0f, 4.0f, 5.0f, 2.0f, 3.0f, 5.0f, 6.0f };

        raul::Tensor image(3 * 3);
        raul::Common::col2im(matrix.data(), 3, 3, 1, 2, 2, 2, 1, 0, 0, image.data());

        ASSERT_EQ(image.size(), static_cast<size_t>(3 * 3));

        raul::Tensor imageGold = { 1.0f, 4.0f, 0.0f, 4.0f, 10.0f, 0.0f, 3.0f, 6.0f, 0.0f };

        for (size_t q = 0; q < 3 * 3; ++q)
        {
            ASSERT_EQ(image[q], imageGold[q]);
        }
    }

    // not symmetric stride
    {
        raul::Tensor matrix = {
            1.0f, 4.0f, 4.0f, 7.0f, 2.0f, 5.0f, 5.0f, 8.0f,
        };

        raul::Tensor image(3 * 3);
        raul::Common::col2im(matrix.data(), 3, 3, 1, 2, 2, 1, 2, 0, 0, image.data());

        ASSERT_EQ(image.size(), static_cast<size_t>(3 * 3));

        raul::Tensor imageGold = { 1.0f, 8.0f, 7.0f, 2.0f, 10.0f, 8.0f, 0.0f, 0.0f, 0.0f };

        for (size_t q = 0; q < 3 * 3; ++q)
        {
            ASSERT_EQ(image[q], imageGold[q]);
        }
    }

    // not square kernel
    {
        raul::Tensor matrix = { 1.0f, 2.0f, 4.0f, 5.0f, 7.0f, 8.0f, 2.0f, 3.0f, 5.0f, 6.0f, 8.0f, 9.0f };

        raul::Tensor image(3 * 3);
        raul::Common::col2im(matrix.data(), 3, 3, 1, 3, 2, 1, 1, 0, 0, image.data());

        ASSERT_EQ(image.size(), static_cast<size_t>(3 * 3));

        raul::Tensor imageGold = { 1.0f, 4.0f, 7.0f, 4.0f, 10.0f, 16.0f, 3.0f, 6.0f, 9.0f };

        for (size_t q = 0; q < 3 * 3; ++q)
        {
            ASSERT_EQ(image[q], imageGold[q]);
        }
    }

    // not square kernel
    {
        raul::Tensor matrix = { 1.0f, 4.0f, 4.0f, 7.0f, 2.0f, 5.0f, 5.0f, 8.0f, 3.0f, 6.0f, 6.0f, 9.0f };

        raul::Tensor image(3 * 3);
        raul::Common::col2im(matrix.data(), 3, 3, 1, 2, 3, 1, 1, 0, 0, image.data());

        ASSERT_EQ(image.size(), static_cast<size_t>(3 * 3));

        raul::Tensor imageGold = { 1.0f, 8.0f, 7.0f, 2.0f, 10.0f, 8.0f, 3.0f, 12.0f, 9.0f };

        for (size_t q = 0; q < 3 * 3; ++q)
        {
            ASSERT_EQ(image[q], imageGold[q]);
        }
    }

    // not symmetric stride & not square kernel
    {
        raul::Tensor matrix = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f };

        raul::Tensor image(3 * 3);
        raul::Common::col2im(matrix.data(), 3, 3, 1, 3, 2, 1, 2, 0, 0, image.data());

        ASSERT_EQ(image.size(), static_cast<size_t>(3 * 3));

        raul::Tensor imageGold = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 0.0f, 0.0f, 0.0f };

        for (size_t q = 0; q < 3 * 3; ++q)
        {
            ASSERT_EQ(image[q], imageGold[q]);
        }
    }

    // not symmetric stride & not square kernel
    {
        raul::Tensor matrix = { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f };

        raul::Tensor image(3 * 3);
        raul::Common::col2im(matrix.data(), 3, 3, 1, 2, 3, 2, 1, 0, 0, image.data());

        ASSERT_EQ(image.size(), static_cast<size_t>(3 * 3));

        raul::Tensor imageGold = { 1.0f, 4.0f, 0.0f, 2.0f, 5.0f, 0.0f, 3.0f, 6.0f, 0.0f };

        for (size_t q = 0; q < 3 * 3; ++q)
        {
            ASSERT_EQ(image[q], imageGold[q]);
        }
    }

    // symmetric padding
    {
        raul::Tensor matrix = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 4.0f,
                                7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 0.0f, 0.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        raul::Tensor image(3 * 3);
        raul::Common::col2im(matrix.data(), 3, 3, 1, 2, 2, 1, 1, 1, 1, image.data());

        ASSERT_EQ(image.size(), static_cast<size_t>(3 * 3));

        raul::Tensor imageGold = { 4.0f, 16.0f, 28.0f, 8.0f, 20.0f, 32.0f, 12.0f, 24.0f, 36.0f };

        for (size_t q = 0; q < 3 * 3; ++q)
        {
            ASSERT_EQ(image[q], imageGold[q]);
        }
    }

    // not symmetric padding
    {
        raul::Tensor matrix = { 0.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 1.0f, 4.0f, 7.0f, 0.0f, 2.0f, 5.0f, 8.0f, 0.0f,
                                0.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 2.0f, 5.0f, 8.0f, 0.0f, 3.0f, 6.0f, 9.0f, 0.0f };

        raul::Tensor image(3 * 3);
        raul::Common::col2im(matrix.data(), 3, 3, 1, 2, 2, 1, 1, 1, 0, image.data());

        ASSERT_EQ(image.size(), static_cast<size_t>(3 * 3));

        raul::Tensor imageGold = { 2.0f, 8.0f, 14.0f, 8.0f, 20.0f, 32.0f, 6.0f, 12.0f, 18.0f };

        for (size_t q = 0; q < 3 * 3; ++q)
        {
            ASSERT_EQ(image[q], imageGold[q]);
        }
    }

    // not symmetric padding
    {
        raul::Tensor matrix = { 0.0f, 0.0f, 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, 0.0f, 0.0f, 4.0f, 7.0f, 5.0f, 8.0f, 6.0f, 9.0f,
                                1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, 0.0f, 0.0f, 4.0f, 7.0f, 5.0f, 8.0f, 6.0f, 9.0f, 0.0f, 0.0f };

        raul::Tensor image(3 * 3);
        raul::Common::col2im(matrix.data(), 3, 3, 1, 2, 2, 1, 1, 0, 1, image.data());

        ASSERT_EQ(image.size(), static_cast<size_t>(3 * 3));

        raul::Tensor imageGold = { 2.0f, 16.0f, 14.0f, 4.0f, 20.0f, 16.0f, 6.0f, 24.0f, 18.0f };

        for (size_t q = 0; q < 3 * 3; ++q)
        {
            ASSERT_EQ(image[q], imageGold[q]);
        }
    }

    // larger
    {

        raul::Tensor matrix = {
            1.0f,  2.0f,  3.0f,  5.0f,  6.0f,  7.0f,  9.0f,  10.0f, 11.0f, 2.0f,  3.0f,  4.0f,  6.0f,  7.0f,  8.0f,  10.0f, 11.0f, 12.0f,
            5.0f,  6.0f,  7.0f,  9.0f,  10.0f, 11.0f, 13.0f, 14.0f, 15.0f, 6.0f,  7.0f,  8.0f,  10.0f, 11.0f, 12.0f, 14.0f, 15.0f, 16.0f,

            17.0f, 18.0f, 19.0f, 21.0f, 22.0f, 23.0f, 25.0f, 26.0f, 27.0f, 18.0f, 19.0f, 20.0f, 22.0f, 23.0f, 24.0f, 26.0f, 27.0f, 28.0f,
            21.0f, 22.0f, 23.0f, 25.0f, 26.0f, 27.0f, 29.0f, 30.0f, 31.0f, 22.0f, 23.0f, 24.0f, 26.0f, 27.0f, 28.0f, 30.0f, 31.0f, 32.0f,

            33.0f, 34.0f, 35.0f, 37.0f, 38.0f, 39.0f, 41.0f, 42.0f, 43.0f, 34.0f, 35.0f, 36.0f, 38.0f, 39.0f, 40.0f, 42.0f, 43.0f, 44.0f,
            37.0f, 38.0f, 39.0f, 41.0f, 42.0f, 43.0f, 45.0f, 46.0f, 47.0f, 38.0f, 39.0f, 40.0f, 42.0f, 43.0f, 44.0f, 46.0f, 47.0f, 48.0f,
        };

        raul::Tensor image(3 * 4 * 4);
        raul::Common::col2im(matrix.data(), 4, 4, 3, 2, 2, 1, 1, 0, 0, image.data());

        ASSERT_EQ(image.size(), static_cast<size_t>(3 * 4 * 4));

        raul::Tensor imageGold = { 1.0f,  4.0f,  6.0f,  4.0f,  10.0f, 24.0f,  28.0f,  16.0f, 18.0f, 40.0f,  44.0f,  24.0f, 13.0f, 28.0f, 30.0f, 16.0f,

                                   17.0f, 36.0f, 38.0f, 20.0f, 42.0f, 88.0f,  92.0f,  48.0f, 50.0f, 104.0f, 108.0f, 56.0f, 29.0f, 60.0f, 62.0f, 32.0f,

                                   33.0f, 68.0f, 70.0f, 36.0f, 74.0f, 152.0f, 156.0f, 80.0f, 82.0f, 168.0f, 172.0f, 88.0f, 45.0f, 92.0f, 94.0f, 48.0f };

        for (size_t q = 0; q < 3 * 4 * 4; ++q)
        {
            ASSERT_EQ(image[q], imageGold[q]);
        }
    }
}

TEST(TestCommon, CheckHadamardTrivialUnit)
{
    PROFILE_TEST

    const raul::Tensor vecA = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    const raul::Tensor vecB = { 10.0f, 20.0f, 40.0f, 50.0f, 70.0f };
    raul::Tensor vecC(vecA.size());

    const raul::Tensor realVecC = { 10.0f, 40.0f, 120.0f, 200.0f, 350.0f };

    // Trivial
    raul::Common::hadamard(5, 1.0f, vecA.data(), vecB.data(), 1, 0.0f, vecC.data(), 1);
    for (size_t q = 0; q < vecC.size(); ++q)
    {
        EXPECT_EQ(vecC[q], realVecC[q]);
    }
}

TEST(TestCommon, CheckHadamardUnit)
{
    PROFILE_TEST

    const raul::Tensor vecA = { 10.0f, 20.0f, 40.0f, 50.0f, 70.0f };
    const raul::Tensor vecB = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f };
    raul::Tensor vecC(vecA.size() * 2);

    const raul::Tensor realVecC = { 20.0f, 0.0f, 160.0f, 0.0f, 560.0f, 0.0f, 1000.0f, 0.0f, 1820.0f, 0.0f };

    // Non-trivial
    raul::Common::hadamard(5, 2.0f, vecA.data(), vecB.data(), 3, 0.0f, vecC.data(), 2);
    for (size_t q = 0; q < vecC.size(); ++q)
    {
        EXPECT_EQ(vecC[q], realVecC[q]);
    }
}

} // UT namespace
