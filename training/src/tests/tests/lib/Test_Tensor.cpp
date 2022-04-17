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
#include <utility>

#include <training/base/common/Common.h>
#include <training/base/common/Conversions.h>
#include <training/base/common/MemoryManager.h>
#include <training/system/TypeHalf.h>

namespace UT
{

TEST(TestTensor, TensorIOUnit)
{
    raul::Tensor tensor("test_name", 1, 1, 1, 5);
    tensor = { 1, 2, 3, 4, 5 };

    raul::Tensor noname_tensor(1, 1, 1, 5);
    noname_tensor = { 1, 2, 3, 4, 5 };

    raul::Tensor noname_big_tensor(1, 1, 1, 20);
    noname_big_tensor = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };

    // Default aka brief
    {
        testing::internal::CaptureStdout();
        std::cout << tensor;
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_STREQ(output.c_str(), "Tensor 'test_name' (1,1,1,5)");
    }

    {
        testing::internal::CaptureStdout();
        std::cout << noname_tensor;
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_STREQ(output.c_str(), "Tensor (1,1,1,5)");
    }

    // Explicit brief
    {
        testing::internal::CaptureStdout();
        std::cout << raul::io::tensor::brief << tensor;
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_STREQ(output.c_str(), "Tensor 'test_name' (1,1,1,5)");
    }

    // Full
    {
        testing::internal::CaptureStdout();
        std::cout << raul::io::tensor::full << tensor;
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_STREQ(output.c_str(), "Tensor 'test_name' (1,1,1,5), size: 20\n[1,2,3,4,5]");
    }

    {
        testing::internal::CaptureStdout();
        std::cout << raul::io::tensor::full << noname_big_tensor;
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_STREQ(output.c_str(), "Tensor (1,1,1,20), size: 80\n[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]");
    }

    // Compact
    {
        testing::internal::CaptureStdout();
        std::cout << raul::io::tensor::compact << tensor;
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_STREQ(output.c_str(), "Tensor 'test_name' (1,1,1,5), size: 20\n[1,2,3,4,5]");
    }

    {
        testing::internal::CaptureStdout();
        std::cout << raul::io::tensor::compact << noname_big_tensor;
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_STREQ(output.c_str(), "Tensor (1,1,1,20), size: 80\n[1,2,3,4,5,...,16,17,18,19,20]");
    }

    // Manual flags
    {
        testing::internal::CaptureStdout();
        std::cout << raul::io::tensor::setview(raul::io::tensor::TensorView::content | raul::io::tensor::TensorView::reduced) << tensor;
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_STREQ(output.c_str(), "Tensor 'test_name' (1,1,1,5)\n[1,2,3,4,5]");
    }
}

TEST(TestTensor, TensorFP16IOUnit)
{
    raul::TensorFP16 noname_big_tensor = { 1_hf, 2_hf, 3_hf, 4_hf, 5_hf, 6_hf, 7_hf, 8_hf, 9_hf, 10_hf, 11_hf, 12_hf, 13_hf, 14_hf, 15_hf, 16_hf, 17_hf, 18_hf, 19_hf, 20_hf };

    {
        testing::internal::CaptureStdout();
        std::cout << raul::io::tensor::compact << noname_big_tensor;
        std::string output = testing::internal::GetCapturedStdout();
        EXPECT_STREQ(output.c_str(), "Tensor (1,1,1,20), size: 40\n[1,2,3,4,5,...,16,17,18,19,20]");
    }

}

TEST(TestTensor, ConstructorUnit)
{
    PROFILE_TEST
    // Tensor(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, bool isAllocate = true)
    raul::Tensor t("test", 1, 2, 3, 4);

    EXPECT_EQ(t.size(), 24u);

    EXPECT_EQ(t.getBatchSize(), 1u);
    EXPECT_EQ(t.getDepth(), 2u);
    EXPECT_EQ(t.getHeight(), 3u);
    EXPECT_EQ(t.getWidth(), 4u);
}

TEST(TestTensor, Constructor2Unit)
{
    PROFILE_TEST
    // Tensor(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, dt filler)
    raul::Tensor t("test", 1, 2, 3, 4, 2.0_dt);

    EXPECT_EQ(t.size(), 24u);

    EXPECT_EQ(t.getBatchSize(), 1u);
    EXPECT_EQ(t.getDepth(), 2u);
    EXPECT_EQ(t.getHeight(), 3u);
    EXPECT_EQ(t.getWidth(), 4u);

    size_t count = 0;
    for (auto d : t)
    {
        ++count;
        EXPECT_EQ(d, 2.0_dt);
    }

    EXPECT_EQ(count, 24u);
}

TEST(TestTensor, Constructor3Unit)
{
    PROFILE_TEST
    raul::dtype data[] = { 1.0_dt, 2.0_dt, 3.0_dt };

    // Tensor(dt_range beginEnd)
    raul::Tensor t(raul::Tensor::dt_range(data, data + 3));
    // Tensor(size_t size, dt filler)
    raul::Tensor t2(10, 1.0_dt);
    // TensorImpl(size_t size)
    raul::Tensor t3(10);

    EXPECT_EQ(t.size(), 3u);
    EXPECT_EQ(t2.size(), 10u);
    EXPECT_EQ(t3.size(), 10u);

    EXPECT_EQ(t.getBatchSize(), 1u);
    EXPECT_EQ(t.getDepth(), 1u);
    EXPECT_EQ(t.getHeight(), 1u);
    EXPECT_EQ(t.getWidth(), 3u);

    EXPECT_EQ(t2.getBatchSize(), 1u);
    EXPECT_EQ(t2.getDepth(), 1u);
    EXPECT_EQ(t2.getHeight(), 1u);
    EXPECT_EQ(t2.getWidth(), 10u);

    EXPECT_EQ(t3.getBatchSize(), 1u);
    EXPECT_EQ(t3.getDepth(), 1u);
    EXPECT_EQ(t3.getHeight(), 1u);
    EXPECT_EQ(t3.getWidth(), 10u);

    EXPECT_EQ(t[0], 1.0_dt);
    EXPECT_EQ(t[1], 2.0_dt);
    EXPECT_EQ(t[2], 3.0_dt);

    size_t count = 0;
    for (auto d : t2)
    {
        ++count;
        EXPECT_EQ(d, 1.0_dt);
    }

    EXPECT_EQ(count, 10u);
}

TEST(TestTensor, Constructor4Unit)
{
    PROFILE_TEST
    raul::dtype data[] = { 1.0_dt, 2.0_dt, 3.0_dt };

    // Tensor(shape inShape, dt_range beginEnd)
    raul::Tensor t(yato::dims(1, 3, 1, 1), raul::Tensor::dt_range(data, data + 3));

    EXPECT_EQ(t.size(), 3u);

    EXPECT_EQ(t.getBatchSize(), 1u);
    EXPECT_EQ(t.getDepth(), 3u);
    EXPECT_EQ(t.getHeight(), 1u);
    EXPECT_EQ(t.getWidth(), 1u);

    EXPECT_EQ(t[0], 1.0_dt);
    EXPECT_EQ(t[1], 2.0_dt);
    EXPECT_EQ(t[2], 3.0_dt);
}

TEST(TestTensor, InitializerListUnit)
{
    PROFILE_TEST
    // Tensor(const Name& name, std::initializer_list<dt> list)
    raul::Tensor t("test", { 1.0_dt, 2.0_dt });
    // Tensor(std::initializer_list<dt> list)
    raul::Tensor t3({ 1.0_dt, 2.0_dt });
    raul::Tensor t2{ 1.0_dt, 2.0_dt };

    EXPECT_EQ(t.size(), 2u);
    EXPECT_EQ(t2.size(), 2u);
    EXPECT_EQ(t3.size(), 2u);

    EXPECT_EQ(t.getBatchSize(), 1u);
    EXPECT_EQ(t.getDepth(), 1u);
    EXPECT_EQ(t.getHeight(), 1u);
    EXPECT_EQ(t.getWidth(), 2u);

    EXPECT_EQ(t2.getBatchSize(), 1u);
    EXPECT_EQ(t2.getDepth(), 1u);
    EXPECT_EQ(t2.getHeight(), 1u);
    EXPECT_EQ(t2.getWidth(), 2u);

    EXPECT_EQ(t3.getBatchSize(), 1u);
    EXPECT_EQ(t3.getDepth(), 1u);
    EXPECT_EQ(t3.getHeight(), 1u);
    EXPECT_EQ(t3.getWidth(), 2u);

    EXPECT_EQ(t[0], 1.0_dt);
    EXPECT_EQ(t[1], 2.0_dt);

    EXPECT_EQ(t2[0], 1.0_dt);
    EXPECT_EQ(t2[1], 2.0_dt);

    EXPECT_EQ(t3[0], 1.0_dt);
    EXPECT_EQ(t3[1], 2.0_dt);
}

TEST(TestTensor, InitializerList2Unit)
{
    PROFILE_TEST
    // Tensor(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, std::initializer_list<dtype> list)
    raul::Tensor t("test", 2, 1, 1, 1, { 1.0_dt, 2.0_dt });
    // Tensor(size_t batchSize, size_t depth, size_t height, size_t width, std::initializer_list<dtype> list)
    EXPECT_THROW(raul::Tensor t3(2, 1, 1, 2, { 1.0_dt, 2.0_dt }), raul::Exception);
    raul::Tensor t2(2, 1, 1, 1, { 1.0_dt, 2.0_dt });

    EXPECT_EQ(t.getShape(), yato::dims(2, 1, 1, 1));
    EXPECT_EQ(t2.getShape(), yato::dims(2, 1, 1, 1));

    EXPECT_EQ(t[0], 1.0_dt);
    EXPECT_EQ(t[1], 2.0_dt);

    EXPECT_EQ(t2[0], 1.0_dt);
    EXPECT_EQ(t2[1], 2.0_dt);
}

TEST(TestTensor, InitializerList3Unit)
{
    PROFILE_TEST
    // Tensor(const Name& name, shape inShape, std::initializer_list<dtype> list)
    raul::Tensor t("test", yato::dims(2, 1, 1, 1), { 1.0_dt, 2.0_dt });
    // Tensor(shape inShape, std::initializer_list<dtype> list)
    EXPECT_THROW(raul::Tensor t3(t.getShape(), { 1.0_dt, 2.0_dt, 1.0_dt }), raul::Exception);
    raul::Tensor t2(t.getShape(), { 1.0_dt, 2.0_dt });

    EXPECT_EQ(t.getShape(), yato::dims(2, 1, 1, 1));
    EXPECT_EQ(t2.getShape(), yato::dims(2, 1, 1, 1));

    EXPECT_EQ(t[0], 1.0_dt);
    EXPECT_EQ(t[1], 2.0_dt);

    EXPECT_EQ(t2[0], 1.0_dt);
    EXPECT_EQ(t2[1], 2.0_dt);
}

TEST(TestTensor, CompressDecompressUnit)
{
    PROFILE_TEST
    const size_t size = 100;
    constexpr raul::dtype minVal = -1000.0_dt;
    constexpr raul::dtype maxVal = 1000.0_dt;
    constexpr raul::dtype unit = (maxVal - minVal) / TODTYPE(size - 1);

    constexpr raul::dtype epsFP16 = 0.5_dt;
    constexpr raul::dtype epsINT8 = 10.0_dt;

    raul::Tensor t(size);

    for (size_t q = 0; q < size; ++q)
    {
        t[q] = minVal + static_cast<raul::dtype>(q) * unit;
    }

    raul::Tensor t2(TORANGE(t));
    raul::Tensor t3(TORANGE(t));

    ASSERT_NO_THROW(t2.compress(raul::CompressionMode::FP16));
    ASSERT_NO_THROW(t2.decompress(raul::CompressionMode::FP16));

    ASSERT_NO_THROW(t3.compress(raul::CompressionMode::INT8));
    ASSERT_NO_THROW(t3.decompress(raul::CompressionMode::INT8));

    for (size_t q = 0; q < size; ++q)
    {
        EXPECT_NEAR(t[q], t2[q], epsFP16);
        EXPECT_NEAR(t[q], t3[q], epsINT8);
    }

    raul::Tensor t4(0);
    ASSERT_THROW(t4.compress(raul::CompressionMode::FP16), raul::Exception);
    ASSERT_NO_THROW(t4.decompress(raul::CompressionMode::FP16));

    ASSERT_THROW(t4.compress(raul::CompressionMode::INT8), raul::Exception);
    ASSERT_NO_THROW(t4.decompress(raul::CompressionMode::INT8));

    raul::Tensor t5(10, 0.0_dt);
    ASSERT_NO_THROW(t5.compress(raul::CompressionMode::FP16));
    ASSERT_NO_THROW(t5.decompress(raul::CompressionMode::FP16));
    for (size_t q = 0; q < 10; ++q)
    {
        EXPECT_EQ(t5[q], 0.0_dt);
    }

    ASSERT_NO_THROW(t5.compress(raul::CompressionMode::INT8));
    ASSERT_NO_THROW(t5.decompress(raul::CompressionMode::INT8));
    for (size_t q = 0; q < 10; ++q)
    {
        EXPECT_EQ(t5[q], 0.0_dt);
    }

    raul::Tensor t6(10, 1.55_dt);
    ASSERT_NO_THROW(t6.compress(raul::CompressionMode::FP16));
    ASSERT_NO_THROW(t6.decompress(raul::CompressionMode::FP16));
    for (size_t q = 0; q < 10; ++q)
    {
        EXPECT_NEAR(t6[q], 1.55_dt, epsFP16);
    }

    raul::Tensor t7(10, 1.55_dt);
    ASSERT_NO_THROW(t7.compress(raul::CompressionMode::INT8));
    ASSERT_NO_THROW(t7.decompress(raul::CompressionMode::INT8));
    for (size_t q = 0; q < 10; ++q)
    {
        EXPECT_EQ(t7[q], 1.55_dt); // no accuracy loss
    }
}

TEST(TestTensor, BroadcastCaseFrom12To22Unit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto* tensor = memory_manager.createTensor("x", 1, 2, 1, 1);
    std::generate(tensor->begin(), tensor->end(), [n = 1.0_dt]() mutable { return n++; });

    // Broadcasting
    const auto tensor_viewer = tensor->getBroadcastedViewer(yato::dims(2, 2, 1, 1));

    const auto mapping = { std::make_pair(0, 0), std::make_pair(1, 1), std::make_pair(2, 0), std::make_pair(3, 1) };

    // Check
    for (auto test_case : mapping)
    {
        EXPECT_EQ(tensor_viewer[test_case.first], (*tensor)[test_case.second]);
    }
}

TEST(TestTensor, BroadcastCaseFrom21To22Unit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto* tensor = memory_manager.createTensor("x", 2, 1, 1, 1);
    std::generate(tensor->begin(), tensor->end(), [n = 1.0_dt]() mutable { return n++; });

    // Broadcasting
    auto tensor_viewer = tensor->getBroadcastedViewer(yato::dims(2, 2, 1, 1));

    const auto mapping = { std::make_pair(0, 0), std::make_pair(1, 0), std::make_pair(2, 1), std::make_pair(3, 1) };

    // Check
    for (auto test_case : mapping)
    {
        EXPECT_EQ(tensor_viewer[test_case.first], (*tensor)[test_case.second]);
    }
}

TEST(TestTensor, BroadcastCaseFrom22To22Unit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto* tensor = memory_manager.createTensor("x", 2, 2, 1, 1);
    std::generate(tensor->begin(), tensor->end(), [n = 1.0_dt]() mutable { return n++; });

    // Broadcasting
    auto tensor_viewer = tensor->getBroadcastedViewer(yato::dims(2, 2, 1, 1));

    const auto mapping = { std::make_pair(0, 0), std::make_pair(1, 1), std::make_pair(2, 2), std::make_pair(3, 3) };

    // Check
    for (auto test_case : mapping)
    {
        EXPECT_EQ(tensor_viewer[test_case.first], (*tensor)[test_case.second]);
    }
}

TEST(TestTensor, BroadcastCaseFrom212To222Unit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto* tensor = memory_manager.createTensor("x", 2, 1, 2, 1);
    std::generate(tensor->begin(), tensor->end(), [n = 1.0_dt]() mutable { return n++; });

    // Broadcasting
    auto tensor_viewer = tensor->getBroadcastedViewer(yato::dims(2, 2, 2, 1));

    const auto mapping = { std::make_pair(0, 0), std::make_pair(1, 1), std::make_pair(2, 0), std::make_pair(3, 1),
                           std::make_pair(4, 2), std::make_pair(5, 3), std::make_pair(6, 2), std::make_pair(7, 3) };

    // Check
    for (auto test_case : mapping)
    {
        EXPECT_EQ(tensor_viewer[test_case.first], (*tensor)[test_case.second]);
    }
}

TEST(TestTensor, BroadcastCaseFrom122To222Unit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto* tensor = memory_manager.createTensor("x", 1, 2, 2, 1);
    std::generate(tensor->begin(), tensor->end(), [n = 1.0_dt]() mutable { return n++; });

    // Broadcasting
    auto tensor_viewer = tensor->getBroadcastedViewer(yato::dims(2, 2, 2, 1));

    const auto mapping = { std::make_pair(0, 0), std::make_pair(1, 1), std::make_pair(2, 2), std::make_pair(3, 3),
                           std::make_pair(4, 0), std::make_pair(5, 1), std::make_pair(6, 2), std::make_pair(7, 3) };

    // Check
    for (auto test_case : mapping)
    {
        EXPECT_EQ(tensor_viewer[test_case.first], (*tensor)[test_case.second]);
    }
}

TEST(TestTensor, BroadcastCaseFrom111To122Unit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto* tensor = memory_manager.createTensor("x", 1, 1, 1, 1);
    std::generate(tensor->begin(), tensor->end(), [n = 1.0_dt]() mutable { return n++; });

    // Broadcasting
    auto tensor_viewer = tensor->getBroadcastedViewer(yato::dims(1, 2, 2, 1));

    const auto mapping = {
        std::make_pair(0, 0),
        std::make_pair(1, 0),
        std::make_pair(2, 0),
        std::make_pair(3, 0),
    };

    // Check
    for (auto test_case : mapping)
    {
        EXPECT_EQ(tensor_viewer[test_case.first], (*tensor)[test_case.second]);
    }
}

TEST(TestTensor, BroadcastWriteFrom11To22Unit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto* tensor = memory_manager.createTensor("x", 1, 1, 1, 1);
    std::fill(tensor->begin(), tensor->end(), 0.0_dt);

    // Broadcasting
    auto tensor_viewer = tensor->getBroadcastedViewer(yato::dims(2, 2, 1, 1));

    for (size_t i = 0; i < tensor_viewer.size(); ++i)
    {
        tensor_viewer[0] += 1.0_dt;
    }

    // Check
    EXPECT_EQ((*tensor)[0], static_cast<raul::dtype>(tensor_viewer.size()));
}

TEST(TestTensor, PlusEqualOperatorUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto* tensorA = memory_manager.createTensor("x", 1, 1, 1, 10);
    auto* tensorB = memory_manager.createTensor("y", 1, 1, 1, 10);
    std::fill(tensorA->begin(), tensorA->end(), 1.0_dt);
    std::fill(tensorB->begin(), tensorB->end(), 2.0_dt);

    EXPECT_EQ(tensorA->size(), tensorB->size());
    EXPECT_EQ(tensorA->size(), 10u);

    memory_manager["y"] += memory_manager["x"];

    for (size_t i = 0; i < tensorA->size(); ++i)
    {
        EXPECT_EQ((*tensorA)[i], 1.0_dt);
        EXPECT_EQ((*tensorB)[i], 3.0_dt);
    }

    memory_manager["y"] += 10.0_dt;

    for (size_t i = 0; i < tensorA->size(); ++i)
    {
        EXPECT_EQ((*tensorB)[i], 13.0_dt);
    }
}

TEST(TestTensor, MinusEqualOperatorUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto* tensorA = memory_manager.createTensor("x", 1, 1, 1, 10);
    auto* tensorB = memory_manager.createTensor("y", 1, 1, 1, 10);
    std::fill(tensorA->begin(), tensorA->end(), 1.0_dt);
    std::fill(tensorB->begin(), tensorB->end(), 2.0_dt);

    EXPECT_EQ(tensorA->size(), tensorB->size());
    EXPECT_EQ(tensorA->size(), 10u);

    memory_manager["y"] -= memory_manager["x"];

    for (size_t i = 0; i < tensorA->size(); ++i)
    {
        EXPECT_EQ((*tensorA)[i], 1.0_dt);
        EXPECT_EQ((*tensorB)[i], 1.0_dt);
    }

    memory_manager["y"] -= 10.0_dt;

    for (size_t i = 0; i < tensorA->size(); ++i)
    {
        EXPECT_EQ((*tensorB)[i], -9.0_dt);
    }
}

TEST(TestTensor, MulEqualOperatorUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto* tensorA = memory_manager.createTensor("x", 1, 1, 1, 10);
    auto* tensorB = memory_manager.createTensor("y", 1, 1, 1, 10);
    std::fill(tensorA->begin(), tensorA->end(), 1.0_dt);
    std::fill(tensorB->begin(), tensorB->end(), 2.0_dt);

    EXPECT_EQ(tensorA->size(), tensorB->size());
    EXPECT_EQ(tensorA->size(), 10u);

    memory_manager["y"] *= memory_manager["x"];

    for (size_t i = 0; i < tensorA->size(); ++i)
    {
        EXPECT_EQ((*tensorA)[i], 1.0_dt);
        EXPECT_EQ((*tensorB)[i], 2.0_dt);
    }

    memory_manager["y"] *= 10.0_dt;

    for (size_t i = 0; i < tensorA->size(); ++i)
    {
        EXPECT_EQ((*tensorB)[i], 20.0_dt);
    }
}

TEST(TestTensor, DivEqualOperatorUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto* tensorA = memory_manager.createTensor("x", 1, 1, 1, 10);
    auto* tensorB = memory_manager.createTensor("y", 1, 1, 1, 10);
    std::fill(tensorA->begin(), tensorA->end(), 1.0_dt);
    std::fill(tensorB->begin(), tensorB->end(), 2.0_dt);

    EXPECT_EQ(tensorA->size(), tensorB->size());
    EXPECT_EQ(tensorA->size(), 10u);

    memory_manager["x"] /= memory_manager["y"];

    for (size_t i = 0; i < tensorA->size(); ++i)
    {
        EXPECT_EQ((*tensorB)[i], 2.0_dt);
        EXPECT_EQ((*tensorA)[i], 0.5_dt);
    }

    memory_manager["x"] /= 10.0_dt;

    for (size_t i = 0; i < tensorA->size(); ++i)
    {
        EXPECT_EQ((*tensorA)[i], 0.05_dt);
    }
}

TEST(TestTensor, MaxIndexUnit)
{
    PROFILE_TEST
    raul::MemoryManager memory_manager;
    auto& t = *memory_manager.createTensor(1, 1, 1, 10, 0.0_dt);

    EXPECT_EQ(t.size(), 10u);

    t[3] = 10.0_dt;
    t[5] = 15.0_dt;
    t[7] = 5.0_dt;

    EXPECT_EQ(t.getMaxIndex(), 5u);
    EXPECT_EQ(t.getMaxIndex(6, 10), 1u);
}

TEST(TestTensor, TensorU8GeneralUnit)
{
    PROFILE_TEST
    raul::TensorU8 t(10, 255);

    EXPECT_EQ(t.size(), 10u);

    EXPECT_EQ(t.getBatchSize(), 1u);
    EXPECT_EQ(t.getDepth(), 1u);
    EXPECT_EQ(t.getHeight(), 1u);
    EXPECT_EQ(t.getWidth(), 10u);

    size_t count = 0;
    for (auto d : t)
    {
        ++count;
        EXPECT_EQ(d, 255);
    }

    EXPECT_EQ(count, 10u);
}

TEST(TestTensor, FindUnit)
{
    PROFILE_TEST
    raul::Tensor::iterator it;
    raul::Tensor t{ 1.0_dt, 2.0_dt, 3.0_dt, 4.0_dt, 5.0_dt };
    EXPECT_EQ(t.size(), 5U);

    for (size_t i = 0; i < t.size(); ++i)
    {
        it = std::find(t.begin(), t.end(), t[i]);
        const auto pos = std::distance(t.begin(), it);
        EXPECT_TRUE(it != t.end());
        EXPECT_EQ(static_cast<ptrdiff_t>(i), pos);
    }

    it = std::find(t.begin(), t.end(), -10.0_dt);
    EXPECT_TRUE(it == t.end());
}

TEST(TestTensor, PoolAllocationUnit)
{
    PROFILE_TEST

    raul::Tensor t("", 10u, 1u, 1u, 1u, raul::AllocationMode::POOL, false);

    std::vector<raul::dtype> pool(t.getShape().total_size(), 10_dt);

    EXPECT_EQ(t.size(), 0u);
    EXPECT_TRUE(t.empty());

    t.memAllocate(pool.data());

    EXPECT_EQ(t.size(), 10u);
    EXPECT_FALSE(t.empty());

    {
        size_t counter = 0;
        for (auto data : t)
        {
            ++counter;
            EXPECT_EQ(data, 0_dt);
        }

        EXPECT_EQ(counter, 10u);
    }

    std::fill(pool.begin(), pool.end(), 10_dt);

    {
        size_t counter = 0;
        for (auto data : t)
        {
            ++counter;
            EXPECT_EQ(data, 10_dt);
        }

        EXPECT_EQ(counter, 10u);
    }

    t = 12_dt;

    for (auto data : pool)
    {
        EXPECT_EQ(data, 12_dt);
    }

    {
        raul::Tensor tt(10u, 1u, 1u, 1u);
        tt = 22_dt;
        t = TORANGE(tt);
    }

    for (auto data : pool)
    {
        EXPECT_EQ(data, 22_dt);
    }

    // wrong size
    {
        raul::Tensor tt(11u, 1u, 1u, 1u);
        tt = 22_dt;
        EXPECT_THROW(t = TORANGE(tt), raul::Exception);
    }

    t.memClear();

    EXPECT_EQ(t.size(), 0u);

    EXPECT_EQ(pool.size(), 10u);

    pool.push_back(11_dt);
    EXPECT_EQ(pool.size(), 11u);

    t.memAllocate(pool.data());

    std::fill(pool.begin(), pool.end(), 22_dt);

    EXPECT_EQ(t.size(), 10u);

    {
        size_t counter = 0;
        for (auto data : t)
        {
            ++counter;
            EXPECT_EQ(data, 22_dt);
        }

        EXPECT_EQ(counter, 10u);
    }

    {
        raul::Tensor t2("", 1u, 1u, 1u, 1u, raul::AllocationMode::POOL, true);

        EXPECT_EQ(t2.size(), 0u); // not allocated with POOL
        EXPECT_TRUE(t2.empty());

        t2.memAllocate(&pool[10]);
        EXPECT_EQ(t2[0], 0_dt);
        pool[10] = 11_dt;

        EXPECT_EQ(t2.size(), 1u);
        EXPECT_EQ(t2[0], 11_dt);
    }
}

TEST(TestTensor, ConversionsFP16FP32)
{
    raul::Tensor t = { 1.0_dt, 2.0_dt };
    raul::TensorFP16 t16(t.size());
    raul::Tensor t2(t.size());

    t16 = TORANGE(t);
    t2 = TORANGE_FP16(t16);

    EXPECT_EQ(t16.size(), 2u);
    EXPECT_EQ(raul::toFloat32(t16[0]), 1_dt);
    EXPECT_EQ(raul::toFloat32(t16[1]), 2_dt);

    EXPECT_EQ(t2.size(), 2u);
    EXPECT_EQ(t2[0], 1_dt);
    EXPECT_EQ(t2[1], 2_dt);
}

TEST(TestTensor, PerformanceFP16FP32)
{
    // d.polubotko: make sure -D__ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1 -march=armv8.2-a+fp16+dotprod added into cxx flags

    const size_t size = 100000000;

    raul::TensorFP16 tA16(size);
    raul::Tensor tA(size);
    raul::TensorFP16 tB16(size);
    raul::Tensor tB(size);
    raul::TensorFP16 tC16(size);
    raul::Tensor tC(size);

    auto gen = raul::random::getGenerator();
    std::uniform_real_distribution distrib(1.0, 2.0);

    for (size_t q = 0; q < size; ++q)
    {
        tA[q] = static_cast<float>(distrib(gen));
        tB[q] = static_cast<float>(distrib(gen));
        tC[q] = static_cast<float>(distrib(gen));
        tA16[q] = raul::toFloat16(tA[q]);
        tB16[q] = raul::toFloat16(tB[q]);
        tC16[q] = raul::toFloat16(tC[q]);
    }

    std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();
    for (size_t q = 0; q < size; ++q)
    {
        tC[q] = tA[q] * tB[q];
    }
    printf("FP32: %.6f\n", static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count()) / 1000.0f);

    timeStart = std::chrono::steady_clock::now();
    for (size_t q = 0; q < size; ++q)
    {
        tC16[q] = tA16[q] * tB16[q];
    }
    printf("FP16: %.6f\n", static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count()) / 1000.0f);
}

TEST(TestTensor, ScaleUnscaleUnit)
{
    raul::Tensor original = { 1.0_dt, 2.0_dt };
    raul::Tensor scaled = { 2.0_dt, 4.0_dt };

    raul::Tensor tensor = original;

    tensor.scale(2.0_dt);
    EXPECT_EQ(tensor.size(), scaled.size());

    for (size_t i = 0; i < tensor.size(); ++i)
    {
        EXPECT_EQ(tensor[i], scaled[i]);
    }

    tensor.unscale();

    EXPECT_EQ(tensor.size(), tensor.size());

    for (size_t i = 0; i < tensor.size(); ++i)
    {
        EXPECT_EQ(tensor[i], original[i]);
    }
}

TEST(TestTensor, ScaleUnscaleTwiceUnit)
{
    raul::Tensor original = { 1.0_dt, 2.0_dt };
    raul::Tensor scaled_twice = { 6.0_dt, 12.0_dt };

    raul::Tensor tensor = original;

    tensor.scale(2.0_dt);
    tensor.scale(3.0_dt);

    EXPECT_EQ(tensor.size(), scaled_twice.size());

    for (size_t i = 0; i < tensor.size(); ++i)
    {
        EXPECT_EQ(tensor[i], scaled_twice[i]);
    }

    tensor.unscale();

    EXPECT_EQ(tensor.size(), tensor.size());

    for (size_t i = 0; i < tensor.size(); ++i)
    {
        EXPECT_EQ(tensor[i], original[i]);
    }

    tensor.unscale();

    EXPECT_EQ(tensor.size(), tensor.size());

    for (size_t i = 0; i < tensor.size(); ++i)
    {
        EXPECT_EQ(tensor[i], original[i]);
    }
}

TEST(TestTensor, FP16DivEqualOperatorUnit)
{
    PROFILE_TEST
    raul::MemoryManagerFP16 memory_manager;
    auto* tensorA = memory_manager.createTensor("x", 1, 1, 1, 10);
    auto* tensorB = memory_manager.createTensor("y", 1, 1, 1, 10);
    std::fill(tensorA->begin(), tensorA->end(), 1.0_hf);
    std::fill(tensorB->begin(), tensorB->end(), 2.0_hf);

    EXPECT_EQ(tensorA->size(), tensorB->size());
    EXPECT_EQ(tensorA->size(), 10u);

    memory_manager["x"] /= memory_manager["y"];

    for (size_t i = 0; i < tensorA->size(); ++i)
    {
        EXPECT_EQ((*tensorB)[i], 2.0_hf);
        EXPECT_EQ((*tensorA)[i], 0.5_hf);
    }

    memory_manager["x"] /= 10.0_hf;

    for (size_t i = 0; i < tensorA->size(); ++i)
    {
        EXPECT_EQ((*tensorA)[i], 0.05_hf);
    }
}

TEST(TestTensor, TensorFP32FP16PromotionsSumUnit)
{
    std::cout << raul::io::tensor::full;

    const auto abs_err = 1e-3_dt;
    const auto arg_a = 0.1_dt;
    const auto arg_b = 0.2_dt;
    const auto res_c = arg_a+arg_b;

    // Tensor[dtype] ?= dtype
    {
        raul::Tensor result{static_cast<raul::dtype>(arg_a)};
        result += static_cast<raul::dtype>(arg_b);
        EXPECT_NEAR(result[0], res_c, abs_err);
        std::cout << result << std::endl;
    }

    // Tensor[half] ?= half
    {
        raul::TensorFP16 result{static_cast<raul::half>(arg_a)};
        result += static_cast<raul::half>(arg_b);
        EXPECT_NEAR(result[0], res_c, abs_err);
        std::cout << result << std::endl;
    }

    // Tensor[dtype] ?= half (promote to dtype)
    {
        raul::Tensor result{static_cast<raul::dtype>(arg_a)};
        result += static_cast<raul::half>(arg_b);
        EXPECT_NEAR(result[0], res_c, abs_err);
        std::cout << result << std::endl;
    }

    // Tensor[half] ?= dtype (promote to dtype and result convert to half)
    {
        raul::TensorFP16 result{static_cast<raul::half>(arg_a)};
        result += static_cast<raul::dtype>(arg_b);
        EXPECT_NEAR(result[0], res_c, abs_err);
        std::cout << result << std::endl;
    }

    std::cout << raul::io::tensor::brief;
}

TEST(TestTensor, TensorFP32FP16PromotionsSubUnit)
{
    std::cout << raul::io::tensor::full;

    const auto abs_err = 1e-3_dt;
    const auto arg_a = 0.1_dt;
    const auto arg_b = 0.2_dt;
    const auto res_c = arg_a-arg_b;

    // Tensor[dtype] ?= dtype
    {
        raul::Tensor result{static_cast<raul::dtype>(arg_a)};
        result -= static_cast<raul::dtype>(arg_b);
        EXPECT_NEAR(result[0], res_c, abs_err);
        std::cout << result << std::endl;
    }

    // Tensor[half] ?= half
    {
        raul::TensorFP16 result{static_cast<raul::half>(arg_a)};
        result -= static_cast<raul::half>(arg_b);
        EXPECT_NEAR(result[0], res_c, abs_err);
        std::cout << result << std::endl;
    }

    // Tensor[dtype] ?= half (promote to dtype)
    {
        raul::Tensor result{static_cast<raul::dtype>(arg_a)};
        result -= static_cast<raul::half>(arg_b);
        EXPECT_NEAR(result[0], res_c, abs_err);
        std::cout << result << std::endl;
    }

    // Tensor[half] ?= dtype (promote to dtype and result convert to half)
    {
        raul::TensorFP16 result{static_cast<raul::half>(arg_a)};
        result -= static_cast<raul::dtype>(arg_b);
        EXPECT_NEAR(result[0], res_c, abs_err);
        std::cout << result << std::endl;
    }

    std::cout << raul::io::tensor::brief;
}

TEST(TestTensor, TensorFP32FP16PromotionsMulUnit)
{
    std::cout << raul::io::tensor::full;

    const auto abs_err = 1e-3_dt;
    const auto arg_a = 5.0_dt;
    const auto arg_b = 1e-1_dt;
    const auto res_c = arg_a*arg_b;

    // Tensor[dtype] ?= dtype
    {
        raul::Tensor result{static_cast<raul::dtype>(arg_a)};
        result *= static_cast<raul::dtype>(arg_b);
        EXPECT_NEAR(result[0], res_c, abs_err);
        std::cout << result << std::endl;
    }

    // Tensor[half] ?= half
    {
        raul::TensorFP16 result{static_cast<raul::half>(arg_a)};
        result *= static_cast<raul::half>(arg_b);
        EXPECT_NEAR(result[0], res_c, abs_err);
        std::cout << result << std::endl;
    }

    // Tensor[dtype] ?= half (promote to dtype)
    {
        raul::Tensor result{static_cast<raul::dtype>(arg_a)};
        result *= static_cast<raul::half>(arg_b);
        EXPECT_NEAR(result[0], res_c, abs_err);
        std::cout << result << std::endl;
    }

    // Tensor[half] ?= dtype (promote to dtype and result convert to half)
    {
        raul::TensorFP16 result{static_cast<raul::half>(arg_a)};
        result *= static_cast<raul::dtype>(arg_b);
        EXPECT_NEAR(result[0], res_c, abs_err);
        std::cout << result << std::endl;
    }

    std::cout << raul::io::tensor::brief;
}

TEST(TestTensor, TensorFP32FP16PromotionsDivUnit)
{
    std::cout << raul::io::tensor::full;

    const auto abs_err = 1e-3;
    const auto arg_a = 0.5;
    const auto arg_b = 3.0;
    const auto res_c = arg_a/arg_b;

    // Tensor[dtype] ?= dtype
    {
        raul::Tensor result{static_cast<raul::dtype>(arg_a)};
        result /= static_cast<raul::dtype>(arg_b);
        EXPECT_NEAR(result[0], res_c, abs_err);
        std::cout << result << std::endl;
    }

    // Tensor[half] ?= half
    {
        raul::TensorFP16 result{static_cast<raul::half>(arg_a)};
        result /= static_cast<raul::half>(arg_b);
        EXPECT_NEAR(result[0], res_c, abs_err);
        std::cout << result << std::endl;
    }

    // Tensor[dtype] ?= half (promote to dtype)
    {
        raul::Tensor result{static_cast<raul::dtype>(arg_a)};
        result /= static_cast<raul::half>(arg_b);
        EXPECT_NEAR(result[0], res_c, abs_err);
        std::cout << result << std::endl;
    }

    // Tensor[half] ?= dtype (promote to dtype and result convert to half)
    {
        raul::TensorFP16 result{static_cast<raul::half>(arg_a)};
        result /= static_cast<raul::dtype>(arg_b);
        EXPECT_NEAR(result[0], res_c, abs_err);
        std::cout << result << std::endl;
    }

    std::cout << raul::io::tensor::brief;
}

} // UT namespace