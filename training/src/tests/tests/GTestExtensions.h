// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GTESTEXTENSIONS_H
#define GTESTEXTENSIONS_H

#include <gtest/gtest.h>

#include <training/base/common/Tensor.h>

#define ASSERT_U8_TENSORS_EQ(reference_value, target_value)                                                                                                                                            \
    {                                                                                                                                                                                                  \
        static_assert(std::is_same_v<std::decay_t<decltype(reference_value)>, raul::TensorU8>, #reference_value " should have type raul::Tensor");                                                     \
        static_assert(std::is_same_v<std::decay_t<decltype(target_value)>, raul::TensorU8>, #target_value " should have type raul::Tensor");                                                           \
                                                                                                                                                                                                       \
        if (reference_value.getBatchSize() != target_value.getBatchSize())                                                                                                                             \
        {                                                                                                                                                                                              \
            FAIL() << "expected that " #reference_value " and " #target_value " have the same batch size, but they have not\n"                                                                         \
                   << #reference_value " has batch size " << reference_value.getBatchSize() << "\n"                                                                                                    \
                   << #target_value " has batch size " << target_value.getBatchSize() << "\n";                                                                                                         \
        }                                                                                                                                                                                              \
                                                                                                                                                                                                       \
        if (reference_value.getDepth() != target_value.getDepth())                                                                                                                                     \
        {                                                                                                                                                                                              \
            FAIL() << "expected that " #reference_value " and " #target_value " have the same depth, but they have not\n"                                                                              \
                   << #reference_value " has depth " << reference_value.getDepth() << "\n"                                                                                                             \
                   << #target_value " has depth " << target_value.getDepth() << "\n";                                                                                                                  \
        }                                                                                                                                                                                              \
                                                                                                                                                                                                       \
        if (reference_value.getHeight() != target_value.getHeight())                                                                                                                                   \
        {                                                                                                                                                                                              \
            FAIL() << "expected that " #reference_value " and " #target_value " have the same height, but they have not\n"                                                                             \
                   << #reference_value " has height " << reference_value.getHeight() << "\n"                                                                                                           \
                   << #target_value " has height " << target_value.getHeight() << "\n";                                                                                                                \
        }                                                                                                                                                                                              \
                                                                                                                                                                                                       \
        if (reference_value.getWidth() != target_value.getWidth())                                                                                                                                     \
        {                                                                                                                                                                                              \
            FAIL() << "expected that " #reference_value " and " #target_value " have the same width, but they have not\n"                                                                              \
                   << #reference_value " has width " << reference_value.getWidth() << "\n"                                                                                                             \
                   << #target_value " has width " << target_value.getWidth() << "\n";                                                                                                                  \
        }                                                                                                                                                                                              \
                                                                                                                                                                                                       \
        for (size_t i = 0; i < reference_value.size(); ++i)                                                                                                                                            \
        {                                                                                                                                                                                              \
            if (reference_value[i] != target_value[i])                                                                                                                                                 \
            {                                                                                                                                                                                          \
                FAIL() << #reference_value " and " #target_value << " have different values at position " << i << "\n"                                                                                 \
                       << #reference_value "[" << i << "] evaluates to " << reference_value[i] << "\n"                                                                                                 \
                       << #target_value "[" << i << "] evaluates to " << target_value[i] << "\n";                                                                                                      \
            }                                                                                                                                                                                          \
        }                                                                                                                                                                                              \
    }

#define ASSERT_FLOAT_TENSORS_EQ(reference_value, target_value, epsilon)                                                                                                                                \
    {                                                                                                                                                                                                  \
        static_assert(std::is_same_v<std::decay_t<decltype(reference_value)>, raul::Tensor>, #reference_value " should have type raul::Tensor");                                                       \
        static_assert(std::is_same_v<std::decay_t<decltype(target_value)>, raul::Tensor>, #target_value " should have type raul::Tensor");                                                             \
        static_assert(std::is_same_v<std::decay_t<decltype(epsilon)>, float>, "epsilon should have type float");                                                                                       \
                                                                                                                                                                                                       \
        if (reference_value.getBatchSize() != target_value.getBatchSize())                                                                                                                             \
        {                                                                                                                                                                                              \
            FAIL() << "expected that " #reference_value " and " #target_value " have the same batch size, but they have not\n"                                                                         \
                   << #reference_value " has batch size " << reference_value.getBatchSize() << "\n"                                                                                                    \
                   << #target_value " has batch size " << target_value.getBatchSize() << "\n";                                                                                                         \
        }                                                                                                                                                                                              \
                                                                                                                                                                                                       \
        if (reference_value.getDepth() != target_value.getDepth())                                                                                                                                     \
        {                                                                                                                                                                                              \
            FAIL() << "expected that " #reference_value " and " #target_value " have the same depth, but they have not\n"                                                                              \
                   << #reference_value " has depth " << reference_value.getDepth() << "\n"                                                                                                             \
                   << #target_value " has depth " << target_value.getDepth() << "\n";                                                                                                                  \
        }                                                                                                                                                                                              \
                                                                                                                                                                                                       \
        if (reference_value.getHeight() != target_value.getHeight())                                                                                                                                   \
        {                                                                                                                                                                                              \
            FAIL() << "expected that " #reference_value " and " #target_value " have the same height, but they have not\n"                                                                             \
                   << #reference_value " has height " << reference_value.getHeight() << "\n"                                                                                                           \
                   << #target_value " has height " << target_value.getHeight() << "\n";                                                                                                                \
        }                                                                                                                                                                                              \
                                                                                                                                                                                                       \
        if (reference_value.getWidth() != target_value.getWidth())                                                                                                                                     \
        {                                                                                                                                                                                              \
            FAIL() << "expected that " #reference_value " and " #target_value " have the same width, but they have not\n"                                                                              \
                   << #reference_value " has width " << reference_value.getWidth() << "\n"                                                                                                             \
                   << #target_value " has width " << target_value.getWidth() << "\n";                                                                                                                  \
        }                                                                                                                                                                                              \
                                                                                                                                                                                                       \
        for (size_t i = 0; i < reference_value.size(); ++i)                                                                                                                                            \
        {                                                                                                                                                                                              \
            if (std::abs(reference_value[i] - target_value[i]) > epsilon)                                                                                                                              \
            {                                                                                                                                                                                          \
                FAIL() << #reference_value " and " #target_value << " have different values at position " << i << "\n"                                                                                 \
                       << #reference_value "[" << i << "] evaluates to " << reference_value[i] << "\n"                                                                                                 \
                       << #target_value "[" << i << "] evaluates to " << target_value[i] << "\n"                                                                                                       \
                       << #epsilon " evaluates to " << epsilon << "\n";                                                                                                                                \
            }                                                                                                                                                                                          \
        }                                                                                                                                                                                              \
    }

#define ASSERT_STD_STR_EQ(reference_value, target_value)                                                                                                                                               \
    {                                                                                                                                                                                                  \
        static_assert(std::is_same_v<std::decay_t<decltype(reference_value)>, std::string>, #reference_value " should have type std::string");                                                         \
        static_assert(std::is_same_v<std::decay_t<decltype(target_value)>, std::string> || std::is_convertible_v<decltype(target_value), std::string>,                                                 \
                      #target_value " should have type std::string or can be converted to it");                                                                                                        \
        if (reference_value != target_value)                                                                                                                                                           \
        {                                                                                                                                                                                              \
            FAIL() << "Expected that " #reference_value << " and " << #target_value << " are equal, but they are differ\n"                                                                             \
                   << #reference_value " is\n"                                                                                                                                                         \
                   << "\t" << reference_value << "\n"                                                                                                                                                  \
                   << #target_value " is\n"                                                                                                                                                            \
                   << "\t" << target_value << "\n";                                                                                                                                                    \
        }                                                                                                                                                                                              \
    }

#define ASSERT_FLOAT_VECTORS_EQ(reference_value, target_value, epsilon)                                                                                                                                \
    {                                                                                                                                                                                                  \
        static_assert(std::is_same_v<std::decay_t<decltype(reference_value)>, std::vector<float>>, #reference_value " should have type std::vector<float>");                                           \
        static_assert(std::is_same_v<std::decay_t<decltype(target_value)>, std::vector<float>>, #target_value " should have type std::vector<float>");                                                 \
        static_assert(std::is_same_v<std::decay_t<decltype(epsilon)>, float>, "epsilon should have type float");                                                                                       \
                                                                                                                                                                                                       \
        if (reference_value.size() != target_value.size())                                                                                                                                             \
        {                                                                                                                                                                                              \
            FAIL() << "expected that " #reference_value " and " #target_value " have the same size, but they have not\n"                                                                               \
                   << #reference_value " has size " << reference_value.size() << "\n"                                                                                                                  \
                   << #target_value " has size " << target_value.size() << "\n";                                                                                                                       \
        }                                                                                                                                                                                              \
                                                                                                                                                                                                       \
        for (size_t i = 0; i < reference_value.size(); ++i)                                                                                                                                            \
        {                                                                                                                                                                                              \
            if (std::abs(reference_value[i] - target_value[i]) > epsilon)                                                                                                                              \
            {                                                                                                                                                                                          \
                FAIL() << #reference_value " and " #target_value << " have different values at position " << i << "\n"                                                                                 \
                       << #reference_value "[" << i << "] evaluates to " << reference_value[i] << "\n"                                                                                                 \
                       << #target_value "[" << i << "] evaluates to " << target_value[i] << "\n"                                                                                                       \
                       << #epsilon " evaluates to " << epsilon << "\n";                                                                                                                                \
            }                                                                                                                                                                                          \
        }                                                                                                                                                                                              \
    }

#define ASSERT_INTERVALS_NEAR(i1_begin, i1_end, i2_begin, i2_end, epsilon)                                                                                                                             \
    {                                                                                                                                                                                                  \
        size_t i1_size = i1_end - i1_begin;                                                                                                                                                            \
        size_t i2_size = i2_end - i2_begin;                                                                                                                                                            \
        if (i1_size != i2_size)                                                                                                                                                                        \
        {                                                                                                                                                                                              \
            FAIL() << "expected that intervals have the same size, but they have different\n"                                                                                                          \
                   << "size of first interval evaluated to:\n"                                                                                                                                         \
                   << "\t" << i1_size << "\n"                                                                                                                                                          \
                   << "size of second interval evaluated to:\n"                                                                                                                                        \
                   << "\t" << i2_size << "\n";                                                                                                                                                         \
        }                                                                                                                                                                                              \
                                                                                                                                                                                                       \
        size_t pos = 0;                                                                                                                                                                                \
        auto it1 = i1_begin;                                                                                                                                                                           \
        auto it2 = i2_begin;                                                                                                                                                                           \
        while (it1 != i1_end)                                                                                                                                                                          \
        {                                                                                                                                                                                              \
            if (std::abs(*it1 - *it2) >= epsilon)                                                                                                                                                      \
            {                                                                                                                                                                                          \
                FAIL() << "intervals differ in position " << pos << "\n";                                                                                                                              \
            }                                                                                                                                                                                          \
            ++it1;                                                                                                                                                                                     \
            ++it2;                                                                                                                                                                                     \
            ++pos;                                                                                                                                                                                     \
        }                                                                                                                                                                                              \
    }

#endif // GTESTEXTENSIONS_H
