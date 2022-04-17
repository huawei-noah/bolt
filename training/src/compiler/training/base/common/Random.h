// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef RANDOM_H
#define RANDOM_H

#include "Common.h"

#include <optional>
#include <random>
#include <type_traits>

namespace raul::random
{

template<typename T>
using dataRange = std::pair<T, T>;
using dtypeRange = dataRange<raul::dtype>;
using halfRange = dataRange<raul::half>;
using intRange = dataRange<int>;

// Seed management

/**
 * @brief Get global Raul seed value
 * @return seed (non-negative integer value)
 *
 * If the value is not set before the random device is used.
 *
 */
[[nodiscard]] size_t getGlobalSeed();

/**
 * @brief Set global Raul seed value
 * @param seed
 */
void setGlobalSeed(size_t seed);

/**
 * @brief Get local thread seed value
 * @return seed (non-negative integer value)
 */
size_t getThreadSeed();

// Generators

[[nodiscard]] std::mt19937_64& getGenerator(std::optional<size_t> seed = std::nullopt);

namespace uniform
{

template<typename T>
[[nodiscard]] T rand(T from, T to)
{
    static thread_local auto gen = getGenerator();
    if constexpr (std::is_integral_v<T>)
    {
        std::uniform_int_distribution<T> dis(from, to);
        return dis(gen);
    }
    else
    {
        std::uniform_real_distribution<raul::dtype> dis(from, to);
        return static_cast<T>(dis(gen));
    }
}

template<typename T>
[[nodiscard]] T rand(dataRange<T> randomRange)
{
    if constexpr (std::is_integral_v<T>)
    {
        return rand<T>(randomRange.first, randomRange.second);
    }
    else
    {
        const auto first = TODTYPE(randomRange.first);
        constexpr auto limit = std::numeric_limits<raul::dtype>::max();
        const auto second = std::nextafter(TODTYPE(randomRange.second), limit);
        return static_cast<T>(rand<raul::dtype>(first, second));
    }
}

}

namespace bernoulli
{
[[nodiscard]] bool randBool(dtype p);
[[nodiscard]] bool randBool(dtype p, std::mt19937_64& gen);
}

} // namespace raul::random

#endif // RANDOM_H
