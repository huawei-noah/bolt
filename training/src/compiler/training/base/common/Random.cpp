// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Random.h"

namespace
{
std::optional<size_t> globalSeed = std::nullopt;
}

namespace raul::random
{

size_t getGlobalSeed()
{
    static std::random_device rd;
    if (!globalSeed)
    {
        setGlobalSeed(rd());
    }
    return *globalSeed;
}

void setGlobalSeed(size_t seed)
{
    globalSeed = seed;
}

size_t getThreadSeed()
{
    static size_t cnt = 0;
    static thread_local size_t seed = getGlobalSeed() + (++cnt);
    return seed;
}

std::mt19937_64& getGenerator(std::optional<size_t> seed)
{
    const size_t localSeed = seed ? *seed : getThreadSeed();
    thread_local static std::mt19937_64 generator(localSeed);
    return generator;
}

namespace bernoulli
{

bool randBool(dtype p)
{
    static thread_local auto gen = getGenerator();
    std::bernoulli_distribution dis;
    return dis(gen, decltype(dis)::param_type{ p });
}

bool randBool(dtype p, std::mt19937_64& gen)
{
    std::bernoulli_distribution dis;
    return dis(gen, decltype(dis)::param_type{ p });
}

}

} // namespace raul::random
