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
#include <training/base/common/Random.h>

#include <future>
#include <set>
#include <thread>

namespace UT
{

TEST(TestRandom, SeedSetGetUnit)
{
    auto checkGlobalSeed = [](std::optional<size_t> seed = std::nullopt) {
        auto currentSeedCheck1 = raul::random::getGlobalSeed();
        auto currentSeedCheck2 = raul::random::getGlobalSeed();
        if (seed)
        {
            EXPECT_EQ(currentSeedCheck1, *seed);
            EXPECT_EQ(currentSeedCheck2, *seed);
        }
        else
        {
            EXPECT_EQ(currentSeedCheck1, currentSeedCheck2);
        }
    };

    size_t manualSeed = 12345;

    checkGlobalSeed();
    raul::random::setGlobalSeed(manualSeed);
    checkGlobalSeed(manualSeed);
}

TEST(TestRandom, SeedSetGetThreadUnit)
{
    auto checkGlobalSeed = [](size_t seed) {
        auto currentSeedCheck1 = raul::random::getGlobalSeed();
        auto currentSeedCheck2 = raul::random::getGlobalSeed();
        EXPECT_EQ(currentSeedCheck1, seed);
        EXPECT_EQ(currentSeedCheck2, seed);
    };

    size_t numberOfThreads = 100U;
    size_t seed = raul::random::getGlobalSeed();

    std::vector<std::thread> threads;

    for (size_t i = 0; i < numberOfThreads; ++i)
    {
        threads.emplace_back(std::thread(checkGlobalSeed, seed));
    }
    for (auto& t : threads)
    {
        t.join();
    }
}

TEST(TestRandom, LocalSeedThreadUnit)
{
    auto checkLocalSeed = [](size_t seed) {
        const size_t localSeed = raul::random::getThreadSeed();
        EXPECT_NE(localSeed, seed);
    };

    size_t numberOfThreads = 100U;
    size_t seed = raul::random::getGlobalSeed();

    std::vector<std::thread> threads;

    for (size_t i = 0; i < numberOfThreads; ++i)
    {
        threads.emplace_back(std::thread(checkLocalSeed, seed));
    }
    for (auto& t : threads)
    {
        t.join();
    }
}

TEST(TestRandom, GeneratorsThreadUnit)
{
    const raul::dtype p = 0.5_dt;
    size_t numberOfValues = 100U;
    size_t numberOfThreads = 10U;

    auto generateSequence = [](const raul::dtype p, const size_t num) -> std::vector<bool> {
        auto gen = raul::random::getGenerator();
        std::vector<bool> container;
        std::bernoulli_distribution distrib(p);
        for (size_t i = 0; i < num; ++i)
        {
            container.push_back(distrib(gen));
        }
        return container;
    };

    std::vector<std::future<std::vector<bool>>> sequences;

    for (size_t i = 0; i < numberOfThreads; ++i)
    {
        sequences.emplace_back(std::async(generateSequence, p, numberOfValues));
    }
    std::set<std::vector<bool>> seqSet;
    for (auto& seq : sequences)
    {
        std::vector<bool> data = seq.get();
        EXPECT_TRUE(seqSet.find(data) == seqSet.cend());
        seqSet.insert(data);
    }
}

// TEST(TestRandom, GeneratorRestoreSequenceUnit)
//{
//    const raul::dtype p = 0.5_dt;
//    const size_t numberOfThreads = 10U;
//    const size_t goldenSeed = 2477580634U;
//    const std::vector<bool> goldenSequence{false, false, true, true, true, false, true, true, false, true};
//    raul::random::setGlobalSeed(goldenSeed);
//    auto gen = raul::random::getGenerator();
//    std::vector<bool> container;
//    std::bernoulli_distribution distrib(p);
//
//    for(size_t i=0; i<numberOfThreads; ++i)
//    {
//        container.push_back(distrib(gen));
//    }
//
//    EXPECT_EQ(goldenSequence, container);
//}

} // UT namespace