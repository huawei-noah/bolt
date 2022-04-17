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

#include <training/base/layers/BasicLayer.h>
#include <training/compiler/Layers.h>

namespace UT::tools
{

std::map<string, string> ARGS::ARGUMENTS;

using namespace std;
using namespace raul;

void checkTensors(const vector<pair<Name, Name>>& tensors, MemoryManager& m, dtype eps)
{
    for (const auto& it : tensors)
    {
        Tensor& t = m[it.first];
        Tensor& golden = m[it.second];
        EXPECT_EQ(t.getShape(), golden.getShape());
        for (size_t i = 0; i < golden.size(); ++i)
        {
            CHECK_NEAR(t[i], golden[i], eps);
        }
        cout << "Tensor '" << it.first << "' checked" << endl;
    }
}

size_t get_size_of_trainable_params(const Workflow& network)
{
    size_t amount_of_trainable_parameters = 0U;
    for (const auto& name : network.getTrainableParameterNames())
    {
        amount_of_trainable_parameters += network.getMemoryManager()[name].getShape().total_size();
    }
    return amount_of_trainable_parameters;
}

size_t get_size_of_trainable_params_mixed_precision(const Workflow& network)
{
    size_t amount_of_trainable_parameters = 0U;
    for (const auto& name : network.getTrainableParameterNames())
    {
        if (network.getMemoryManager().tensorExists(name))
            amount_of_trainable_parameters += network.getMemoryManager()[name].getShape().total_size();
        else if (network.getMemoryManager<MemoryManagerFP16>().tensorExists(name))
            amount_of_trainable_parameters += network.getMemoryManager<MemoryManagerFP16>()[name].getShape().total_size();
    }
    return amount_of_trainable_parameters;
}

dtype TensorDiff(const Tensor& a, const Tensor& b)
{
    if (a.size() != b.size() || a.empty() || b.empty())
    {
        THROW_NONAME("TestTools", "empty tensor");
    }
    dtype sum = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

dtype TensorNorm(const Tensor& a)
{
    if (a.empty())
    {
        THROW_NONAME("TestTools", "empty tensor");
    }
    dtype sum = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        sum += a[i] * a[i];
    }
    return std::sqrt(sum);
}

dtype TensorNorm(const TensorFP16& a)
{
    if (a.empty())
    {
        THROW_NONAME("TestTools", "empty tensor");
    }
    dtype sum = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        sum += TODTYPE(a[i]) * TODTYPE(a[i]);
    }
    return std::sqrt(sum);
}

#if !defined(_WIN32)
long getPeakOfMemory()
{
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
}

Timestamp getCPUTimestamp()
{
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return { ru.ru_utime, ru.ru_stime };
}

std::pair<double, double> getElapsedTime(Timestamp begin, Timestamp end)
{
    const auto userDiff = (end.user.tv_sec - begin.user.tv_sec) + (end.user.tv_usec - begin.user.tv_usec) / 1e6;
    const auto systemDiff = end.system.tv_sec - begin.system.tv_sec + (end.system.tv_usec - begin.system.tv_usec) / 1e6;
    return make_pair(userDiff, systemDiff);
}

#endif

} // UT::tools

namespace raul::helpers
{

using namespace std;

ProfilerGuard::ProfilerGuard(const string& testName)
    : mWasEnabled(false)
    , mTestName(testName)
{
    if (Profiler::getInstance().isDisabled())
    {
        return;
    }

    mWasEnabled = Profiler::getInstance().getState();
    if (Profiler::getInstance().useJsonFormat())
    {
        Profiler::getInstance().increasePrefix(mTestName);
    }
    else
    {
        *Profiler::getInstance().getOstream() << mTestName + "\n";
    }
    Profiler::getInstance().enableProfiler();
}

ProfilerGuard::~ProfilerGuard()
{
    if (!mWasEnabled && !Profiler::getInstance().isDisabled())
    {
        if (Profiler::getInstance().useJsonFormat())
        {
            Profiler::getInstance().clearPrefix();
        }
        Profiler::getInstance().disableProfiler();
    }
}

void ListenerHelper::addProcesser(const ListenerHelper::ProcessData& checkData)
{
    mProcessers.push_back(checkData);
}

void Checker::check(const Workflow& work) const
{
    const auto& memory_manager = work.getMemoryManager();

    if (mProcessers.empty())
    {
        THROW_NONAME("Checker", "empty processors");
    }

    for (const auto& checkData : mProcessers)
    {
        const auto& tensor = memory_manager[checkData.first];
        EXPECT_EQ(tensor.size(), checkData.second.size());

        for (size_t i = 0; i < tensor.size(); ++i)
        {
            ASSERT_TRUE(UT::tools::expect_near_relative(tensor[i], checkData.second[i], mEps)) << "at " << i << ", expected: " << checkData.second[i] << ", got: " << tensor[i];
        }
    }
}

void FillerBeforeBackward::BeforeBackward(Workflow& work)
{
    auto& memory_manager = work.getMemoryManager();

    if (mProcessers.empty())
    {
        THROW_NONAME("FillerBeforeBackward", "empty processors");
    }

    for (const auto& checkData : mProcessers)
    {
        auto& tensor = memory_manager[checkData.first];
        EXPECT_EQ(tensor.size(), checkData.second.size());

        tensor = TORANGE(checkData.second);
    }
}

} // namespace UT::tools