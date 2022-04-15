// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TESTTOOLS_H
#define TESTTOOLS_H

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <training/common/Random.h>
#include <training/layers/parameters/LayerParameters.h>
#include <training/network/Workflow.h>
#include <training/network/WorkflowEager.h>
#include <training/tools/Profiler.h>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#define TEST_NAME (::testing::UnitTest::GetInstance()->current_test_info()->test_case_name() + std::string(".") + ::testing::UnitTest::GetInstance()->current_test_info()->name())

using namespace raul;

namespace UT::tools
{

class ARGS
{
  public:
    static std::map<string, string> ARGUMENTS;
};

template<typename T>
inline T getArg(const string&, T defaultVal)
{
    return defaultVal;
}

template<>
inline std::string getArg(const string& name, std::string defaultVal)
{
    auto v = ARGS::ARGUMENTS.find(name);
    if (v == ARGS::ARGUMENTS.end())
    {
        return defaultVal;
    }
    return v->second;
}

inline std::string getArg(const string& name, const char* defaultVal)
{
    auto v = ARGS::ARGUMENTS.find(name);
    if (v == ARGS::ARGUMENTS.end())
    {
        return string(defaultVal);
    }
    return v->second;
}

template<>
inline size_t getArg(const string& name, size_t defaultVal)
{
    auto v = ARGS::ARGUMENTS.find(name);
    if (v == ARGS::ARGUMENTS.end())
    {
        return defaultVal;
    }
    return stoul(v->second);
}

template<>
inline int getArg(const string& name, int defaultVal)
{
    auto v = ARGS::ARGUMENTS.find(name);
    if (v == ARGS::ARGUMENTS.end())
    {
        return defaultVal;
    }
    return stoi(v->second);
}

template<>
inline bool getArg(const string& name, bool defaultVal)
{
    auto v = ARGS::ARGUMENTS.find(name);
    if (v == ARGS::ARGUMENTS.end())
    {
        return defaultVal;
    }
    return v->second == "1" || v->second == "true" || v->second == "yes";
}

using optional_path = std::optional<std::filesystem::path>;

std::filesystem::path getTempDir();

std::filesystem::path getTestAssetsDir();

void print_tensor(const raul::Tensor& t, const std::filesystem::path& s);
void print_tensor(const raul::TensorGPUHelper& t, const std::filesystem::path& s);

void checkTensors(const std::vector<std::pair<raul::Name, raul::Name>>& tensors, raul::MemoryManagerGPU& m, raul::dtype eps);
void checkTensors(const std::vector<std::pair<raul::Name, raul::Name>>& tensors, raul::MemoryManager& m, raul::dtype eps);
decltype(raul::NetworkParameters::mCallbackGPU) createLayerTracerCallbackGPU(raul::Workflow*);

template<typename MM>
void init_rand_tensor(const std::string& tensor_name, const raul::random::dataRange<typename MM::type> random_range, MM& memory_manager)
{
    auto& tensor = memory_manager[tensor_name];
    for (auto& val : tensor)
    {
        val = raul::random::uniform::rand<typename MM::type>(random_range);
    }
}

template<>
void init_rand_tensor(const std::string& tensor_name, const raul::random::dataRange<raul::MemoryManagerGPU::type> random_range, raul::MemoryManagerGPU& memory_manager);

size_t get_size_of_trainable_params(const raul::Workflow& network);
size_t get_size_of_trainable_params_mixed_precision(const raul::Workflow& network);

std::string read_file_to_string(const std::filesystem::path& path);

size_t countLines(const std::filesystem::path& p);

template<typename T>
bool expect_near_relative(const T a, const T b, const T epsilon)
{
    const auto diff = static_cast<T>(std::abs(TODTYPE(a - b)));
    constexpr auto min_t = std::numeric_limits<T>::min();
    if (a == b)
    {
        return true;
    }
    if (a == static_cast<T>(0.0) || b == static_cast<T>(0.0) || diff < min_t)
    {
        return diff < (epsilon * min_t);
    }
    const auto abs_a = static_cast<T>(std::abs(TODTYPE(a)));
    const auto abs_b = static_cast<T>(std::abs(TODTYPE(b)));
    return (diff / abs_a <= epsilon) && (diff / abs_b <= epsilon);
}

#if !defined(_WIN32)
/**
 * Return the peak of really used RAM (RSS)
 * This is a high water mark for RSS.
 * @return maximum resident set size in kilobytes
 */
long getPeakOfMemory();

struct Timestamp
{
    timeval user;
    timeval system;
};
/**
 * Get timestamp form the kernel
 * @return timestamp (see Timestamp)
 */
Timestamp getCPUTimestamp();

/**
 * Calculate elapsed time
 *
 * @param begin timestamp of start
 * @param end timestamp of stop
 * @return a pair with elapsed time in seconds (user, system)
 */
std::pair<double, double> getElapsedTime(Timestamp begin, Timestamp end);
#endif

namespace toynet
{
enum class Nonlinearity
{
    relu,
    sigmoid,
    hsigmoid,
    hswish,
    swish
};

std::unique_ptr<raul::Workflow> buildUnique(const size_t width,
                                            const size_t height,
                                            const size_t channels,
                                            const size_t batch_size,
                                            const size_t classes,
                                            const size_t hidden_size,
                                            const Nonlinearity nl_type,
                                            const bool usePool = false,
                                            const optional_path& weights_path = std::nullopt,
                                            const std::unordered_map<raul::Name, raul::ScalingStrategy>& scaling = {});
}
} // namespace UT::tools

namespace raul::helpers
{

class ProfilerGuard
{
  public:
    ProfilerGuard(const std::string& testName);

    ~ProfilerGuard();

  private:
    bool mWasEnabled;
    std::string mTestName;
};

#define PROFILE_TEST raul::helpers::ProfilerGuard guard{ TEST_NAME };
#define GPU_ONLY_TEST                                                                                                                                                                                  \
    if (!raul::Common::hasOpenCL())                                                                                                                                                                    \
    {                                                                                                                                                                                                  \
        std::cout << "Test " << TEST_NAME << " skipped as no GPU found" << std::endl;                                                                                                                  \
        return;                                                                                                                                                                                        \
    }

class ListenerHelper : public WorkflowListener
{
  public:
    typedef std::pair<raul::Name, const raul::Tensor&> ProcessData;

    void addProcesser(const ProcessData& checkData);

  protected:
    std::vector<ProcessData> mProcessers;
};

class Checker : public ListenerHelper
{
  public:
    Checker()
        : mEps(1e-5_dt)
    {
    }

    Checker(raul::dtype eps)
        : mEps(eps)
    {
    }

  protected:
    void check(const Workflow& work) const;

    raul::dtype mEps;
};

class CheckerAfterForward : public Checker
{
  public:
    CheckerAfterForward(raul::dtype eps)
        : Checker(eps)
    {
    }

    void AfterForward(Workflow& work) override { check(work); }
};

class CheckerAfterBackward : public Checker
{
  public:
    CheckerAfterBackward(raul::dtype eps)
        : Checker(eps)
    {
    }

    void AfterBackward(Workflow& work) override { check(work); }
};

class FillerBeforeBackward : public ListenerHelper
{
  public:
    FillerBeforeBackward() {}

    void BeforeBackward(Workflow& work) override;
};
}

#endif
