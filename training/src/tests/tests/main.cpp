// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tclap/CmdLine.h>

#include <tests/tools/TestTools.h>
#include <training/base/common/Conversions.h>
#include <training/base/common/Random.h>
#include <training/system/Profiler.h>
#include <Version.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(ANDROID)
#include <android/log.h>
#include <dlfcn.h>
#include <signal.h>
#include <unwind.h>
#endif

struct cmdParam
{
    bool mProfile = false;
    std::string mProfileOutput;
    bool mUseJsonFormat = false;
    std::optional<size_t> mSeed = std::nullopt;
};

cmdParam parseCMD(int argc, char* argv[])
{
    cmdParam res;
    TCLAP::CmdLine cmd("", '=', Conversions::toString(raul::Version::getNumber()), false);

    TCLAP::UnlabeledMultiArg<std::string> argsToIgnore("", "Capture all unmatched values", false, "string");
    cmd.add(argsToIgnore);

    TCLAP::SwitchArg profile("", "profile", "Enable internal profiler", false);
    cmd.add(profile);

    TCLAP::ValueArg<std::string> profileOutput("", "profile_output", "Profiler output location", false, "", "string");
    cmd.add(profileOutput);

    TCLAP::SwitchArg useJsonFormat("", "use_json_format", "Print output as json", false);
    cmd.add(useJsonFormat);

    TCLAP::ValueArg<size_t> useSeed("", "raul_seed", "Raul seed", false, 0, "number");
    cmd.add(useSeed);

    cmd.parse(argc, argv);

    res.mProfile = profile.getValue();
    res.mProfileOutput = profileOutput.getValue();
    res.mUseJsonFormat = useJsonFormat.getValue();
    res.mSeed = useSeed.isSet() ? std::optional<size_t>(useSeed.getValue()) : std::nullopt;

    for (auto s : argsToIgnore.getValue())
    {
        if (raul::Common::startsWith(s, "--gtest"))
        {
            continue;
        }
        if (!raul::Common::startsWith(s, "--") || s.find("=") == string::npos)
        {
            throw runtime_error("Bad argument: " + s);
        }
        s = s.substr(2);
        auto pos = s.find("=");
        auto key = s.substr(0, pos);
        auto val = s.substr(pos + 1);
        UT::tools::ARGS::ARGUMENTS[key] = val;
    }

    return res;
}

#if defined(ANDROID)
namespace
{

struct BacktraceState
{
    void** current;
    void** end;
};

static _Unwind_Reason_Code unwindCallback(struct _Unwind_Context* context, void* arg)
{
    BacktraceState* state = static_cast<BacktraceState*>(arg);
    uintptr_t pc = _Unwind_GetIP(context);
    if (pc)
    {
        if (state->current == state->end)
        {
            return _URC_END_OF_STACK;
        }
        else
        {
            *state->current++ = reinterpret_cast<void*>(pc);
        }
    }
    return _URC_NO_REASON;
}

}

size_t captureBacktrace(void** buffer, size_t max)
{
    BacktraceState state = { buffer, buffer + max };
    _Unwind_Backtrace(unwindCallback, &state);

    return state.current - buffer;
}

void dumpBacktrace(std::ostream& os, void** buffer, size_t count)
{
    for (size_t idx = 0; idx < count; ++idx)
    {
        const void* addr = buffer[idx];
        const char* symbol = "";

        Dl_info info;
        if (dladdr(addr, &info) && info.dli_sname)
        {
            symbol = info.dli_sname;
        }

        os << "  #" << std::setw(2) << idx << ": " << addr << "  " << symbol << "\n";
    }
}

void backtraceToLogcat()
{
    const size_t max = 30;
    void* buffer[max];
    // std::ostringstream oss;

    dumpBacktrace(std::cout, buffer, captureBacktrace(buffer, max));

    //__android_log_print(ANDROID_LOG_INFO, "app_name", "%s", oss.str().c_str());
}

void sighandler([[maybe_unused]] int signum, siginfo_t* info, [[maybe_unused]] void* f)
{
    if (info->si_signo == SIGSEGV)
    {
        printf("Invalid access of address %p, ", info->si_addr);
        switch (info->si_code)
        {
            case SEGV_MAPERR:
                printf("SEGV_MAPERR\n");
                break;
            case SEGV_ACCERR:
                printf("SEGV_ACCERR\n");
                break;
            default:
                printf("unknown si_code\n");
        } /* switch() */
    }
    backtraceToLogcat();
    exit(1);

    return;
}
#endif
inline void printTestConfig()
{
    const auto width = 10;
#if defined(ANDROID)
    std::cout << std::setw(width) << "Library: "
              << "Raul Android" << std::endl;
#elif defined(_WIN32)
    std::cout << std::setw(width) << "Library: "
              << "Raul Windows" << std::endl;
#elif defined(__linux__)
    std::cout << std::setw(width) << "Library: "
              << "Raul Linux" << std::endl;
#else
    std::cout << std::setw(width) << "Library: "
              << "Raul Unknown platform" << std::endl;
#endif
    std::cout << std::setw(width) << "Version: " << raul::Version::getNumber() << std::endl;
    std::cout << std::setw(width) << "Revision: " << raul::Version::getRevisionStr() << std::endl;
    std::cout << std::setw(width) << "Date: " << raul::Version::getDateStr() << std::endl;
    std::cout << std::setw(width) << "Seed: " << raul::random::getGlobalSeed() << std::endl;
#if defined(_OPENMP)
    std::cout << std::setw(width) << "OpenMP: "
              << "On (threads: " << omp_get_max_threads() << ")" << std::endl;
#else
    std::cout << std::setw(width) << "OpenMP: "
              << "Off" << std::endl;
#endif
#if defined(_BLAS_ENHANCE)
    std::cout << std::setw(width) << "BLAS: "
              << "On (Enhance)" << std::endl;
#elif defined(_BLAS)
    std::cout << std::setw(width) << "BLAS: "
              << "On" << std::endl;
#else
    std::cout << std::setw(width) << "BLAS: "
              << "Off" << std::endl;
#endif
}

std::string printMemory(long value_kB)
{
    std::stringstream os;
    os << value_kB << " kB";
    if (value_kB > 1024 * 1024)
    {
        os << " (" << value_kB / (1024 * 1024) << " GB)";
    }
    else if (value_kB > 1024)
    {
        os << " (" << value_kB / 1024 << " MB)";
    }
    return os.str();
}

std::string printTime(double value_s)
{
    std::stringstream os;
    os << value_s << " s";
    if (value_s > 3600.0)
    {
        os << " (" << value_s / (3600.0) << " h)";
    }
    else if (value_s > 60.0)
    {
        os << " (" << value_s / 60.0 << " min)";
    }
    return os.str();
}

inline void printPerformanceMetrics(size_t repeat, double real_time_s, double cpu_user_time_s, double cpu_system_time_s, long peak)
{
    const auto width = 18;
    std::cout << "Metrics [config]" << std::endl;
    std::cout << std::setw(width) << "Test repeats: " << repeat << std::endl;
#if defined(_OPENMP)
    std::cout << std::setw(width) << "OpenMP threads: " << omp_get_max_threads() << std::endl;
#else 
    std::cout << std::setw(width) << "OpenMP threads: 1" << std::endl;
#endif
    std::cout << "Metrics [total]" << std::endl;
    std::cout << std::setw(width) << "Memory peak: " << printMemory(peak) << endl;

    std::cout << std::setw(width) << "CPU usr time: " << printTime(cpu_user_time_s) << std::endl;
    std::cout << std::setw(width) << "CPU sys time: " << printTime(cpu_system_time_s) << std::endl;
    std::cout << std::setw(width) << "CPU sum time: " << printTime(cpu_user_time_s + cpu_system_time_s) << std::endl;
    std::cout << std::setw(width) << "Real time: " << printTime(real_time_s) << std::endl;
    if (repeat > 1)
    {
        std::cout << "Metrics [1 iter, average]" << std::endl;
        const auto div = static_cast<double>(repeat);
        std::cout << std::setw(width) << "CPU usr time: " << printTime(cpu_user_time_s / div) << std::endl;
        std::cout << std::setw(width) << "CPU sys time: " << printTime(cpu_system_time_s / div) << std::endl;
        std::cout << std::setw(width) << "CPU sum time: " << printTime((cpu_user_time_s + cpu_system_time_s) / div) << std::endl;
        std::cout << std::setw(width) << "Real time: " << printTime(real_time_s / div) << std::endl;
    }
}

int main(int argc, char* argv[])
{
#if !defined(_WIN32)
    const auto beginWallClock = std::chrono::steady_clock::now();
    const auto beginTimestamp = UT::tools::getCPUTimestamp();
#endif

#if defined(ANDROID)
    struct sigaction act;
    memset(&act, 0, sizeof(act));
    act.sa_sigaction = sighandler;
    act.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &act, NULL);
#endif

    cmdParam params = parseCMD(argc, argv);

    if (params.mSeed)
    {
        raul::random::setGlobalSeed(*params.mSeed);
    }

    printTestConfig();

    // Create profiler
    [[maybe_unused]] raul::Profiler& profiler = raul::Profiler::getInstance();
    std::filebuf fb;
    fb.open(params.mProfileOutput, std::ios::out);
    std::ostream os(&fb);
    if (fb.is_open())
    {
        // JSON format
        if (params.mUseJsonFormat)
        {
            os << "[\n";
        }
        profiler.initialize(&os, true, false, !params.mProfile, params.mUseJsonFormat);
    }
    else
    {
        profiler.initialize(&std::cout, true, false, !params.mProfile, params.mUseJsonFormat);
    }

    // --gtest_filter="type.name"
    ::testing::InitGoogleTest(&argc, argv);
#ifdef _DEBUG
    ::testing::FLAGS_gtest_break_on_failure = true;
#endif
    int res = RUN_ALL_TESTS();

    if (params.mUseJsonFormat)
    {
        os << "}\n]";
    }

#if !defined(_WIN32)
    const auto endTimestamp = UT::tools::getCPUTimestamp();
    const auto endWallClock = std::chrono::steady_clock::now();
    const auto [userTime, systemTime] = UT::tools::getElapsedTime(beginTimestamp, endTimestamp);
    const std::chrono::duration<double> duration = endWallClock - beginWallClock;
    const auto peak = UT::tools::getPeakOfMemory();
    printPerformanceMetrics(::testing::FLAGS_gtest_repeat, duration.count(), userTime, systemTime, peak);
#endif

    return res;
}
