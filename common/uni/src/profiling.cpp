// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <map>
#include <vector>
#include <algorithm>
#include <sys/time.h>

#include "profiling.h"
#include "error.h"
#include "thread_affinity.h"

int OMP_NUM_THREADS = OMP_MAX_NUM_THREADS;

#ifdef _THREAD_SAFE
pthread_mutex_t uniThreadMutex = PTHREAD_MUTEX_INITIALIZER;
#endif
static std::map<std::string, double> time_statistics;
static bool time_statistics_flag = true;
#ifndef _EAGER_LOG
static std::vector<std::string> logs;
#endif

#ifdef _USE_MEM_CHECK
#include "memory_cpu.h"
std::map<std::string, size_t> mem_statistics;
#endif

double ut_time_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double time = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    return time;
}

void ut_time_init()
{
#ifndef _USE_LITE
    UNI_THREAD_SAFE({
        time_statistics.clear();
        time_statistics_flag = true;
    });
#endif
}

void ut_time_start()
{
    UNI_THREAD_SAFE({ time_statistics_flag = true; });
}

void ut_time_stop()
{
    UNI_THREAD_SAFE({ time_statistics_flag = false; });
}

inline std::string ut_profile_log(const std::string &name,
    const std::string &category,
    const int &pid,
    const int &tid,
    const double &start_vs,
    const double &duration_vs)
{
    std::string log = "{\"name\": \"" + name + "\", \"cat\": \"" + category +
        "\", \"ph\": \"X\", \"pid\": \"" + std::to_string(pid) + "\", \"tid\": \"" +
        std::to_string(tid) + "\", \"ts\": " + std::to_string(start_vs) +
        ", \"dur\": " + std::to_string(duration_vs) + "},";
    return log;
}

#ifdef _PROFILE
#define UNI_PROFILE_LOG(...)                       \
    {                                              \
        UNI_THREADID                               \
        UNI_THREAD_SAFE({                          \
            UNI_LOGD("[PROFILE] thread %d ", tid); \
            UNI_LOGD(__VA_ARGS__);                 \
        })                                         \
    }
#else
#define UNI_PROFILE_LOG(...)
#endif

void ut_time_process(
    const std::string &name, const std::string &category, double time_start_ms, double time_end_ms)
{
#ifdef _PROFILE
    UNI_THREADID
    std::string log = ut_profile_log(
        name, category, 0, tid, time_start_ms * 1000, (time_end_ms - time_start_ms) * 1000);
#ifdef _EAGER_LOG
    UNI_PROFILE_LOG("%s\n", log.c_str());
#else
    logs.push_back(log);
#endif
#endif

    if (!time_statistics_flag) {
        return;
    }
#ifdef _PROFILE_STATISTICS
    double duration = time_end_ms - time_start_ms;
    UNI_THREAD_SAFE({
        if (time_statistics.find(category) == time_statistics.end()) {
            time_statistics[category] = duration;
        } else {
            time_statistics[category] += duration;
        }
    });
#endif
}

void ut_time_statistics()
{
#ifndef _EAGER_LOG
    printf("\nFunction Time:\n{\"name\": function name, \"cat\": function category, \"ph\": "
           "function type, \"pid\": process id, \"tid\": thread id, \"ts\": start time(ms), "
           "\"dur\": duration time(vs, gpu will have 1 ms synchronization overhead)\n");
    for (unsigned int i = 0; i < logs.size(); i++) {
        UNI_PROFILE_LOG("%s\n", logs[i].c_str());
    }
#endif
    std::vector<std::pair<std::string, double>> vec(time_statistics.begin(), time_statistics.end());
    sort(vec.begin(), vec.end(),
        [&](const std::pair<std::string, double> &a, const std::pair<std::string, double> &b) {
            return (a.second > b.second);
        });
    printf("\nStatistics Time:\n");
    printf("%32s %16s\n", "category", "time(ms)");
    for (unsigned int i = 0; i < vec.size(); ++i) {
        printf("%32s %16.4lf\n", vec[i].first.c_str(), vec[i].second);
    }
    printf("\n");
}

#undef UNI_PROFILE_LOG
