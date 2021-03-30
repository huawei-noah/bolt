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

#ifdef _THREAD_SAFE
pthread_mutex_t uniThreadMutex = PTHREAD_MUTEX_INITIALIZER;
#endif

double ut_time_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double time = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    return time;
}

std::map<std::string, double> time_statistics;

void ut_time_init()
{
    UNI_THREAD_SAFE(time_statistics.clear());
}

void ut_time_process(
    const std::string &name, const std::string &category, double time_start_ms, double time_end_ms)
{
#ifdef _PROFILE
    UNI_PROFILE_INFO(
        name.c_str(), category.c_str(), time_start_ms * 1000, (time_end_ms - time_start_ms) * 1000);
#endif
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
    std::vector<std::pair<std::string, double>> vec(time_statistics.begin(), time_statistics.end());
    sort(vec.begin(), vec.end(),
        [&](const std::pair<std::string, double> &a, const std::pair<std::string, double> &b) {
            return (a.second > b.second);
        });
    for (int i = 0; i < (int)vec.size(); ++i) {
        UNI_INFO_LOG("%s\t%lfms\n", vec[i].first.c_str(), vec[i].second);
    }
}
