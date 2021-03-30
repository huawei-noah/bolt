// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_PROFILING
#define _H_PROFILING

#include <string>

double ut_time_ms();
void ut_time_init();
void ut_time_process(
    const std::string &name, const std::string &category, double time_start_ms, double time_end_ms);
void ut_time_statistics();

#ifdef _PROFILE_STATISTICS
#define UNI_TIME_INIT ut_time_init();
#define UNI_TIME_STATISTICS ut_time_statistics();
#else
#define UNI_TIME_INIT
#define UNI_TIME_STATISTICS
#endif

#ifdef _PROFILE
#define UNI_PROFILE(func, name, category)        \
    double profile_time_start_ms = ut_time_ms(); \
    func;                                        \
    double profile_time_end_ms = ut_time_ms();   \
    ut_time_process(name, category, profile_time_start_ms, profile_time_end_ms);
#else
#define UNI_PROFILE(func, name, category) func;

#endif
#endif  // _H_PROFILING
