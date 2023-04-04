// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_AFFINITY_POLICY
#define _H_AFFINITY_POLICY

#include "sys.h"
#ifdef _USE_OPENMP
#include <omp.h>
#define OMP_MAX_NUM_THREADS \
    (getenv("OMP_NUM_THREADS") == NULL ? omp_get_num_procs() : atoi(getenv("OMP_NUM_THREADS")))
#else
#define OMP_MAX_NUM_THREADS 1
#endif
extern int OMP_NUM_THREADS;
const int CPU_MAX_NUMBER = 128;

typedef enum {
    AFFINITY_CPU = 0,
    AFFINITY_CPU_LOW_POWER = 1,
    AFFINITY_CPU_HIGH_PERFORMANCE = 2,
    AFFINITY_GPU = 3
} AffinityPolicy;

typedef struct CpuStat {
    unsigned long idle;
    unsigned long total;
} CpuStat;

typedef struct DeviceInfo {
    int cpuNum;
    Arch archs[CPU_MAX_NUMBER];
    long freqs[CPU_MAX_NUMBER];
    float occupys[CPU_MAX_NUMBER];
    int cpuids[CPU_MAX_NUMBER];
    CpuStat cpuStats[CPU_MAX_NUMBER];

    float maxOccupy;
    AffinityPolicy affinityPolicy;
    Arch schedule;
} DeviceInfo;

inline const char *const *AffinityPolicyNames()
{
    static const char *const names[] = {
        "CPU", "CPU_AFFINITY_LOW_POWER", "CPU_AFFINITY_HIGH_PERFORMANCE", "GPU"};
    return names;
}

inline const AffinityPolicy *AffinityPolicies()
{
    static const AffinityPolicy policies[] = {
        AFFINITY_CPU, AFFINITY_CPU_LOW_POWER, AFFINITY_CPU_HIGH_PERFORMANCE, AFFINITY_GPU};
    return policies;
}

inline AffinityPolicy thread_affinity_get_policy_by_name(const char *name)
{
    for (int i = 0; i < 4; i++) {
        const char *target = AffinityPolicyNames()[i];
        if (strcmp(target, name) == 0) {
            return AffinityPolicies()[i];
        }
    }
    UNI_WARNING_LOG("can not recognize affinity policy:%s, use default "
                    "value:CPU_AFFINITY_HIGH_PERFORMANCE.\n",
        name);
    return AFFINITY_CPU_HIGH_PERFORMANCE;
}

inline void set_cpu_num_threads(int threadNum)
{
#ifndef _USE_OPENMP
    if (threadNum > 1) {
        UNI_WARNING_LOG("this library not support multi-threads parallel, please rebuild with "
                        "--openmp option.\n");
    }
#endif
    if (threadNum < 0) {
        threadNum = 1;
    }
    if (threadNum > OMP_MAX_NUM_THREADS) {
        threadNum = OMP_MAX_NUM_THREADS;
    }
    OMP_NUM_THREADS = threadNum;
}
#endif
