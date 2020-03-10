// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_THREAD_AFFINITY
#define _H_THREAD_AFFINITY

#include <sys/syscall.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <sched.h>
#include "sys.h"


typedef enum {
    CPU_AFFINITY_LOW_POWER = 0,
    CPU_AFFINITY_HIGH_PERFORMANCE = 1
} CpuAffinityPolicy;

inline const char * const *CpuAffinityPolicyNames() {
    static const char * const names[] = {
        "CPU_AFFINITY_LOW_POWER",
        "CPU_AFFINITY_HIGH_PERFORMANCE"
    };
    return names;
}
inline const CpuAffinityPolicy* CpuAffinityPolicies() {
    static const CpuAffinityPolicy policies[] = {
        CPU_AFFINITY_LOW_POWER,
        CPU_AFFINITY_HIGH_PERFORMANCE
    };
    return policies;
}

inline int get_cpus_num() {
    const int bufferSize = 1024;
    char buffer[bufferSize];
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp) {
        return 1;
    }

    int cpuNum = 0;
    while (!feof(fp)) {
        char* status = fgets(buffer, bufferSize, fp);
        if (!status)
            break;

        if (memcmp(buffer, "processor", 9) == 0) {
            cpuNum++;
        }
    }
    fclose(fp);
    return cpuNum;
}

inline void get_cpus_arch(Arch *archs)
{
    const int bufferSize = 1024;
    char buffer[bufferSize];
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp) {
        *archs = CPU_GENERAL;
        return;
    }

    int cpuid = 0;
    while (!feof(fp)) {
        char* status = fgets(buffer, bufferSize, fp);
        if (!status)
            break;

        if (memcmp(buffer, "CPU part", 8) == 0) {
            Arch arch = ARM_V8;
            int id = 0;
            sscanf(buffer, "CPU part\t: %x", &id);
            switch (id) {
                case 0xd05:
                    arch = ARM_A55;
                    break;
                case 0xd40:
                    arch = ARM_A76;
                    break;
                default:
                    printf("[WARNING] unknown CPU %d arch %x\n Default to ARM_V8\n", cpuid, id);
                    break;
            }
            archs[cpuid++] = arch;
        }
    }
    fclose(fp);
}

inline long get_cpu_freq(int cpuid) {
    char path[256];
    FILE *fp = NULL;
    if (fp == NULL) {
        snprintf(path,
                 sizeof(path),
                 "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state",
                 cpuid);
        fp = fopen(path, "rb");
    }
    if (fp == NULL) {
        snprintf(path,
                 sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state",
                 cpuid);
        fp = fopen(path, "rb");
    }
    if (fp == NULL) {
        snprintf(path,
                 sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",
                 cpuid);
        fp = fopen(path, "rb");
    }

    long maxFrequency = -1;
    if (fp == NULL) {
        printf("[WARNING] can not get CPU max frequency\n");
    } else {
        fscanf(fp, "%ld", &maxFrequency);
        fclose(fp);
    }
    return maxFrequency;
}

inline void get_cpus_freq(long *freqs, int cpuNum) {
    for (int i = 0; i < cpuNum; i++) {
        freqs[i] = get_cpu_freq(i);
    }
}

inline void sort_cpus_by_arch_freq(Arch *archs, long *freqs, int *cpuids, int cpuNum) {
    for (int i = 0; i < cpuNum; i++) {
        cpuids[i] = i;
    }

    for (int i = 1; i < cpuNum; i++) {
        for (int j = i - 1; j >= 0; j--) {
            if (archs[j+1] < archs[j]) {
                Arch tmpArch = archs[j];
                archs[j] = archs[j+1];
                archs[j+1] = tmpArch;
                long tmpFreq = freqs[j];
                freqs[j] = freqs[j+1];
                freqs[j+1] = tmpFreq;
                int tmpCpuid = cpuids[j];
                cpuids[j] = cpuids[j+1];
                cpuids[j+1] = tmpCpuid;
                continue;
            }
            if (archs[j+1] == archs[j]) {
                if (freqs[j+1] < freqs[j]) {
                    Arch tmpArch = archs[j];
                    archs[j] = archs[j+1];
                    archs[j+1] = tmpArch;
                    long tmpFreq = freqs[j];
                    freqs[j] = freqs[j+1];
                    freqs[j+1] = tmpFreq;
                    int tmpCpuid = cpuids[j];
                    cpuids[j] = cpuids[j+1];
                    cpuids[j+1] = tmpCpuid;
                    continue;
                }
                if (freqs[j+1] >= freqs[j]) {
                    continue;
                }
            }
            if (archs[j+1] > archs[j]) {
                continue;
            }
        }
    }
}

inline int set_thread_affinity(int cpuid) {
#ifdef __GLIBC__
    pid_t tid = syscall(SYS_gettid);
#else
    pid_t tid = gettid();
#endif
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpuid, &mask);
    int status = syscall(__NR_sched_setaffinity, tid, sizeof(mask), &mask);
    if (status) {
        printf("[WARNING] fail to set affinity %d\n", status);
        return -1;
    }
    return 0;
}

inline CpuAffinityPolicy  thread_affinity_get_policy_by_name(const char *name)
{
    int nameLength = strlen(name);
    for (int i = 0; i < 2; i++) {
        const char *target = CpuAffinityPolicyNames()[i];
        int targetLength = strlen(target);
        if (nameLength < targetLength) continue;
        int match = 1;
        for (int j = 0; j < targetLength; j++) {
            if (name[j] == target[j] || name[j] == target[j] + 32) {
                continue;
            } else {
                match = 0;
                break;
            }
        }
        if (match) {
            return CpuAffinityPolicies()[i];
        }
    }
    return CPU_AFFINITY_HIGH_PERFORMANCE;
}

inline void thread_affinity_init(int *cpuNum, Arch **archs, int **cpuids) {
    *cpuNum = get_cpus_num();
    *archs  = (Arch *)malloc(sizeof(Arch) * (*cpuNum));
    *cpuids = (int *)malloc(sizeof(int) * (*cpuNum));
    long *freqs  = (long *)malloc(sizeof(long) * (*cpuNum));
    get_cpus_arch(*archs);
    get_cpus_freq(freqs, *cpuNum);
    sort_cpus_by_arch_freq(*archs, freqs, *cpuids, *cpuNum);
    free(freqs);
}

inline Arch thread_affinity_set_by_policy(int cpuNum, Arch *archs, int *cpuids, CpuAffinityPolicy policy, int threadId) {
    if (threadId >= cpuNum) {
        printf("[WARNING] can not allocate more cores for thread %d\n", threadId);
        return CPU_GENERAL;
    }

    int cpuid;
    Arch arch;
    switch (policy) {
        case CPU_AFFINITY_LOW_POWER: {
            cpuid = cpuids[threadId];
            arch = archs[threadId];
            break;
        }
        case CPU_AFFINITY_HIGH_PERFORMANCE: {
            cpuid = cpuids[cpuNum-1-threadId];
            arch = archs[cpuNum-1-threadId];
            break;
        }
        default: {
            cpuid = cpuids[cpuNum-1-threadId];
            arch = archs[cpuNum-1-threadId];
            break;
        }
    }
    set_thread_affinity(cpuid);
    return arch;
}

inline void thread_affinity_set_by_arch(int cpuNum, Arch *archs, int *cpuids, Arch arch, int threadId)
{
    if (threadId >= cpuNum) {
        printf("[WARNING] can not allocate more cores for thread %d\n", threadId);
        return;
    }
    int count = 0;
    int cpuid = -1;
    for (int i=0; i < cpuNum; i++) {
        if (archs[i] == arch) {
            if (count == threadId) {
                cpuid = cpuids[i];
                break;
            } else {
                count++;
            }
        }
    }
    if (cpuid != -1) {
        set_thread_affinity(cpuid);
    } else {
        printf("[WARNING] there is not enough %d arch cores for thread %d", arch, threadId);
    }
}

inline void thread_affinity_destroy(int *cpuNum, Arch **archs, int **cpuids) {
    if (*cpuids != NULL) {
        free(*cpuids);
        *cpuids = NULL;
    }
    if (*archs != NULL) {
        free(*archs);
        *archs = NULL;
    }
    *cpuNum = 0;
}
#endif
