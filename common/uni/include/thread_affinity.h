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

#ifndef _WIN32
#include <sys/syscall.h>
#include <sched.h>
#endif
#include <unistd.h>
#include <string.h>
#include "sys.h"
#include "error.h"
#include "data_type.h"

#ifdef _USE_X86
#define __cpuid(data, eaxIn, ecxIn)                                                   \
    __asm__ __volatile__("cpuid\n"                                                    \
                         : "=a"(data[0]), "=b"(data[1]), "=c"(data[2]), "=d"(data[3]) \
                         : "0"(eaxIn), "2"(ecxIn))
#endif

const int CPU_MAX_NUMBER = 128;
#ifdef _USE_OPENMP
const int OMP_NUM_THREADS = 2;
#else
const int OMP_NUM_THREADS = 1;
#endif

typedef enum {
    AFFINITY_CPU_LOW_POWER = 0,
    AFFINITY_CPU_HIGH_PERFORMANCE = 1,
    AFFINITY_GPU = 2
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
        "CPU_AFFINITY_LOW_POWER", "CPU_AFFINITY_HIGH_PERFORMANCE", "GPU"};
    return names;
}

inline const AffinityPolicy *AffinityPolicies()
{
    static const AffinityPolicy policies[] = {
        AFFINITY_CPU_LOW_POWER, AFFINITY_CPU_HIGH_PERFORMANCE, AFFINITY_GPU};
    return policies;
}

inline int get_cpus_num()
{
    int cpuNum = 0;
#if defined(__APPLE__)
    cpuNum = 6;
#elif defined(_WIN32)
    cpuNum = atoi(getenv("NUMBER_OF_PROCESSORS"));
#else
    cpuNum = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    if (cpuNum == 0) {
        UNI_ERROR_LOG("can not get cpu processor number.\n");
    }
    if (cpuNum > CPU_MAX_NUMBER) {
        cpuNum = CPU_MAX_NUMBER;
    }
    return cpuNum;
}

inline void get_cpus_arch(Arch *archs, int cpuNum)
{
#ifdef __APPLE__
    for (int cpuid = 0; cpuid < cpuNum; cpuid++) {
        archs[cpuid] = ARM_A76;
    }
    return;
#endif
    *archs = CPU_GENERAL;

#if defined(_USE_FP32) && defined(_USE_X86)
    //_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    U32 data[4] = {};
    const U32 &ebx = data[1];
    const U32 &ecx = data[2];

    const U32 osxsave = 1U << 0;
    const U32 avx = 1U << 1;
    const U32 avx2 = 1U << 2;

    U32 cpuArch = 0;
    __cpuid(data, 0, 0);
    __cpuid(data, 1, 0);
    if (ecx & (1U << 27)) {
        cpuArch |= osxsave;
    }
    if (cpuArch & osxsave) {
        if (ecx & (1U << 28)) {
            cpuArch |= avx;
        }
    }
    __cpuid(data, 7, 0);
    if ((cpuArch & avx) && (ebx & (1U << 5))) {
        cpuArch |= avx2;
    }

    if (cpuArch & avx2) {
        archs[0] = X86_AVX2;
    } else {
        UNI_WARNING_LOG("AVX2 is not available, use general implementation.\n");
    }
#endif

    int cpuid = 0;
#ifdef _USE_NEON
    FILE *fp = fopen("/proc/cpuinfo", "rb");
    if (!fp) {
        return;
    }
    const int bufferSize = 1024;
    char buffer[bufferSize];
    while (!feof(fp)) {
        char *status = fgets(buffer, bufferSize, fp);
        if (!status) {
            break;
        }

        if (memcmp(buffer, "CPU part", 8) == 0) {
            Arch arch = ARM_V8;
            int id = 0;
            sscanf(buffer, "CPU part\t: %x", &id);
            switch (id) {
                case 0xc07:
                    arch = ARM_V7;
                    break;
                case 0xc0f:
                    arch = ARM_V7;
                    break;
                case 0xd01:
                    arch = ARM_A76;
                    break;
                case 0xd03:
                    arch = ARM_V8;
                    break;
                case 0xd04:
                    arch = ARM_V7;
                    break;
                case 0xd05:
                    arch = ARM_A55;
                    break;
                case 0xd07:
                    arch = ARM_V8;
                    break;
                case 0xd08:
                    arch = ARM_V8;
                    break;
                case 0xd09:
                    arch = ARM_V8;
                    break;
                case 0xd0a:
                    arch = ARM_A76;
                    break;
                case 0xd0b:
                    arch = ARM_A76;
                    break;
                case 0xd0d:
                    arch = ARM_A76;
                    break;
                case 0xd40:
                    arch = ARM_A76;
                    break;
                case 0xd41:
                    arch = ARM_A76;
                    break;
                case 0xd44:
                    arch = ARM_A76;
                    break;
                case 0x804:
                    arch = ARM_A76;
                    break;
                case 0x805:
                    arch = ARM_A55;
                    break;
                case 0x802:
                    arch = ARM_A76;
                    break;
                case 0x803:
                    arch = ARM_A55;
                    break;
                case 0x801:
                    arch = ARM_V8;
                    break;
                case 0x800:
                    arch = ARM_V8;
                    break;
                case 0x205:
                    arch = ARM_V8;
                    break;
                default:
                    UNI_WARNING_LOG("unknown CPU %d arch %x, set to ARM_V8\n", cpuid, id);
                    break;
            }
            archs[cpuid++] = arch;
        }
    }
    fclose(fp);
#endif
    for (; cpuid < cpuNum; cpuid++) {
        archs[cpuid] = archs[0];
    }
}

inline long get_cpu_freq(int cpuid)
{
    char path[256];
    FILE *fp = NULL;
    if (fp == NULL) {
        snprintf(
            path, sizeof(path), "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuid);
        fp = fopen(path, "rb");
    }
    if (fp == NULL) {
        snprintf(
            path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state", cpuid);
        fp = fopen(path, "rb");
    }
    if (fp == NULL) {
        snprintf(
            path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuid);
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

inline void get_cpus_freq(long *freqs, int cpuNum)
{
    for (int i = 0; i < cpuNum; i++) {
        freqs[i] = get_cpu_freq(i);
    }
}

inline void get_cpus_occupy(CpuStat *cpuStat, float *cpuOccupy, int cpuNum)
{
    const int bufferSize = 1024;
    char buffer[bufferSize];
    char name[32];
    unsigned long user, nice, system, idle, iowait, irq, softirq, total;
    FILE *fp = fopen("/proc/stat", "rb");
    if (!fp) {
        for (int i = 0; i < cpuNum; i++) {
            cpuOccupy[i] = 0;
        }
        return;
    }

    // skip total statistics
    fgets(buffer, bufferSize, fp);

    for (int i = 0; i < cpuNum; i++) {
        fgets(buffer, bufferSize, fp);
        sscanf(buffer, "%s %lu %lu %lu %lu %lu %lu %lu", name, &user, &nice, &system, &idle,
            &iowait, &irq, &softirq);
        total = user + nice + system + idle + iowait + irq + softirq;
        cpuOccupy[i] = 0;
        if (cpuStat[i].total != 0) {
            float idleTime = idle - cpuStat[i].idle;
            float totalTime = total - cpuStat[i].total;
            if (totalTime != 0) {
                cpuOccupy[i] = 1.0 - idleTime / totalTime;
            }
        }
        cpuStat[i].idle = idle;
        cpuStat[i].total = total;
    }
    fclose(fp);
}

inline void swap_variable(void *a, void *b, const int size)
{
    char buffer[size];
    memcpy(buffer, a, size);
    memcpy(a, b, size);
    memcpy(b, buffer, size);
}

inline void disable_cpus(float *occupys, int *cpuids, int cpuNum, float cpuOccupyMax)
{
    for (int i = 0; i < cpuNum; i++) {
        if (occupys[i] > cpuOccupyMax) {
            cpuids[i] = -1;
        }
    }
}

inline void sort_cpus_by_arch_freq_occupy(
    Arch *archs, long *freqs, float *occupys, int *cpuids, int cpuNum, float cpuOccupyMax)
{
    for (int i = 0; i < cpuNum; i++) {
        cpuids[i] = i;
    }

    for (int i = 1; i < cpuNum; i++) {
        for (int j = i - 1; j >= 0; j--) {
            if (archs[j + 1] < archs[j]) {
                swap_variable(&archs[j], &archs[j + 1], sizeof(Arch));
                swap_variable(&freqs[j], &freqs[j + 1], sizeof(long));
                swap_variable(&cpuids[j], &cpuids[j + 1], sizeof(int));
                swap_variable(&occupys[j], &occupys[j + 1], sizeof(float));
                continue;
            }
            if (archs[j + 1] == archs[j]) {
                if (freqs[j + 1] < freqs[j]) {
                    swap_variable(&archs[j], &archs[j + 1], sizeof(Arch));
                    swap_variable(&freqs[j], &freqs[j + 1], sizeof(long));
                    swap_variable(&cpuids[j], &cpuids[j + 1], sizeof(int));
                    swap_variable(&occupys[j], &occupys[j + 1], sizeof(float));
                    continue;
                }
                if (freqs[j + 1] >= freqs[j]) {
                    continue;
                }
            }
            if (archs[j + 1] > archs[j]) {
                continue;
            }
        }
    }
    disable_cpus(occupys, cpuids, cpuNum, cpuOccupyMax);
}

inline int set_thread_affinity(int threadid, const int *cpuids, int num)
{
#if !(defined(__APPLE__) || defined(_WIN32))
    UNI_THREADID;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int i = 0; i < num; i++) {
        UNI_DEBUG_LOG("bind thread %d to core %d\n", threadid, cpuids[i]);
        CPU_SET(cpuids[i], &mask);
    }
    int status = syscall(__NR_sched_setaffinity, tid, sizeof(mask), &mask);
    if (status) {
        UNI_WARNING_LOG("fail to set affinity %d\n", status);
        return -1;
    }
#endif
    return 0;
}

inline AffinityPolicy thread_affinity_get_policy_by_name(const char *name)
{
    int nameLength = strlen(name);
    for (int i = 0; i < 3; i++) {
        const char *target = AffinityPolicyNames()[i];
        int targetLength = strlen(target);
        if (nameLength < targetLength) {
            continue;
        }
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
            return AffinityPolicies()[i];
        }
    }
    return AFFINITY_CPU_HIGH_PERFORMANCE;
}

inline Arch thread_affinity_set_by_policy(
    Arch *archs, int *cpuids, int cpuNum, AffinityPolicy policy, int threadId)
{
    if (threadId >= cpuNum) {
        UNI_WARNING_LOG("can not allocate more cores for thread %d\n", threadId);
        return CPU_GENERAL;
    }
    if (policy == AFFINITY_GPU) {
        return MALI;
    }
#ifndef _USE_OPENMP
    int cpuid;
    Arch arch;
    int i = cpuNum - 1 - threadId;
    switch (policy) {
        case AFFINITY_CPU_LOW_POWER: {
            i = threadId;
            while (cpuids[i] == -1 && i < cpuNum - 1) {
                i++;
            }
            break;
        }
        case AFFINITY_CPU_HIGH_PERFORMANCE: {
            i = cpuNum - 1 - threadId;
            while (cpuids[i] == -1 && i > 0) {
                i--;
            }
            break;
        }
        default: {
            break;
        }
    }
    cpuid = cpuids[i];
    arch = archs[i];
    set_thread_affinity(threadId, &cpuid, 1);
#else
    int index = 0;
    for (int i = 0; i < cpuNum; i++) {
        if (policy == AFFINITY_CPU_LOW_POWER && archs[index] > archs[i]) {
            index = i;
        }
        if (policy == AFFINITY_CPU_HIGH_PERFORMANCE && archs[index] < archs[i]) {
            index = i;
        }
    }
    int count = 0;
    int candidates[CPU_MAX_NUMBER];
    for (int i = 0; i < cpuNum; i++) {
        if (archs[index] == archs[i]) {
            candidates[count++] = i;
        }
    }
    set_thread_affinity(threadId, candidates, count);
    Arch arch = archs[index];
#endif
    return arch;
}

inline void thread_affinity_set_by_arch(
    Arch *archs, int *cpuids, int cpuNum, Arch arch, int threadId)
{
    if (threadId >= cpuNum) {
        UNI_WARNING_LOG("can not allocate more cores for thread %d\n", threadId);
        return;
    }
    if (IS_MALI_GPU(arch)) {
        return;
    }
    int count = 0;
    int cpuid = -1;
    for (int i = 0; i < cpuNum; i++) {
        if (archs[i] == arch && cpuids[i] != -1) {
            if (count == threadId) {
                cpuid = cpuids[i];
                break;
            } else {
                count++;
            }
        }
    }
    if (cpuid != -1) {
        set_thread_affinity(threadId, &cpuid, 1);
    } else {
        UNI_WARNING_LOG("there is not enough %d arch cores for thread %d", arch, threadId);
    }
}

inline DeviceInfo get_cpu_info(AffinityPolicy affinityPolicy)
{
    DeviceInfo deviceInfo;
    deviceInfo.affinityPolicy = affinityPolicy;
    deviceInfo.cpuNum = get_cpus_num();
    deviceInfo.maxOccupy = 0.5;
    get_cpus_arch(deviceInfo.archs, deviceInfo.cpuNum);
    get_cpus_freq(deviceInfo.freqs, deviceInfo.cpuNum);
    for (int i = 0; i < deviceInfo.cpuNum; i++) {
        deviceInfo.cpuStats[i].total = 0;
    }
    get_cpus_occupy(deviceInfo.cpuStats, deviceInfo.occupys, deviceInfo.cpuNum);
    return deviceInfo;
}

inline void set_cpu_dynamic(DeviceInfo *deviceInfo, int threadId)
{
    if (deviceInfo->affinityPolicy == AFFINITY_GPU) {
        deviceInfo->schedule = MALI;
        return;
    }
    get_cpus_occupy(deviceInfo->cpuStats, deviceInfo->occupys, deviceInfo->cpuNum);
    sort_cpus_by_arch_freq_occupy(deviceInfo->archs, deviceInfo->freqs, deviceInfo->occupys,
        deviceInfo->cpuids, deviceInfo->cpuNum, deviceInfo->maxOccupy);
    deviceInfo->schedule = thread_affinity_set_by_policy(deviceInfo->archs, deviceInfo->cpuids,
        deviceInfo->cpuNum, deviceInfo->affinityPolicy, threadId);
}
#endif
