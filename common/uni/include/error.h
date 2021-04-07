// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_ERROR
#define _H_ERROR

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef _WIN32
#define UNI_THREADID int tid = 0;
#elif defined(__GLIBC__) || defined(__linux__)
#include <sys/syscall.h>
#define UNI_THREADID pid_t tid = syscall(SYS_gettid);
#elif defined(__APPLE__)
#include <thread>
#define UNI_THREADID                   \
    uint64_t tid64;                    \
    pthread_threadid_np(NULL, &tid64); \
    pid_t tid = (pid_t)tid64;
#else
#define UNI_THREADID pid_t tid = gettid();
#endif

#ifdef _THREAD_SAFE
#include <pthread.h>
extern pthread_mutex_t uniThreadMutex;
#endif

#ifdef _USE_ANDROID_LOG
#include <android/log.h>
#define UNI_LOGD(...)                                                \
    {                                                                \
        __android_log_print(ANDROID_LOG_DEBUG, "Bolt", __VA_ARGS__); \
        printf(__VA_ARGS__);                                         \
        fflush(stdout);                                              \
    }
#else
#define UNI_LOGD(...)        \
    {                        \
        printf(__VA_ARGS__); \
        fflush(stdout);      \
    }
#endif
#define UNI_EXIT exit(1);

#ifdef __cplusplus
extern "C" {
#endif
#ifdef _THREAD_SAFE
#define UNI_THREAD_SAFE(func)            \
    pthread_mutex_lock(&uniThreadMutex); \
    func;                                \
    pthread_mutex_unlock(&uniThreadMutex);
#else
#define UNI_THREAD_SAFE(func) func;
#endif
#define UNI_CI_LOG(...) printf(__VA_ARGS__);
#define UNI_INFO_LOG(...)                       \
    {                                           \
        UNI_THREADID                            \
        UNI_THREAD_SAFE({                       \
            UNI_LOGD("[INFO] thread %d ", tid); \
            UNI_LOGD(__VA_ARGS__);              \
        })                                      \
    }
#define UNI_WARNING_LOG(...)                                                           \
    {                                                                                  \
        UNI_THREADID                                                                   \
        UNI_THREAD_SAFE({                                                              \
            UNI_LOGD("[WARNING] thread %d file %s line %d ", tid, __FILE__, __LINE__); \
            UNI_LOGD(__VA_ARGS__);                                                     \
        })                                                                             \
    }
#define UNI_ERROR_LOG(...)                                                           \
    {                                                                                \
        UNI_THREADID                                                                 \
        UNI_THREAD_SAFE({                                                            \
            UNI_LOGD("[ERROR] thread %d file %s line %d ", tid, __FILE__, __LINE__); \
            UNI_LOGD(__VA_ARGS__);                                                   \
        })                                                                           \
        UNI_EXIT;                                                                    \
    }
#ifdef _DEBUG
#define UNI_DEBUG_LOG(...)                       \
    {                                            \
        UNI_THREADID                             \
        UNI_THREAD_SAFE({                        \
            UNI_LOGD("[DEBUG] thread %d ", tid); \
            UNI_LOGD(__VA_ARGS__);               \
        })                                       \
    }
#else
#define UNI_DEBUG_LOG(...)
#endif
#define CHECK_REQUIREMENT(status)                 \
    if (!(status)) {                              \
        UNI_ERROR_LOG("requirement mismatch.\n"); \
    }
#define CHECK_STATUS(ee)                                         \
    {                                                            \
        EE status = (ee);                                        \
        if (status != SUCCESS) {                                 \
            UNI_ERROR_LOG("got an error: %s\n", ee2str(status)); \
        }                                                        \
    }

inline void UNI_PROFILE_INFO(const char *name, const char *category, long start, long duration)
{
#ifdef _PROFILE
    int pid = 0;
    UNI_THREADID;
    UNI_THREAD_SAFE({
        UNI_LOGD("[PROFILE] thread %d ", tid);
        UNI_LOGD("{\"name\": \"%s\", \"cat\": \"%s\", \"ph\": \"X\", \"pid\": \"%d\", \"tid\": "
                 "\"%d\", \"ts\": %ld, \"dur\": %ld},\n",
            name, category, pid, tid, start, duration);
    });
#endif
}

typedef enum {
    SUCCESS = 0,
    NULL_POINTER = 1,
    NOT_MATCH = 2,
    NOT_FOUND = 3,
    ALLOC_FAILED = 4,
    NOT_IMPLEMENTED = 50,
    NOT_SUPPORTED = 51,
    GCL_ERROR = 52,
    FILE_ERROR = 53,
    UNKNOWN = 99
} EE;

inline const char *ee2str(EE ee)
{
    const char *ret = 0;
    switch (ee) {
        case SUCCESS:
            ret = "SUCCESS";
            break;
        case NULL_POINTER:
            ret = "Null Pointer";
            break;
        case NOT_MATCH:
            ret = "Not Match";
            break;
        case NOT_FOUND:
            ret = "Not Found";
            break;
        case NOT_IMPLEMENTED:
            ret = "Not Implemented";
            break;
        case NOT_SUPPORTED:
            ret = "Not Supported";
            break;
        case FILE_ERROR:
            ret = "Error with file system";
            break;
        default:
            ret = "Unknown";
            break;
    }
    return ret;
}
#ifdef __cplusplus
}
#endif

#endif
