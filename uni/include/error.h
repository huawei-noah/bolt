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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#if defined(_DEBUG) && defined(__ANDROID__)
#include <android/log.h>
#define LOG_TAG "Bolt"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(...) printf(__VA_ARGS__)
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_DEBUG) && defined(__ANDROID__)
    #define CHECK_REQUIREMENT(status) if (!(status)) {\
                LOGD("[ERROR] %s %s line %d requirement mismatch\n", __FILE__, __func__, __LINE__);\
            }
#else
    #define CHECK_REQUIREMENT(status) if (!(status)) {\
                LOGD("[ERROR] %s %s line %d requirement mismatch\n", __FILE__, __func__, __LINE__);\
                exit(1);\
            }
#endif

#if defined(_DEBUG) && defined(__ANDROID__)
    #define CHECK_STATUS(ee) {\
                EE status = (ee); \
                if (status != SUCCESS) {\
                    LOGD("[ERROR] %s %s line %d got an error: %s\n", __FILE__, __func__, __LINE__, ee2str(status));\
                }\
            }
#else
    #define CHECK_STATUS(ee) {\
                EE status = (ee); \
                if (status != SUCCESS) {\
                    LOGD("[ERROR] %s %s line %d got an error: %s\n", __FILE__, __func__, __LINE__, ee2str(status));\
                    exit(1);\
                }\
            }
#endif

    typedef enum {
        SUCCESS = 0,
        NULL_POINTER = 1,
        NOT_MATCH = 2,
        NOT_FOUND = 3,
        ALLOC_FAILED = 4,
        NOT_IMPLEMENTED = 50,
        NOT_SUPPORTED = 51,
        GCL_ERROR = 52,
        UNKNOWN = 99
    } EE;

    inline const char* ee2str(EE ee) {
        const char* ret = 0;
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
            default:
                ret = "Unknown";
                break;
        }
        return ret;
    }

    #define CI_info(x) do { std::cout << x << std::endl; } while (0)
    
    #ifdef _DEBUG
    #define DEBUG_info(x) do { std::cout << x << std::endl; } while (0)
    #define DEBUG_info_s(x) do { std::cout << x << " "; } while (0)
    #else
    #define DEBUG_info(x) do { } while (0)
    #define DEBUG_info_s(x) do { } while (0)
    #endif

#ifdef __cplusplus
}
#endif

#endif
