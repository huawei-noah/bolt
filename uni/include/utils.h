// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_UTILS
#define _H_UTILS

#include <math.h>
#include <string>
#include <sys/time.h>
#include <iostream>

#include "sys.h"
#include "type.h"
#include "error.h"

const Arch UT_ARCH = ARM_A76;

// whether to check right
const int UT_CHECK = 1;

// toop times to benchmark
const int UT_LOOPS = 6;

// init data type
typedef enum UT_RANDOM_TYPE{
    UT_INIT_RANDOM,   // random
    UT_INIT_NEG,      // random & < 0
    UT_INIT_POS,      // random & > 0
    UT_INIT_ZERO      // 0
} UT_RANDOM_TYPE;

// generate random data
template<typename T>
inline T ut_init_s(UT_RANDOM_TYPE type) {
    if (type == UT_INIT_ZERO) {
        return 0;
    }

    T s = (T)0.5;

    if (s == 0.5) {
        s = rand() % 1000 / 1000.0 - 0.5;
    } else {
        s = rand() % 100 - 50;
    }

    if (type == UT_INIT_NEG) {
        s = (s > 0) ? (s * -1) : s;
    }
    if (type == UT_INIT_POS) {
        s = (s < 0) ? (s * -1) : s;
    }
    return s;
}


// generate random array
template<typename T>
inline void ut_init_v(T* A, U32 len, UT_RANDOM_TYPE type) {
    if (A == nullptr)
        return;

    for (U32 i = 0; i < len; i++) {
        A[i] = ut_init_s<T>(type);
    }
}

template<typename T>
inline void ut_set_v(T* A, U32 len, T a) {
    if (A == nullptr)
        return;

    for (U32 i = 0; i < len; i++) {
        A[i] = a;
    }
}


template<typename T>
inline T* ut_sinput_v(U32 len, F16 value) {
    T *A = (T *)malloc(sizeof(T) * len);
    ut_set_v<T>(A, len, value);

    return A;
}


template<typename T>
inline T* ut_input_v(U32 len, UT_RANDOM_TYPE type) {
    T *A = (T *)malloc(sizeof(T) * len);
    ut_init_v<T>(A, len, type);

    return A;
}


// unit test element check
template<typename T>
inline void ut_check_s(T a, T b, T threshold, std::string file, int line) {
    if(! ((a <= b + threshold) && (a >= b - threshold)))
    {
        std::cerr << "[ERROR] check in " << file << " at line " << line << " " \
                  << a << " " << b << std::endl;
        exit(1);
    }
}


// unit test array check
template<typename T>
inline void ut_check_v(T *A, T *B, U32 len, T threshold, std::string file, int line) {
    for (U32 i = 0; i < len; i++)
        ut_check_s<T>(A[i], B[i], threshold, file, line);
}

template<typename T>
inline void ut_check_v(T *A, T val, U32 len, std::string file, int line) {
    for (U32 i = 0; i < len; i++)
        ut_check_s<T>(A[i], val, 0, file, line);
}



// benchmark time
inline double ut_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double time = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    return time;
}
inline double ut_time_s() {
    return ut_time_ms() / 1000.0;
}


// calculate GFLOPS
inline double ut_gflops(double ops, double time_ms) {
    return 1e-6 * ops / time_ms;
}

// uniform log message
template<typename T>
inline void ut_log(char *call, double ops, double time_ms) {
    char buffer[200];
    sprintf(buffer, "%dbit, %s,\tTIME %10.6lfms,\tGFLOPS %10.6lf",
            sizeof(T)*8, call, time_ms,
            ut_gflops(ops, time_ms));
    std::cout << buffer << std::endl;
}

#endif
