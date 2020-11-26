// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_UT_UTIL
#define _H_UT_UTIL

#include <math.h>
#include <string>
#include <sys/time.h>
#include <iostream>
#include <cstring>

#include "sys.h"
#include "types.h"
#include "error.h"

#if defined(_USE_NEON)
const Arch UT_ARCH = ARM_A76;
#elif defined(_USE_X86)
const Arch UT_ARCH = X86_AVX2;
#else
const Arch UT_ARCH = CPU_GENERAL;
#endif

// whether to check right
const int UT_CHECK = 1;

// loop times to benchmark
const int UT_LOOPS = 6;

// init data type
typedef enum UT_RANDOM_TYPE {
    UT_INIT_RANDOM,  // random
    UT_INIT_NEG,     // random & < 0
    UT_INIT_POS,     // random & > 0
    UT_INIT_ZERO     // 0
} UT_RANDOM_TYPE;

// generate random data
inline F32 ut_init_s(DataType dt, UT_RANDOM_TYPE type)
{
    if (type == UT_INIT_ZERO) {
        return 0;
    }

    F32 s = 0;
    if (0
#ifdef _USE_FP32
        || dt == DT_F32
#endif
#ifdef _USE_FP16
        || dt == DT_F16
#endif
    ) {
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
inline void ut_init_v(U8 *data, U32 len, DataType dt, UT_RANDOM_TYPE type)
{
    if (data == nullptr) {
        return;
    }

    for (U32 i = 0; i < len; i++) {
        switch (dt) {
#ifdef _USE_FP32
            case DT_F32: {
                F32 *dataPtr = (F32 *)data;
                dataPtr[i] = ut_init_s(dt, type);
                break;
            }
#endif
#ifdef _USE_FP16
            case DT_F16: {
                F16 *dataPtr = (F16 *)data;
                dataPtr[i] = ut_init_s(dt, type);
                break;
            }
#endif
            case DT_I32: {
                I32 *dataPtr = (I32 *)data;
                dataPtr[i] = ut_init_s(dt, type);
                break;
            }
            case DT_U32: {
                U32 *dataPtr = (U32 *)data;
                dataPtr[i] = ut_init_s(dt, type);
                break;
            }
            case DT_I8: {
                INT8 *dataPtr = (INT8 *)data;
                dataPtr[i] = ut_init_s(dt, type);
                break;
            }
            case DT_BIN11: {
                BIN8 *dataPtr = (BIN8 *)data;
                dataPtr[i] = ut_init_s(dt, type);
                break;
            }
            case DT_BIN01: {
                BIN8 *dataPtr = (BIN8 *)data;
                dataPtr[i] = ut_init_s(dt, type);
                break;
            }
            default:
                UNI_ERROR_LOG("unsupported data type in ut_init_v\n");
        }
    }
}

inline U8 *ut_input_v(U32 len, DataType dt, UT_RANDOM_TYPE type)
{
    U8 *data = (U8 *)malloc(len * bytesOf(dt));
    ut_init_v(data, len, dt, type);

    return data;
}

// unit test element check
inline void ut_check_s(F32 a, F32 b, F32 threshold, std::string file, int line, int index)
{
    if (!((a <= b + threshold) && (a >= b - threshold))) {
        UNI_ERROR_LOG("check in %s at line %d, %d @ %f %f\n", file.c_str(), line, index, a, b);
    }
}

// unit test array check
inline void ut_check_v(
    void *A, void *B, U32 len, DataType dt, F32 threshold, std::string file, int line)
{
    F32 a = 0, b = 0;
    for (U32 i = 0; i < len; i++) {
        switch (dt) {
#ifdef _USE_FP32
            case DT_F32:
                a = ((F32 *)A)[i];
                b = ((F32 *)B)[i];
                break;
#endif
#ifdef _USE_FP16
            case DT_F16:
                a = ((F16 *)A)[i];
                b = ((F16 *)B)[i];
                break;
#endif
            case DT_I32:
                a = ((I32 *)A)[i];
                b = ((I32 *)B)[i];
                break;
            case DT_U32:
                a = ((U32 *)A)[i];
                b = ((U32 *)B)[i];
                break;
            case DT_I8:
                a = ((INT8 *)A)[i];
                b = ((INT8 *)B)[i];
                break;
            case DT_BIN11:
                a = ((BIN8 *)A)[i];
                b = ((BIN8 *)B)[i];
                break;
            case DT_BIN01:
                a = ((BIN8 *)A)[i];
                b = ((BIN8 *)B)[i];
                break;
            default:
                UNI_ERROR_LOG("unsupported data type in ut_check_v(array, array)\n");
        }
        ut_check_s(a, b, threshold, file, line, i);
    }
}

inline void ut_check_v(void *A, F32 val, U32 len, DataType dt, std::string file, int line)
{
    F32 a;
    for (U32 i = 0; i < len; i++) {
        switch (dt) {
#ifdef _USE_FP32
            case DT_F32:
                a = ((F32 *)A)[i];
                break;
#endif
#ifdef _USE_FP16
            case DT_F16:
                a = ((F16 *)A)[i];
                break;
#endif
            case DT_I32:
                a = ((I32 *)A)[i];
                break;
            case DT_U32:
                a = ((U32 *)A)[i];
                break;
            case DT_BIN11:
                a = ((BIN8 *)A)[i];
                break;
            case DT_BIN01:
                a = ((BIN8 *)A)[i];
                break;
            default:
                UNI_ERROR_LOG("unsupported data type in ut_check_v(array, scalar)\n");
        }
        ut_check_s(a, val, 0, file, line, i);
    }
}

inline void ut_check_a(void *A, void *B, U32 len, DataType dt)
{
    U32 e0, e1, e2, e3, e4, e5, e6;
    e0 = 0;
    e1 = 0;
    e2 = 0;
    e3 = 0;
    e4 = 0;
    e5 = 0;
    e6 = 0;
    F32 a = 1, b = 0, diff;
    F32 d0, d1, d2, d3, d4, d5;
    F32 maxrel = -1.0;
    F32 maxabs = -1.0;
    F32 max_a0, max_b0, max_a1, max_b1;
    U32 max_n0, max_n1;
    switch (dt) {
        case DT_F32:
            d0 = 1;
            d1 = 0.1;
            d2 = 0.01;
            d3 = 0.001;
            d4 = 0.0001;
            d5 = 0.00001;
            break;
#ifdef _USE_FP16
        case DT_F16:
            d0 = 1;
            d1 = 0.1;
            d2 = 0.01;
            d3 = 0.001;
            d4 = 0.0001;
            d5 = 0.00001;
            break;
#endif
        case DT_U8:
            d0 = 30;
            d1 = 20;
            d2 = 10;
            d3 = 5;
            d4 = 3;
            d5 = 2;
            break;
        default:
            UNI_ERROR_LOG("unsupported data type in ut_check_a(array, array)\n");
    }

    for (U32 i = 0; i < len; i++) {
        switch (dt) {
            case DT_F32:
                a = ((F32 *)A)[i];
                b = ((F32 *)B)[i];
                break;
#ifdef _USE_FP16
            case DT_F16:
                a = ((F16 *)A)[i];
                b = ((F16 *)B)[i];
                break;
#endif
            case DT_U8:
                a = ((U8 *)A)[i];
                b = ((U8 *)B)[i];
                break;
            default:
                break;
        }

        if (UNI_ISNAN((float)a) || UNI_ISINF((float)a)) {
            UNI_ERROR_LOG("nan or inf value in ut_check_a of input A\n");
            return;
        }
        if (UNI_ISNAN((float)b) || UNI_ISINF((float)b)) {
            UNI_ERROR_LOG("nan or inf value in ut_check_a of input B\n");
            return;
        }

        diff = a - b;
        if (diff < 0) {
            diff = -diff;
        }
        if (diff > maxabs) {
            maxabs = diff;
            max_a0 = a;
            max_b0 = b;
            max_n0 = i;
        }
        F32 tmp = diff * 2 / (a + b + 0.000001);
        if (tmp > maxrel) {
            maxrel = tmp;
            max_a1 = a;
            max_b1 = b;
            max_n1 = i;
        }
        if (diff >= d0) {
            e0++;
            continue;
        }
        if (diff >= d1) {
            e1++;
            continue;
        }
        if (diff >= d2) {
            e2++;
            continue;
        }
        if (diff >= d3) {
            e3++;
            continue;
        }
        if (diff >= d4) {
            e4++;
            continue;
        }
        if (diff >= d5) {
            e5++;
            continue;
        }
        e6++;
    }
    std::cout << "abs(diff) >= " << std::scientific << d0 << " number = " << std::dec << e0
              << std::endl;
    std::cout << "abs(diff) >= " << std::scientific << d1 << " number = " << std::dec << e1
              << std::endl;
    std::cout << "abs(diff) >= " << std::scientific << d2 << " number = " << std::dec << e2
              << std::endl;
    std::cout << "abs(diff) >= " << std::scientific << d3 << " number = " << std::dec << e3
              << std::endl;
    std::cout << "abs(diff) >= " << std::scientific << d4 << " number = " << std::dec << e4
              << std::endl;
    std::cout << "abs(diff) >= " << std::scientific << d5 << " number = " << std::dec << e5
              << std::endl;
    std::cout << "others number = " << e6 << std::endl;
    std::cout << "number " << max_n0 << " is "
              << "maxabs = " << std::fixed << maxabs << " a = " << max_a0 << " b = " << max_b0
              << std::endl;
    std::cout << "number " << max_n1 << " is "
              << "maxrel = " << std::fixed << maxrel << " a = " << max_a1 << " b = " << max_b1
              << std::endl;
}
// benchmark time
inline double ut_time_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double time = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    return time;
}

inline double ut_time_s()
{
    return ut_time_ms() / 1000.0;
}

// calculate GFLOPS
inline double ut_gflops(double ops, double time_ms)
{
    return 1e-6 * ops / time_ms;
}

// uniform log message
inline void ut_log(DataType dt, char *call, double ops, double time_ms)
{
    UNI_INFO_LOG("%ubit, %s,\tTIME %10.6lfms,\tGFLOPS %10.6lf\n", (U32)bytesOf(dt) * 8, call,
        time_ms, ut_gflops(ops, time_ms));
}

inline void initialization_zero(void *ptr, int bytesOfNum)
{
    memset(ptr, 0, bytesOfNum);
    return;
}

#endif
