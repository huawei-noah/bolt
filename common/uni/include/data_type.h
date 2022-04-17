// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_DATA_TYPE
#define _H_DATA_TYPE

#include <bitset>
#include <math.h>
#include <limits.h>
#ifdef _USE_FP16
#include <arm_neon.h>
typedef __fp16 F16;
#endif
#ifdef _USE_X86
#include <immintrin.h>
#include <xmmintrin.h>
#define FTZ _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif
#define _USE_ULTRA_OPTIMIZATION
#include "secure_c_wrapper.h"

typedef int8_t INT8;
typedef uint8_t UINT8;
typedef unsigned char U8;
typedef const unsigned char CU8;
typedef char I8;
typedef const char CI8;
typedef unsigned int U32;
typedef const unsigned int CU32;
typedef int32_t I32;
typedef const int CI32;
typedef float F32;
typedef double F64;
typedef int64_t I64;
typedef uint64_t U64;
typedef unsigned char BIN8;

typedef enum {
    DT_U8 = 0,
    DT_I8 = 1,
    DT_U32 = 2,
    DT_I32 = 3,
    DT_F16 = 4,
    DT_F16_8Q = 5,
    DT_F32 = 6,
    DT_BIN01 = 7,
    DT_BIN11 = 8,
    DT_F32_8Q = 9,
    DT_U8_Q = 10,
    DT_I64 = 11,
    DT_U64 = 12,
    DT_F64 = 13,
    DT_NUM = 14
} DataType;

inline const char *const *DataTypeName()
{
    static const char *const names[] = {"DT_U8", "DT_I8", "DT_U32", "DT_I32", "DT_F16", "DT_F16_8Q",
        "DT_F32", "DT_BIN01", "DT_BIN11", "DT_F32_8Q", "DT_U8_Q", "DT_I64", "DT_U64", "DT_F64",
        "DT_NUM"};
    return names;
}

inline U32 bytesOf(DataType dt)
{
    // Please divide number of elements by 8 first in the case of binary data types
    U32 bytes[] = {1, 1, 4, 4, 2, 2, 4, 1, 1, 4, 1, 8, 8, 8};
    U32 ret;
    if (dt < DT_NUM) {
        ret = bytes[dt];
    } else {
        ret = 0;
        printf("[ERROR] try to get unknown type:%s bytes.\n", DataTypeName()[dt]);
        exit(1);
    }
    return ret;
}

#ifdef _USE_FP16
inline void transformFromHalf(DataType dataType, const F16 *src, void *dst, int num)
{
    if (num <= 0) {
        return;
    }
    if (num % 8 != 0) {
        printf("[ERROR] can not support to transformFromHalf for array(length(%d) mod 8 != 0).\n",
            num);
        exit(1);
    }
    float threshold = 0;
    switch (dataType) {
        case DT_BIN11: {
            threshold = 0;
            break;
        }
        case DT_BIN01: {
            threshold = 0.5;
            break;
        }
        default: {
            printf("[ERROR] can not transform half to %s.\n", DataTypeName()[dataType]);
            exit(1);
        }
    }
    BIN8 *ptr = (BIN8 *)dst;
    for (int i = 0, k = 0; i < num / 8; i++) {
        BIN8 temp = 0;
        for (int j = 0; j < 8; j++, k++) {
            if (src[k] >= threshold) {
                temp |= (1 << (7 - j));
            }
        }
        ptr[i] = temp;
    }
}

inline void transformToHalf(DataType dataType, const void *src, F16 *dst, int num)
{
    if (num <= 0) {
        return;
    }
    if (num % 8 != 0) {
        printf(
            "[ERROR] can not support to transformToHalf for array(length(%d) mod 8 != 0).\n", num);
        exit(1);
    }
    switch (dataType) {
        case DT_BIN01: {
            const BIN8 *ptr = (const BIN8 *)src;
            for (int i = 0; i < num; i++) {
                std::bitset<8> Val(((BIN8 *)ptr)[i / 8]);
                if (Val.test(7 - (i % 8))) {
                    dst[i] = 1.0;
                } else {
                    dst[i] = 0;
                }
            }
            break;
        }
        case DT_BIN11: {
            const BIN8 *ptr = (const BIN8 *)src;
            for (int i = 0; i < num; i++) {
                std::bitset<8> Val(((BIN8 *)ptr)[i / 8]);
                if (Val.test(7 - (i % 8))) {
                    dst[i] = 1.0;
                } else {
                    dst[i] = -1.0;
                }
            }
            break;
        }
        default: {
            printf("[ERROR] can not transform %s to half.\n", DataTypeName()[dataType]);
            exit(1);
        }
    }
}
#endif

inline void transformToInt(DataType dataType, const void *src, int *dst, int num)
{
    if (num <= 0) {
        return;
    }
    switch (dataType) {
        case DT_I64: {
            I64 value;
            const U8 *ptr = (const U8 *)src;
            for (int i = 0; i < num; i++) {
                UNI_MEMCPY(&value, ptr, sizeof(I64));
                ptr += sizeof(I64);
                value = value > INT_MAX ? INT_MAX : value;
                dst[i] = value < INT_MIN ? INT_MIN : value;
            }
            break;
        }
        case DT_U32:
        case DT_I32: {
            UNI_MEMCPY(dst, src, sizeof(int) * num);
            break;
        }
        default: {
            printf("[ERROR] can not transform %s to int.\n", DataTypeName()[dataType]);
            exit(1);
        }
    }
}

inline unsigned short float32ToFloat16(float value)
{
    const U32 *word = (const U32 *)(&value);
    unsigned short sign = (word[0] & 0x80000000) >> 31;
    unsigned short exponent = (word[0] & 0x7F800000) >> 23;
    unsigned int significand = word[0] & 0x7FFFFF;

    unsigned short u;
    if (exponent == 0) {
        u = (sign << 15) | (0x00 << 10) | 0x00;
    } else if (exponent == 0xFF) {
        u = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    } else {
        short newexp = exponent + (-127 + 15);
        if (newexp >= 31) {
            u = (sign << 15) | (0x1F << 10) | 0x00;
        } else if (newexp <= 0) {
            if (newexp >= -10) {
                unsigned short sig = (significand | 0x800000) >> (14 - newexp);
                u = (sign << 15) | (0x00 << 10) | sig;
            } else {
                u = (sign << 15) | (0x00 << 10) | 0x00;
            }
        } else {
            u = (sign << 15) | (newexp << 10) | (significand >> 13);
        }
    }
    return u;
}

inline void transformFromFloat(
    DataType dataType, const float *src, void *dst, int num, float scale = 1)
{
    if (num <= 0) {
        return;
    }
    switch (dataType) {
        case DT_F32: {
            UNI_MEMCPY(dst, src, sizeof(float) * num);
            break;
        }
        case DT_I64: {
            I64 *ptr = (I64 *)dst;
            for (int i = 0; i < num; i++) {
                ptr[i] = src[i];
            }
            break;
        }
        case DT_U32: {
            U32 *ptr = (U32 *)dst;
            for (int i = 0; i < num; i++) {
                ptr[i] = src[i];
            }
            break;
        }
        case DT_I32: {
            I32 *ptr = (I32 *)dst;
            for (int i = 0; i < num; i++) {
                ptr[i] = src[i];
            }
            break;
        }
        case DT_F16_8Q:
        case DT_F16: {
#ifdef _USE_FP16
            F16 *ptr = (F16 *)dst;
#else
            unsigned short *q = (unsigned short *)dst;
#endif
            for (int i = 0; i < num; i++) {
#ifdef _USE_FP16
                ptr[i] = src[i];
#else
                q[i] = float32ToFloat16(src[i]);
#endif
            }
            break;
        }
        case DT_I8: {
            INT8 *ptr = (INT8 *)dst;
            for (int i = 0; i < num; i++) {
                ptr[i] = src[i] * scale;
            }
            break;
        }
        case DT_U8: {
            U8 *ptr = (U8 *)dst;
            for (int i = 0; i < num; i++) {
                ptr[i] = src[i];
            }
            break;
        }
        default: {
            printf("[ERROR] can not transform float to %s.\n", DataTypeName()[dataType]);
            exit(1);
        }
    }
}

inline void transformToFloat(
    DataType dataType, const void *src, float *dst, int num, float scale = 1)
{
    if (num <= 0) {
        return;
    }
    switch (dataType) {
        case DT_F32_8Q:
        case DT_F32: {
            UNI_MEMCPY(dst, src, sizeof(float) * num);
            break;
        }
        case DT_I64: {
            const I64 *ptr = (const I64 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_U32: {
            const U32 *ptr = (const U32 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_I32: {
            const I32 *ptr = (const I32 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_F16_8Q:
        case DT_F16: {
#ifdef _USE_FP16
            const F16 *ptr = (const F16 *)src;
#else
            const unsigned short *q = (const unsigned short *)src;
            U32 *word = (U32 *)dst;
#endif
            for (int i = 0; i < num; i++) {
#ifdef _USE_FP16
                dst[i] = ptr[i];
#else
                unsigned short value = q[i];
                unsigned short sign = (value & 0x8000) >> 15;
                unsigned short exponent = (value & 0x7c00) >> 10;
                unsigned short significand = value & 0x03FF;

                U32 u;
                if (exponent == 0) {
                    if (significand == 0) {
                        u = sign << 31;
                    } else {
                        exponent = 0;
                        while (0 == (significand & 0x200)) {
                            significand <<= 1;
                            exponent++;
                        }
                        significand <<= 1;
                        significand &= 0x3FF;
                        u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
                    }
                } else if (exponent == 0x1F) {
                    u = (sign << 31) | (0xFF << 23) | (significand << 13);
                } else {
                    u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
                }
                word[i] = u;
#endif
            }
            break;
        }
        case DT_I8: {
            const INT8 *ptr = (const INT8 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = ptr[i] / scale;
            }
            break;
        }
        case DT_U8: {
            const U8 *ptr = (const U8 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_U8_Q: {
            const U8 *ptr = (const U8 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = ptr[i] - 128;
            }
            break;
        }
        case DT_BIN01: {
            const BIN8 *ptr = (const BIN8 *)src;
            for (int i = 0; i < num; i++) {
                std::bitset<8> Val(((BIN8 *)ptr)[i / 8]);
                if (Val.test(7 - (i % 8))) {
                    dst[i] = 1.0;
                } else {
                    dst[i] = 0;
                }
            }
            break;
        }
        case DT_BIN11: {
            const BIN8 *ptr = (const BIN8 *)src;
            for (int i = 0; i < num; i++) {
                std::bitset<8> Val(((BIN8 *)ptr)[i / 8]);
                if (Val.test(7 - (i % 8))) {
                    dst[i] = 1.0;
                } else {
                    dst[i] = -1.0;
                }
            }
            break;
        }
        default: {
            printf("[ERROR] can not transform %s to float.\n", DataTypeName()[dataType]);
            exit(1);
        }
    }
}

inline void UNI_INIT(U32 num, DataType dt, F32 val, void *dst)
{
    if (num <= 0) {
        return;
    }
    if (val == 0) {
        UNI_MEMSET(dst, 0, bytesOf(dt) * num);
        return;
    }
    switch (dt) {
        case DT_F16: {
            unsigned short mem = float32ToFloat16(val);
            unsigned short *arr = (unsigned short *)dst;
            for (U32 i = 0; i < num; i++) {
                arr[i] = mem;
            }
            break;
        }
        case DT_F32: {
            F32 *arr = (F32 *)dst;
            for (U32 i = 0; i < num; i++) {
                arr[i] = val;
            }
            break;
        }
        case DT_U32: {
            U32 *arr = (U32 *)dst;
            for (U32 i = 0; i < num; i++) {
                arr[i] = val;
            }
            break;
        }
        case DT_I32: {
            I32 *arr = (I32 *)dst;
            for (U32 i = 0; i < num; i++) {
                arr[i] = val;
            }
            break;
        }
        default: {
            printf("[ERROR] can not init %s type data.\n", DataTypeName()[dt]);
            exit(1);
        }
    }
}

inline INT8 round_towards_zero(F32 num, bool clamp = true)
{
    INT8 ret;
    if (clamp) {
        if (num > 127.0) {
            return 127;
        } else if (num < -127.0) {
            return -127;
        }
    }
    if (num > 0) {
        ret = floor(num);
    } else {
        ret = ceil(num);
    }
    return ret;
}
#endif
