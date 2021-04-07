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
#include <string.h>
#if defined(_USE_NEON) || defined(_USE_MALI)
#include <arm_neon.h>
#ifdef __aarch64__
typedef __fp16 F16;
#endif
typedef int8_t INT8;
#else
typedef char INT8;
#endif
#ifdef _USE_X86
#include <immintrin.h>
#include <xmmintrin.h>
#endif

typedef unsigned char U8;
typedef const unsigned char CU8;
typedef char I8;
typedef const char CI8;
typedef unsigned int U32;
typedef const unsigned int CU32;
typedef int I32;
typedef const int CI32;
typedef float F32;
typedef double F64;
typedef int64_t I64;
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
    DT_NUM = 9
} DataType;

inline const char *const *DataTypeName()
{
    static const char *const names[] = {"DT_U8", "DT_I8", "DT_U32", "DT_I32", "DT_F16", "DT_F16_8Q",
        "DT_F32", "DT_BIN01", "DT_BIN11", "DT_NUM"};
    return names;
}

inline U32 bytesOf(DataType dt)
{
    // Please divide number of elements by 8 first in the case of binary data types
    U32 bytes[] = {1, 1, 4, 4, 2, 2, 4, 1, 1, 8};
    return dt < DT_NUM ? bytes[dt] : 0;
}

inline void transformFromFloat(DataType dataType, float *src, void *dst, int num, float scale = 1)
{
    switch (dataType) {
        case DT_F32: {
            memcpy(dst, src, sizeof(float) * num);
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
#ifdef __aarch64__
            F16 *ptr = (F16 *)dst;
#else
            U32 *word = (U32 *)src;
            unsigned short *q = (unsigned short *)dst;
#endif
            for (int i = 0; i < num; i++) {
#ifdef __aarch64__
                ptr[i] = src[i];
#else
                unsigned short sign = (word[i] & 0x80000000) >> 31;
                unsigned short exponent = (word[i] & 0x7F800000) >> 23;
                unsigned int significand = word[i] & 0x7FFFFF;

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
                q[i] = u;
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
            printf("can not transform float to %s.\n", DataTypeName()[dataType]);
            exit(1);
        }
    }
}

inline void transformToFloat(DataType dataType, void *src, float *dst, int num, float scale = 1)
{
    switch (dataType) {
        case DT_F32: {
            memcpy(dst, src, sizeof(float) * num);
            break;
        }
        case DT_U32: {
            U32 *ptr = (U32 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_I32: {
            I32 *ptr = (I32 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_F16_8Q:
        case DT_F16: {
#ifdef __aarch64__
            F16 *ptr = (F16 *)src;
#else
            unsigned short *q = (unsigned short *)src;
            U32 *word = (U32 *)dst;
#endif
            for (int i = 0; i < num; i++) {
#ifdef __aarch64__
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
            INT8 *ptr = (INT8 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = ptr[i] / scale;
            }
            break;
        }
        case DT_U8: {
            U8 *ptr = (U8 *)src;
            for (int i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_BIN01: {
            BIN8 *ptr = (BIN8 *)src;
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
            BIN8 *ptr = (BIN8 *)src;
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
            printf("can not transform %s to float.\n", DataTypeName()[dataType]);
            exit(1);
        }
    }
}

inline void UNI_INIT(U32 num, DataType dt, F32 val, void *dst)
{
    switch (dt) {
        case DT_F16: {
            unsigned int short mem;
            transformFromFloat(DT_F16, &val, &mem, 1);
            U8 *arr = (U8 *)dst;
            for (U32 i = 0; i < num; i++) {
                memcpy(arr + i * bytesOf(DT_F16), &mem, bytesOf(DT_F16));
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
            printf("can not init %s type data.\n", DataTypeName()[dt]);
            exit(1);
        }
    }
}
#endif
