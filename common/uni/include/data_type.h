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
#include <float.h>

#if defined(_USE_NEON)
#include <arm_neon.h>
#define _USE_FP16_TYPE
typedef __fp16 F16;
#if defined(__ARM_FEATURE_BF16)
typedef __bf16 BF16;
#endif
#else
typedef unsigned short F16;
#endif

#if defined(_USE_X86)
#include <immintrin.h>
#include <xmmintrin.h>
#define FTZ _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif

#define _USE_ULTRA_OPTIMIZATION

#include "secure_c_wrapper.h"

typedef int8_t INT8;
typedef uint8_t UINT8;
typedef int16_t I16;
typedef uint16_t U16;
typedef int32_t I32;
typedef uint32_t U32;
typedef int64_t I64;
typedef uint64_t U64;
typedef float F32;
typedef double F64;

typedef unsigned char BIN8;
typedef unsigned char U8;
typedef char I8;

#ifdef _USE_LITE
#define NAME_LEN 4
#define DIM_LEN 4
#define ENUM_TYPE uint8_t
#else
#define NAME_LEN 128
#define DIM_LEN 20
#define ENUM_TYPE uint32_t
#endif
typedef enum DataType : uint32_t {
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
    DT_BF16 = 14,
    DT_I4 = 15,
    DT_NUM = 16
} DataType;

inline const char *const *DataTypeName()
{
    static const char *const names[] = {"DT_U8", "DT_I8", "DT_U32", "DT_I32", "DT_F16", "DT_F16_8Q",
        "DT_F32", "DT_BIN01", "DT_BIN11", "DT_F32_8Q", "DT_U8_Q", "DT_I64", "DT_U64", "DT_F64",
        "DT_BF16", "DT_I4", "DT_NUM"};
    return names;
}

inline U32 bytesOf(DataType dt)
{
    // Please divide number of elements by 8 first in the case of binary data types
    U32 bytes[] = {1, 1, 4, 4, 2, 2, 4, 1, 1, 4, 1, 8, 8, 8, 2, 1};
    U32 ret;
    if (dt < sizeof(bytes) / sizeof(U32)) {
        ret = bytes[dt];
    } else {
        ret = 0;
        UNI_ERROR_LOG("try to get unknown type:%s bytes.\n", DataTypeName()[dt]);
    }
    return ret;
}

inline unsigned short float32ToFloat16(float value)
{
    U16 u;
#if defined(_USE_X86) && defined(__F16C__)
    u = _cvtss_sh(value, 0);
#else
    const U32 *word = (const U32 *)(&value);
    U16 sign = (word[0] & 0x80000000) >> 31;
    U16 exponent = (word[0] & 0x7F800000) >> 23;
    U32 significand = word[0] & 0x7FFFFF;

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
#endif
    return u;
}

inline float float16ToFloat32(unsigned short value)
{
    float ret;
#if defined(_USE_X86) && defined(__F16C__)
    ret = _cvtsh_ss(value);
#else
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
    ret = *((float *)&u);
#endif
    return ret;
}

inline unsigned short float32ToBfloat16(const float v)
{
    const U32 *ptr = (const U32 *)(&v);
    U16 res = (*ptr >> 16);
    const U16 error = (*ptr & 0x0000ffff);
    U16 bf_l = res & 0x0001;
    if ((error > 0x8000) || ((error == 0x8000) && (bf_l != 0))) {
        res += 1;
    }
    return res;
}

inline float bfloat16ToFloat32(const unsigned short v)
{
    U32 lv = (v << 16);
    float res = 0;
    UNI_MEMCPY(&res, &lv, sizeof(float));
    return res;
}

#ifdef _USE_FP16_TYPE
inline int transformFromHalf(DataType dt, const F16 *src, void *dst, I32 num)
{
    if (num <= 0) {
        return 0;
    }
    int ret = 0;
    if (dt == DT_BIN01 || dt == DT_BIN11) {
        if (num % 8 != 0) {
            UNI_ERROR_LOG(
                "can not support to transformFromHalf for array(length(%d) mod 8 != 0).\n", num);
            return 1;
        }
        float threshold = 0.5;
        if (dt == DT_BIN11) {
            threshold = 0;
        }
        BIN8 *ptr = (BIN8 *)dst;
        for (I32 i = 0, k = 0; i < num / 8; i++) {
            BIN8 temp = 0;
            for (I32 j = 0; j < 8; j++, k++) {
                if (src[k] >= threshold) {
                    temp |= (1 << (7 - j));
                }
            }
            ptr[i] = temp;
        }
    } else if (dt == DT_U8) {
        U8 *ptr = (U8 *)dst;
        for (I32 i = 0; i < num; i++) {
            ptr[i] = (U8)src[i];
        }
    } else {
        UNI_ERROR_LOG("can not transform from half to %s.\n", DataTypeName()[dt]);
        ret = 1;
    }
    return ret;
}

inline int transformToHalf(DataType dt, const void *src, F16 *dst, I32 num)
{
    if (num <= 0) {
        return 0;
    }
    if ((dt == DT_BIN01 || dt == DT_BIN11) && num % 8 != 0) {
        UNI_ERROR_LOG("can not support to transformToHalf for array(length(%d) mod 8 != 0).\n", num);
        return 1;
    }
    int ret = 0;
    switch (dt) {
        case DT_BIN01: {
            const BIN8 *ptr = (const BIN8 *)src;
            for (I32 i = 0; i < num; i++) {
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
            for (I32 i = 0; i < num; i++) {
                std::bitset<8> Val(((BIN8 *)ptr)[i / 8]);
                if (Val.test(7 - (i % 8))) {
                    dst[i] = 1;
                } else {
                    dst[i] = -1;
                }
            }
            break;
        }
        case DT_U8: {
            const U8 *ptr = (const U8 *)src;
            for (I32 i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        default: {
            UNI_ERROR_LOG("can not transform %s to half.\n", DataTypeName()[dt]);
            ret = 1;
            break;
        }
    }
    return ret;
}
#endif

inline int transformFromFloat(DataType dt, const float *src, void *dst, I32 num, float scale = 1)
{
    if (num <= 0) {
        return 0;
    }
    int ret = 0;
    switch (dt) {
        case DT_F32_8Q:
        case DT_F32: {
            UNI_MEMCPY(dst, src, sizeof(float) * num);
            break;
        }
        case DT_I64: {
            I64 *ptr = (I64 *)dst;
            for (I32 i = 0; i < num; i++) {
                ptr[i] = (I64)src[i];
            }
            break;
        }
        case DT_U32: {
            U32 *ptr = (U32 *)dst;
            for (I32 i = 0; i < num; i++) {
                ptr[i] = (U32)src[i];
            }
            break;
        }
        case DT_I32: {
            I32 *ptr = (I32 *)dst;
            for (I32 i = 0; i < num; i++) {
                ptr[i] = (I32)src[i];
            }
            break;
        }
        case DT_F16_8Q:
        case DT_F16: {
            F16 *ptr = (F16 *)dst;
            I32 i = 0;
#if defined(_USE_NEON)
            for (i = 0; i < num - 3; i += 4) {
                float32x4_t in = vld1q_f32(src + i);
                float16x4_t out = vcvt_f16_f32(in);
                vst1_f16(ptr + i, out);
            }
#elif defined(_USE_X86) && defined(__F16C__)
            for (i = 0; i < num - 7; i += 8) {
                __m256 in = _mm256_loadu_ps(src + i);
                __m128i out = _mm256_cvtps_ph(in, 0);
                _mm_storeu_si128((__m128i *)(ptr + i), out);
            }
#endif
            for (; i < num; i++) {
#ifdef _USE_FP16_TYPE
                ptr[i] = (F16)src[i];
#else
                ptr[i] = float32ToFloat16(src[i]);
#endif
            }
            break;
        }
        case DT_BF16: {
            unsigned short *ptr = (unsigned short *)dst;
            for (I32 i = 0; i < num; i++) {
                ptr[i] = float32ToBfloat16(src[i]);
            }
            break;
        }
        case DT_I8: {
            INT8 *ptr = (INT8 *)dst;
            for (I32 i = 0; i < num; i++) {
                ptr[i] = (INT8)(src[i] * scale);
            }
            break;
        }
        case DT_U8: {
            U8 *ptr = (U8 *)dst;
            for (I32 i = 0; i < num; i++) {
                ptr[i] = (U8)src[i];
            }
            break;
        }
        default: {
            UNI_ERROR_LOG("can not transform float to %s.\n", DataTypeName()[dt]);
            ret = 1;
            break;
        }
    }
    return ret;
}

inline int transformToFloat(DataType dt, const void *src, float *dst, I32 num, float scale = 1)
{
    if (num <= 0) {
        return 0;
    }
    int ret = 0;
    switch (dt) {
        case DT_F32_8Q:
        case DT_F32: {
            UNI_MEMCPY(dst, src, sizeof(float) * num);
            break;
        }
        case DT_I64: {
            const I64 *ptr = (const I64 *)src;
            for (I32 i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_U32: {
            const U32 *ptr = (const U32 *)src;
            for (I32 i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_I32: {
            const I32 *ptr = (const I32 *)src;
            for (I32 i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_F16_8Q:
        case DT_F16: {
            const F16 *ptr = (const F16 *)src;
            I32 i = 0;
#if defined(_USE_NEON)
            for (i = 0; i < num - 3; i += 4) {
                float16x4_t in = vld1_f16(ptr + i);
                float32x4_t out = vcvt_f32_f16(in);
                vst1q_f32(dst + i, out);
            }
#elif defined(_USE_X86) && defined(__F16C__)
            for (i = 0; i < num - 7; i += 8) {
                __m128i in = _mm_loadu_si128((const __m128i *)(ptr + i));
                __m256 out = _mm256_cvtph_ps(in);
                _mm256_storeu_ps(dst + i, out);
            }
#endif
            for (; i < num; i++) {
#ifdef _USE_FP16_TYPE
                dst[i] = ptr[i];
#else
                dst[i] = float16ToFloat32(ptr[i]);
#endif
            }
            break;
        }
        case DT_BF16: {
            const unsigned short *ptr = (const unsigned short *)src;
            for (I32 i = 0; i < num; i++) {
                dst[i] = bfloat16ToFloat32(ptr[i]);
            }
            break;
        }
        case DT_I8: {
            const INT8 *ptr = (const INT8 *)src;
            for (I32 i = 0; i < num; i++) {
                dst[i] = ptr[i] / scale;
            }
            break;
        }
        case DT_U8: {
            const U8 *ptr = (const U8 *)src;
            for (I32 i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_U8_Q: {
            const U8 *ptr = (const U8 *)src;
            for (I32 i = 0; i < num; i++) {
                dst[i] = (ptr[i] - 128) / scale;
            }
            break;
        }
        case DT_BIN01: {
            const BIN8 *ptr = (const BIN8 *)src;
            for (I32 i = 0; i < num; i++) {
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
            for (I32 i = 0; i < num; i++) {
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
            UNI_ERROR_LOG("can not transform %s to float.\n", DataTypeName()[dt]);
            ret = 1;
            break;
        }
    }
    return ret;
}

inline int transformToInt(DataType dt, const void *src, I32 *dst, I32 num)
{
    if (num <= 0) {
        return 0;
    }
    int ret = 0;
    switch (dt) {
        case DT_I64: {
            I64 value;
            const U8 *ptr = (const U8 *)src;
            for (I32 i = 0; i < num; i++) {
                UNI_MEMCPY(&value, ptr, sizeof(I64));
                ptr += sizeof(I64);
                value = value > INT_MAX ? INT_MAX : value;
                dst[i] = value < INT_MIN ? INT_MIN : value;
            }
            break;
        }
        case DT_U32:
        case DT_I32: {
            UNI_MEMCPY(dst, src, sizeof(I32) * num);
            break;
        }
        case DT_F32: {
            F32 value;
            const U8 *ptr = (const U8 *)src;
            for (I32 i = 0; i < num; i++) {
                UNI_MEMCPY(&value, ptr, sizeof(F32));
                ptr += sizeof(F32);
                dst[i] = (I32)value;
            }
            break;
        }
        default: {
            UNI_ERROR_LOG("can not transform %s to int.\n", DataTypeName()[dt]);
            ret = 1;
            break;
        }
    }
    return ret;
}

inline int transformToUInt(DataType dt, const void *src, U32 *dst, I32 num)
{
    if (num <= 0) {
        return 0;
    }
    int ret = 0;
    switch (dt) {
        case DT_F32_8Q:
        case DT_F32: {
            const F32 *ptr = (const F32 *)src;
            for (I32 i = 0; i < num; i++) {
                dst[i] = (U32)ptr[i];
            }
            break;
        }
        case DT_I64: {
            const I64 *ptr = (const I64 *)src;
            for (I32 i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_U32: {
            const U32 *ptr = (const U32 *)src;
            for (I32 i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_I32: {
            const I32 *ptr = (const I32 *)src;
            for (I32 i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_F16_8Q:
        case DT_F16: {
#ifdef _USE_FP16_TYPE
            const F16 *ptr = (const F16 *)src;
            for (I32 i = 0; i < num; i++) {
                dst[i] = (U32)ptr[i];
            }
#else
            const unsigned short *ptr = (const unsigned short *)src;
            for (I32 i = 0; i < num; i++) {
                dst[i] = (U32)float16ToFloat32(ptr[i]);
            }
#endif
            break;
        }
        case DT_I8: {
            const INT8 *ptr = (const INT8 *)src;
            for (I32 i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        case DT_U8: {
            const U8 *ptr = (const U8 *)src;
            for (I32 i = 0; i < num; i++) {
                dst[i] = ptr[i];
            }
            break;
        }
        default: {
            UNI_ERROR_LOG("can not transform %s to unsigned int.\n", DataTypeName()[dt]);
            ret = 1;
            break;
        }
    }
    return ret;
}

inline int transformDataType(DataType idt,
    const void *input,
    float *iscale,
    DataType odt,
    void *output,
    float *oscale,
    U32 length)
{
    int ret = 1;
    float scale = 1;
    if (idt == DT_F32) {
        if (oscale != NULL) {
            scale = *oscale;
        }
        ret = transformFromFloat(odt, (const F32 *)input, output, length, scale);
    } else if (odt == DT_F32) {
        if (iscale != NULL) {
            scale = *iscale;
        }
        ret = transformToFloat(idt, input, (F32 *)output, length, scale);
#ifdef _USE_FP16_TYPE
    } else if (idt == DT_F16) {
        ret = transformFromHalf(odt, (const F16 *)input, output, length);
    } else if (odt == DT_F16) {
        ret = transformToHalf(idt, input, (F16 *)output, length);
#endif
    }
    return ret;
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
                arr[i] = (U32)val;
            }
            break;
        }
        case DT_I32: {
            I32 *arr = (I32 *)dst;
            for (U32 i = 0; i < num; i++) {
                arr[i] = (I32)val;
            }
            break;
        }
        case DT_I8: {
            INT8 *arr = (INT8 *)dst;
            for (U32 i = 0; i < num; i++) {
                arr[i] = (INT8)val;
            }
            break;
        }
        case DT_U8: {
            U8 *arr = (U8 *)dst;
            for (U32 i = 0; i < num; i++) {
                arr[i] = (U8)val;
            }
            break;
        }
        default: {
            UNI_ERROR_LOG("can not init %s type data.\n", DataTypeName()[dt]);
            return;
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
        ret = (INT8)floor(num);
    } else {
        ret = (INT8)ceil(num);
    }
    return ret;
}

inline I32 isQuantMixDataType(DataType dt)
{
    I32 ret = (dt == DT_F16_8Q || dt == DT_F32_8Q) ? 1 : 0;
    return ret;
}

inline DataType noQuantDataType(DataType dt)
{
    DataType ret = (dt == DT_F16_8Q) ? DT_F16 : ((dt == DT_F32_8Q) ? DT_F32 : dt);
    return ret;
}
#endif
