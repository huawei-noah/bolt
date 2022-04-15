// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TYPE_HALF_H
#define TYPE_HALF_H

#include <cstdint>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4804)
#pragma warning(disable : 4805)
#endif // _MSC_VER

#if !defined(ANDROID)
#include <half.hpp>
#endif

namespace raul
{

#if defined(ANDROID)
typedef _Float16 half;
#else
typedef half_float::half half;
#endif

inline float toFloat32(float value)
{
    return value;
}

#if defined(ANDROID)
inline float toFloat32(half value)
{
    return static_cast<float>(value);
}
#else
inline float toFloat32(half value)
{
    return half_float::half_cast<float>(value);
}
#endif

inline half toFloat16(half& f)
{
    return f;
}

#if defined(ANDROID)
inline half toFloat16(float const& f)
{
    return static_cast<half>(f);
}
#else
inline half toFloat16(float const& f)
{
    return half_float::half_cast<half>(f);
}
#endif

// left without implementation on purpose to allow only float <-> half conversion
template<typename T, typename F>
inline T toFloat(const F&);

template<>
inline float toFloat<float>(const half& val)
{
    return toFloat32(val);
}

template<>
inline half toFloat<half>(const float& val)
{
    return toFloat16(val);
}

template<>
inline float toFloat<float>(const float& val)
{
    return val;
}

template<>
inline half toFloat<half>(const half& val)
{
    return val;
}

#if !defined(ANDROID)
template <typename T>
class castHelper
{
public:
    template <typename U>
    static T cast(const U& val) { return static_cast<T>(val); }

    static T cast(const half& val) { return half_float::half_cast<T>(val); }
};

template <>
class castHelper<half>
{
public:
    template <typename U>
    static half cast(const U& val) { return half_float::half_cast<half>(val); }

    static half cast(const bool& val) { return half_float::half_cast<half>(static_cast<int>(val)); }

    static half cast(const half& val) { return val; }
};
#endif

} // namespace raul

#if defined(ANDROID)
#define TOHTYPE(var) static_cast<raul::half>(var)
#else
#define TOHTYPE(var) castHelper<raul::half>::cast(var)
#endif

inline raul::half operator"" _hf(long double value)
{
    return raul::toFloat16(static_cast<float>(value));
}

inline raul::half operator"" _hf(unsigned long long value)
{
    return raul::toFloat16(static_cast<float>(value));
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

#endif // TYPE_HALF_H
