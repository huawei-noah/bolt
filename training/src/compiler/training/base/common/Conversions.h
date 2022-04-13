// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ConversionsH
#define ConversionsH

#include <iomanip>
#include <sstream>

#include <training/base/common/Common.h>

class Conversions
{
  public:
    template<class T>
    static inline std::string toString(const T& t)
    {
        std::ostringstream o;
        o << t;
        return o.str();
    }

    template<class T>
    static inline std::string toString(const T& t, int width)
    {
        std::ostringstream o;
        o << std::setw(width);
        o << t;
        return o.str();
    }

    template<class T>
    static inline std::string toString(const T& t, int width, int precision)
    {
        std::ostringstream o;
        o << std::setw(width);
        o << std::setprecision(precision);
        o << t;
        return o.str();
    }

    template<typename T>
    static inline bool fromString(const std::string& str, T& res)
    {
        bool isCorrect = true;

        if (false == str.empty())
        {
            std::istringstream ss(str);
            ss >> res;
            if (ss.fail())
            {
                isCorrect = false;
            }
        }
        else
        {
            isCorrect = false;
        }
        return isCorrect;
    }

    template<class TT, size_t N>
    static inline std::string toString(yato::dimensionality<N, TT>& t)
    {
        std::ostringstream o;
        o << "[" << t[0];
        for (size_t i = 1; i < N; ++i)
        {
            o << ", " << t[i];
        }
        o << "]";
        return o.str();
    }

    template<class TT>
    static inline std::string toString(yato::dimensionality<3U, TT>& t)
    {
        std::ostringstream o;
        o << "[N";
        for (size_t i = 0; i < 3; ++i)
        {
            o << ", " << t[i];
        }
        o << "]";
        return o.str();
    }
};

template<>
inline std::string Conversions::toString<raul::half>(const raul::half& t)
{
    std::ostringstream o;
    o << static_cast<raul::dtype>(t);
    return o.str();
}

template<>
inline std::string Conversions::toString<raul::LayerExecutionTarget>(const raul::LayerExecutionTarget& t)
{
    std::ostringstream o;
    switch (t)
    {
        case raul::LayerExecutionTarget::CPU:
            o << "CPU";
            break;
        case raul::LayerExecutionTarget::CPUFP16:
            o << "CPUFP16";
            break;
        case raul::LayerExecutionTarget::Default:
            o << "Default";
            break;
    }
    return o.str();
}

template<>
inline std::string Conversions::toString(const raul::shape& t)
{
    std::ostringstream o;
    o << "[" << t[0] << ", " << t[1] << "," << t[2] << "," << t[3] << "]";
    return o.str();
}

template<>
inline std::string Conversions::toString<bool>(const bool& t)
{
    std::ostringstream o;
    o << (t ? "true" : "false");
    return o.str();
}

template<>
inline bool Conversions::fromString<bool>(const std::string& str, bool& res)
{
    bool isCorrect = true;

    if (str == "1" || str == "true")
    {
        res = true;
    }
    else if (str == "0" || str == "false")
    {
        res = false;
    }
    else
    {
        isCorrect = false;
    }
    return isCorrect;
}

#endif
