// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_STRING_FUNCTIONS
#define _H_STRING_FUNCTIONS

#include <string>
#include <algorithm>
#include <vector>

inline std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

inline std::string upper(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return s;
}

inline int contains(std::string s, std::string sub)
{
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    std::transform(sub.begin(), sub.end(), sub.begin(), ::tolower);
    return s.find(sub) != std::string::npos;
}

inline int startswith(std::string s, std::string sub)
{
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    std::transform(sub.begin(), sub.end(), sub.begin(), ::tolower);
    return s.find(sub) == 0;
}

inline int endswith(std::string s, std::string sub)
{
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    std::transform(sub.begin(), sub.end(), sub.begin(), ::tolower);
    return s.rfind(sub) == (s.length() - sub.length());
}

inline std::string strip(const std::string &s)
{
    const std::string WHITESPACE = " \n\r\t\f\v";
    size_t start = s.find_first_not_of(WHITESPACE);
    size_t end = s.find_last_not_of(WHITESPACE);
    std::string ret;
    if (start == std::string::npos || end == std::string::npos) {
        ret = "";
    } else {
        ret = s.substr(start, end + 1);
    }
    return ret;
}

inline std::vector<std::string> split(const std::string &s, const std::string &pattern)
{
    std::vector<std::string> ret;
    std::string str = strip(s);
    if (str == "") {
        return ret;
    }
    std::string line = str + pattern;
    size_t pos = line.find(pattern);
    while (pos != line.npos) {
        std::string t = line.substr(0, pos);
        if (t != "" && t != pattern) {
            ret.push_back(t);
        }
        line = line.substr(pos + 1);
        pos = line.find(pattern);
    }
    return ret;
}

inline std::string int2Any(int val, const int &radix)
{
    std::string ret = "";
    do {
        int r = val % radix;
        if (r >= 0 && r <= 9) {
            ret = std::string(1, r + '0') + ret;
        } else if (r <= 35) {
            ret = std::string(1, r - 10 + 'a') + ret;
        } else if (r <= 71) {
            ret = std::string(1, r - 36 + 'A') + ret;
        } else {
            ret += ".";
        }
        val /= radix;
    } while (val != 0);
    return ret;
}

#endif
