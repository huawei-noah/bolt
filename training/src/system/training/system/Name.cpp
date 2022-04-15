// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Name.h"

namespace raul
{
Name Name::getPrefix() const
{
    size_t pos = 0;

    std::string tmpStr(string);

    size_t finalPos = 0;
    const size_t separatorLength = std::string(TENSOR_SEP).length();

    while ((pos = tmpStr.find(TENSOR_SEP)) != std::string::npos)
    {
        tmpStr.erase(0, pos + separatorLength);

        finalPos += pos + separatorLength;
    }

    if (finalPos == std::string::npos || finalPos == 0)
    {
        return string;
    }

    return string.substr(0, finalPos - separatorLength);
}

Name Name::getLastName() const
{
    size_t pos = 0;

    std::string tmpStr(string);

    size_t finalPos = 0;
    const size_t separatorLength = std::string(TENSOR_SEP).length();

    while ((pos = tmpStr.find(TENSOR_SEP)) != std::string::npos)
    {
        tmpStr.erase(0, pos + separatorLength);

        finalPos += pos + separatorLength;
    }

    if (finalPos == std::string::npos || finalPos == 0)
    {
        return string;
    }

    return string.substr(finalPos, string.length());
}
} // namespace raul