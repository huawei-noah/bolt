// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Errors.h"
#include <sstream>

namespace raul
{

std::string Exception::getMessage(size_t level) const
{
    const auto end = '\n';
    std::stringstream ss;

    if (level == 0)
    {
        ss << "Runtime error:";
    }

    ss << end << std::string(level, ' ');

    if (typeName)
    {
        ss << *typeName;
    }
    else
    {
        ss << "NoneType";
    }
    ss << "[";
    if (objName)
    {
        ss << *objName;
    }
    if (funcName)
    {
        ss << "::" << *funcName;
    }
    ss << "]: ";

    ss << message;

    if (fileName && lineNumber)
    {
        ss << " - " << *fileName << ":" << *lineNumber;
    }

    try
    {
        if (innerException)
        {
            std::rethrow_exception(innerException);
        }
    }
    catch (const Exception& e)
    {
        ss << e.getMessage(level + 1);
    }
    catch (const std::exception& e)
    {
        ss << end << std::string(level + 1, ' ') << "Exception: " << e.what();
    }

    return ss.str();
}

const char* Exception::what() const noexcept
{
    return errorText.c_str();
}

} // namespace raul
