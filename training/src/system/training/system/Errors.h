// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ERRORS_H
#define ERRORS_H

#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "TypeHalf.h"
#include "Types.h"

namespace raul
{

/**
 * Exception with tracing
 */
class Exception : public std::exception
{
    std::string message;
    std::optional<std::string> typeName;
    std::optional<std::string> objName;
    std::optional<std::string> funcName;
    std::optional<std::string> fileName;
    std::optional<size_t> lineNumber;

    std::string errorText;
    std::exception_ptr innerException;

    [[nodiscard]] std::string getMessage(size_t level = 0) const;

  public:
    explicit Exception(std::string message)
        : message{ std::move(message) }
        , typeName{ std::nullopt }
        , objName{ std::nullopt }
        , funcName{ std::nullopt }
        , fileName{ std::nullopt }
        , lineNumber{ std::nullopt }
    {
        errorText = getMessage();
    }

    Exception setType(std::string name)
    {
        typeName = std::move(name);
        errorText = getMessage();
        return *this;
    }

    Exception setObject(std::string name)
    {
        objName = std::move(name);
        errorText = getMessage();
        return *this;
    }

    Exception setFunction(std::string name)
    {
        funcName = std::move(name);
        errorText = getMessage();
        return *this;
    }

    Exception setPosition(std::string name, size_t line)
    {
        fileName = std::move(name);
        lineNumber = line;
        errorText = getMessage();
        return *this;
    }

    Exception setInnerException(std::exception_ptr exception)
    {
        if (exception)
        {
            innerException = std::move(exception);
            errorText = getMessage();
        }
        return *this;
    }

    [[nodiscard]] const char* what() const noexcept override;
    ~Exception(){}
};

#define BASE_TYPE_NAME(TYPE) (std::is_same_v<TYPE, dtype> ? "dtype" : std::is_same_v<TYPE, half> ? "half" : "T")

#define THROW(TYPE, OBJECT, MESSAGE) throw raul::Exception(MESSAGE).setType(TYPE).setObject(OBJECT).setFunction(__func__).setPosition(__FILE__, __LINE__).setInnerException(std::current_exception());
#define THROW_NONAME(TYPE, MESSAGE) throw raul::Exception(MESSAGE).setType(TYPE).setFunction(__func__).setPosition(__FILE__, __LINE__).setInnerException(std::current_exception());

} // namespace raul

#endif // ERRORS_H
