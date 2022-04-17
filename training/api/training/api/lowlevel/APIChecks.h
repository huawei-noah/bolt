// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef API_CHECKS_H
#define API_CHECKS_H

#include <string>

#define CHECK_PRECONDITION_M(x, m)                                                                                                                                                                     \
    do                                                                                                                                                                                                 \
    {                                                                                                                                                                                                  \
        if (!(x))                                                                                                                                                                                      \
        {                                                                                                                                                                                              \
            set_last_error(m);                                                                                                                                                                         \
            return STATUS_ERROR;                                                                                                                                                                       \
        }                                                                                                                                                                                              \
    } while (false)

#define CHECK_PRECONDITION(x) CHECK_PRECONDITION_M(x, "Condition " #x " violated")

#define CHECK_NOT_NULL(x) CHECK_PRECONDITION_M(x, #x " is NULL")

#define CHECK_STRING(x)                                                                                                                                                                                \
    do                                                                                                                                                                                                 \
    {                                                                                                                                                                                                  \
        if (!(x))                                                                                                                                                                                      \
        {                                                                                                                                                                                              \
            set_last_error(#x " is NULL");                                                                                                                                                             \
            return STATUS_ERROR_BAD_NAME;                                                                                                                                                              \
        }                                                                                                                                                                                              \
        if (std::string_view(x) == "")                                                                                                                                                                 \
        {                                                                                                                                                                                              \
            set_last_error(#x " is empty");                                                                                                                                                            \
            return STATUS_ERROR_BAD_NAME;                                                                                                                                                              \
        }                                                                                                                                                                                              \
    } while (false)

#define FORWARD_ERROR(x)                                                                                                                                                                               \
    do                                                                                                                                                                                                 \
    {                                                                                                                                                                                                  \
        auto status = x;                                                                                                                                                                               \
        if (status != STATUS_OK) { return status; }                                                                                                                                                    \
    } while (false)

// Error strings
#define SE_LAYER_ALREADY_ADDED "Altering params of layer already added to graph is prohibited"

#endif