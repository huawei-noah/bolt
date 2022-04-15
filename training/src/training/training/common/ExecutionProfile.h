// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef EXECUTION_PROFILE_H
#define EXECUTION_PROFILE_H

#include <cfloat>
#include <float.h>
#include <iostream>
#include <string>
#include <unordered_map>

#include <training/common/Common.h>

namespace raul
{

struct ExecutionHint
{
    ExecutionHint()
        : Target(ExecutionTarget::GPU)
        , LocalSize(cl::NullRange)
        , BestTimeNS(FLT_MAX)
    {
    }
    ExecutionTarget Target;
    cl::NDRange LocalSize;
    float BestTimeNS; // best time in nanoseconds
    std::string Data;
};

class ExecutionProfile
{
  public:
    ExecutionProfile() {}

    ExecutionHint& operator[](const std::string& context) { return mProfile[context]; }

  private:
    std::unordered_map<std::string, ExecutionHint> mProfile;
};

}

#endif // EXECUTION_PROFILE_H
