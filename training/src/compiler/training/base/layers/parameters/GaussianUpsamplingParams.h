// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GAUSSIAN_UPSAMPLING_PARAMS_H
#define GAUSSIAN_UPSAMPLING_PARAMS_H

#include <optional>
#include <string>

#include "BasicParameters.h"

namespace raul
{

struct GaussianUpsamplingParams : public BasicParams
{
    GaussianUpsamplingParams() = delete;
    GaussianUpsamplingParams(const Names& inputs, const Name output, const size_t melLen, const dtype eps = 0.00001_dt)
        : BasicParams(inputs, Names(1, output))
        , mMelLen(melLen)
        , mEps(eps)
    {
    }

    void print(std::ostream& stream) const override;
    size_t mMelLen;
    dtype mEps;
};

} // raul namespace

#endif // GAUSSIAN_UPSAMPLING_PARAMS_H