// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ZONEOUT_PARAMS_H
#define ZONEOUT_PARAMS_H

#include <string>
#include <vector>

#include "BasicParameters.h"

namespace raul
{

struct ZoneoutParams : public BasicParams
{
    ZoneoutParams() = delete;
    /** Parameters for Zoneout layer
     * @param inputs vector of names of input tensors
     * @param outputs vector of names of output tensors
     * @param param probability of keeping previous value
     */
    ZoneoutParams(const Names& inputs, const Names& outputs, const dtype probability)
        : BasicParams(inputs, outputs)
        , mProbability(probability)
    {
    }

    void print(std::ostream& stream) const override;

    dtype mProbability;
};

} // raul namespace

#endif