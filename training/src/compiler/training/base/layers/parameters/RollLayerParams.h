// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ROLL_LAYER_PARAMS_H
#define ROLL_LAYER_PARAMS_H

#include "BasicParameters.h"
#include <string>

namespace raul
{

/**
 * @param input name of input tensor
 * @param output name of output tensor
 * @param dimension specifies the dimension to shift
 * @param shift number of elements to shift
 */

struct RollLayerParams : public BasicParamsWithDim
{
    RollLayerParams() = delete;
    RollLayerParams(const raul::Name& input, const raul::Name& output, raul::Dimension dimension, size_t shift, bool cycled = true, raul::dtype val = 1.0_dt)
        : BasicParamsWithDim(Names(1, input), Names(1, output), dimension)
        , mShift(shift)
        , mCycled(cycled)
        , mValueToFill(val)
    {
    }

    size_t mShift;
    bool mCycled;
    raul::dtype mValueToFill;

    void print(std::ostream& stream) const override;
};

} // raul namespace

#endif // ROLL_LAYER_PARAMS_H
