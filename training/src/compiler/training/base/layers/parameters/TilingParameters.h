// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TILING_PARAMETERS_H
#define TILING_PARAMETERS_H

#include "BasicParameters.h"

namespace raul
{

/** Parameters for Tile Layer
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param repeats contains number of repetitions
 * @param paramDim says what dimension to repeat
 */
struct TilingParams : public BasicParamsWithDim
{
    TilingParams() = delete;

    TilingParams(const raul::Name& input, const raul::Name& output, const std::initializer_list<size_t> repeats);
    TilingParams(const raul::Name& input, const raul::Name& output, size_t multiple, raul::Dimension paramDim);
    TilingParams(const raul::Name& input, const raul::Name& output, size_t multiple, const std::string& paramDim = "default");

    size_t mRepeatDepth;
    size_t mRepeatHeight;
    size_t mRepeatWidth;

    void print(std::ostream& stream) const override;
};

}

#endif