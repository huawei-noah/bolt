// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TilingParameters.h"

namespace
{

raul::Dimension stringToDimension(const std::string& dim)
{
    if (dim == "width")
    {
        return raul::Dimension::Width;
    }
    if (dim == "height")
    {
        return raul::Dimension::Height;
    }
    if (dim == "depth")
    {
        return raul::Dimension::Depth;
    }
    if (dim == "batch")
    {
        return raul::Dimension::Batch;
    }
    return raul::Dimension::Default;
}

}

namespace raul
{

TilingParams::TilingParams(const raul::Name& input, const raul::Name& output, const std::initializer_list<size_t> repeats)
    : BasicParamsWithDim(Names(1, input), Names(1, output), raul::Dimension::Default)
{
    if (repeats.size() != 3)
    {
        THROW_NONAME("TilingParams", "3 numbers should be provided to duplicate whole tensor");
    }

    mRepeatDepth = *repeats.begin();
    mRepeatHeight = *(repeats.begin() + 1);
    mRepeatWidth = *(repeats.begin() + 2);
}

TilingParams::TilingParams(const raul::Name& input, const raul::Name& output, size_t multiple, raul::Dimension paramDim)
    : BasicParamsWithDim(Names(1, input), Names(1, output), paramDim)
    , mRepeatDepth(1u)
    , mRepeatHeight(1u)
    , mRepeatWidth(1u)
{
    if (dim == raul::Dimension::Depth)
    {
        mRepeatDepth *= multiple;
    }
    else if (dim == raul::Dimension::Height)
    {
        mRepeatHeight *= multiple;
    }
    else if (dim == raul::Dimension::Width)
    {
        mRepeatWidth *= multiple;
    }
    else
    {
        mRepeatDepth *= multiple;
        mRepeatHeight *= multiple;
        mRepeatWidth *= multiple;
    }
}

TilingParams::TilingParams(const raul::Name& input, const raul::Name& output, size_t multiple, const std::string& paramDim)
    : TilingParams(input, output, multiple, stringToDimension(paramDim))
{
}

void TilingParams::print(std::ostream& stream) const
{
    BasicParamsWithDim::print(stream);
    stream << "repeats: " << mRepeatDepth << " (Depth) " << mRepeatHeight << " (Height) " << mRepeatWidth << " (Width)";
}

}