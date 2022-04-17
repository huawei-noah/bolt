// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "RepeatInterleaveParams.h"

namespace raul
{

RepeatInterleaveParams::RepeatInterleaveParams(const raul::Name& input, const raul::Name& output, std::vector<size_t> repeats, raul::Dimension paramDim)
    : BasicParamsWithDim(Names(1, input), Names(1, output), paramDim)
    , mRepeats(std::move(repeats))
{
}

RepeatInterleaveParams::RepeatInterleaveParams(const raul::Name& input, const raul::Name& output, std::vector<size_t> repeats, const std::string& paramDim)
    : BasicParamsWithDim(Names(1, input), Names(1, output), paramDim)
    , mRepeats(std::move(repeats))
{
}

void RepeatInterleaveParams::print(std::ostream& stream) const
{
    BasicParamsWithDim::print(stream);
    stream << ", repeats: ";
    for (auto& val : mRepeats)
    {
        stream << val << " ";
    }
}

} // namespace raul