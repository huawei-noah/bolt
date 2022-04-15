// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SlicingParams.h"

namespace raul
{

SlicingParams::SlicingParams(const Name& input, const Names& outputs, const std::string& paramDimStr, std::vector<int> slice_sizes)
    : BasicParamsWithDim(Names(1, input), outputs, paramDimStr == "default" ? "width" : paramDimStr)
    , sliceSize(slice_sizes)
{
}

SlicingParams::SlicingParams(const Name& input, const Names& outputs, Dimension paramDim, std::vector<int> slice_sizes)
    : BasicParamsWithDim(Names(1, input), outputs, paramDim == Dimension::Default ? Dimension::Width : paramDim)
    , sliceSize(slice_sizes)
{
}

void SlicingParams::print(std::ostream& stream) const
{
    BasicParamsWithDim::print(stream);
    if (sliceSize.empty())
    {
        stream << ", slices: " << getOutputs().size();
    }
    else
    {
        stream << ", slice_size: ";
        for (auto& i : sliceSize)
            stream << i << " ";
    }
}

} // namespace raul
