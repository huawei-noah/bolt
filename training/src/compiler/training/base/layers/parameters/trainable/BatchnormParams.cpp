// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "BatchnormParams.h"

namespace raul
{

TrainableBasicParamsWithDim::TrainableBasicParamsWithDim(const Names& inputs, const Names& outputs, const Name& sharedLayer, const std::string& paramDim, bool frozen)
    : TrainableParams(inputs, outputs, sharedLayer, frozen)
{
    std::string r = paramDim;
    if (r == "width")
        dim = Dimension::Width;
    else if (r == "height")
        dim = Dimension::Height;
    else if (r == "depth")
        dim = Dimension::Depth;
    else if (r == "batch")
        dim = Dimension::Batch;
    else if (r == "default")
        dim = Dimension::Default;
    else
        THROW_NONAME("TrainableBasicParamsWithDim", "Unknown dimension: " + r);
}

TrainableBasicParamsWithDim::TrainableBasicParamsWithDim(const Names& inputs, const Names& outputs, const Name& sharedLayer, Dimension paramDim, bool frozen)
    : TrainableParams(inputs, outputs, sharedLayer, frozen)
    , dim(paramDim)
{
}

void TrainableBasicParamsWithDim::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    std::string s = std::map<Dimension, std::string>{
        { Dimension::Default, "default" }, { Dimension::Width, "width" }, { Dimension::Height, "height" }, { Dimension::Depth, "depth" }, { Dimension::Batch, "batch" }
    }[dim];
    stream << "dim: " << s;
}

void BatchnormParams::print(std::ostream& stream) const
{
    TrainableBasicParamsWithDim::print(stream);
    stream << "momentum: " << momentum << " eps:" << eps;
}

} // namespace raul