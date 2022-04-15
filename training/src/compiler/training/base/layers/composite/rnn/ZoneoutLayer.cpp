// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ZoneoutLayer.h"
#include <training/base/layers/basic/DropoutLayer.h>
#include <training/base/layers/basic/ElementWiseSumLayer.h>
#include <training/base/layers/basic/RandomSelectLayer.h>
#include <training/base/layers/basic/ScaleLayer.h>
#include <training/base/layers/basic/SelectLayer.h>

namespace raul
{

ZoneoutLayer::ZoneoutLayer(const Name& name, const ZoneoutParams& params, NetworkParameters& networkParameters)
    : RandomSelectLayer(name, RandomSelectParams{ params.getInputs(), params.getOutputs(), 1_dt - params.mProbability }, networkParameters)
{
}

void ZoneoutLayer::forwardComputeImpl(NetworkMode mode)
{
    if (mode == NetworkMode::Train || mode == NetworkMode::TrainCheckpointed)
    {
        RandomSelectLayer::forwardComputeImpl(mode);
    }
    else
    {
        THROW("ZoneoutLayer", mName, "Test mode not implemented");
    }
}

} // namespace raul
