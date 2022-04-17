// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "DataLayer.h"

#include <algorithm>

#include <training/base/impl/ImplFactory.h>

namespace
{
// to override compiler checks
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::DataLayer, raul::DummyImpl>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::DataLayer, raul::DummyImpl>();
} // anonymous namespace

namespace raul
{
DataLayer::DataLayer(const Name& name, const DataParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "Data", params, networkParameters, { false, true })
{
    if (!params.getInputs().empty())
    {
        THROW("DataLayer", mName, "input names not allowed");
    }
    if (params.getOutputs().empty())
    {
        THROW("DataLayer", mName, "no output names");
    }

    if (std::any_of(params.getOutputs().begin(), params.getOutputs().end(), [](const auto& s) { return s.empty(); }))
    {
        THROW("DataLayer", mName, "empty output name");
    }

    auto end = params.getOutputs().end();
    if (params.labelsCount > 0)
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, params.getOutputs().back(), WShape{ BS(), 1u, 1u, params.labelsCount }, DEC_FRBC_READ_NOMEMOPT);
        end -= 1;
    }

    for (auto s = params.getOutputs().begin(); s != end; ++s)
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, *s, WShape{ BS(), params.depth, params.height, params.width }, DEC_FORW_READ_NOMEMOPT);
    }
}
} // namespace raul