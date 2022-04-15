// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "PostNet.h"

#include <training/api/API.h>
#include <training/layers/activations/TanhActivation.h>
#include <training/layers/basic/DropoutLayer.h>
#include <training/layers/basic/trainable/Batchnorm.h>
#include <training/layers/basic/trainable/Convolution1DLayer.h>

namespace
{
using namespace std;
using namespace raul;

void AddConv1d(Workflow* work, const Name& name, const BasicParams& params, size_t kernelSize, size_t channels, float dropout, bool batchNormBeforeActivation, bool activation, bool frozen = false)
{
    auto input = params.getInputs()[0];
    auto output = params.getOutputs()[0];
    auto nextName = name / "conv1d_output";

    auto parent = params.getSharedLayer().empty() ? Name{} : params.getSharedLayer() / "conv1d";

    work->add<Convolution1DLayer>(name / "conv1d", Convolution1DParams{ input, nextName, parent, kernelSize, channels, 1U, kernelSize / 2, 1U, 1U, true, false, true, frozen });

    if (activation && !batchNormBeforeActivation)
    {
        work->add<TanhActivation>(name / "activation", BasicParams{ { nextName }, { name / "activated" } });
        nextName = name / "activated";
    }

    parent = params.getSharedLayer().empty() ? Name{} : params.getSharedLayer() / "batch_norm";
    work->add<BatchNormLayer>(name / "batch_norm", BatchnormParams{ { nextName }, { name / "batched" }, 0.01f, 1e-3f, "width", frozen });
    nextName = name / "batched";

    if (activation && batchNormBeforeActivation)
    {
        work->add<TanhActivation>(name / "activation", BasicParams{ { nextName }, { name / "activated" } });
        nextName = name / "activated";
    }

    work->add<DropoutLayer>(name / "dropout", DropoutParams{ { nextName }, { output }, dropout });
}
}

namespace raul::tacotron
{
using namespace std;

void AddPostNet(Workflow* work, const Name& name, const BasicParams& params, const TacotronParams& tparams)
{
    auto input = params.getInputs()[0];
    auto output = params.getOutputs()[0];
    auto inp = input;

    for (size_t i = 0; i < tparams.postnetKernelSize.size(); ++i)
    {
        auto parent = params.getSharedLayer().empty() ? Name() : params.getSharedLayer() / ("conv1d[" + to_string(i + 1) + "]");
        auto opName = name / ("conv1d[" + to_string(i + 1) + "]");
        auto outp = opName / "out";
        bool activation = true;

        if (i == tparams.postnetKernelSize.size() - 1)
        {
            activation = false;
            outp = output;
        }

        AddConv1d(work,
                  opName,
                  BasicParams{ { inp }, { outp }, parent },
                  tparams.postnetKernelSize[i],
                  tparams.postnetChannels,
                  tparams.postnetDropoutRate,
                  tparams.batchnormBeforeActivation,
                  activation,
                  tparams.frozen);

        inp = outp;
    }
}

} // namespace raul::tacotron
