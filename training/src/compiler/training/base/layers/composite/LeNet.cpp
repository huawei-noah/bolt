// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LeNet.h"

#include <training/compiler/Layers.h>

namespace raul
{

void AddLeNetModel(Workflow* work, const Name& name, const BasicParams& params, size_t numClasses, bool includeTop)
{
    if (params.getInputs().size() != 1)
    {
        THROW_NONAME("AddLeNetModel", "Wrong number of inputs: 1 expected, " + std::to_string(params.getInputs().size()) + " provided");
    }
    if (params.getOutputs().size() != 1)
    {
        THROW_NONAME("AddLeNetModel", "Wrong number of outputs: 1 expected, " + std::to_string(params.getOutputs().size()) + " provided");
    }

    const auto& input = params.getInputs()[0];
    const auto& output = params.getOutputs()[0];

    work->add<Convolution2DLayer>(name / "conv1", Convolution2DParams{ { input }, { name / "c1_out" }, 5, 6 });
    work->add<ReLUActivation>(name / "relu1", BasicParams{ { name / "c1_out" }, { name / "c1_relu" } });
    work->add<MaxPoolLayer2D>(name / "pool1", Pool2DParams{ { name / "c1_relu" }, { name / "pool1_out" }, 2, 2 });

    work->add<Convolution2DLayer>(name / "conv2", Convolution2DParams{ { name / "pool1_out" }, { name / "c2_out" }, 5, 16 });
    work->add<ReLUActivation>(name / "relu2", BasicParams{ { name / "c2_out" }, { name / "c2_relu" } });
    work->add<MaxPoolLayer2D>(name / "pool2", Pool2DParams{ { name / "c2_relu" }, { includeTop ? name / "pool2_out" : output }, 2, 2 });

    if (includeTop)
    {
        work->add<ReshapeLayer>(name / "flatten", ViewParams{ name / "pool2_out", name / "flatten", 1, 1, -1 });
        work->add<LinearLayer>(name / "fc1", LinearParams{ name / "flatten", name / "fc1_out", 120 });
        work->add<ReLUActivation>(name / "relu3", BasicParams{ { name / "fc1_out" }, { name / "fc1_relu" } });
        work->add<LinearLayer>(name / "fc2", LinearParams{ name / "fc1_relu", name / "fc2_out", 84 });
        work->add<ReLUActivation>(name / "relu4", BasicParams{ { name / "fc2_out" }, { name / "fc2_relu" } });
        work->add<LinearLayer>(name / "fc3", LinearParams{ name / "fc2_relu", output, numClasses });
    }
}

}