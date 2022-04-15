// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <tests/tools/TestTools.h>

#include <ModelConverter.h>
#include <training/common/Common.h>
#include <training/common/Conversions.h>
#include <training/layers/BasicLayer.h>
#include <training/layers/parameters/Parameters.h>
#include <training/network/Graph.h>
#include <training/tools/Datasets.h>

namespace UT
{

TEST(TestConverter, Unit)
{
    PROFILE_TEST("TestConverter.Unit")
    const size_t BATCH_SIZE = 50;
    const size_t NUM_CLASSES = 10;
    const size_t MNIST_SIZE = 28;
    const size_t FC1_SIZE = 500;
    const size_t FC2_SIZE = 100;

    raul::NetDef netdef;
    netdef.addOp("data", "OpDataLayer", createParam(raul::DataLayerParams{ {}, { "data", "labels" }, MNIST_SIZE, MNIST_SIZE, 1, NUM_CLASSES, BATCH_SIZE }));
    netdef.addOp("fc1", "OpFCLayer", createParam(raul::OutputParams{ { "data" }, { "fc1" }, FC1_SIZE }));
    netdef.addOp("tanh", "OpTanhActivation", createParam(raul::BasicParams{ { "fc1" }, { "tanh" } }));
    netdef.addOp("fc2", "OpFCLayer", createParam(raul::OutputParams{ { "tanh" }, { "fc2" }, FC2_SIZE }));
    netdef.addOp("sigmoid", "OpSigmoidActivation", createParam(raul::BasicParams{ { "fc2" }, { "sigmoid" } }));
    netdef.addOp("fc3", "OpFCLayer", createParam(raul::OutputParams{ { "sigmoid" }, { "fc3" }, NUM_CLASSES }));
    netdef.addOp("softmax", "OpSoftmaxActivation", createParam(raul::BasicParams{ { "fc3" }, { "softmax" } }));
    netdef.addOp("loss", "OpCrossEntropyLoss", createParam(raul::BasicParams{ { "softmax", "labels" }, { "loss" } }));

    raul::Graph network(std::move(netdef));

    network("fc1")->setWeights(raul::Common::loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc1.weight.data", FC1_SIZE, MNIST_SIZE * MNIST_SIZE));
    network("fc1")->setBiases(raul::Common::loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc1.bias.data", 1, FC1_SIZE));
    network("fc2")->setWeights(raul::Common::loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc2.weight.data", FC2_SIZE, FC1_SIZE));
    network("fc2")->setBiases(raul::Common::loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc2.bias.data", 1, FC2_SIZE));
    network("fc3")->setWeights(raul::Common::loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc3.weight.data", NUM_CLASSES, FC2_SIZE));
    network("fc3")->setBiases(raul::Common::loadData(tools::getTestAssetsDir() / "test_fc_layer" / "mnist" / "0_fc3.bias.data", 1, NUM_CLASSES));

    std::ofstream s("f.bin", std::ios::binary);

    raul::SerializeToBolt(network, s);
}

} // UT namespace