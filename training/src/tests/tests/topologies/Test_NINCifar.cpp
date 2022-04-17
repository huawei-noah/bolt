// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <training/base/layers/BasicLayer.h>
#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>
#include <training/base/optimizers/SGD.h>

namespace UT
{

TEST(TestNINCifar, TopologyUnit)
{
    PROFILE_TEST
    const size_t golden_trainable_parameters = 966986U;

    const size_t BATCH_SIZE = 64;

    const size_t NUM_CLASSES = 10;
    const size_t IMAGE_SIZE = 32;
    const size_t IMAGE_CHANNELS = 3;

    const size_t CONV1_FILTERS = 192;
    const size_t CONV1_KERNEL_SIZE = 5;
    const size_t CONV1_STRIDE = 1;
    const size_t CONV1_PADDING = 2;

    const size_t CONV2_FILTERS = 160;
    const size_t CONV2_KERNEL_SIZE = 1;

    const size_t CONV3_FILTERS = 96;
    const size_t CONV3_KERNEL_SIZE = 1;

    const size_t MAXPOOL_KERNEL = 3;
    const size_t MAXPOOL_STRIDE = 2;
    const size_t MAXPOOL_PADDING = 1;

    const size_t CONV4_FILTERS = 192;
    const size_t CONV4_KERNEL_SIZE = 5;
    const size_t CONV4_STRIDE = 1;
    const size_t CONV4_PADDING = 2;

    const size_t CONV5_FILTERS = 192;
    const size_t CONV5_KERNEL_SIZE = 1;

    const size_t CONV6_FILTERS = 192;
    const size_t CONV6_KERNEL_SIZE = 1;

    const size_t AVGPOOL1_KERNEL = 3;
    const size_t AVGPOOL1_STRIDE = 2;
    const size_t AVGPOOL1_PADDING = 1;

    const size_t CONV7_FILTERS = 192;
    const size_t CONV7_KERNEL_SIZE = 3;
    const size_t CONV7_STRIDE = 1;
    const size_t CONV7_PADDING = 1;

    const size_t CONV8_FILTERS = 192;
    const size_t CONV8_KERNEL_SIZE = 1;

    const size_t CONV9_FILTERS = 10;
    const size_t CONV9_KERNEL_SIZE = 1;

    const size_t AVGPOOL2_KERNEL = 8;
    const size_t AVGPOOL2_STRIDE = 1;

    printf("begin creating graph\n");
    raul::Workflow work;
    work.add<raul::DataLayer>("data", raul::DataParams{ { "data", "labels" }, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES });

    work.add<raul::Convolution2DLayer>("conv1", raul::Convolution2DParams{ { "data" }, { "conv1" }, CONV1_KERNEL_SIZE, CONV1_FILTERS, CONV1_STRIDE, CONV1_PADDING });
    work.add<raul::ReLUActivation>("relu1", raul::BasicParams{ { "conv1" }, { "relu1" } });
    work.add<raul::Convolution2DLayer>("conv2", raul::Convolution2DParams{ { "relu1" }, { "conv2" }, CONV2_KERNEL_SIZE, CONV2_FILTERS });
    work.add<raul::ReLUActivation>("relu2", raul::BasicParams{ { "conv2" }, { "relu2" } });
    work.add<raul::Convolution2DLayer>("conv3", raul::Convolution2DParams{ { "relu2" }, { "conv3" }, CONV3_KERNEL_SIZE, CONV3_FILTERS });
    work.add<raul::ReLUActivation>("relu3", raul::BasicParams{ { "conv3" }, { "relu3" } });
    work.add<raul::MaxPoolLayer2D>("mp", raul::Pool2DParams{ { "relu3" }, { "mp" }, MAXPOOL_KERNEL, MAXPOOL_STRIDE, MAXPOOL_PADDING });

    work.add<raul::Convolution2DLayer>("conv4", raul::Convolution2DParams{ { "mp" }, { "conv4" }, CONV4_KERNEL_SIZE, CONV4_FILTERS, CONV4_STRIDE, CONV4_PADDING });
    work.add<raul::ReLUActivation>("relu4", raul::BasicParams{ { "conv4" }, { "relu4" } });
    work.add<raul::Convolution2DLayer>("conv5", raul::Convolution2DParams{ { "relu4" }, { "conv5" }, CONV5_KERNEL_SIZE, CONV5_FILTERS });
    work.add<raul::ReLUActivation>("relu5", raul::BasicParams{ { "conv5" }, { "relu5" } });
    work.add<raul::Convolution2DLayer>("conv6", raul::Convolution2DParams{ { "relu5" }, { "conv6" }, CONV6_KERNEL_SIZE, CONV6_FILTERS });
    work.add<raul::ReLUActivation>("relu6", raul::BasicParams{ { "conv6" }, { "relu6" } });
    work.add<raul::AveragePoolLayer>("avg1", raul::Pool2DParams{ { "relu6" }, { "avg1" }, AVGPOOL1_KERNEL, AVGPOOL1_STRIDE, AVGPOOL1_PADDING });

    work.add<raul::Convolution2DLayer>("conv7", raul::Convolution2DParams{ { "avg1" }, { "conv7" }, CONV7_KERNEL_SIZE, CONV7_FILTERS, CONV7_STRIDE, CONV7_PADDING });
    work.add<raul::ReLUActivation>("relu7", raul::BasicParams{ { "conv7" }, { "relu7" } });
    work.add<raul::Convolution2DLayer>("conv8", raul::Convolution2DParams{ { "relu7" }, { "conv8" }, CONV8_KERNEL_SIZE, CONV8_FILTERS });
    work.add<raul::ReLUActivation>("relu8", raul::BasicParams{ { "conv8" }, { "relu8" } });
    work.add<raul::Convolution2DLayer>("conv9", raul::Convolution2DParams{ { "relu8" }, { "conv9" }, CONV9_KERNEL_SIZE, CONV9_FILTERS });
    work.add<raul::ReLUActivation>("relu9", raul::BasicParams{ { "conv9" }, { "relu9" } });
    work.add<raul::AveragePoolLayer>("avg2", raul::Pool2DParams{ { "relu9" }, { "avg2" }, AVGPOOL2_KERNEL, AVGPOOL2_STRIDE });

    work.add<raul::SoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "avg2" }, { "softmax" } });
    work.add<raul::CrossEntropyLoss>("loss", raul::LossParams{ { "softmax", "labels" }, { "loss" } });

    work.preparePipelines();
    work.setBatchSize(BATCH_SIZE);
    work.prepareMemoryForTraining();

    printf("end creating graph\n");

    // Checks
    EXPECT_EQ(tools::get_size_of_trainable_params(work), golden_trainable_parameters);
}

} // UT namespace