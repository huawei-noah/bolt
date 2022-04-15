// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef API_HYPERPARAMS_H
#define API_HYPERPARAMS_H

struct NIN_hyperparams_t
{
    size_t conv1_filters;
    size_t conv1_kernel_size;
    size_t conv1_stride;
    size_t conv1_padding;

    size_t conv2_filters;
    size_t conv2_kernel_size;

    size_t conv3_filters;
    size_t conv3_kernel_size;

    size_t maxpool_kernel;
    size_t maxpool_stride;
    size_t maxpool_padding;

    size_t conv4_filters;
    size_t conv4_kernel_size;
    size_t conv4_stride;
    size_t conv4_padding;

    size_t conv5_filters;
    size_t conv5_kernel_size;

    size_t conv6_filters;
    size_t conv6_kernel_size;

    size_t avgpool1_kernel;
    size_t avgpool1_stride;
    size_t avgpool1_padding;

    size_t conv7_filters;
    size_t conv7_kernel_size;
    size_t conv7_stride;
    size_t conv7_padding;

    size_t conv8_filters;
    size_t conv8_kernel_size;

    size_t conv9_filters;
    size_t conv9_kernel_size;

    size_t avgpool2_kernel;
    size_t avgpool2_stride;
};

struct MobileNetV2_hyperparams_t
{
    float bnMomentum;

    size_t num_classes;

    static const size_t reproduceLayers = 16;
    size_t filterSizes[reproduceLayers][3];
    size_t strideSizes[reproduceLayers];
    bool residual[reproduceLayers];

    size_t lastLayerSize;

    size_t avgWidth;
};

struct ResNet18_hyperparams_t
{
};

#endif