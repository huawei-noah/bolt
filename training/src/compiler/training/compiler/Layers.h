// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LAYERS_H
#define LAYERS_H

#include <training/base/layers/basic/ArgExtremumLayer.h>
#include <training/base/layers/basic/ArgMaxLayer.h>
#include <training/base/layers/basic/ArgMinLayer.h>
#include <training/base/layers/basic/AveragePoolLayer.h>
#include <training/base/layers/basic/BatchExpanderLayer.h>
#include <training/base/layers/basic/ClampLayer.h>
#include <training/base/layers/basic/ConcatenationLayer.h>
#include <training/base/layers/basic/CumSumLayer.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/layers/basic/DropoutLayer.h>
#include <training/base/layers/basic/DynamicDepthwiseConvolution2DLayer.h>
#include <training/base/layers/basic/ElementWiseCompareLayer.h>
#include <training/base/layers/basic/ElementWiseDivLayer.h>
#include <training/base/layers/basic/ElementWiseExtremumLayer.h>
#include <training/base/layers/basic/ElementWiseMaxLayer.h>
#include <training/base/layers/basic/ElementWiseMinLayer.h>
#include <training/base/layers/basic/ElementWiseMulLayer.h>
#include <training/base/layers/basic/ElementWiseSubLayer.h>
#include <training/base/layers/basic/ElementWiseSumLayer.h>
#include <training/base/layers/basic/ExpLayer.h>
#include <training/base/layers/basic/FakeQuantLayer.h>
#include <training/base/layers/basic/FixedBiasLayer.h>
#include <training/base/layers/basic/GlobalAveragePoolLayer.h>
#include <training/base/layers/basic/IndexFillLayer.h>
#include <training/base/layers/basic/L2NormLayer.h>
#include <training/base/layers/basic/L2SquaredNormLayer.h>
#include <training/base/layers/basic/LabelSmoothing.h>
#include <training/base/layers/basic/LogLayer.h>
#include <training/base/layers/basic/LossWrapperHelperLayer.h>
#include <training/base/layers/basic/MaskedFillLayer.h>
#include <training/base/layers/basic/MatMulLayer.h>
#include <training/base/layers/basic/MaxPoolLayer.h>
#include <training/base/layers/basic/PaddingLayer.h>
#include <training/base/layers/basic/PositionalEncoding.h>
#include <training/base/layers/basic/RSqrtLayer.h>
#include <training/base/layers/basic/RandomChoiceLayer.h>
#include <training/base/layers/basic/RandomSelectLayer.h>
#include <training/base/layers/basic/RandomTensorLayer.h>
#include <training/base/layers/basic/ReduceArithmeticLayer.h>
#include <training/base/layers/basic/ReduceBatchMeanLayer.h>
#include <training/base/layers/basic/ReduceExtremumLayer.h>
#include <training/base/layers/basic/ReduceMaxLayer.h>
#include <training/base/layers/basic/ReduceMeanLayer.h>
#include <training/base/layers/basic/ReduceMinLayer.h>
#include <training/base/layers/basic/ReduceNonZeroLayer.h>
#include <training/base/layers/basic/ReduceStdLayer.h>
#include <training/base/layers/basic/ReduceSumLayer.h>
#include <training/base/layers/basic/RepeatInterleaveLayer.h>
#include <training/base/layers/basic/ReshapeLayer.h>
#include <training/base/layers/basic/ReverseLayer.h>
#include <training/base/layers/basic/RollLayer.h>
#include <training/base/layers/basic/ScaleLayer.h>
#include <training/base/layers/basic/SelectLayer.h>
#include <training/base/layers/basic/SlicerLayer.h>
#include <training/base/layers/basic/SplitterLayer.h>
#include <training/base/layers/basic/SqrtLayer.h>
#include <training/base/layers/basic/SquareLayer.h>
#include <training/base/layers/basic/TensorLayer.h>
#include <training/base/layers/basic/TileLayer.h>
#include <training/base/layers/basic/TransposeLayer.h>

#include <training/base/layers/basic/trainable/Batchnorm.h>
#include <training/base/layers/basic/trainable/Convolution1DLayer.h>
#include <training/base/layers/basic/trainable/Convolution2DLayer.h>
#include <training/base/layers/basic/trainable/ConvolutionDepthwiseLayer.h>
#include <training/base/layers/basic/trainable/Embedding.h>
#include <training/base/layers/basic/trainable/LayerNorm.h>
#include <training/base/layers/basic/trainable/LayerNorm2D.h>
#include <training/base/layers/basic/trainable/LinearLayer.h>
#include <training/base/layers/basic/trainable/TransposedConvolution1DLayer.h>
#include <training/base/layers/basic/trainable/TransposedConvolution2DLayer.h>

#include <training/base/layers/composite/AdditiveAttentionLayer.h>
#include <training/base/layers/composite/rnn/GRUFusedGatesCalcLayer.h>
#include <training/base/layers/composite/rnn/GRULayer.h>

#include <training/base/layers/activations/GeLUActivation.h>
#include <training/base/layers/activations/HSigmoidActivation.h>
#include <training/base/layers/activations/HSwishActivation.h>
#include <training/base/layers/activations/LeakyReLUActivation.h>
#include <training/base/layers/activations/LogSoftMaxActivation.h>
#include <training/base/layers/activations/ReLUActivation.h>
#include <training/base/layers/activations/SigmoidActivation.h>
#include <training/base/layers/activations/SoftMaxActivation.h>
#include <training/base/layers/activations/SwishActivation.h>
#include <training/base/layers/activations/TanhActivation.h>

#include <training/base/loss/BinaryCrossEntropyLoss.h>
#include <training/base/loss/CrossEntropyLoss.h>
#include <training/base/loss/KLDivLoss.h>
#include <training/base/loss/L1Loss.h>
#include <training/base/loss/MSELoss.h>
#include <training/base/loss/NegativeLogLikelihoodLoss.h>
#include <training/base/loss/SigmoidCrossEntropyLoss.h>
#include <training/base/loss/SoftmaxCrossEntropyLoss.h>

#endif