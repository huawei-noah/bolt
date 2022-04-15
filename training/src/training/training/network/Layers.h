// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

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

#include <training/layers/basic/ArgExtremumLayer.h>
#include <training/layers/basic/ArgMaxLayer.h>
#include <training/layers/basic/ArgMinLayer.h>
#include <training/layers/basic/AveragePoolLayer.h>
#include <training/layers/basic/BatchExpanderLayer.h>
#include <training/layers/basic/ClampLayer.h>
#include <training/layers/basic/ConcatenationLayer.h>
#include <training/layers/basic/ConvertPrecisionLayer.h>
#include <training/layers/basic/CumSumLayer.h>
#include <training/layers/basic/DataLayer.h>
#include <training/layers/basic/DropoutLayer.h>
#include <training/layers/basic/DynamicDepthwiseConvolution2DLayer.h>
#include <training/layers/basic/ElementWiseCompareLayer.h>
#include <training/layers/basic/ElementWiseDivLayer.h>
#include <training/layers/basic/ElementWiseExtremumLayer.h>
#include <training/layers/basic/ElementWiseMaxLayer.h>
#include <training/layers/basic/ElementWiseMinLayer.h>
#include <training/layers/basic/ElementWiseMulLayer.h>
#include <training/layers/basic/ElementWiseSubLayer.h>
#include <training/layers/basic/ElementWiseSumLayer.h>
#include <training/layers/basic/ExpLayer.h>
#include <training/layers/basic/FakeQuantLayer.h>
#include <training/layers/basic/FixedBiasLayer.h>
#include <training/layers/basic/GlobalAveragePoolLayer.h>
#include <training/layers/basic/IndexFillLayer.h>
#include <training/layers/basic/L2NormLayer.h>
#include <training/layers/basic/L2SquaredNormLayer.h>
#include <training/layers/basic/LabelSmoothing.h>
#include <training/layers/basic/LogLayer.h>
#include <training/layers/basic/LossWrapperHelperLayer.h>
#include <training/layers/basic/MaskedFillLayer.h>
#include <training/layers/basic/MatMulLayer.h>
#include <training/layers/basic/MaxPoolLayer.h>
#include <training/layers/basic/PaddingLayer.h>
#include <training/layers/basic/PositionalEncoding.h>
#include <training/layers/basic/RSqrtLayer.h>
#include <training/layers/basic/RandomChoiceLayer.h>
#include <training/layers/basic/RandomSelectLayer.h>
#include <training/layers/basic/RandomTensorLayer.h>
#include <training/layers/basic/ReduceArithmeticLayer.h>
#include <training/layers/basic/ReduceBatchMeanLayer.h>
#include <training/layers/basic/ReduceExtremumLayer.h>
#include <training/layers/basic/ReduceMaxLayer.h>
#include <training/layers/basic/ReduceMeanLayer.h>
#include <training/layers/basic/ReduceMinLayer.h>
#include <training/layers/basic/ReduceNonZeroLayer.h>
#include <training/layers/basic/ReduceStdLayer.h>
#include <training/layers/basic/ReduceSumLayer.h>
#include <training/layers/basic/RepeatInterleaveLayer.h>
#include <training/layers/basic/ReshapeLayer.h>
#include <training/layers/basic/ReverseLayer.h>
#include <training/layers/basic/RollLayer.h>
#include <training/layers/basic/ScaleLayer.h>
#include <training/layers/basic/SelectLayer.h>
#include <training/layers/basic/SlicerLayer.h>
#include <training/layers/basic/SplitterLayer.h>
#include <training/layers/basic/SqrtLayer.h>
#include <training/layers/basic/SquareLayer.h>
#include <training/layers/basic/TensorLayer.h>
#include <training/layers/basic/TileLayer.h>
#include <training/layers/basic/TransposeLayer.h>

#include <training/layers/basic/trainable/Batchnorm.h>
#include <training/layers/basic/trainable/Convolution1DLayer.h>
#include <training/layers/basic/trainable/Convolution2DLayer.h>
#include <training/layers/basic/trainable/ConvolutionDepthwiseLayer.h>
#include <training/layers/basic/trainable/Embedding.h>
#include <training/layers/basic/trainable/LayerNorm.h>
#include <training/layers/basic/trainable/LayerNorm2D.h>
#include <training/layers/basic/trainable/LinearLayer.h>
#include <training/layers/basic/trainable/TransposedConvolution1DLayer.h>
#include <training/layers/basic/trainable/TransposedConvolution2DLayer.h>

#include <training/layers/composite/AdditiveAttentionLayer.h>
#include <training/layers/composite/rnn/GRUFusedGatesCalcLayer.h>
#include <training/layers/composite/rnn/GRULayer.h>

#include <training/layers/activations/GeLUActivation.h>
#include <training/layers/activations/HSigmoidActivation.h>
#include <training/layers/activations/HSwishActivation.h>
#include <training/layers/activations/LeakyReLUActivation.h>
#include <training/layers/activations/LogSoftMaxActivation.h>
#include <training/layers/activations/ReLUActivation.h>
#include <training/layers/activations/SigmoidActivation.h>
#include <training/layers/activations/SoftMaxActivation.h>
#include <training/layers/activations/SwishActivation.h>
#include <training/layers/activations/TanhActivation.h>

#include <training/loss/BinaryCrossEntropyLoss.h>
#include <training/loss/CrossEntropyLoss.h>
#include <training/loss/KLDivLoss.h>
#include <training/loss/L1Loss.h>
#include <training/loss/MSELoss.h>
#include <training/loss/NegativeLogLikelihoodLoss.h>
#include <training/loss/SigmoidCrossEntropyLoss.h>
#include <training/loss/SoftmaxCrossEntropyLoss.h>

#endif
