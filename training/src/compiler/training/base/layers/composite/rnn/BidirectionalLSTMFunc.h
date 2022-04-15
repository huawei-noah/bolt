// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef BIDIRECTIONAL_LSTM_FUNC_H
#define BIDIRECTIONAL_LSTM_FUNC_H

#include "training/base/layers/parameters/BasicParameters.h"
#include "training/base/layers/parameters/trainable/LSTMCellParams.h"
#include <training/base/common/Common.h>
#include <training/base/layers/basic/ConcatenationLayer.h>
#include <training/base/layers/composite/rnn/LSTMLayer.h>

namespace raul
{

/**
 * @brief Bidirectional configuration for sequence processing layers
 *
 */

enum class BidirectionalMergeType
{
    ConcatWidth,
    ConcatHeight,
    ConcatDepth,
    Sum,
    Mul
};

void BidirectionalLSTMFunc(const Name& name, const LSTMParams& params, NetworkParameters& networkParameters, BidirectionalMergeType mergeType = BidirectionalMergeType::ConcatWidth);

} // raul namespace

#endif
