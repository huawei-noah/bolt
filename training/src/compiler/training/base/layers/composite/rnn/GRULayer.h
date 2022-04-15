// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GRU_LAYER_H
#define GRU_LAYER_H

#include "training/base/layers/parameters/trainable/GRUParams.h"
#include <training/base/common/Common.h>
#include <training/base/layers/composite/rnn/GRUCellLayer.h>

namespace raul
{

/**
 * @brief Gated Recurrent Unit Cell
 *
 * @see
 * - K. Cho, B. Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio
 * “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation” arXiv preprint arXiv:1406.1078 (2014).
 */
class GRULayer
{

  public:
    GRULayer(const Name& name, const GRUParams& params, NetworkParameters& networkParameters);

    GRULayer(GRULayer&&) = default;
    GRULayer(const GRULayer&) = delete;
    GRULayer& operator=(const GRULayer&) = delete;

  private:
    void verifyInOut(const Name& name, const GRUParams& params) const;
    void initLocalState(const Name& name, const GRUParams& params, NetworkParameters& networkParameters);
    void buildGRULayer(const Name& name, const GRUParams& params, NetworkParameters& networkParameters) const;

    Name mHidden;
    bool mIsExternalState;
    size_t mLengthSequence;
    std::string mSequenceDimension;
};
} // raul namespace

#endif
