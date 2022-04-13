// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LSTM_CELL_LAYER_H
#define LSTM_CELL_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/TrainableLayer.h>
#include <training/base/layers/parameters/trainable/LSTMCellParams.h>

#include <optional>

namespace raul
{

/**
 * @brief Long Short-Term Memory Cell
 *
 * @see
 * - S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory” Neural Comput., vol. 9, no. 8, pp. 1735–1780, Nov. 1997, doi: 10.1162/neco.1997.9.8.1735.
 */
class LSTMCellLayer
{
  public:
    LSTMCellLayer(const Name& name, const LSTMCellParams& params, NetworkParameters& networkParameters);

    LSTMCellLayer(LSTMCellLayer&&) = default;
    LSTMCellLayer(const LSTMCellLayer&) = delete;
    LSTMCellLayer& operator=(const LSTMCellLayer&) = delete;

  private:
    void buildLayer(const Name& name, const LSTMCellParams& params, NetworkParameters& networkParameters);
};

} // raul namespace

#endif