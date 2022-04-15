// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef RECURRENT_META_LAYER_H
#define RECURRENT_META_LAYER_H

#include "training/layers/MetaLayer.h"
// d.polubotko(TODO): implement
#if 0
#include <training/layers/parameters/RecurrentParams.h>

namespace raul
{

/**
 * @brief Recurrent Meta Layer
 *
 * This layer represents subgraph in which operator is mapped over one dimension
 * which represents a sequence. It is a so-called sequence to sequence mapping.
 *
 * Additionally to operator description, it has to be provided the length of the sequence,
 * dimension over which the operator is mapped and the name of sequence input tensor and output sequence tensor.
 *
 * Technically, it is implemented using graph unrolling mechanism. If the numbers of inputs and outputs tensors of
 * the mapping operator are not equal this can lead to ambiguity in interconnect.
 * To resolve this, the in-out name dictionary must be provided.
 *
 */
class RecurrentMetaLayer : public MetaLayer
{

  public:
    RecurrentMetaLayer(const Name& name, const RecurrentParams& params, raul::NetworkParameters& networkParameters);

    RecurrentMetaLayer(RecurrentMetaLayer&&) = default;
    RecurrentMetaLayer(const RecurrentMetaLayer&) = delete;
    RecurrentMetaLayer& operator=(const RecurrentMetaLayer&) = delete;
};
} // raul namespace

#endif

#endif // SEQUENTIAL_META_LAYER_H