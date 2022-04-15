// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef STACKED_META_LAYER_H
#define STACKED_META_LAYER_H

#include "training/layers/MetaLayer.h"
// d.polubotko(TODO): implement
#if 0
#include <training/layers/parameters/StackedParams.h>

namespace raul
{

/**
 * @brief Stacked Meta Layer
 *
 * Creates stack of operators of the same type.
 * For instance, a stack of 3 equivalent operators
 * Op performs y = Op(Op(Op(x)))
 *
 * @note: the operators in the stack do not share their parameters.
 *
 */
class StackedMetaLayer : public MetaLayer
{

  public:
    StackedMetaLayer(const Name& name, const StackedParams& params, raul::NetworkParameters& networkParameters);

    StackedMetaLayer(StackedMetaLayer&&) = default;
    StackedMetaLayer(const StackedMetaLayer&) = delete;
    StackedMetaLayer& operator=(const StackedMetaLayer&) = delete;
};
} // raul namespace
#endif

#endif // STACKED_META_LAYER_H