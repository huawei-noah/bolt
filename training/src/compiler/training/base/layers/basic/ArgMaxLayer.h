// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ARG_MAX_LAYER_H
#define ARG_MAX_LAYER_H

#include "ArgExtremumLayer.h"

namespace raul
{

/**
 * @brief ArgMax Layer
 * Returns indices where values is the maximum value of each row
 * of the input tensor in the given dimension dim. Indices are the index locations
 * of each maximum value found.
 * If two outputs are supplied, the second will contain corresponding maximum values.
 * Axis should be specified: raul::Dimension::Default causes an exception.
 *
 * @see
 * https://pytorch.org/docs/master/generated/torch.max.html
 */

using ArgMaxLayer = ArgExtremumLayer<ArgMax>;

} // raul namespace

#endif