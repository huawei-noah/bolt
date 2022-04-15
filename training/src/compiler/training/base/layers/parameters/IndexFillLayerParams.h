// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef INDEX_FILL_LAYER_PARAMS_H
#define INDEX_FILL_LAYER_PARAMS_H

#include "BasicParameters.h"
#include <string>
#include <vector>

namespace raul
{
/**
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param dimension specifies the dimension to fill
 * @param indices vector of indices in chose dim to fill
 */
struct IndexFillLayerParams : public BasicParamsWithDim
{
    IndexFillLayerParams() = delete;
    IndexFillLayerParams(const raul::Name& input, const raul::Name& output, raul::Dimension dimension, const std::unordered_set<size_t>& indices, raul::dtype value = 1.0_dt)
        : BasicParamsWithDim(Names(1, input), Names(1, output), dimension)
        , mIndices(indices)
        , mFillValue(value)
    {
    }

    std::unordered_set<size_t> mIndices;
    raul::dtype mFillValue;

    void print(std::ostream& stream) const override;
};

} // raul namespace

#endif // INDEX_FILL_LAYER_PARAMS_H
