// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TILE_LAYER_H
#define TILE_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>
#include <training/base/layers/parameters/TilingParameters.h>

namespace raul
{

/**
 * @brief Tile Layer
 * This operation creates a new tensor by replicating input multiples times. The output tensor's i'th dimension
 * has input.dims(i) * multiples[i] elements, and the values of input are replicated multiples[i] times along the 'i'th dimension.
 * If no dimension specified, whole tensor will be replicated, otherwise - only needed dimension.
 *
 *
 */
class TileLayer : public BasicLayer
{
  public:
    TileLayer(const Name& name, const TilingParams& params, NetworkParameters& networkParameters);

    TileLayer(TileLayer&&) = default;
    TileLayer(const TileLayer&) = delete;
    TileLayer& operator=(const TileLayer&) = delete;

  private:
    raul::Dimension mDimension;
    size_t mRepeatDepth;
    size_t mRepeatHeight;
    size_t mRepeatWidth;

    template<typename MM>
    friend class TileLayerCPU;
};

} // raul namespace

#endif