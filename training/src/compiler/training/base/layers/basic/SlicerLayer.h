// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef SLICER_LAYER_H
#define SLICER_LAYER_H

#include <training/base/layers/BasicLayer.h>

#include <training/base/layers/parameters/SlicingParams.h>

namespace raul
{

/**
 * @brief Slicing Layer
 *
 * The layer allows extracting sub-tensors. It is the opposite operation of concatenation.
 */
class SlicerLayer : public BasicLayer
{
  public:
    SlicerLayer(const Name& name, const SlicingParams& params, NetworkParameters& networkParameters);

    SlicerLayer(SlicerLayer&&) = default;
    SlicerLayer(const SlicerLayer&) = delete;
    SlicerLayer& operator=(const SlicerLayer&) = delete;

  private:
    std::vector<size_t> mSlices;
    Dimension mDirection;
    size_t mDimIndex;

    Name mBackwardTmpBufferName;

    template<typename MM>
    friend class SlicerLayerCPU;
};

} // raul namespace
#endif