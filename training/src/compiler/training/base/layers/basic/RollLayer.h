// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ROLL_LAYER_H
#define ROLL_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>
#include <training/base/layers/parameters/RollLayerParams.h>

namespace raul
{

/**
 * @brief Roll Layer
 *
 * Roll the tensor along the given dimension. Elements that are shifted
 * beyond the last position are re-introduced at the first position or
 * can be replaced by specified value.
 *
 */
class RollLayer : public BasicLayer
{
  public:
    RollLayer(const Name& name, const RollLayerParams& params, NetworkParameters& networkParameters);

    RollLayer(RollLayer&&) = default;
    RollLayer(const RollLayer&) = delete;
    RollLayer& operator=(const RollLayer&) = delete;

    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override;

  private:
    raul::Dimension mDimension;
    size_t mShift;
    bool mCycled;
    raul::dtype mValueToFill;
};

}

#endif // raul namespace