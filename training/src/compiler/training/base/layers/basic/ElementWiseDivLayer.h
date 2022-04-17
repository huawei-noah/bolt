// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ELEMENT_WISE_DIV_LAYER_H
#define ELEMENT_WISE_DIV_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/BroadcastingLayer.h>

namespace raul
{

/**
 * @brief Element-wise division Layer
 *
 * Implements element-wise division of two tensors.
 */
class ElementWiseDivLayer : public BroadcastingLayer
{
  public:
    ElementWiseDivLayer(const Name& name, const ElementWiseLayerParams& params, NetworkParameters& networkParameters);

    ElementWiseDivLayer(ElementWiseDivLayer&&) = default;
    ElementWiseDivLayer(const ElementWiseDivLayer&) = delete;
    ElementWiseDivLayer& operator=(const ElementWiseDivLayer&) = delete;

  private:
    bool mBroadcast;

    Name mBackwardTmpBufferName;

    size_t mDepth;
    size_t mHeight;
    size_t mWidth;

    template<typename MM>
    friend class ElementWiseDivLayerCPU;
};

} // raul namespace
#endif