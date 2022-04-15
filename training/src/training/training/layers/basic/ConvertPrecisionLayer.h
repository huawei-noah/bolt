// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef CAST_LAYER_H
#define CAST_LAYER_H

#include <training/common/Common.h>
#include <training/layers/BasicLayer.h>
#include <training/layers/parameters/ConvertPrecisionParams.h>

namespace raul
{

/**
 * @brief  ConvertPrecisionLayer
 *
 * The layer converts input tensor precision to output tensor precision using OverrideLayerExecutionTarget logic
 * Class should be used between calls overrideLayerExecutionTarget(...) and resetLayerExecutionTargetOverride()
 * Use invertDirection to cast precision in opposite way (create layer before resetLayerExecutionTargetOverride())
 *
 */
class ConvertPrecisionLayer : public BasicLayer
{
  public:
    ConvertPrecisionLayer(const Name& name, const ConvertPrecisionParams& params, NetworkParameters& networkParameters);

    ConvertPrecisionLayer(ConvertPrecisionLayer&&) = default;
    ConvertPrecisionLayer(const ConvertPrecisionLayer&) = delete;
    ConvertPrecisionLayer& operator=(const ConvertPrecisionLayer&) = delete;

    void forwardComputeImpl(NetworkMode) override;
    void backwardComputeImpl() override;

  private:
    LayerExecutionTarget mTarget;
    bool mInvertDirection;
};

} // raul namespace

#endif // CAST_LAYER_H
