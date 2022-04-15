// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TRAINABLE_LAYER_H
#define TRAINABLE_LAYER_H

#include <training/common/Common.h>
#include <training/common/MemoryManager.h>
#include <training/network/NetworkParameters.h>

#include "parameters/DataParams.h"
#include "parameters/trainable/TrainableParams.h"

#include "BasicLayer.h"

#include <optional>

namespace raul
{

/**
 * @brief Basic trainable layer
 *
 * @todo(ck): split to interfaces: ITrainable, ILayer and so on.
 *
 */
class TrainableLayer : public BasicLayer
{
  public:
    TrainableLayer(const raul::Name& name, const std::string& typeName, const TrainableParams& params, NetworkParameters& networkParams, std::pair<bool, bool> doChecks = { true, true });

    [[nodiscard]] bool isTrainable() const override { return true; }
    [[nodiscard]] virtual bool isFrozen() const { return mFrozen; }

  protected:
    bool mFrozen;
    /// @todo(ck): use container because the following solution is not general
    Name mWeightsName;
    Name mBiasesName;
};

} // raul namespace

#endif // TRAINABLE_LAYER_H
