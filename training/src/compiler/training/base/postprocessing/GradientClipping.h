// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GRADIENT_CLIPPING_H
#define GRADIENT_CLIPPING_H

#include "GradientPostprocessor.h"
#include <optional>

namespace raul::postprocessing
{

/**
 * @brief Gradient Clipping
 * Given a vector of gradients, and a clipping ratio clipNorm, this object
 * normalize gradients like grad[i] * clipNorm / max(globalNorm, clipNorm),
 * where globalNorm = sqrt(sum([l2norm(g)**2 for g in grads]))
 *
 */

struct GradientClipping : public GradientPostprocessor
{
    explicit GradientClipping(raul::dtype clipNorm = std::numeric_limits<dtype>::max(), std::optional<raul::dtype> globalNorm = std::nullopt);
    void processGradients(std::vector<ParamAndGrad>&, NetworkParameters& networkParameters) override;
    void processGradients(std::vector<ParamAndGradImpl<TensorFP16>>&, NetworkParameters& networkParameters) override;
    void processGradientsMixedPrecision(std::vector<ParamAndGrad>&, std::vector<ParamAndGradImpl<TensorFP16>>&, NetworkParameters& networkParameters);
    raul::dtype calcGlobalNorm(const std::vector<ParamAndGrad>&, const NetworkParameters& networkParameters) const;
    raul::dtype calcGlobalNorm(const std::vector<ParamAndGradImpl<TensorFP16>>&, const NetworkParameters& networkParameters) const;
    raul::dtype calcGlobalNormMixedPrecision(std::vector<ParamAndGrad>&, std::vector<ParamAndGradImpl<TensorFP16>>&, const NetworkParameters& networkParameters) const;
    raul::dtype getGlobalNorm() const;

  private:
    std::ostream& as_ostream(std::ostream& out) const override;
    const raul::dtype mClipNorm;
    const std::optional<raul::dtype> mGlobalNorm;

    raul::dtype mCurrentGlobalNorm;
};

} // raul::postprocessing

#endif