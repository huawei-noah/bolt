// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef BATCHNORM_PARAMS_H
#define BATCHNORM_PARAMS_H

#include "TrainableParams.h"
#include <string>
#include <vector>

namespace raul
{
struct TrainableBasicParamsWithDim : public TrainableParams
{
    TrainableBasicParamsWithDim() = delete;

    TrainableBasicParamsWithDim(const Names& inputs, const Names& outputs, const Name& sharedLayer, const std::string& paramDim = "default", bool frozen = false);

    TrainableBasicParamsWithDim(const Names& inputs, const Names& outputs, const Name& sharedLayer, Dimension paramDim, bool frozen = false);

    Dimension dim = Dimension::Default;

    void print(std::ostream& stream) const override;
};

/**
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param paramMomentum statistics computation for evaluation (LayerName::MeanEval, LayerName::VarianceEval)
 * @param paramEps a value added to the denominator for numerical stability
 */
struct BatchnormParams : public TrainableBasicParamsWithDim
{
    BatchnormParams() = delete;
    BatchnormParams(const Names& inputs, const Names& outputs, float paramMomentum, float paramEps, raul::Dimension dim, bool frozen = false)
        : TrainableBasicParamsWithDim(inputs, outputs, Name{}, dim, frozen)
        , momentum(paramMomentum)
        , eps(paramEps)
    {
    }

    BatchnormParams(const Names& inputs, const Names& outputs, float paramMomentum = 0.0f, float paramEps = 1e-5f, const std::string& dim = "depth", bool frozen = false)
        : BatchnormParams(inputs, outputs, Name{}, paramMomentum, paramEps, dim, frozen)
    {
    }

    BatchnormParams(const Names& inputs, const Names& outputs, const Name& sharedLayer, float paramMomentum = 0.0f, float paramEps = 1e-5f, const std::string& dim = "depth", bool frozen = false)
        : TrainableBasicParamsWithDim(inputs, outputs, sharedLayer, dim, frozen)
        , momentum(paramMomentum)
        , eps(paramEps)
    {
    }

    float momentum;
    float eps;

    void print(std::ostream& stream) const override;
};

} // raul namespace

#endif // BATCHNORM_PARAMS_H