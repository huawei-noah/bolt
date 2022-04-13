// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LAYERNORM_PARAMS_H
#define LAYERNORM_PARAMS_H

#include "TrainableParams.h"
#include <string>
#include <vector>

namespace raul
{
/**
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param paramEps a value added to the denominator for numerical stability
 * @param paramTFStyle TF style - eps added to biased dispersion or Transformer style (eps added to unbiased stddev), useBesselCorrection is ignored
 */
struct LayerNormParams : public TrainableParams
{
    LayerNormParams() = delete;
    LayerNormParams(const Name& input, const Name& output, float paramEps = 1e-5f, bool paramTFStyle = false, bool useBesselCorrection = true, bool frozen = false)
        : TrainableParams(Names(1, input), Names(1, output), frozen)
        , eps(paramEps)
        , useTFstyle(paramTFStyle)
        , useBesselCorrection(useBesselCorrection)
    {
    }

    float eps;
    bool useTFstyle;
    bool useBesselCorrection;

    void print(std::ostream& stream) const override;
};
} // raul namespace

#endif // LAYERNORM_PARAMS_H