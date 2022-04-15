// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TACOTRON_TRAINING_PARAMS_H
#define TACOTRON_TRAINING_PARAMS_H

#include <string>
#include <vector>

#include <training/common/Common.h>

namespace raul
{

struct TacotronTrainingParams
{
    TacotronTrainingParams(dtype initialLearningRate = 1e-2_dt,
                           dtype clipNorm = 0.0_dt,
                           bool clipGradients = true,
                           bool decayLearningRate = true,
                           size_t decaySteps = 1000,
                           size_t decayStartStep = 200,
                           dtype decayRate = 0.5,
                           dtype finalLearningRate = 1e-4_dt,
                           bool warmupLearningRate = true,
                           size_t warmupSteps = 400)
        : initialLearningRate(initialLearningRate)
        , clipNorm(clipNorm)
        , clipGradients(clipGradients)
        , decayLearningRate(decayLearningRate)
        , decaySteps(decaySteps)
        , decayStartStep(decayStartStep)
        , decayRate(decayRate)
        , finalLearningRate(finalLearningRate)
        , warmupLearningRate(warmupLearningRate)
        , warmupSteps(warmupSteps)
    {
    }

    dtype initialLearningRate;

    dtype clipNorm;
    bool clipGradients; // whether to clip gradients

    bool decayLearningRate;
    size_t decaySteps;
    size_t decayStartStep;
    dtype decayRate;
    dtype finalLearningRate;

    bool warmupLearningRate;
    size_t warmupSteps;
};

} // raul namespace
#endif