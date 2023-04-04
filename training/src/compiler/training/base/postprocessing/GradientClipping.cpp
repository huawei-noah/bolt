// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "GradientClipping.h"
#include <training/compiler/Workflow.h>

#ifdef _MSC_VER
#pragma warning(disable : 4938)
#endif

namespace raul::postprocessing
{

GradientClipping::GradientClipping(raul::dtype clipNorm, std::optional<raul::dtype> globalNorm)
    : mClipNorm(clipNorm)
    , mGlobalNorm(globalNorm)
    , mCurrentGlobalNorm(0.0_dt)
{
    if (mClipNorm <= 0.0_dt)
    {
        THROW_NONAME("GradientClipping", "clip norm should be more than zero");
    }
    if (mGlobalNorm && mGlobalNorm.value() <= 0.0_dt)
    {
        THROW_NONAME("GradientClipping", "global norm should be more than zero");
    }
}

raul::dtype GradientClipping::calcGlobalNorm(const std::vector<ParamAndGrad>& trainableParams, const NetworkParameters& networkParameters) const
{
    raul::dtype qSum = 0.0_dt;
    if (networkParameters.mCalculationMode == CalculationMode::DETERMINISTIC)
    {
        for (size_t j = 0; j < trainableParams.size(); ++j)
        {
            auto& grad = trainableParams[j].Gradient;
            for (size_t i = 0; i < grad.size(); ++i)
            {
                qSum += grad[i] * grad[i];
            }
        }
    }
#if defined(_OPENMP)
    else if (networkParameters.mCalculationMode == CalculationMode::FAST)
    {
#pragma omp parallel for reduction(+ : qSum)
        for (size_t j = 0; j < trainableParams.size(); ++j)
        {
            auto& grad = trainableParams[j].Gradient;
            for (size_t i = 0; i < grad.size(); ++i)
            {
                qSum += grad[i] * grad[i];
            }
        }
    }
#endif
    else
    {
        THROW_NONAME("GradientClipping", "unexpected calculation mode");
    }
    qSum = std::sqrt(qSum);

    return qSum;
}

raul::dtype GradientClipping::calcGlobalNorm(const std::vector<ParamAndGradImpl<TensorFP16>>& trainableParams, const NetworkParameters& networkParameters) const
{
    raul::dtype qSum = 0.0_dt;
    if (networkParameters.mCalculationMode == CalculationMode::DETERMINISTIC)
    {
        for (size_t j = 0; j < trainableParams.size(); ++j)
        {
            auto& grad = trainableParams[j].Gradient;
            for (size_t i = 0; i < grad.size(); ++i)
            {
                qSum += TODTYPE(grad[i]) * TODTYPE(grad[i]);
            }
        }
    }
#if defined(_OPENMP)
    else if (networkParameters.mCalculationMode == CalculationMode::FAST)
    {
#pragma omp parallel for reduction(+ : qSum)
        for (size_t j = 0; j < trainableParams.size(); ++j)
        {
            auto& grad = trainableParams[j].Gradient;
            for (size_t i = 0; i < grad.size(); ++i)
            {
                qSum += TODTYPE(grad[i]) * TODTYPE(grad[i]);
            }
        }
    }
#endif
    else
    {
        THROW_NONAME("GradientClipping", "unexpected calculation mode");
    }
    qSum = std::sqrt(qSum);

    return qSum;
}

raul::dtype GradientClipping::calcGlobalNormMixedPrecision(std::vector<ParamAndGrad>& trainableParams,
                                                           std::vector<ParamAndGradImpl<TensorFP16>>& trainableParamsFP16,
                                                           const NetworkParameters& networkParameters) const
{
    raul::dtype qSum = 0.0_dt;
    if (networkParameters.mCalculationMode == CalculationMode::DETERMINISTIC)
    {
        for (size_t j = 0; j < trainableParams.size(); ++j)
        {
            auto& grad = trainableParams[j].Gradient;
            for (size_t i = 0; i < grad.size(); ++i)
            {
                qSum += grad[i] * grad[i];
            }
        }

        for (size_t j = 0; j < trainableParamsFP16.size(); ++j)
        {
            auto& grad = trainableParamsFP16[j].Gradient;
            for (size_t i = 0; i < grad.size(); ++i)
            {
                qSum += grad[i] * grad[i];
            }
        }
    }
#if defined(_OPENMP)
    else if (networkParameters.mCalculationMode == CalculationMode::FAST)
    {
#pragma omp parallel for reduction(+ : qSum)
        for (size_t j = 0; j < trainableParams.size(); ++j)
        {
            auto& grad = trainableParams[j].Gradient;
            for (size_t i = 0; i < grad.size(); ++i)
            {
                qSum += grad[i] * grad[i];
            }
        }

#pragma omp parallel for reduction(+ : qSum)
        for (size_t j = 0; j < trainableParamsFP16.size(); ++j)
        {
            auto& grad = trainableParamsFP16[j].Gradient;
            for (size_t i = 0; i < grad.size(); ++i)
            {
                qSum += grad[i] * grad[i];
            }
        }
    }
#endif
    else
    {
        THROW_NONAME("GradientClipping", "unexpected calculation mode");
    }
    qSum = std::sqrt(qSum);

    return qSum;
}

void GradientClipping::processGradients(std::vector<ParamAndGrad>& trainableParams, NetworkParameters& networkParameters)
{
    if (!mGlobalNorm)
    {
        mCurrentGlobalNorm = calcGlobalNorm(trainableParams, networkParameters);
        if (mCurrentGlobalNorm == 0.0_dt)
        {
            // If all zeroes just return
            return;
        }
    }

    const auto factor = mClipNorm / std::max(mGlobalNorm ? mGlobalNorm.value() : mCurrentGlobalNorm, mClipNorm);
    if (factor == 1.0_dt)
    {
        return;
    }
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t j = 0; j < trainableParams.size(); ++j)
    {
        auto& grad = trainableParams[j].Gradient;
        std::transform(grad.begin(), grad.end(), grad.begin(), [&factor](auto& element) { return element * factor; });
    }
}

void GradientClipping::processGradients(std::vector<ParamAndGradImpl<TensorFP16>>& trainableParams, NetworkParameters& networkParameters)
{
    if (!mGlobalNorm)
    {
        mCurrentGlobalNorm = calcGlobalNorm(trainableParams, networkParameters);
        if (mCurrentGlobalNorm == 0.0_dt)
        {
            // If all zeroes just return
            return;
        }
    }

    const auto factor = mClipNorm / std::max(mGlobalNorm ? mGlobalNorm.value() : mCurrentGlobalNorm, mClipNorm);
    if (factor == 1.0_dt)
    {
        return;
    }
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t j = 0; j < trainableParams.size(); ++j)
    {
        auto& grad = trainableParams[j].Gradient;
        std::transform(grad.begin(), grad.end(), grad.begin(), [&factor](auto& element) -> half { return element * TOHTYPE(factor); });
    }
}

void GradientClipping::processGradientsMixedPrecision(std::vector<ParamAndGrad>& trainableParams, std::vector<ParamAndGradImpl<TensorFP16>>& trainableParamsFP16, NetworkParameters& networkParameters)
{
    if (!mGlobalNorm)
    {
        mCurrentGlobalNorm = calcGlobalNormMixedPrecision(trainableParams, trainableParamsFP16, networkParameters);
        if (mCurrentGlobalNorm == 0.0_dt)
        {
            // If all zeroes just return
            return;
        }
    }

    const auto factor = mClipNorm / std::max(mGlobalNorm ? mGlobalNorm.value() : mCurrentGlobalNorm, mClipNorm);
    if (factor == 1.0_dt)
    {
        return;
    }

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t j = 0; j < trainableParams.size(); ++j)
    {
        auto& grad = trainableParams[j].Gradient;
        std::transform(grad.begin(), grad.end(), grad.begin(), [&factor](auto& element) { return element * factor; });
    }

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t j = 0; j < trainableParamsFP16.size(); ++j)
    {
        auto& grad = trainableParamsFP16[j].Gradient;
        std::transform(grad.begin(), grad.end(), grad.begin(), [&factor](auto& element) -> half { return element * TOHTYPE(factor); });
    }
}

raul::dtype GradientClipping::getGlobalNorm() const
{
    if (!mGlobalNorm)
    {
        return mCurrentGlobalNorm;
    }
    return mGlobalNorm.value();
}

std::ostream& GradientClipping::as_ostream(std::ostream& out) const
{
    out << "GradientClipping(clip norm = " << mClipNorm;
    if (mGlobalNorm)
    {
        out << ", mGlobalNorm = " << mGlobalNorm.value();
    }
    out << ")";
    return out;
}

}
