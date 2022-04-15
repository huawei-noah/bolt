// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "RMSprop.h"

namespace
{

constexpr raul::dtype boundary = 0.0_dt;

}

namespace raul::optimizers
{

RMSprop::RMSprop(const dtype lr, const dtype alpha, const dtype eps, const dtype weightDecay, const dtype momentum, bool centered, bool tfStyle)
    : mLearningRate(lr)
    , mAlpha(alpha)
    , mEps(eps)
    , mWeightDecay(weightDecay)
    , mMomentum(momentum)
    , mCentered(centered)
    , mTFStyle(tfStyle)
{
    std::string prefix = "RMSProp[ctor]: ";
    if (lr < boundary)
    {
        THROW_NONAME("RMSProp", "reset lr>" + Conversions::toString(boundary) + " (current lr=" + Conversions::toString(lr) + ")");
    }
    if (alpha < boundary)
    {
        THROW_NONAME("RMSProp", "reset alpha>" + Conversions::toString(boundary) + " (current alpha=" + Conversions::toString(alpha) + ")");
    }
    if (eps < boundary)
    {
        THROW_NONAME("RMSProp", "reset eps>" + Conversions::toString(boundary) + " (current eps=" + Conversions::toString(eps) + ")");
    }
    if (weightDecay < boundary)
    {
        THROW_NONAME("RMSProp", "reset weightDecay>" + Conversions::toString(boundary) + " (current weightDecay=" + Conversions::toString(weightDecay) + ")");
    }
    if (momentum < boundary)
    {
        THROW_NONAME("RMSProp", "reset momentum>" + Conversions::toString(boundary) + " (current momentum=" + Conversions::toString(momentum) + ")");
    }
}

void RMSprop::optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("RMSprop[optimize]", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    if (!memory_manager.tensorExists(Name("RMSprop") / param.getName() / "square_avg"))
    {
        memory_manager.createTensor(Name("RMSprop") / param.getName() / "square_avg", 1, param.size(), 1, 1, 0.0_dt);
        if (mMomentum != 0.0_dt)
        {
            memory_manager.createTensor(Name("RMSprop") / param.getName() / "momentum_buffer", 1, param.size(), 1, 1, 0.0_dt);
        }
        if (mCentered)
        {
            memory_manager.createTensor(Name("RMSprop") / param.getName() / "grad_avg", 1, param.size(), 1, 1, 0.0_dt);
        }
    }

    Tensor& squareAvg = memory_manager[Name("RMSprop") / param.getName() / "square_avg"];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < param.size(); i++)
    {
        const auto resultGrad = grad[i] + mWeightDecay * param[i];
        squareAvg[i] = squareAvg[i] * mAlpha + resultGrad * resultGrad * (1.0_dt - mAlpha);

        dtype avg = 0.0_dt;
        if (mCentered)
        {
            auto& avgGrad = memory_manager[Name("RMSprop") / param.getName() / "grad_avg"][i];
            avgGrad = avgGrad * mAlpha + resultGrad * (1.0_dt - mAlpha);
            if (mTFStyle)
            {
                avg = std::sqrt(squareAvg[i] - avgGrad * avgGrad + mEps);
            }
            else
            {
                avg = std::sqrt(squareAvg[i] - avgGrad * avgGrad) + mEps;
            }
        }
        else
        {
            if (mTFStyle)
            {
                avg = std::sqrt(squareAvg[i] + mEps);
            }
            else
            {
                avg = std::sqrt(squareAvg[i]) + mEps;
            }
        }
        if (mMomentum != 0.0_dt)
        {
            auto& momentum = memory_manager[Name("RMSprop") / param.getName() / "momentum_buffer"][i];
            momentum = momentum * mMomentum + resultGrad / avg;
            param[i] -= mLearningRate * momentum;
        }
        else
        {
            param[i] -= mLearningRate * resultGrad / avg;
        }
    }
}

void RMSprop::optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("RMSprop[optimize]", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    if (!memory_manager.tensorExists(Name("RMSprop") / param.getName() / "square_avg"))
    {
        memory_manager.createTensor(Name("RMSprop") / param.getName() / "square_avg", 1, param.size(), 1, 1, 0.0_hf);
        if (mMomentum != 0.0_dt)
        {
            memory_manager.createTensor(Name("RMSprop") / param.getName() / "momentum_buffer", 1, param.size(), 1, 1, 0.0_hf);
        }
        if (mCentered)
        {
            memory_manager.createTensor(Name("RMSprop") / param.getName() / "grad_avg", 1, param.size(), 1, 1, 0.0_hf);
        }
    }

    TensorFP16& squareAvg = memory_manager[Name("RMSprop") / param.getName() / "square_avg"];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < param.size(); i++)
    {
        const auto resultGrad = TODTYPE(grad[i]) + mWeightDecay * TODTYPE(param[i]);
        const auto squareAvgTmp = TODTYPE(squareAvg[i]) * mAlpha + resultGrad * resultGrad * (1.0_dt - mAlpha);
        squareAvg[i] = TOHTYPE(squareAvgTmp);

        dtype avg = 0.0_dt;
        if (mCentered)
        {
            auto& avgGrad = memory_manager[Name("RMSprop") / param.getName() / "grad_avg"][i];
            const auto avgGradTmp = TODTYPE(avgGrad) * mAlpha + resultGrad * (1.0_dt - mAlpha);
            avgGrad = TOHTYPE(avgGradTmp);
            if (mTFStyle)
            {
                avg = std::sqrt(squareAvgTmp - avgGradTmp * avgGradTmp + mEps);
            }
            else
            {
                avg = std::sqrt(squareAvgTmp - avgGradTmp * avgGradTmp) + mEps;
            }
        }
        else
        {
            if (mTFStyle)
            {
                avg = std::sqrt(squareAvgTmp + mEps);
            }
            else
            {
                avg = std::sqrt(squareAvgTmp) + mEps;
            }
        }
        if (mMomentum != 0.0_dt)
        {
            auto& momentum = memory_manager[Name("RMSprop") / param.getName() / "momentum_buffer"][i];
            momentum = TOHTYPE(TODTYPE(momentum) * mMomentum + resultGrad / avg);
            param[i] -= TOHTYPE(mLearningRate) * momentum;
        }
        else
        {
            param[i] -= TOHTYPE(mLearningRate * resultGrad / avg);
        }
    }
}

std::ostream& RMSprop::as_ostream(std::ostream& out) const
{
    out << "RMSprop(lr=" << std::scientific << mLearningRate << ", alpha=" << mAlpha << ", eps=" << mEps << ", weight decay=" << mWeightDecay << ", momentum: " << mMomentum
        << ", centered: " << (mCentered ? "true" : "false") << ", style: " << (mTFStyle ? "tensorflow" : "pytorch") << ")";
    return out;
}

} // raul::optimizers