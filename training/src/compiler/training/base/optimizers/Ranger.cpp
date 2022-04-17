// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Ranger.h"
#include <cmath>

using namespace raul;

namespace
{
constexpr raul::dtype LB = 0.0_dt;
constexpr raul::dtype UB = 1.0_dt;
enum buffFields : int
{
    stepNumField,
    smaValField,
    stepSizeField
};
}

namespace raul::optimizers
{
Ranger::Ranger(const dtype lr, const dtype alpha, const size_t k, const dtype nSmaThreshold, const dtype beta1, const dtype beta2, const dtype eps, const dtype weightDecay, bool useGc)
    : mLearningRate(lr)
    , mAlpha(alpha)
    , mK(k)
    , mNSmaThreshold(nSmaThreshold)
    , mBeta1(beta1)
    , mBeta2(beta2)
    , mEpsilon(eps)
    , mWeightDecay(weightDecay)
    , mUseGc(useGc)
    , mStep(0)
{
    if (lr <= LB)
    {
        THROW_NONAME("Ranger", "reset lr>" + Conversions::toString(LB) + " (current lr=" + Conversions::toString(lr) + ")");
    }
    if (alpha < LB || alpha > UB)
    {
        THROW_NONAME("Ranger", "reset alpha from [" + Conversions::toString(LB) + ", " + Conversions::toString(UB) + "] (current alpha=" + Conversions::toString(alpha) + ")");
    }
    if (k < 1)
    {
        THROW_NONAME("Ranger", "Invalid lookahead steps=" + Conversions::toString(k));
    }
    if (eps <= LB)
    {
        THROW_NONAME("Ranger", "reset eps>" + Conversions::toString(LB) + " (current eps=" + Conversions::toString(eps) + ")");
    }
    if (beta1 < LB || beta1 >= UB)
    {
        THROW_NONAME("Ranger", "reset beta1 from [" + Conversions::toString(LB) + ", " + Conversions::toString(UB) + ") (current beta1=" + Conversions::toString(beta1) + ")");
    }
    if (beta2 < LB || beta2 >= UB)
    {
        THROW_NONAME("Ranger", "reset beta2 from [" + Conversions::toString(LB) + ", " + Conversions::toString(UB) + ") (current beta2=" + Conversions::toString(beta2) + ")");
    }
    if (weightDecay < LB)
    {
        THROW_NONAME("Ranger", "reset weightDecay>=" + Conversions::toString(LB) + " (current weightDecay=" + Conversions::toString(weightDecay) + ")");
    }
}

void Ranger::optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Ranger", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    if (!memory_manager.tensorExists(Name("Ranger") / param.getName() / "exp_avg"))
    {
        memory_manager.createTensor(Name("Ranger") / param.getName() / "exp_avg", param.getShape());
        memory_manager.createTensor(Name("Ranger") / param.getName() / "exp_avg_sq", param.getShape());
        // In order to reduce calculations
        memory_manager.createTensor(Name("Ranger") / param.getName() / "beta1T", 1, 1, 1, 1, mBeta1);
        memory_manager.createTensor(Name("Ranger") / param.getName() / "beta2T", 1, 1, 1, 1, mBeta2);
        // For lookahead part
        memory_manager.createTensor(Name("Ranger") / param.getName() / "slow_buffer", param.getShape(), param);
    }

    mStep++;

    // Calculate values needed for denominator
    auto& beta1T = memory_manager[Name("Ranger") / param.getName() / "beta1T"];
    auto& beta2T = memory_manager[Name("Ranger") / param.getName() / "beta2T"];
    dtype nSma = 0.0_dt;
    dtype stepSize = 1.0_dt;
    auto& buff = mBuffered[mStep % 10];
    if (mStep == std::get<stepNumField>(buff))
    {
        nSma = std::get<smaValField>(buff);
        stepSize = std::get<stepSizeField>(buff);
    }
    else
    {
        auto nSmaMax = 2.0_dt / (1.0_dt - mBeta2) - 1.0_dt;
        nSma = nSmaMax - 2.0_dt * static_cast<dtype>(mStep) * beta2T[0] / (1.0_dt - beta2T[0]);
        auto stepSizeDivisor = 1.0_dt - beta1T[0];
        if (nSma > mNSmaThreshold)
        {
            stepSize = std::sqrt((1.0_dt - beta2T[0]) * (nSma - 4.0_dt) / (nSmaMax - 4.0_dt) * (nSma - 2.0_dt) / nSma * nSmaMax / (nSmaMax - 2.0_dt));
        }
        stepSize /= stepSizeDivisor;
        std::get<stepNumField>(buff) = mStep;
        std::get<smaValField>(buff) = nSma;
        std::get<stepSizeField>(buff) = stepSize;
    }

    auto gradMean = 0.0_dt;
    // If use GC option (in original implementation it is used for convolutions and FC)
    if (mUseGc)
    {
        gradMean = std::accumulate(grad.begin(), grad.end(), 0.0_dt) / static_cast<dtype>(grad.size());
    }

    // Update param
    auto& m = memory_manager[Name("Ranger") / param.getName() / "exp_avg"];
    auto& v = memory_manager[Name("Ranger") / param.getName() / "exp_avg_sq"];
    auto& slowBuffer = memory_manager[Name("Ranger") / param.getName() / "slow_buffer"];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < param.size(); ++i)
    {
        // m_new = beta1 * m + (1 - beta1) * grad
        m[i] = mBeta1 * m[i] + (1.0_dt - mBeta1) * (grad[i] - gradMean);
        // v_new = beta2 * v + (1 - beta2) * grad * grad
        v[i] = mBeta2 * v[i] + (1.0_dt - mBeta2) * (grad[i] - gradMean) * (grad[i] - gradMean);
        // Param decay
        if (mWeightDecay != 0.0_dt)
        {
            param[i] = param[i] * (1.0_dt - mWeightDecay * mLearningRate);
        }
        auto denominator = 1.0_dt;
        if (nSma > mNSmaThreshold)
        {
            // param_new = param - lr * step_size * m_new / (sqrt(v_new) + epsilon)
            // else param_new = param - lr * step_size * m_new
            denominator = std::sqrt(v[i]) + mEpsilon;
        }
        param[i] = param[i] - mLearningRate * stepSize * m[i] / denominator;

        // Lookahead part
        if (mStep % mK == 0)
        {
            slowBuffer[i] += mAlpha * (param[i] - slowBuffer[i]);
            param[i] = slowBuffer[i];
        }
    }
    beta1T[0] *= mBeta1;
    beta2T[0] *= mBeta2;
}

void Ranger::optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Ranger", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    if (!memory_manager.tensorExists(Name("Ranger") / param.getName() / "exp_avg"))
    {
        memory_manager.createTensor(Name("Ranger") / param.getName() / "exp_avg", param.getShape());
        memory_manager.createTensor(Name("Ranger") / param.getName() / "exp_avg_sq", param.getShape());
        // In order to reduce calculations
        memory_manager.createTensor(Name("Ranger") / param.getName() / "beta1T", 1, 1, 1, 1, TOHTYPE(mBeta1));
        memory_manager.createTensor(Name("Ranger") / param.getName() / "beta2T", 1, 1, 1, 1, TOHTYPE(mBeta2));
        // For lookahead part
        memory_manager.createTensor(Name("Ranger") / param.getName() / "slow_buffer", param.getShape(), param);
    }

    mStep++;

    // Calculate values needed for denominator
    auto& beta1T = memory_manager[Name("Ranger") / param.getName() / "beta1T"];
    auto& beta2T = memory_manager[Name("Ranger") / param.getName() / "beta2T"];
    dtype nSma = 0.0_dt;
    dtype stepSize = 1.0_dt;
    auto& buff = mBuffered[mStep % 10];
    if (mStep == std::get<stepNumField>(buff))
    {
        nSma = std::get<smaValField>(buff);
        stepSize = std::get<stepSizeField>(buff);
    }
    else
    {
        auto nSmaMax = 2.0_dt / (1.0_dt - mBeta2) - 1.0_dt;
        nSma = nSmaMax - 2.0_dt * static_cast<dtype>(mStep) * TODTYPE(beta2T[0] / (1.0_hf - beta2T[0]));
        auto stepSizeDivisor = 1.0_dt - TODTYPE(beta1T[0]);
        if (nSma > mNSmaThreshold)
        {
            stepSize = std::sqrt((1.0_dt - TODTYPE(beta2T[0])) * (nSma - 4.0_dt) / (nSmaMax - 4.0_dt) * (nSma - 2.0_dt) / nSma * nSmaMax / (nSmaMax - 2.0_dt));
        }
        stepSize /= stepSizeDivisor;
        std::get<stepNumField>(buff) = mStep;
        std::get<smaValField>(buff) = nSma;
        std::get<stepSizeField>(buff) = stepSize;
    }

    auto gradMean = 0.0_dt;
    // If use GC option (in original implementation it is used for convolutions and FC)
    if (mUseGc)
    {
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : gradMean)
#endif
        for (size_t i = 0; i < grad.size(); ++i)
        {
            gradMean += TODTYPE(grad[i]);
        }
        gradMean /= TODTYPE(grad.size());
    }

    // Update param
    auto& m = memory_manager[Name("Ranger") / param.getName() / "exp_avg"];
    auto& v = memory_manager[Name("Ranger") / param.getName() / "exp_avg_sq"];
    auto& slowBuffer = memory_manager[Name("Ranger") / param.getName() / "slow_buffer"];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < param.size(); ++i)
    {
        // m_new = beta1 * m + (1 - beta1) * grad
        const auto mTmp = mBeta1 * TODTYPE(m[i]) + (1.0_dt - mBeta1) * (TODTYPE(grad[i]) - gradMean);
        m[i] = TOHTYPE(mTmp);
        // v_new = beta2 * v + (1 - beta2) * grad * grad
        const auto vTmp = mBeta2 * TODTYPE(v[i]) + (1.0_dt - mBeta2) * (TODTYPE(grad[i]) - gradMean) * (TODTYPE(grad[i]) - gradMean);
        v[i] = TOHTYPE(vTmp);
        // Param decay
        if (mWeightDecay != 0.0_dt)
        {
            param[i] = param[i] * TOHTYPE(1.0_dt - mWeightDecay * mLearningRate);
        }
        auto denominator = 1.0_dt;
        if (nSma > mNSmaThreshold)
        {
            // param_new = param - lr * step_size * m_new / (sqrt(v_new) + epsilon)
            // else param_new = param - lr * step_size * m_new
            denominator = std::sqrt(vTmp) + mEpsilon;
        }
        param[i] = param[i] - TOHTYPE(mLearningRate * stepSize * mTmp / denominator);

        // Lookahead part
        if (mStep % mK == 0)
        {
            slowBuffer[i] += TOHTYPE(mAlpha) * (param[i] - slowBuffer[i]);
            param[i] = slowBuffer[i];
        }
    }
    beta1T[0] = TOHTYPE(mBeta1 * TODTYPE(beta1T[0]));
    beta2T[0] = TOHTYPE(mBeta2 * TODTYPE(beta2T[0]));
}

std::ostream& Ranger::as_ostream(std::ostream& out) const
{
    out << "Ranger(lr=" << std::scientific << mLearningRate << ", alpha=" << mAlpha << ", k=" << mK << ", nSMaThreshold=" << mNSmaThreshold << ", beta1=" << mBeta1;
    out << ", beta2=" << mBeta2 << ", epsilon=" << mEpsilon << ", weightDecay=" << mWeightDecay << ", useGc=" << (mUseGc ? "true" : "false") << ")";
    return out;
}

} // raul::optimizers