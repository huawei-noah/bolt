// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "GRUFusedGatesCalcLayerCPU.h"
#include "../GRUFusedGatesCalcLayer.h"

#include <training/base/impl/ImplFactory.h>
namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::GRUFusedGatesCalcLayer, raul::GRUFusedGatesCalcLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::GRUFusedGatesCalcLayer, raul::GRUFusedGatesCalcLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
void GRUFusedGatesCalcLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    // Process input
    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
    auto& linearIH = work.getMemoryManager<MM>()[mLayer.mLinearIHTmp];

    const auto& weightsIH = work.getMemoryManager<MM>().getTensor(mLayer.mWeightsNameIH);

    const auto batchSize = work.getBatchSize();
    size_t N = batchSize * input.getDepth() * input.getHeight();

    auto beta = 0.0_dt;
    if (mLayer.mUseBiasForInput)
    {
        const auto& biasesIH = work.getMemoryManager<MM>().getTensor(mLayer.mBiasesNameIH);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < N; i++)
        {
            std::copy(biasesIH.cbegin(), biasesIH.cend(), linearIH.begin() + i * mLayer.mOutputsCount);
        }

        beta = 1.0_dt;
    }

    Common::gemm(CblasNoTrans,
                 CblasTrans,
                 N,
                 mLayer.mOutputsCount,
                 input.getWidth(),
                 1.0_dt,
                 input.getBuffer(),
                 weightsIH.getBuffer(),
                 beta,
                 linearIH.getBuffer());

    // Process hidden
    const auto& hiddenState = work.getMemoryManager<MM>()[mLayer.mInputs[1]];
    auto& linearHH = work.getMemoryManager<MM>()[mLayer.mLinearHHTmp];

    const auto& weightsHH = work.getMemoryManager<MM>().getTensor(mLayer.mWeightsNameHH);

    beta = 0_dt;
    if (mLayer.mUseBiasForHidden)
    {
        const auto& biasesHH = work.getMemoryManager<MM>().getTensor(mLayer.mBiasesNameHH);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < N; i++)
        {
            std::copy(biasesHH.cbegin(), biasesHH.cend(), linearHH.begin() + i * mLayer.mOutputsCount);
        }

        beta = 1.0_dt;
    }

    Common::gemm(CblasNoTrans,
                 CblasTrans,
                 N,
                 mLayer.mOutputsCount,
                 hiddenState.getWidth(),
                 1.0_dt,
                 hiddenState.getBuffer(),
                 weightsHH.getBuffer(),
                 beta,
                 linearHH.getBuffer());

    auto& newHiddenState = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];

    const auto sliceSize = linearIH.getWidth() / 3;

    const auto linearIH2D = linearIH.reshape(yato::dims(batchSize, sliceSize * 3));
    const auto linearHH2D = linearHH.reshape(yato::dims(batchSize, sliceSize * 3));
    const auto hiddenState2D = hiddenState.reshape(yato::dims(batchSize, sliceSize));
    const auto newHiddenState2D = newHiddenState.reshape(yato::dims(batchSize, sliceSize));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < batchSize; ++i)
    {
        for (size_t j = 0; j < sliceSize; ++j)
        {
            auto sigmoidGates0 = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(linearIH2D[i][j]) - TODTYPE(linearHH2D[i][j])));
            auto sigmoidGates1 = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(linearIH2D[i][sliceSize + j]) - TODTYPE(linearHH2D[i][sliceSize + j])));
            auto tanhGates2 = std::tanh(sigmoidGates0 * TODTYPE(linearHH2D[i][sliceSize * 2 + j]) + TODTYPE(linearIH2D[i][sliceSize * 2 + j]));
            newHiddenState2D[i][j] = TOMMTYPE(sigmoidGates1 * TODTYPE(hiddenState2D[i][j]) + tanhGates2 * (1.0_dt - sigmoidGates1));
        }
    }
}

template<typename MM>
void GRUFusedGatesCalcLayerCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltasHidden = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];
    const auto& linearIH = work.getMemoryManager<MM>()[mLayer.mLinearIHTmp];
    const auto& linearHH = work.getMemoryManager<MM>()[mLayer.mLinearHHTmp];
    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
    const auto& hiddenState = work.getMemoryManager<MM>()[mLayer.mInputs[1]];

    const auto& weightsIH = work.getMemoryManager<MM>()[mLayer.mWeightsNameIH];
    const auto& weightsHH = work.getMemoryManager<MM>()[mLayer.mWeightsNameHH];

    const auto batchSize = work.getBatchSize();
    const auto sliceSize = linearIH.getWidth() / 3;
    size_t N = batchSize * input.getDepth() * input.getHeight();

    const auto deltasHidden2D = deltasHidden.reshape(yato::dims(batchSize, sliceSize));
    const auto linearIH2D = linearIH.reshape(yato::dims(batchSize, sliceSize * 3));
    const auto linearHH2D = linearHH.reshape(yato::dims(batchSize, sliceSize * 3));
    const auto hiddenState2D = hiddenState.reshape(yato::dims(batchSize, sliceSize));

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[0]) || !mLayer.mFrozen)
    {
        auto& linearIHGrad = work.getMemoryManager<MM>()[mLayer.mLinearIHTmp.grad()];
        auto linearIHGrad2D = linearIHGrad.reshape(yato::dims(batchSize, sliceSize * 3));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < batchSize; ++i)
        {
            for (size_t j = 0; j < sliceSize; ++j)
            {
                auto sigmoidGates0 = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(linearIH2D[i][j]) - TODTYPE(linearHH2D[i][j])));
                auto sigmoidGates1 = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(linearIH2D[i][sliceSize + j]) - TODTYPE(linearHH2D[i][sliceSize + j])));
                auto tanhGates2 = std::tanh(sigmoidGates0 * TODTYPE(linearHH2D[i][sliceSize * 2 + j]) + TODTYPE(linearIH2D[i][sliceSize * 2 + j]));

                const auto coeff = (1.0_dt - sigmoidGates1) * TODTYPE(deltasHidden2D[i][j]);

                linearIHGrad2D[i][j] += TOMMTYPE(sigmoidGates0 * (1.0_dt - sigmoidGates0) * TODTYPE(linearHH2D[i][sliceSize * 2 + j]) * (1.0_dt - tanhGates2 * tanhGates2) * coeff);
                linearIHGrad2D[i][sliceSize + j] += TOMMTYPE(sigmoidGates1 * (TODTYPE(hiddenState2D[i][j]) - tanhGates2) * coeff);
                linearIHGrad2D[i][sliceSize * 2 + j] += TOMMTYPE((1.0_dt - tanhGates2 * tanhGates2) * coeff);
            }
        }

        // if ((mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
        {
            auto& inputGrad = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()];

            Common::gemm(CblasNoTrans,
                         CblasNoTrans,
                         N,
                         inputGrad.getWidth(),
                         mLayer.mOutputsCount,
                         1.0_dt,
                         linearIHGrad.getBuffer(),
                         weightsIH.getBuffer(),
                         1.0_dt,
                         inputGrad.getBuffer());
        }

        if (!mLayer.mFrozen)
        {
            auto& gradWeightsIH = work.getMemoryManager<MM>()[mLayer.mWeightsNameIH.grad()];

            Common::gemm(CblasTrans,
                         CblasNoTrans,
                         mLayer.mOutputsCount,
                         input.getWidth(),
                         N,
                         1.0_dt,
                         linearIHGrad.getBuffer(),
                         input.getBuffer(),
                         1.0_dt,
                         gradWeightsIH.getBuffer());

            if (mLayer.mUseBiasForInput)
            {
                auto& gradBiasesIH = work.getMemoryManager<MM>()[mLayer.mBiasesNameIH.grad()];
                //#if defined(_OPENMP)
                //#pragma omp parallel for
                //#endif
                for (size_t i = 0; i < N; i++)
                {
                    std::transform(linearIHGrad.cbegin() + i * mLayer.mOutputsCount,
                                   linearIHGrad.cbegin() + i * mLayer.mOutputsCount + mLayer.mOutputsCount,
                                   gradBiasesIH.cbegin(),
                                   gradBiasesIH.begin(),
                                   std::plus<typename MM::type>());
                }
            }
        }
    }

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[1]) || !mLayer.mFrozen)
    {
        auto& linearHHGrad = work.getMemoryManager<MM>()[mLayer.mLinearHHTmp.grad()];
        auto linearHHGrad2D = linearHHGrad.reshape(yato::dims(batchSize, sliceSize * 3));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < batchSize; ++i)
        {
            for (size_t j = 0; j < sliceSize; ++j)
            {
                auto sigmoidGates0 = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(linearIH2D[i][j]) - TODTYPE(linearHH2D[i][j])));
                auto sigmoidGates1 = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(linearIH2D[i][sliceSize + j]) - TODTYPE(linearHH2D[i][sliceSize + j])));
                auto tanhGates2 = std::tanh(sigmoidGates0 * TODTYPE(linearHH2D[i][sliceSize * 2 + j]) + TODTYPE(linearIH2D[i][sliceSize * 2 + j]));

                const auto coeff = (1.0_dt - sigmoidGates1) * TODTYPE(deltasHidden2D[i][j]);

                linearHHGrad2D[i][j] += TOMMTYPE(sigmoidGates0 * (1.0_dt - sigmoidGates0) * TODTYPE(linearHH2D[i][sliceSize * 2 + j]) * (1.0_dt - tanhGates2 * tanhGates2) * coeff);
                linearHHGrad2D[i][sliceSize + j] += TOMMTYPE(sigmoidGates1 * (TODTYPE(hiddenState2D[i][j]) - tanhGates2) * coeff);
                linearHHGrad2D[i][sliceSize * 2 + j] += TOMMTYPE(sigmoidGates0 * (1.0_dt - tanhGates2 * tanhGates2) * coeff);
            }
        }

        // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[1]))
        {
            auto& hiddenStateGrad = work.getMemoryManager<MM>()[mLayer.mInputs[1].grad()];
            auto hiddenStateGrad2D = hiddenStateGrad.reshape(yato::dims(batchSize, sliceSize));

#if defined(_OPENMP)
#pragma omp parallel for ordered
#endif
            for (size_t i = 0; i < batchSize; ++i)
            {
                for (size_t j = 0; j < sliceSize; ++j)
                {
                    auto sigmoidGates1 = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(linearIH2D[i][sliceSize + j]) - TODTYPE(linearHH2D[i][sliceSize + j])));
                    hiddenStateGrad2D[i][j] += TOMMTYPE(sigmoidGates1 * TODTYPE(deltasHidden2D[i][j]));
                }
            }

            Common::gemm(CblasNoTrans,
                         CblasNoTrans,
                         N,
                         hiddenStateGrad.getWidth(),
                         mLayer.mOutputsCount,
                         1.0_dt,
                         linearHHGrad.getBuffer(),
                         weightsHH.getBuffer(),
                         1.0_dt,
                         hiddenStateGrad.getBuffer());
        }

        if (!mLayer.mFrozen)
        {
            auto& gradWeightsHH = work.getMemoryManager<MM>()[mLayer.mWeightsNameHH.grad()];

            Common::gemm(CblasTrans,
                         CblasNoTrans,
                         mLayer.mOutputsCount,
                         hiddenState.getWidth(),
                         N,
                         1.0_dt,
                         linearHHGrad.getBuffer(),
                         hiddenState.getBuffer(),
                         1.0_dt,
                         gradWeightsHH.getBuffer());

            if (mLayer.mUseBiasForHidden)
            {
                auto& gradBiasesHH = work.getMemoryManager<MM>()[mLayer.mBiasesNameHH.grad()];
                //#if defined(_OPENMP)
                //#pragma omp parallel for
                //#endif
                for (size_t i = 0; i < N; i++)
                {
                    std::transform(linearHHGrad.cbegin() + i * mLayer.mOutputsCount,
                                   linearHHGrad.cbegin() + i * mLayer.mOutputsCount + mLayer.mOutputsCount,
                                   gradBiasesHH.cbegin(),
                                   gradBiasesHH.begin(),
                                   std::plus<typename MM::type>());
                }
            }
        }
    }
}

template class GRUFusedGatesCalcLayerCPU<MemoryManager>;
template class GRUFusedGatesCalcLayerCPU<MemoryManagerFP16>;

} // namespace raul