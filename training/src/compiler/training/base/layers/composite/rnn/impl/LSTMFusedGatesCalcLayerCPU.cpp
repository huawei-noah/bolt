// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LSTMFusedGatesCalcLayerCPU.h"
#include "../LSTMFusedGatesCalcLayer.h"

#include <training/base/common/Random.h>

namespace raul
{

template<typename MM>
void LSTMFusedGatesCalcLayerCPU<MM>::forwardComputeImpl(NetworkMode mode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    // Process gates
    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
    const auto& hiddenState = work.getMemoryManager<MM>()[mLayer.mInputs[1]];

    auto& gates = work.getMemoryManager<MM>()[mLayer.mGatesName];
    auto& tmp = work.getMemoryManager<MM>()[mLayer.mTmpCalculationsName];

    const auto batchSize = work.getBatchSize();
    size_t N = batchSize * input.getDepth() * input.getHeight();

    // If use single matrix
    if (mLayer.mUseSingleParamTensor)
    {
        auto tmp2D = tmp.reshape(yato::dims(batchSize, tmp.getWidth()));
        const auto input2D = input.reshape(yato::dims(batchSize, input.getWidth()));
        const auto hiddenState2D = hiddenState.reshape(yato::dims(batchSize, hiddenState.getWidth()));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < batchSize; ++i)
        {
            std::copy(input2D[i].begin(), input2D[i].end(), tmp2D[i].begin());
            std::copy(hiddenState2D[i].begin(), hiddenState2D[i].end(), &tmp2D[i][input.getWidth()]);
        }

        const auto& weights = work.getMemoryManager<MM>().getTensor(mLayer.mTrainableParamsNames[0]);

        Common::gemm(CblasNoTrans,
                     CblasTrans,
                     N,
                     mLayer.mOutputsCount,
                     tmp.getWidth(),
                     1.0_dt,
                     tmp.getBuffer(),
                     weights.getBuffer(),
                     0.0_dt,
                     gates.getBuffer());

        if (mLayer.mUseBias)
        {
            const auto& biases = work.getMemoryManager<MM>().getTensor(mLayer.mTrainableParamsNames[1]);

            for (size_t index = 0; index < N; ++index)
            {
                Common::axpy(mLayer.mOutputsCount, 1.0_dt, biases.getBuffer(), 1, gates.getBuffer(), 1, 0, index * mLayer.mOutputsCount);
            }
        }
    }
    else
    {
        // Process input
        const auto& weightsIH = work.getMemoryManager<MM>().getTensor(mLayer.mTrainableParamsNames[0]);

        Common::gemm(CblasNoTrans,
                     CblasTrans,
                     N,
                     mLayer.mOutputsCount,
                     input.getWidth(),
                     1.0_dt,
                     input.getBuffer(),
                     weightsIH.getBuffer(),
                     0.0_dt,
                     tmp.getBuffer());

        if (mLayer.mUseBias)
        {
            const auto& biasesIH = work.getMemoryManager<MM>().getTensor(mLayer.mTrainableParamsNames[1]);

            for (size_t index = 0; index < N; ++index)
            {
                Common::axpy(mLayer.mOutputsCount, 1.0_dt, biasesIH.getBuffer(), 1, tmp.getBuffer(), 1, 0, index * mLayer.mOutputsCount);
            }
        }

        // Process hidden
        const auto& weightsHH = work.getMemoryManager<MM>().getTensor(mLayer.mTrainableParamsNames[2]);

        Common::gemm(CblasNoTrans,
                     CblasTrans,
                     N,
                     mLayer.mOutputsCount,
                     hiddenState.getWidth(),
                     1.0_dt,
                     hiddenState.getBuffer(),
                     weightsHH.getBuffer(),
                     0.0_dt,
                     gates.getBuffer());

        if (mLayer.mUseBias)
        {
            const auto& biasesHH = work.getMemoryManager<MM>().getTensor(mLayer.mTrainableParamsNames[3]);

            for (size_t index = 0; index < N; ++index)
            {
                Common::axpy(mLayer.mOutputsCount, 1.0_dt, biasesHH.getBuffer(), 1, gates.getBuffer(), 1, 0, index * mLayer.mOutputsCount);
            }
        }

        // Final gates is sum of tmp and gates
        gates += tmp;
    }

    const auto& cellState = work.getMemoryManager<MM>()[mLayer.mInputs[2]];
    auto& newCellState = mLayer.mUseZoneout ? work.getMemoryManager<MM>()[mLayer.mNoZoneoutNewCellName] : work.getMemoryManager<MM>()[mLayer.mOutputs[1]];

    auto& newHiddenState = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];

    const auto sliceSize = gates.getWidth() / 4;

    const auto gates2D = gates.reshape(yato::dims(batchSize, sliceSize * 4));
    auto newHiddenState2D = newHiddenState.reshape(yato::dims(batchSize, sliceSize));
    const auto cellState2D = cellState.reshape(yato::dims(batchSize, sliceSize));
    auto newCellState2D = newCellState.reshape(yato::dims(batchSize, sliceSize));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < batchSize; ++i)
    {
        for (size_t j = 0; j < sliceSize; ++j)
        {
            auto newCState = std::tanh(TODTYPE(gates2D[i][sliceSize * 2 + j])) / (1.0_dt + std::exp(-TODTYPE(gates2D[i][j]))) +
                             TODTYPE(cellState2D[i][j]) / (1.0_dt + std::exp(-(TODTYPE(gates2D[i][sliceSize + j]) + mLayer.mForgetBias)));
            newCellState2D[i][j] = TOMMTYPE(newCState);
            newHiddenState2D[i][j] = TOMMTYPE(std::tanh(newCState) / (1.0_dt + std::exp(-TODTYPE(gates2D[i][sliceSize * 3 + j]))));
        }
    }

    if (mLayer.mUseZoneout)
    {
        if (mode == NetworkMode::Test)
        {
            throw std::runtime_error(mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]: Test mode with zoneout is not implemented");
        }

        auto& newCellStateFinal = work.getMemoryManager<MM>()[mLayer.mOutputs[1]];
        auto& mRandomCPUHidden = work.getMemoryManager<MM>()[mLayer.mRandomNameHidden];
        auto& mRandomCPUCell = work.getMemoryManager<MM>()[mLayer.mRandomNameCell];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < hiddenState.size(); ++q)
        {
            mRandomCPUHidden[q] = TOMMTYPE(random::bernoulli::randBool(1.0_dt - mLayer.mZoneout) ? 1.0_dt : 0.0_dt);
            mRandomCPUCell[q] = TOMMTYPE(random::bernoulli::randBool(1.0_dt - mLayer.mZoneout) ? 1.0_dt : 0.0_dt);
            newHiddenState[q] = TOMMTYPE(newHiddenState[q] * mRandomCPUHidden[q] + (1.0_dt - mRandomCPUHidden[q]) * hiddenState[q]);
            newCellStateFinal[q] = TOMMTYPE(newCellState[q] * mRandomCPUCell[q] + (1.0_dt - mRandomCPUCell[q]) * cellState[q]);
        }
    }
}

template<typename MM>
void LSTMFusedGatesCalcLayerCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    const auto& deltasHidden = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];
    const auto& deltasCell = work.getMemoryManager<MM>()[mLayer.mOutputs[1].grad()];
    const auto& gates = work.getMemoryManager<MM>()[mLayer.mGatesName];
    const auto& input = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
    const auto& hiddenState = work.getMemoryManager<MM>()[mLayer.mInputs[1]];
    const auto& cellState = work.getMemoryManager<MM>()[mLayer.mInputs[2]];
    const auto& newCellState = mLayer.mUseZoneout ? work.getMemoryManager<MM>()[mLayer.mNoZoneoutNewCellName] : work.getMemoryManager<MM>()[mLayer.mOutputs[1]];

    const auto batchSize = work.getBatchSize();
    const auto sliceSize = gates.getWidth() / 4;
    size_t N = batchSize * input.getDepth() * input.getHeight();

    if (mLayer.mUseZoneout)
    {
        auto& deltasHiddenNoZoneout = work.getMemoryManager<MM>()[mLayer.mNoZoneoutNewHiddenGradName];
        auto& deltasCellNoZoneout = work.getMemoryManager<MM>()[mLayer.mNoZoneoutNewCellGradName];
        auto& mRandomCPUHidden = work.getMemoryManager<MM>()[mLayer.mRandomNameHidden];
        auto& mRandomCPUCell = work.getMemoryManager<MM>()[mLayer.mRandomNameCell];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < batchSize * sliceSize; ++i)
        {
            if (mRandomCPUHidden[i] == TOMMTYPE(1.0_dt))
            {
                deltasHiddenNoZoneout[i] += deltasHidden[i];
            }
            if (mRandomCPUCell[i] == TOMMTYPE(1.0_dt))
            {
                deltasCellNoZoneout[i] += deltasCell[i];
            }
        }
    }
    const auto deltasHidden2D =
        mLayer.mUseZoneout ? work.getMemoryManager<MM>()[mLayer.mNoZoneoutNewHiddenGradName].reshape(yato::dims(batchSize, sliceSize)) : deltasHidden.reshape(yato::dims(batchSize, sliceSize));
    const auto deltasCell2D =
        mLayer.mUseZoneout ? work.getMemoryManager<MM>()[mLayer.mNoZoneoutNewCellGradName].reshape(yato::dims(batchSize, sliceSize)) : deltasCell.reshape(yato::dims(batchSize, sliceSize));
    const auto gates2D = gates.reshape(yato::dims(batchSize, sliceSize * 4));
    const auto cellState2D = cellState.reshape(yato::dims(batchSize, sliceSize));
    const auto newCellState2D = newCellState.reshape(yato::dims(batchSize, sliceSize));

    // Calculate gradients for tmp storages
    auto& gatesGrad = work.getMemoryManager<MM>()[mLayer.mGatesName.grad()];
    auto gatesGrad2D = gatesGrad.reshape(yato::dims(batchSize, sliceSize * 4));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < batchSize; ++i)
    {
        for (size_t j = 0; j < sliceSize; ++j)
        {
            auto globalGrad = TODTYPE(deltasHidden2D[i][j]) / (1.0_dt + std::exp(-TODTYPE(gates2D[i][sliceSize * 3 + j]))) *
                                  (1.0_dt - std::tanh(TODTYPE(newCellState2D[i][j])) * std::tanh(TODTYPE(newCellState2D[i][j]))) +
                              TODTYPE(deltasCell2D[i][j]);

            auto tmp = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(gates2D[i][j])));
            gatesGrad2D[i][j] += TOMMTYPE(globalGrad * std::tanh(TODTYPE(gates2D[i][sliceSize * 2 + j])) * tmp * (1.0_dt - tmp));

            tmp = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(gates2D[i][sliceSize + j]) - mLayer.mForgetBias));
            gatesGrad2D[i][sliceSize + j] += TOMMTYPE(globalGrad * TODTYPE(cellState2D[i][j]) * tmp * (1.0_dt - tmp));

            gatesGrad2D[i][sliceSize * 2 + j] +=
                TOMMTYPE(globalGrad / (1.0_dt + std::exp(-TODTYPE(gates2D[i][j]))) * (1.0_dt - std::tanh(TODTYPE(gates2D[i][sliceSize * 2 + j])) * std::tanh(TODTYPE(gates2D[i][sliceSize * 2 + j]))));

            tmp = 1.0_dt / (1.0_dt + std::exp(-TODTYPE(gates2D[i][sliceSize * 3 + j])));
            gatesGrad2D[i][sliceSize * 3 + j] += TOMMTYPE(TODTYPE(deltasHidden2D[i][j]) * std::tanh(TODTYPE(newCellState2D[i][j])) * tmp * (1.0_dt - tmp));
        }
    }

    if (mLayer.mUseSingleParamTensor)
    {
        auto& tmpGrad = work.getMemoryManager<MM>()[mLayer.mTmpCalculationsName.grad()];
        const auto& weights = work.getMemoryManager<MM>()[mLayer.mTrainableParamsNames[0]];
        Common::gemm(CblasNoTrans,
                     CblasNoTrans,
                     N,
                     tmpGrad.getWidth(),
                     mLayer.mOutputsCount,
                     1.0_dt,
                     gatesGrad.getBuffer(),
                     weights.getBuffer(),
                     1.0_dt,
                     tmpGrad.getBuffer());
        const auto tmpGrad2D = tmpGrad.reshape(yato::dims(batchSize, tmpGrad.getWidth()));
        // if ((mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
        {
            auto& inputGrad = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()];
            auto inputGrad2D = inputGrad.reshape(yato::dims(batchSize, inputGrad.getWidth()));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t i = 0; i < batchSize; ++i)
            {
                for (size_t j = 0; j < inputGrad.getWidth(); ++j)
                {
                    inputGrad2D[i][j] += tmpGrad2D[i][j];
                }
            }
        }
        // if ((mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[1]))
        {
            auto& hiddenStateGrad = work.getMemoryManager<MM>()[mLayer.mInputs[1].grad()];
            auto hiddenStateGrad2D = hiddenStateGrad.reshape(yato::dims(batchSize, hiddenStateGrad.getWidth()));

            if (mLayer.mUseZoneout)
            {
                auto& mRandomCPUHidden = work.getMemoryManager<MM>()[mLayer.mRandomNameHidden];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < hiddenStateGrad.size(); ++i)
                {
                    if (mRandomCPUHidden[i] == TOMMTYPE(0.0_dt))
                    {
                        hiddenStateGrad[i] += deltasHidden[i];
                    }
                }
            }
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t i = 0; i < batchSize; ++i)
            {
                for (size_t j = 0; j < hiddenStateGrad.getWidth(); ++j)
                {
                    hiddenStateGrad2D[i][j] += tmpGrad2D[i][input.getWidth() + j];
                }
            }
        }
    }
    else
    {
        // if ((mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[0]))
        {
            auto& inputGrad = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()];
            const auto& weightsIH = work.getMemoryManager<MM>()[mLayer.mTrainableParamsNames[0]];
            Common::gemm(CblasNoTrans,
                         CblasNoTrans,
                         N,
                         inputGrad.getWidth(),
                         mLayer.mOutputsCount,
                         1.0_dt,
                         gatesGrad.getBuffer(),
                         weightsIH.getBuffer(),
                         1.0_dt,
                         inputGrad.getBuffer());
        }
        // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[1]))
        {
            auto& hiddenStateGrad = work.getMemoryManager<MM>()[mLayer.mInputs[1].grad()];

            if (mLayer.mUseZoneout)
            {
                auto& mRandomCPUHidden = work.getMemoryManager<MM>()[mLayer.mRandomNameHidden];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < hiddenStateGrad.size(); ++i)
                {
                    if (mRandomCPUHidden[i] == TOMMTYPE(0.0_dt))
                    {
                        hiddenStateGrad[i] += deltasHidden[i];
                    }
                }
            }

            const auto& weightsHH = work.getMemoryManager<MM>()[mLayer.mTrainableParamsNames[2]];
            Common::gemm(CblasNoTrans,
                         CblasNoTrans,
                         N,
                         hiddenStateGrad.getWidth(),
                         mLayer.mOutputsCount,
                         1.0_dt,
                         gatesGrad.getBuffer(),
                         weightsHH.getBuffer(),
                         1.0_dt,
                         hiddenStateGrad.getBuffer());
        }
    }

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[2]))
    {
        auto& cellStateGrad = work.getMemoryManager<MM>()[mLayer.mInputs[2].grad()];
        auto cellStateGrad2D = cellStateGrad.reshape(yato::dims(batchSize, sliceSize));

        if (mLayer.mUseZoneout)
        {
            auto& mRandomCPUCell = work.getMemoryManager<MM>()[mLayer.mRandomNameCell];
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t i = 0; i < cellStateGrad.size(); ++i)
            {
                if (mRandomCPUCell[i] == TOMMTYPE(0.0_dt))
                {
                    cellStateGrad[i] += deltasCell[i];
                }
            }
        }

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < batchSize; ++i)
        {
            for (size_t j = 0; j < sliceSize; ++j)
            {
                cellStateGrad2D[i][j] += TOMMTYPE((TODTYPE(deltasHidden2D[i][j]) / (1.0_dt + std::exp(-TODTYPE(gates2D[i][sliceSize * 3 + j]))) *
                                                       (1.0_dt - std::tanh(TODTYPE(newCellState2D[i][j])) * std::tanh(TODTYPE(newCellState2D[i][j]))) +
                                                   TODTYPE(deltasCell2D[i][j])) /
                                                  (1.0_dt + std::exp(-(TODTYPE(gates2D[i][sliceSize + j]) + mLayer.mForgetBias))));
            }
        }
    }

    if (!mLayer.mFrozen)
    {
        if (mLayer.mUseSingleParamTensor)
        {
            auto& gradWeights = work.getMemoryManager<MM>()[mLayer.mTrainableParamsNames[0].grad()];
            auto& tmp = work.getMemoryManager<MM>()[mLayer.mTmpCalculationsName];
            Common::gemm(CblasTrans,
                         CblasNoTrans,
                         mLayer.mOutputsCount,
                         tmp.getWidth(),
                         N,
                         1.0_dt,
                         gatesGrad.getBuffer(),
                         tmp.getBuffer(),
                         1.0_dt,
                         gradWeights.getBuffer());

            if (mLayer.mUseBias)
            {
                auto& gradBiases = work.getMemoryManager<MM>()[mLayer.mTrainableParamsNames[1].grad()];
                for (size_t index = 0; index < N; ++index)
                {
                    Common::axpy(mLayer.mOutputsCount, 1.0_dt, gatesGrad.getBuffer(), 1, gradBiases.getBuffer(), 1, index * mLayer.mOutputsCount, 0);
                }
            }
        }
        else
        {
            auto& gradWeightsIH = work.getMemoryManager<MM>()[mLayer.mTrainableParamsNames[0].grad()];

            Common::gemm(CblasTrans,
                         CblasNoTrans,
                         mLayer.mOutputsCount,
                         input.getWidth(),
                         N,
                         1.0_dt,
                         gatesGrad.getBuffer(),
                         input.getBuffer(),
                         1.0_dt,
                         gradWeightsIH.getBuffer());

            auto& gradWeightsHH = work.getMemoryManager<MM>()[mLayer.mTrainableParamsNames[2].grad()];

            Common::gemm(CblasTrans,
                         CblasNoTrans,
                         mLayer.mOutputsCount,
                         hiddenState.getWidth(),
                         N,
                         1.0_dt,
                         gatesGrad.getBuffer(),
                         hiddenState.getBuffer(),
                         1.0_dt,
                         gradWeightsHH.getBuffer());

            if (mLayer.mUseBias)
            {
                auto& gradBiasesIH = work.getMemoryManager<MM>()[mLayer.mTrainableParamsNames[1].grad()];
                for (size_t index = 0; index < N; ++index)
                {
                    Common::axpy(mLayer.mOutputsCount,
                                 1.0_dt,
                                 gatesGrad.getBuffer(),
                                 1,
                                 gradBiasesIH.getBuffer(),
                                 1,
                                 index * mLayer.mOutputsCount,
                                 0);
                }

                auto& gradBiasesHH = work.getMemoryManager<MM>()[mLayer.mTrainableParamsNames[3].grad()];
                for (size_t index = 0; index < N; ++index)
                {
                    Common::axpy(mLayer.mOutputsCount,
                                 1.0_dt,
                                 gatesGrad.getBuffer(),
                                 1,
                                 gradBiasesHH.getBuffer(),
                                 1,
                                 index * mLayer.mOutputsCount,
                                 0);
                }
            }
        }
    }
}

template class LSTMFusedGatesCalcLayerCPU<MemoryManager>;
template class LSTMFusedGatesCalcLayerCPU<MemoryManagerFP16>;

} // namespace raul
