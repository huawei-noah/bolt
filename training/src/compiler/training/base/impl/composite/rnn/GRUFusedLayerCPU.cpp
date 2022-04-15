// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "GRUFusedLayerCPU.h"
#include <training/base/layers/composite/rnn/GRUFusedLayer.h>

#include <training/base/impl/ImplFactory.h>

namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::GRUFusedLayer, raul::GRUFusedLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::GRUFusedLayer, raul::GRUFusedLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
void GRUFusedLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    size_t accumulatedSize = 0;

    for (size_t q = 0; q < mLayer.mLengthSequence; ++q)
    {
        // Process input
        const auto& inputData = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
        typename MM::tensor input(inputData.getBatchSize(), 1u, 1u, inputData.getWidth());
        Common::unpack4D(inputData, input, mLayer.mDirection, q, mLayer.mTypeName, mLayer.mName, true);

        auto& linearIH = work.getMemoryManager<MM>()[mLayer.mLinearIHTmp[q]];

        const auto& weightsIH = work.getMemoryManager<MM>().getTensor(mLayer.mWeightsNameIH);

        const auto batchSize = work.getBatchSize();
        size_t N = batchSize * input.getDepth() * input.getHeight();

        Common::gemm(CblasNoTrans,
                     CblasTrans,
                     N,
                     mLayer.mOutputsCount,
                     input.getWidth(),
                     1.0_dt,
                     input.getBuffer(),
                     weightsIH.getBuffer(),
                     0.0_dt,
                     linearIH.getBuffer());

        if (mLayer.mUseBiasForInput)
        {
            const auto& biasesIH = work.getMemoryManager<MM>().getTensor(mLayer.mBiasesNameIH);

            for (size_t index = 0; index < N; ++index)
            {
                Common::axpy(mLayer.mOutputsCount, 1.0_dt, biasesIH.getBuffer(), 1, linearIH.getBuffer(), 1, 0, index * mLayer.mOutputsCount);
            }
        }

        // Process hidden
        const auto& hiddenState = work.getMemoryManager<MM>()[mLayer.mInputsLocal[q][1]];
        auto& linearHH = work.getMemoryManager<MM>()[mLayer.mLinearHHTmp[q]];

        const auto& weightsHH = work.getMemoryManager<MM>().getTensor(mLayer.mWeightsNameHH);

        Common::gemm(CblasNoTrans,
                     CblasTrans,
                     N,
                     mLayer.mOutputsCount,
                     hiddenState.getWidth(),
                     1.0_dt,
                     hiddenState.getBuffer(),
                     weightsHH.getBuffer(),
                     0.0_dt,
                     linearHH.getBuffer());

        if (mLayer.mUseBiasForHidden)
        {
            const auto& biasesHH = work.getMemoryManager<MM>().getTensor(mLayer.mBiasesNameHH);

            for (size_t index = 0; index < N; ++index)
            {
                Common::axpy(mLayer.mOutputsCount, 1.0_dt, biasesHH.getBuffer(), 1, linearHH.getBuffer(), 1, 0, index * mLayer.mOutputsCount);
            }
        }

        auto& newHiddenState = work.getMemoryManager<MM>()[mLayer.mOutputsLocal[q]];

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

        // concatenate hidden
        auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];
        Common::pack4D(newHiddenState, output, mLayer.mDirection, accumulatedSize, mLayer.mTypeName, mLayer.mName, true);
        accumulatedSize += newHiddenState.getShape()[mLayer.mDimIndex + 1];
    }
}

template<typename MM>
void GRUFusedLayerCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    size_t accumulatedSize = mLayer.mLengthSequence - 1;

    for (size_t q = mLayer.mLengthSequence; q-- > 0;)
    {

        auto& deltasHidden = work.getMemoryManager<MM>()[mLayer.mOutputsLocal[q].grad()];
        const auto& linearIH = work.getMemoryManager<MM>()[mLayer.mLinearIHTmp[q]];
        const auto& linearHH = work.getMemoryManager<MM>()[mLayer.mLinearHHTmp[q]];

        const auto& inputData = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
        typename MM::tensor input(inputData.getBatchSize(), 1u, 1u, inputData.getWidth());
        Common::unpack4D(inputData, input, mLayer.mDirection, q, mLayer.mTypeName, mLayer.mName, true);

        auto& inputGradData = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()];

        const auto& hiddenState = work.getMemoryManager<MM>()[mLayer.mInputsLocal[q][1]];

        const auto& weightsIH = work.getMemoryManager<MM>()[mLayer.mWeightsNameIH];
        const auto& weightsHH = work.getMemoryManager<MM>()[mLayer.mWeightsNameHH];

        const auto batchSize = work.getBatchSize();
        const auto sliceSize = linearIH.getWidth() / 3;
        size_t N = batchSize * input.getDepth() * input.getHeight();

        // concatenate hidden
        const auto& delta = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];
        Common::unpack4D(delta, deltasHidden, mLayer.mDirection, accumulatedSize, mLayer.mTypeName, mLayer.mName, false);
        accumulatedSize -= deltasHidden.getShape()[mLayer.mDimIndex + 1];

        const auto deltasHidden2D = deltasHidden.reshape(yato::dims(batchSize, sliceSize));
        const auto linearIH2D = linearIH.reshape(yato::dims(batchSize, sliceSize * 3));
        const auto linearHH2D = linearHH.reshape(yato::dims(batchSize, sliceSize * 3));
        const auto hiddenState2D = hiddenState.reshape(yato::dims(batchSize, sliceSize));

        // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputsLocal[q][0]) || !mLayer.mFrozen)
        {
            auto& linearIHGrad = work.getMemoryManager<MM>()[mLayer.mLinearIHTmp[q].grad()];
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
                    linearIHGrad2D[i][j] += TOMMTYPE(sigmoidGates0 * (1.0_dt - sigmoidGates0) * TODTYPE(linearHH2D[i][sliceSize * 2 + j]) * (1.0_dt - tanhGates2 * tanhGates2) *
                                                     (1.0_dt - sigmoidGates1) * TODTYPE(deltasHidden2D[i][j]));
                    linearIHGrad2D[i][sliceSize + j] += TOMMTYPE(sigmoidGates1 * (1.0_dt - sigmoidGates1) * (TODTYPE(hiddenState2D[i][j]) - tanhGates2) * TODTYPE(deltasHidden2D[i][j]));
                    linearIHGrad2D[i][sliceSize * 2 + j] += TOMMTYPE((1.0_dt - tanhGates2 * tanhGates2) * (1.0_dt - sigmoidGates1) * TODTYPE(deltasHidden2D[i][j]));
                }
            }

            // if ((mLayer.mNetworkParams.isGradNeeded(mLayer.mInputsLocal[q][0]))
            {
                // auto& inputGrad = work.getMemoryManager<MM>()[mLayer.mInputsLocal[q][0].grad()];
                typename MM::tensor inputGrad(inputGradData.getBatchSize(), 1u, 1u, inputGradData.getWidth());
                Common::unpack4D(inputGradData, inputGrad, mLayer.mDirection, q, mLayer.mTypeName, mLayer.mName, true);

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

                Common::pack4D(inputGrad, inputGradData, mLayer.mDirection, q, mLayer.mTypeName, mLayer.mName, false);
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
                    for (size_t index = 0; index < N; ++index)
                    {
                        Common::axpy(mLayer.mOutputsCount,
                                     1.0_dt,
                                     linearIHGrad.getBuffer(),
                                     1,
                                     gradBiasesIH.getBuffer(),
                                     1,
                                     index * mLayer.mOutputsCount,
                                     0);
                    }
                }
            }
        }

        // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputsLocal[q][1]) || !mLayer.mFrozen)
        {
            auto& linearHHGrad = work.getMemoryManager<MM>()[mLayer.mLinearHHTmp[q].grad()];
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
                    linearHHGrad2D[i][j] += TOMMTYPE(sigmoidGates0 * (1.0_dt - sigmoidGates0) * TODTYPE(linearHH2D[i][sliceSize * 2 + j]) * (1.0_dt - tanhGates2 * tanhGates2) *
                                                     (1.0_dt - sigmoidGates1) * TODTYPE(deltasHidden2D[i][j]));
                    linearHHGrad2D[i][sliceSize + j] += TOMMTYPE(sigmoidGates1 * (1.0_dt - sigmoidGates1) * (TODTYPE(hiddenState2D[i][j]) - tanhGates2) * TODTYPE(deltasHidden2D[i][j]));
                    linearHHGrad2D[i][sliceSize * 2 + j] += TOMMTYPE(sigmoidGates0 * (1.0_dt - tanhGates2 * tanhGates2) * (1.0_dt - sigmoidGates1) * TODTYPE(deltasHidden2D[i][j]));
                }
            }

            // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputsLocal[q][1]))
            {
                auto& hiddenStateGrad = work.getMemoryManager<MM>()[mLayer.mInputsLocal[q][1].grad()];
                auto hiddenStateGrad2D = hiddenStateGrad.reshape(yato::dims(batchSize, sliceSize));

#if defined(_OPENMP)
#pragma omp parallel for
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
                    for (size_t index = 0; index < N; ++index)
                    {
                        Common::axpy(mLayer.mOutputsCount,
                                     1.0_dt,
                                     linearHHGrad.getBuffer(),
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
}

template class GRUFusedLayerCPU<MemoryManager>;
template class GRUFusedLayerCPU<MemoryManagerFP16>;

} // namespace raul