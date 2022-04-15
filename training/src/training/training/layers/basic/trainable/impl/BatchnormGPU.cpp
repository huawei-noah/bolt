// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "BatchnormGPU.h"
#include "../Batchnorm.h"

#include <training/opencl/GPUCommon.h>

namespace
{

std::tuple<size_t, size_t, size_t> reassign(raul::Dimension dim, size_t i, size_t k, size_t q)
{
    if (dim == raul::Dimension::Depth)
    {
        return std::make_tuple(i, k, q);
    }
    if (dim == raul::Dimension::Height)
    {
        return std::make_tuple(k, i, q);
    }
    return std::make_tuple(k, q, i);
}

} // anonymous namespace

namespace raul
{

void BatchNormLayerGPU::initNotBSTensors()
{
    auto caller = mLayer.mTypeName + "[" + mLayer.mName + "::initNotBSTensors]";

    auto weightsBuffer = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mWeightsName).getBuffer();
    mLayer.mNetworkParams.mWorkflow.getKernelManager().fillBuffer(weightsBuffer, 1.0_dt, caller);
    auto biasesBuffer = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mBiasesName).getBuffer();
    mLayer.mNetworkParams.mWorkflow.getKernelManager().fillBuffer(biasesBuffer, 0.0_dt, caller);
    auto varEvalBuffer = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mName / "VarianceEval").getBuffer();
    mLayer.mNetworkParams.mWorkflow.getKernelManager().fillBuffer(varEvalBuffer, 1.0_dt, caller);
    auto meanEvalBuffer = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mName / "MeanEval").getBuffer();
    mLayer.mNetworkParams.mWorkflow.getKernelManager().fillBuffer(meanEvalBuffer, 0.0_dt, caller);
}

void BatchNormLayerGPU::forwardComputeImpl(NetworkMode mode)
{
    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();
    const size_t inputChannels = mLayer.mNetworkParams.mWorkflow.getDepth(mLayer.mInputName);
    const size_t inputHeight = mLayer.mNetworkParams.mWorkflow.getHeight(mLayer.mInputName);
    const size_t inputWidth = mLayer.mNetworkParams.mWorkflow.getWidth(mLayer.mInputName);

    auto& inputs = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mInputName);
    auto& gamma = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mWeightsName);
    auto& beta = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mBiasesName);
    auto& outputs = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mOutputName);

    auto& xHat = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mName / "XHat");

    auto& varSqrt = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mName / "VarianceSqrt");
    auto& mean = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mName / "Mean");
    auto& var = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mName / "Variance");
    auto& meanEval = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mName / "MeanEval");
    auto& varEval = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mName / "VarianceEval");
    auto caller = mLayer.mTypeName + "[" + mLayer.mName + "::forwardComputeImpl]";

    if (mode == NetworkMode::Train || mode == NetworkMode::TrainCheckpointed)
    {
        gpu::batchnorm_forward(mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                               caller,
                               mLayer.mDimension,
                               batchSize,
                               inputChannels,
                               inputHeight,
                               inputWidth,
                               mLayer.mMomentum,
                               mLayer.mEps,
                               mode == NetworkMode::Train,
                               inputs.getBuffer(),
                               beta.getBuffer(),
                               gamma.getBuffer(),
                               mean.getBuffer(),
                               var.getBuffer(),
                               xHat.getBuffer(),
                               varSqrt.getBuffer(),
                               outputs.getBuffer(),
                               meanEval.getBuffer(),
                               varEval.getBuffer());
    }
    else
    {
        gpu::batchnorm_test(mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                            caller,
                            mLayer.mDimension,
                            batchSize,
                            inputChannels,
                            inputHeight,
                            inputWidth,
                            mLayer.mEps,
                            inputs.getBuffer(),
                            beta.getBuffer(),
                            gamma.getBuffer(),
                            meanEval.getBuffer(),
                            varEval.getBuffer(),
                            outputs.getBuffer());
    }
}

void BatchNormLayerGPU::backwardComputeImpl()
{
    auto caller = mLayer.mTypeName + "[" + mLayer.mName + "::backwardComputeImpl]";

    const size_t batchSize = mLayer.mNetworkParams.mWorkflow.getBatchSize();
    const size_t inputChannels = mLayer.mNetworkParams.mWorkflow.getDepth(mLayer.mInputName.grad());
    const size_t inputHeight = mLayer.mNetworkParams.mWorkflow.getHeight(mLayer.mInputName.grad());
    const size_t inputWidth = mLayer.mNetworkParams.mWorkflow.getWidth(mLayer.mInputName.grad());

    auto& xHat = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mName / "XHat");
    auto& varSqrt = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mName / "VarianceSqrt");
    auto& deltas = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mOutputName.grad());
    auto& gamma = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mWeightsName);
    auto& prevLayerDelta = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mInputName.grad());

    if (mLayer.mDimension == raul::Dimension::Depth || mLayer.mDimension == raul::Dimension::Width)
    {
        if (!mLayer.mFrozen)
        {
            auto& nablaBeta = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mBiasesName.grad());
            auto& nablaGamma = mLayer.mNetworkParams.mMemoryManagerGPU(mLayer.mWeightsName.grad());
            gpu::batchnorm_backward(mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                                    caller,
                                    mLayer.mDimension,
                                    batchSize,
                                    inputChannels,
                                    inputHeight,
                                    inputWidth,
                                    deltas.getBuffer(),
                                    xHat.getBuffer(),
                                    varSqrt.getBuffer(),
                                    gamma.getBuffer(),
                                    prevLayerDelta.getBuffer(),
                                    nablaBeta.getBuffer(),
                                    nablaGamma.getBuffer());
        }
        else
        {
            gpu::batchnorm_backward(mLayer.mNetworkParams.mWorkflow.getKernelManager(),
                                    caller,
                                    mLayer.mDimension,
                                    batchSize,
                                    inputChannels,
                                    inputHeight,
                                    inputWidth,
                                    deltas.getBuffer(),
                                    xHat.getBuffer(),
                                    varSqrt.getBuffer(),
                                    gamma.getBuffer(),
                                    prevLayerDelta.getBuffer());
        }
    }
    else
    {
        auto xHatGPU = mLayer.mNetworkParams.mMemoryManagerGPU[mLayer.mName / "XHat"];
        auto varSqrtGPU = mLayer.mNetworkParams.mMemoryManagerGPU[mLayer.mName / "VarianceSqrt"];
        auto deltasGPU = mLayer.mNetworkParams.mMemoryManagerGPU[mLayer.mOutputName.grad()];
        auto gammaGPU = mLayer.mNetworkParams.mMemoryManagerGPU[mLayer.mWeightsName];
        auto prevLayerDeltaGPU = mLayer.mNetworkParams.mMemoryManagerGPU[mLayer.mInputName.grad()];

        const Tensor xHatCPU(xHatGPU);
        const Tensor varSqrtCPU(varSqrtGPU);
        const Tensor deltasCPU(deltasGPU);
        const Tensor gammaCPU(gammaGPU);

        auto deltas4D = deltasCPU.get4DView();
        auto xhat4D = xHatCPU.get4DView();

        if (!mLayer.mFrozen)
        {
            auto nablaBetaGPU = mLayer.mNetworkParams.mMemoryManagerGPU[mLayer.mBiasesName.grad()];
            Tensor nablaBeta(nablaBetaGPU);
            auto nablaGammaGPU = mLayer.mNetworkParams.mMemoryManagerGPU[mLayer.mWeightsName.grad()];
            Tensor nablaGamma(nablaGammaGPU);
#ifdef RAUL_USE_LIB_OPENMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < mLayer.mChosenDimSize; ++i)
            {
                for (size_t j = 0; j < batchSize; ++j)
                {
                    for (size_t k = 0; k < mLayer.mOtherDims[0]; ++k)
                    {
                        for (size_t q = 0; q < mLayer.mOtherDims[1]; ++q)
                        {
                            // Rearrange indices in proper way
                            auto [depth, height, width] = reassign(mLayer.mDimension, i, k, q);
                            nablaBeta[i] += deltas4D[j][depth][height][width];
                            nablaGamma[i] += deltas4D[j][depth][height][width] * xhat4D[j][depth][height][width];
                        }
                    }
                }
            }
            nablaBetaGPU = nablaBeta;
            nablaGammaGPU = nablaGamma;
        }

        ////if (mNetworkParams.isGradNeeded(mInputName))
        {
            Tensor nablaXhat(batchSize, mLayer.mInputDepth, mLayer.mInputHeight, mLayer.mInputWidth);
            auto nablaXhat4D = nablaXhat.get4DView();

            Tensor dvar(mLayer.mChosenDimSize, 0.0_dt);
            Tensor dvar2(mLayer.mChosenDimSize, 0.0_dt);
#ifdef RAUL_USE_LIB_OPENMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < mLayer.mChosenDimSize; ++i)
            {
                for (size_t j = 0; j < batchSize; ++j)
                {
                    for (size_t k = 0; k < mLayer.mOtherDims[0]; ++k)
                    {
                        for (size_t q = 0; q < mLayer.mOtherDims[1]; ++q)
                        {
                            // Rearrange indices in proper way
                            auto [depth, height, width] = reassign(mLayer.mDimension, i, k, q);
                            nablaXhat4D[j][depth][height][width] = deltas4D[j][depth][height][width] * gammaCPU[i];
                            dvar[i] += nablaXhat4D[j][depth][height][width];
                            dvar2[i] += nablaXhat4D[j][depth][height][width] * xhat4D[j][depth][height][width];
                        }
                    }
                }
            }

            Tensor prevLayerDeltaCPU(prevLayerDeltaGPU);
            auto prevDeltas4D = prevLayerDeltaCPU.get4DView();
            const size_t N = batchSize * mLayer.mOtherDims[0] * mLayer.mOtherDims[1];
#ifdef RAUL_USE_LIB_OPENMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < mLayer.mChosenDimSize; ++i)
            {
                for (size_t j = 0; j < batchSize; ++j)
                {
                    for (size_t k = 0; k < mLayer.mOtherDims[0]; ++k)
                    {
                        for (size_t q = 0; q < mLayer.mOtherDims[1]; ++q)
                        {
                            // Rearrange indices in proper way
                            auto [depth, height, width] = reassign(mLayer.mDimension, i, k, q);
                            prevDeltas4D[j][depth][height][width] += (static_cast<raul::dtype>(N) * nablaXhat4D[j][depth][height][width] - dvar[i] - xhat4D[j][depth][height][width] * dvar2[i]) /
                                                                     (static_cast<raul::dtype>(N) * varSqrtCPU[i]);
                        }
                    }
                }
            }
            prevLayerDeltaGPU = prevLayerDeltaCPU;
        }
    }
}

} // namespace raul