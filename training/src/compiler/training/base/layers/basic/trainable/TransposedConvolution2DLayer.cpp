// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TransposedConvolution2DLayer.h"

#include <functional>

#if defined(_OPENMP)
#include <omp.h>
#endif

//#define RAUL_NAIVE_CONV_FORWARD
//#define RAUL_NAIVE_CONV_BACKWARD

namespace raul
{

TransposedConvolution2DLayer::TransposedConvolution2DLayer(const Name& name, const TransposedConvolution2DParams& params, NetworkParameters& networkParameters)
    : TransposedConvolution2DLayer(name, "TransposeConvolution2D", params, networkParameters)
{
}

TransposedConvolution2DLayer::TransposedConvolution2DLayer(const Name& name, const std::string& typeName, const TransposedConvolution2DParams& params, NetworkParameters& networkParameters)
    : TrainableLayer(name, typeName, params, networkParameters)
    , mKernelWidth(params.kernelWidth)
    , mKernelHeight(params.kernelHeight)
    , mKernelsCount(params.kernelsCount)
    , mStrideW(params.strideW)
    , mStrideH(params.strideH)
    , mPaddingW(params.paddingW)
    , mPaddingH(params.paddingH)
    , mOutputPaddingW(params.mOutputPaddingW)
    , mOutputPaddingH(params.mOutputPaddingH)
    , mUseBias(params.bias)
    , mDilationW(params.mDilationW)
    , mDilationH(params.mDilationH)
    , mQuantizeWeights(params.quantizeWeights)
    , mDilationEnabled(false)
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    if (mInputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of input names");
    }
    if (mOutputs.size() != 1)
    {
        THROW(mTypeName, mName, "wrong number of output names");
    }

    mInputName = mInputs[0];
    mOutputName = mOutputs[0];
    mPrevLayerDeltaName = mInputName.grad();

    mNetworkParams.mWorkflow.copyDeclaration(name, mInputName, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
    mNetworkParams.mWorkflow.copyDeclaration(name, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

    mInputWidth = mNetworkParams.mWorkflow.getWidth(mInputName);
    mInputHeight = mNetworkParams.mWorkflow.getHeight(mInputName);
    mInputDepth = mNetworkParams.mWorkflow.getDepth(mInputName);

    mEffectiveReceptiveFieldW = mDilationW * (mKernelWidth - 1) + 1;
    mEffectiveReceptiveFieldH = mDilationH * (mKernelHeight - 1) + 1;

    mOutputWidth = (mInputWidth - 1) * mStrideW - 2 * mPaddingW + mEffectiveReceptiveFieldW + mOutputPaddingW;
    mOutputHeight = (mInputHeight - 1) * mStrideH - 2 * mPaddingH + mEffectiveReceptiveFieldH + mOutputPaddingH;

#if !defined(RAUL_NAIVE_CONV_FORWARD) && !defined(RAUL_NAIVE_CONV_BACKWARD)
    if (mDilationW > 1 || mDilationH > 1)
    {
        mDilationEnabled = true;
        mDilationTensor = "Dilated" + mWeightsName;
        mNetworkParams.mWorkflow.tensorNeeded(mName, mDilationTensor, raul::WShape{ mInputDepth, mKernelsCount, mEffectiveReceptiveFieldH, mEffectiveReceptiveFieldW }, DEC_FRBC_WRIT);
    }
#else
    mNetworkParams.mWorkflow.tensorNeeded(mName, mName / "TMP", raul::WShape{ 1u, 1u, 1u, mKernelsCount * (mOutputHeight + 2 * mPaddingH) * (mOutputWidth + 2 * mPaddingW) }, DEC_FRBC_WRIT);
#endif

    size_t numThreads = 1;
#if defined(_OPENMP)
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
#endif
    for (size_t i = 0; i < numThreads; ++i)
    {
        Name im2ColF = mName / "Im2ColFor" / Conversions::toString(i);
        mIm2ColForward.push_back(im2ColF);

        Name im2ColB = mName / "Im2ColBack" / Conversions::toString(i);
        mIm2ColBackward.push_back(im2ColB);

        mNetworkParams.mWorkflow.tensorNeeded(
            mName, im2ColF, raul::WShape{ 1u, 1u, 1u, mInputHeight * mInputWidth * mKernelsCount * mEffectiveReceptiveFieldH * mEffectiveReceptiveFieldW }, DEC_FORW_WRIT);
        mNetworkParams.mWorkflow.copyDeclaration(mName, im2ColF, im2ColB, DEC_BACK_WRIT);
    }

    mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, raul::WShape{ BS(), mKernelsCount, mOutputHeight, mOutputWidth }, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mWeightsName, raul::WShape{ mInputDepth, mKernelsCount, mKernelHeight, mKernelWidth }, DEC_TRAINABLE);
    if (mUseBias)
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, mBiasesName, raul::WShape{ 1u, mKernelsCount, 1u, 1u }, DEC_TRAINABLE);
    }

    if (!mFrozen)
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, mWeightsName, mWeightsName.grad(), DEC_TRAINABLE_GRAD);
        if (mUseBias)
        {
            mNetworkParams.mWorkflow.copyDeclaration(mName, mBiasesName, mBiasesName.grad(), DEC_TRAINABLE_GRAD);
        }
    }

    if (mQuantizeWeights)
    {
        if (!mNetworkParams.mQuantizerPtr)
        {
            THROW(mTypeName, mName, "quantizer is not defined");
        }
        mWeightsBackup = mWeightsName + "_backup";

        mNetworkParams.mWorkflow.copyDeclaration(mName, mWeightsName, mWeightsBackup, DEC_FORW_WRIT);
    }
}

#ifdef RAUL_NAIVE_CONV_FORWARD
void TransposedConvolution2DLayer::forwardComputeImpl(NetworkMode)
{
    const size_t batchSize = mNetworkParams.mWorkflow.getBatchSize();

    Tensor& output = mNetworkParams.mMemoryManager[mOutputName];

    const Tensor& input = mNetworkParams.mMemoryManager[mInputName];

    auto input4D = input.reshape(yato::dims(batchSize, mInputDepth, mInputHeight, mInputWidth));
    auto output3D = output.reshape(yato::dims(batchSize, mKernelsCount, mOutputHeight * mOutputWidth));

    auto kernelsWeights4D = mNetworkParams.mMemoryManager[mWeightsName].reshape(yato::dims(mInputDepth, mKernelsCount, mKernelHeight, mKernelWidth));

    size_t outputWidthPadded = mOutputWidth + 2 * mPaddingW;
    size_t outputHeightPadded = mOutputHeight + 2 * mPaddingH;

    auto outputTmp3D = mNetworkParams.mMemoryManager[mName / "TMP"].reshape(yato::dims(mKernelsCount, outputHeightPadded, outputWidthPadded));

    for (size_t q = 0; q < batchSize; ++q)
    {
        std::fill(mNetworkParams.mMemoryManager[mName / "TMP"].begin(), mNetworkParams.mMemoryManager[mName / "TMP"].end(), 0.0_dt);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t kernelIndex = 0; kernelIndex < mKernelsCount; ++kernelIndex)
        {
            for (size_t d = 0; d < mInputDepth; ++d)
            {
                for (size_t oy = 0; oy < mInputHeight; ++oy)
                {
                    for (size_t ox = 0; ox < mInputWidth; ++ox)
                    {
                        for (size_t ky = 0; ky < mKernelHeight; ++ky)
                        {
                            for (size_t kx = 0; kx < mKernelWidth; ++kx)
                            {
                                outputTmp3D[kernelIndex][oy * mStrideH + ky * mDilationH][ox * mStrideW + kx * mDilationW] += kernelsWeights4D[d][kernelIndex][ky][kx] * input4D[q][d][oy][ox];
                            }
                        }
                    }
                }
            }
        }
        Common::removePadding2D(&outputTmp3D[0][0][0], &output3D[q][0][0], mKernelsCount, outputWidthPadded, outputHeightPadded, mOutputWidth, mOutputHeight);
    }

    if (mUseBias)
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < batchSize; ++q)
        {
            for (size_t kernelIndex = 0; kernelIndex < mKernelsCount; ++kernelIndex)
            {
                for (size_t oy = 0; oy < mOutputHeight; ++oy)
                {
                    for (size_t ox = 0; ox < mOutputWidth; ++ox)
                    {
                        output3D[q][kernelIndex][oy * mOutputWidth + ox] += mNetworkParams.mMemoryManager[mBiasesName][kernelIndex];
                    }
                }
            }
        }
    }
}
#else
void TransposedConvolution2DLayer::forwardComputeImpl(NetworkMode)
{
    Tensor& weights = mNetworkParams.mMemoryManager[mWeightsName];
    if (mQuantizeWeights)
    {
        mNetworkParams.mMemoryManager[mWeightsBackup] = TORANGE(weights);
        mNetworkParams.mQuantizerPtr->quantize(weights.begin(), weights.end());
        mNetworkParams.mQuantizerPtr->dequantize(weights.begin(), weights.end());
    }
    const size_t batchSize = mNetworkParams.mWorkflow.getBatchSize();

    Tensor& output = mNetworkParams.mMemoryManager[mOutputName];

    const Tensor& input = mNetworkParams.mMemoryManager[mInputName];
    auto input3D = input.reshape(yato::dims(batchSize, mInputDepth, mInputHeight * mInputWidth));
    auto output3D = output.reshape(yato::dims(batchSize, mKernelsCount, mOutputHeight * mOutputWidth));

    // Fill dilated weights if needed
    if (mDilationEnabled)
    {
        std::fill(mNetworkParams.mMemoryManager[mDilationTensor].begin(), mNetworkParams.mMemoryManager[mDilationTensor].end(), 0.0_dt);
        auto kernelsWeights4D = mNetworkParams.mMemoryManager[mWeightsName].reshape(yato::dims(mInputDepth, mKernelsCount, mKernelHeight, mKernelWidth));
        auto dilatedKernelsWeights4D = mNetworkParams.mMemoryManager[mDilationTensor].reshape(yato::dims(mInputDepth, mKernelsCount, mEffectiveReceptiveFieldH, mEffectiveReceptiveFieldW));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t d = 0; d < mInputDepth; ++d)
        {
            for (size_t kernelIndex = 0; kernelIndex < mKernelsCount; ++kernelIndex)
            {
                for (size_t ky = 0; ky < mKernelHeight; ++ky)
                {
                    for (size_t kx = 0; kx < mKernelWidth; ++kx)
                    {
                        dilatedKernelsWeights4D[d][kernelIndex][ky * mDilationH][kx * mDilationW] = kernelsWeights4D[d][kernelIndex][ky][kx];
                    }
                }
            }
        }
    }

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < batchSize; ++i)
    {
        size_t index = 0;
#if defined(_OPENMP)
        index = omp_get_thread_num();
#endif
        Tensor& im2ColFor = mNetworkParams.mMemoryManager[mIm2ColForward[index]];
        Common::gemm(CblasTrans,
                     CblasNoTrans,
                     mEffectiveReceptiveFieldW * mEffectiveReceptiveFieldH * mKernelsCount,
                     mInputWidth * mInputHeight,
                     mInputDepth,
                     1.0_dt,
                     mDilationEnabled ? &mNetworkParams.mMemoryManager[mDilationTensor][0] : &mNetworkParams.mMemoryManager[mWeightsName][0],
                     &input3D[i][0][0],
                     0.0_dt,
                     &im2ColFor[0]);

        Common::col2im(&im2ColFor[0], mOutputWidth, mOutputHeight, mKernelsCount, mEffectiveReceptiveFieldW, mEffectiveReceptiveFieldH, mStrideW, mStrideH, mPaddingW, mPaddingH, &output3D[i][0][0]);
    }
    if (mUseBias)
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < batchSize; ++q)
        {
            for (size_t kernelIndex = 0; kernelIndex < mKernelsCount; ++kernelIndex)
            {
                for (size_t oy = 0; oy < mOutputHeight; ++oy)
                {
                    for (size_t ox = 0; ox < mOutputWidth; ++ox)
                    {
                        output3D[q][kernelIndex][oy * mOutputWidth + ox] += mNetworkParams.mMemoryManager[mBiasesName][kernelIndex];
                    }
                }
            }
        }
    }

    if (mQuantizeWeights)
    {
        weights = TORANGE(mNetworkParams.mMemoryManager[mWeightsBackup]);
    }
}
#endif

#ifdef RAUL_NAIVE_CONV_BACKWARD
void TransposedConvolution2DLayer::backwardComputeImpl()
{
    const size_t batchSize = mNetworkParams.mWorkflow.getBatchSize();

    const Tensor& deltas = mNetworkParams.mMemoryManager[mOutputName.grad()];
    auto deltas3D = deltas.reshape(yato::dims(batchSize, mKernelsCount, mOutputHeight * mOutputWidth));

    size_t outputWidthPadded = mOutputWidth + 2 * mPaddingW;
    size_t outputHeightPadded = mOutputHeight + 2 * mPaddingH;

    // if (mNetworkParams.isGradNeeded(mInputName))
    {
        Tensor& prevLayerDelta = mNetworkParams.mMemoryManager[mPrevLayerDeltaName];

        auto kernelsWeights4D = mNetworkParams.mMemoryManager[mWeightsName].reshape(yato::dims(mInputDepth, mKernelsCount, mKernelHeight, mKernelWidth));
        auto prevLayerDeltas3D = prevLayerDelta.reshape(yato::dims(batchSize, mInputDepth, mInputHeight * mInputWidth));

        for (size_t q = 0; q < batchSize; ++q)
        {
            std::fill(mNetworkParams.mMemoryManager[mName / "TMP"].begin(), mNetworkParams.mMemoryManager[mName / "TMP"].end(), 0.0_dt);
            Common::addPadding2D(&deltas3D[q][0][0], &mNetworkParams.mMemoryManager[mName / "TMP"][0], mKernelsCount, mOutputWidth, mOutputHeight, outputWidthPadded, outputHeightPadded);
            auto deltasPadded2D = mNetworkParams.mMemoryManager[mName / "TMP"].reshape(yato::dims(mKernelsCount, outputHeightPadded * outputWidthPadded));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t d = 0; d < mInputDepth; ++d)
            {
                for (size_t kernelIndex = 0; kernelIndex < mKernelsCount; ++kernelIndex)
                {
                    for (size_t oy = 0; oy < mInputHeight; ++oy)
                    {
                        for (size_t ox = 0; ox < mInputWidth; ++ox)
                        {
                            for (size_t ky = 0; ky < mKernelHeight; ++ky)
                            {
                                for (size_t kx = 0; kx < mKernelWidth; ++kx)
                                {
                                    prevLayerDeltas3D[q][d][oy * mInputWidth + ox] +=
                                        kernelsWeights4D[d][kernelIndex][ky][kx] *
                                        deltasPadded2D[kernelIndex][oy * outputWidthPadded * mStrideH + ky * mDilationH * outputWidthPadded + ox * mStrideW + kx * mDilationW];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (!mFrozen)
    {
        const Tensor& inputs = mNetworkParams.mMemoryManager[mInputName];

        Tensor& gradWeights = mNetworkParams.mMemoryManager[mWeightsName.grad()];
        auto inputs4D = inputs.reshape(yato::dims(batchSize, mInputDepth, mInputHeight, mInputWidth));
        auto gradWeights4D = gradWeights.reshape(yato::dims(mInputDepth, mKernelsCount, mKernelHeight, mKernelWidth));

        if (mNetworkParams.mCalculationMode == CalculationMode::DETERMINISTIC)
        {
            for (size_t i = 0; i < batchSize; ++i)
            {
                std::fill(mNetworkParams.mMemoryManager[mName / "TMP"].begin(), mNetworkParams.mMemoryManager[mName / "TMP"].end(), 0.0_dt);
                Common::addPadding2D(&deltas3D[i][0][0], &mNetworkParams.mMemoryManager[mName / "TMP"][0], mKernelsCount, mOutputWidth, mOutputHeight, outputWidthPadded, outputHeightPadded);
                auto deltasPadded2D = mNetworkParams.mMemoryManager[mName / "TMP"].reshape(yato::dims(mKernelsCount, outputHeightPadded * outputWidthPadded));

                for (size_t d = 0; d < mInputDepth; ++d)
                {
                    for (size_t kernelIndex = 0; kernelIndex < mKernelsCount; ++kernelIndex)
                    {
                        for (size_t ky = 0; ky < mKernelHeight; ++ky)
                        {
                            for (size_t kx = 0; kx < mKernelWidth; ++kx)
                            {
                                for (size_t oy = 0; oy < mInputHeight; ++oy)
                                {
                                    for (size_t ox = 0; ox < mInputWidth; ++ox)
                                    {
                                        gradWeights4D[d][kernelIndex][ky][kx] +=
                                            deltasPadded2D[kernelIndex][oy * outputWidthPadded * mStrideH + ky * mDilationH * outputWidthPadded + ox * mStrideW + kx * mDilationW] *
                                            inputs4D[i][d][oy][ox];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
#if defined(_OPENMP)
        else if (mNetworkParams.mCalculationMode == CalculationMode::FAST)
        {
#pragma omp parallel for
            for (size_t i = 0; i < batchSize; ++i)
            {
                std::fill(mNetworkParams.mMemoryManager[mName / "TMP"].begin(), mNetworkParams.mMemoryManager[mName / "TMP"].end(), 0.0_dt);
                Common::addPadding2D(&deltas3D[i][0][0], &mNetworkParams.mMemoryManager[mName / "TMP"][0], mKernelsCount, mOutputWidth, mOutputHeight, outputWidthPadded, outputHeightPadded);
                auto deltasPadded2D = mNetworkParams.mMemoryManager[mName / "TMP"].reshape(yato::dims(mKernelsCount, outputHeightPadded * outputWidthPadded));

                for (size_t d = 0; d < mInputDepth; ++d)
                {
                    for (size_t kernelIndex = 0; kernelIndex < mKernelsCount; ++kernelIndex)
                    {
                        for (size_t ky = 0; ky < mKernelHeight; ++ky)
                        {
                            for (size_t kx = 0; kx < mKernelWidth; ++kx)
                            {
                                for (size_t oy = 0; oy < mInputHeight; ++oy)
                                {
                                    for (size_t ox = 0; ox < mInputWidth; ++ox)
                                    {
                                        gradWeights4D[d][kernelIndex][ky][kx] +=
                                            deltasPadded2D[kernelIndex][oy * outputWidthPadded * mStrideH + ky * mDilationH * outputWidthPadded + ox * mStrideW + kx * mDilationW] *
                                            inputs4D[i][d][oy][ox];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
#endif
        else
        {
            auto prefix = "TransposedConvolution2DLayer[" + mName + "::backwardCompute]: ";
            THROW(mTypeName, mName, "unexpected calculation mode");
        }

        if (mUseBias)
        {
            Tensor& gradBiases = mNetworkParams.mMemoryManager[mBiasesName.grad()];
            for (size_t i = 0; i < batchSize; ++i)
            {
                for (size_t kernelIndex = 0; kernelIndex < mKernelsCount; ++kernelIndex)
                {
                    gradBiases[kernelIndex] += std::accumulate(deltas3D[i][kernelIndex].begin(), deltas3D[i][kernelIndex].end(), 0.0_dt);
                }
            }
        }
    }
}
#else
void TransposedConvolution2DLayer::backwardComputeImpl()
{
    const size_t batchSize = mNetworkParams.mWorkflow.getBatchSize();

    const Tensor& deltas = mNetworkParams.mMemoryManager[mOutputName.grad()];
    auto deltas3D = deltas.reshape(yato::dims(batchSize, mKernelsCount, mOutputHeight * mOutputWidth));

    // if (mNetworkParams.isGradNeeded(mInputName))
    {
        Tensor& prevLayerDelta = mNetworkParams.mMemoryManager[mPrevLayerDeltaName];

        auto prevLayerDeltas3D = prevLayerDelta.reshape(yato::dims(batchSize, mInputDepth, mInputHeight * mInputWidth));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < batchSize; ++q)
        {
            size_t index = 0;
#if defined(_OPENMP)
            index = omp_get_thread_num();
#endif
            Tensor& im2ColBack = mNetworkParams.mMemoryManager[mIm2ColBackward[index]];
            Common::im2col(
                &deltas3D[q][0][0], mOutputWidth, mOutputHeight, mKernelsCount, mEffectiveReceptiveFieldW, mEffectiveReceptiveFieldH, mStrideW, mStrideH, mPaddingW, mPaddingH, &im2ColBack[0]);

            Common::gemm(CblasNoTrans,
                         CblasNoTrans,
                         mInputDepth,
                         mInputWidth * mInputHeight,
                         mEffectiveReceptiveFieldW * mEffectiveReceptiveFieldH * mKernelsCount,
                         1.0_dt,
                         mDilationEnabled ? &mNetworkParams.mMemoryManager[mDilationTensor][0] : &mNetworkParams.mMemoryManager[mWeightsName][0],
                         &im2ColBack[0],
                         1.0_dt,
                         &prevLayerDeltas3D[q][0][0]);
        }
    }

    if (!mFrozen)
    {
        const Tensor& inputs = mNetworkParams.mMemoryManager[mInputName];

        Tensor& gradWeights = mNetworkParams.mMemoryManager[mWeightsName.grad()];

        auto inputs3D = inputs.reshape(yato::dims(batchSize, mInputDepth, mInputHeight * mInputWidth));

        if (mDilationEnabled)
        {
            std::fill(mNetworkParams.mMemoryManager[mDilationTensor].begin(), mNetworkParams.mMemoryManager[mDilationTensor].end(), 0.0_dt);
        }

        if (mNetworkParams.mCalculationMode == CalculationMode::DETERMINISTIC)
        {
            for (size_t q = 0; q < batchSize; ++q)
            {
                size_t index = 0;
                Tensor& im2ColBack = mNetworkParams.mMemoryManager[mIm2ColBackward[index]];
                Common::im2col(
                    &deltas3D[q][0][0], mOutputWidth, mOutputHeight, mKernelsCount, mEffectiveReceptiveFieldW, mEffectiveReceptiveFieldH, mStrideW, mStrideH, mPaddingW, mPaddingH, &im2ColBack[0]);

                Common::gemm(CblasNoTrans,
                             CblasTrans,
                             mInputDepth,
                             mEffectiveReceptiveFieldW * mEffectiveReceptiveFieldH * mKernelsCount,
                             mInputWidth * mInputHeight,
                             1.0_dt,
                             &inputs3D[q][0][0],
                             &im2ColBack[0],
                             1.0_dt,
                             mDilationEnabled ? &mNetworkParams.mMemoryManager[mDilationTensor][0] : &gradWeights[0]);
            }
        }
#if defined(_OPENMP)
        else if (mNetworkParams.mCalculationMode == CalculationMode::FAST)
        {
#pragma omp parallel for
            for (size_t q = 0; q < batchSize; ++q)
            {
                size_t index = omp_get_thread_num();

                Tensor& im2ColBack = mNetworkParams.mMemoryManager[mIm2ColBackward[index]];
                Common::im2col(
                    &deltas3D[q][0][0], mOutputWidth, mOutputHeight, mKernelsCount, mEffectiveReceptiveFieldW, mEffectiveReceptiveFieldH, mStrideW, mStrideH, mPaddingW, mPaddingH, &im2ColBack[0]);
#pragma omp critical
                Common::gemm(CblasNoTrans,
                             CblasTrans,
                             mInputDepth,
                             mEffectiveReceptiveFieldW * mEffectiveReceptiveFieldH * mKernelsCount,
                             mInputWidth * mInputHeight,
                             1.0_dt,
                             &inputs3D[q][0][0],
                             &im2ColBack[0],
                             1.0_dt,
                             mDilationEnabled ? &mNetworkParams.mMemoryManager[mDilationTensor][0] : &gradWeights[0]);
            }
        }
#endif
        else
        {
            auto prefix = "TransposedConvolution2DLayer[" + mName + "::backwardCompute]: ";
            THROW(mTypeName, mName, "unexpected calculation mode");
        }

        if (mDilationEnabled)
        {
            auto gradWeights4D = gradWeights.reshape(yato::dims(mInputDepth, mKernelsCount, mKernelHeight, mKernelWidth));
            auto dilatedGradWeights4D = mNetworkParams.mMemoryManager[mDilationTensor].reshape(yato::dims(mInputDepth, mKernelsCount, mEffectiveReceptiveFieldH, mEffectiveReceptiveFieldW));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t d = 0; d < mInputDepth; ++d)
            {
                for (size_t kernelIndex = 0; kernelIndex < mKernelsCount; ++kernelIndex)
                {
                    for (size_t ky = 0; ky < mKernelHeight; ++ky)
                    {
                        for (size_t kx = 0; kx < mKernelWidth; ++kx)
                        {
                            gradWeights4D[d][kernelIndex][ky][kx] += dilatedGradWeights4D[d][kernelIndex][ky * mDilationH][kx * mDilationW];
                        }
                    }
                }
            }
        }

        if (mUseBias)
        {
            Tensor& gradBiases = mNetworkParams.mMemoryManager[mBiasesName.grad()];
            for (size_t i = 0; i < batchSize; ++i)
            {
                for (size_t kernelIndex = 0; kernelIndex < mKernelsCount; ++kernelIndex)
                {
                    gradBiases[kernelIndex] += std::accumulate(deltas3D[i][kernelIndex].begin(), deltas3D[i][kernelIndex].end(), 0.0_dt);
                }
            }
        }
    }
}
#endif

} // namespace raul