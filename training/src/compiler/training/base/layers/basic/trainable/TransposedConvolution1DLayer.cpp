// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TransposedConvolution1DLayer.h"

#include <functional>

#if defined(_OPENMP)
#include <omp.h>
#endif

//#define RAUL_NAIVE_CONV_FORWARD
//#define RAUL_NAIVE_CONV_BACKWARD

namespace raul
{

TransposedConvolution1DLayer::TransposedConvolution1DLayer(const Name& name, const TransposedConvolution1DParams& params, NetworkParameters& networkParameters)
    : TrainableLayer(name, "TransposedConvolution1D", params, networkParameters)
    , mOutputChannels(params.kernelsCount)
    , mKernelSize(params.kernelSize)
    , mStride(params.stride)
    , mPadding(params.padding)
    , mOutputPadding(params.outputPadding)
    , mDilation(params.dilation)
    , mGroups(params.groups)
    , mUseBias(params.useBias)
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
    if (mGroups < 1)
    {
        THROW(mTypeName, mName, "zero groups");
    }
    if (mDilation < 1)
    {
        THROW(mTypeName, mName, "dilation parameter should be at least 1");
    }

    mInputName = mInputs[0];
    mOutputName = mOutputs[0];
    mPrevLayerDeltaName = mInputName.grad();

    if (mNetworkParams.mWorkflow.getHeight(mInputName) > 1 && mNetworkParams.mWorkflow.getDepth(mInputName) > 1)
    {
        THROW(mTypeName, mName, "height and depth can't both be > 1");
    }

    mNetworkParams.mWorkflow.copyDeclaration(name, mInputName, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);
    mNetworkParams.mWorkflow.copyDeclaration(name, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

    mInputSize = mNetworkParams.mWorkflow.getWidth(mInputName);
    mInputChannels = mNetworkParams.mWorkflow.getHeight(mInputName) * mNetworkParams.mWorkflow.getDepth(mInputName);

    if (mInputChannels % mGroups != 0 || mOutputChannels % mGroups != 0)
    {
        THROW(mTypeName, mName, "bad number of groups");
    }

    mEffectiveReceptiveField = mDilation * (mKernelSize - 1) + 1;
    mOutputSize = (mInputSize - 1) * mStride - 2 * mPadding + mEffectiveReceptiveField + mOutputPadding;
#if !defined(RAUL_NAIVE_CONV_FORWARD) && !defined(RAUL_NAIVE_CONV_BACKWARD)
    if (mDilation > 1)
    {
        mDilationEnabled = true;
        mDilationTensor = "Dilated" + mWeightsName;
        mNetworkParams.mWorkflow.tensorNeeded(mName, mDilationTensor, raul::WShape{ 1u, mInputChannels, mOutputChannels / mGroups, mEffectiveReceptiveField }, DEC_FRBC_WRIT);
    }
#else
    mNetworkParams.mWorkflow.tensorNeeded(mName, mName / "TMP", raul::WShape{ 1u, 1u, 1u, mOutputChannels * (mOutputSize + 2 * mPadding) }, DEC_FRBC_WRIT);
#endif

    raul::WShape outputShape{ BS(), mOutputChannels, 1u, mOutputSize };
    if (mNetworkParams.mWorkflow.getHeight(mInputName) > 1)
    {
        outputShape = raul::WShape{ BS(), 1u, mOutputChannels, mOutputSize };
    }

    mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, outputShape, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);

    mNetworkParams.mWorkflow.tensorNeeded(mName, mWeightsName, raul::WShape{ 1u, mInputChannels, mOutputChannels / mGroups, mKernelSize }, DEC_TRAINABLE);
    if (mUseBias)
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, mBiasesName, raul::WShape{ 1u, mOutputChannels, 1u, 1u }, DEC_TRAINABLE);
    }

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

        mNetworkParams.mWorkflow.tensorNeeded(mName, im2ColF, raul::WShape{ 1u, 1u, 1u, mInputSize * mOutputChannels * mEffectiveReceptiveField }, DEC_FORW_WRIT);
        mNetworkParams.mWorkflow.copyDeclaration(mName, im2ColF, im2ColB, DEC_BACK_WRIT);
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
void TransposedConvolution1DLayer::forwardComputeImpl(NetworkMode)
{
    const size_t batchSize = mNetworkParams.mWorkflow.getBatchSize();

    Tensor& output = mNetworkParams.mMemoryManager[mOutputName];

    const Tensor& input = mNetworkParams.mMemoryManager[mInputName];

    auto input3D = input.reshape(yato::dims(batchSize, mInputChannels, mInputSize));
    auto output3D = output.reshape(yato::dims(batchSize, mOutputChannels, mOutputSize));

    auto kernelsWeights3D = mNetworkParams.mMemoryManager[mWeightsName].reshape(yato::dims(mInputChannels, mOutputChannels / mGroups, mKernelSize));
    size_t outputSizePadded = mOutputSize + 2 * mPadding;

    auto outputTmp2D = mNetworkParams.mMemoryManager[mName / "TMP"].reshape(yato::dims(mOutputChannels, outputSizePadded));

    for (size_t i = 0; i < batchSize; ++i)
    {
        std::fill(mNetworkParams.mMemoryManager[mName / "TMP"].begin(), mNetworkParams.mMemoryManager[mName / "TMP"].end(), 0.0_dt);

        for (size_t group = 0; group < mGroups; ++group)
        {
            for (size_t d = 0; d < mInputChannels / mGroups; ++d)
            {
                for (size_t kernelIndex = 0; kernelIndex < mOutputChannels / mGroups; ++kernelIndex)
                {
                    for (size_t ox = 0; ox < mInputSize; ++ox)
                    {
                        for (size_t kx = 0; kx < mKernelSize; ++kx)
                        {
                            outputTmp2D[kernelIndex + group * mOutputChannels / mGroups][ox * mStride + kx * mDilation] +=
                                input3D[i][d + group * mInputChannels / mGroups][ox] * kernelsWeights3D[d + group * mInputChannels / mGroups][kernelIndex][kx];
                        }
                    }
                }
            }
        }
        Common::removePadding1D(&outputTmp2D[0][0], &output3D[i][0][0], mOutputChannels, outputSizePadded, mOutputSize);
    }

    if (mUseBias)
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < batchSize; ++q)
        {
            for (size_t kernelIndex = 0; kernelIndex < mOutputChannels; ++kernelIndex)
            {
                const dtype bias = mNetworkParams.mMemoryManager[mBiasesName][kernelIndex];
                std::transform(output3D[q][kernelIndex].begin(), output3D[q][kernelIndex].end(), output3D[q][kernelIndex].begin(), [bias](dtype& val) { return val + bias; });
            }
        }
    }
}
#else
void TransposedConvolution1DLayer::forwardComputeImpl(NetworkMode)
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
    auto input3D = input.reshape(yato::dims(batchSize, mInputChannels, mInputSize));
    auto output3D = output.reshape(yato::dims(batchSize, mOutputChannels, mOutputSize));

    auto finalKernelsWeights3D = mDilationEnabled ? mNetworkParams.mMemoryManager[mDilationTensor].reshape(yato::dims(mInputChannels, mOutputChannels / mGroups, mEffectiveReceptiveField))
                                                  : weights.reshape(yato::dims(mInputChannels, mOutputChannels / mGroups, mKernelSize));

    // Fill dilated weights if needed
    if (mDilationEnabled)
    {
        std::fill(mNetworkParams.mMemoryManager[mDilationTensor].begin(), mNetworkParams.mMemoryManager[mDilationTensor].end(), 0.0_dt);
        auto kernelsWeights3D = weights.reshape(yato::dims(mInputChannels, mOutputChannels / mGroups, mKernelSize));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t d = 0; d < mInputChannels; ++d)
        {
            for (size_t kernelIndex = 0; kernelIndex < mOutputChannels / mGroups; ++kernelIndex)
            {
                for (size_t kx = 0; kx < mKernelSize; ++kx)
                {
                    finalKernelsWeights3D[d][kernelIndex][kx * mDilation] = kernelsWeights3D[d][kernelIndex][kx];
                }
            }
        }
    }
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t q = 0; q < batchSize; ++q)
    {
        size_t index = 0;
#if defined(_OPENMP)
        index = omp_get_thread_num();
#endif
        Tensor& im2ColFor = mNetworkParams.mMemoryManager[mIm2ColForward[index]];
        for (size_t group = 0; group < mGroups; ++group)
        {
            Common::gemm(CblasTrans,
                         CblasNoTrans,
                         mEffectiveReceptiveField * mOutputChannels / mGroups,
                         mInputSize,
                         mInputChannels / mGroups,
                         1.0_dt,
                         &finalKernelsWeights3D[group * mInputChannels / mGroups][0][0],
                         &input3D[q][group * mInputChannels / mGroups][0],
                         0.0_dt,
                         &im2ColFor[0] + group * mEffectiveReceptiveField * mOutputChannels * mInputSize / mGroups);
        }
        Common::col2im(&im2ColFor[0], mOutputSize, 1u, mOutputChannels, mEffectiveReceptiveField, 1u, mStride, 1u, mPadding, 0, &output3D[q][0][0]);
    }

    if (mUseBias)
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < batchSize; ++q)
        {
            for (size_t kernelIndex = 0; kernelIndex < mOutputChannels; ++kernelIndex)
            {
                const dtype bias = mNetworkParams.mMemoryManager[mBiasesName][kernelIndex];
                std::transform(output3D[q][kernelIndex].begin(), output3D[q][kernelIndex].end(), output3D[q][kernelIndex].begin(), [bias](dtype& val) { return val + bias; });
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
void TransposedConvolution1DLayer::backwardComputeImpl()
{
    const size_t batchSize = mNetworkParams.mWorkflow.getBatchSize();

    const Tensor& deltas = mNetworkParams.mMemoryManager[mOutputName.grad()];
    auto deltas3D = deltas.reshape(yato::dims(batchSize, mOutputChannels, mOutputSize));

    size_t outputSizePadded = mOutputSize + 2 * mPadding;

    // if (mNetworkParams.isGradNeeded(mInputName))
    {
        Tensor& prevLayerDelta = mNetworkParams.mMemoryManager[mPrevLayerDeltaName];

        auto kernelsWeights3D = mNetworkParams.mMemoryManager[mWeightsName].reshape(yato::dims(mInputChannels, mOutputChannels / mGroups, mKernelSize));
        auto prevLayerDeltas3D = prevLayerDelta.reshape(yato::dims(batchSize, mInputChannels, mInputSize));

        for (size_t q = 0; q < batchSize; ++q)
        {
            std::fill(mNetworkParams.mMemoryManager[mName / "TMP"].begin(), mNetworkParams.mMemoryManager[mName / "TMP"].end(), 0.0_dt);
            Common::addPadding1D(&deltas3D[q][0][0], &mNetworkParams.mMemoryManager[mName / "TMP"][0], mOutputChannels, mOutputSize, outputSizePadded);
            auto deltasPadded2D = mNetworkParams.mMemoryManager[mName / "TMP"].reshape(yato::dims(mOutputChannels, outputSizePadded));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t group = 0; group < mGroups; ++group)
            {
                for (size_t kernelIndex = 0; kernelIndex < mOutputChannels / mGroups; ++kernelIndex)
                {
                    for (size_t d = 0; d < mInputChannels / mGroups; ++d)
                    {
                        for (size_t ox = 0; ox < mInputSize; ++ox)
                        {
                            for (size_t kx = 0; kx < mKernelSize; ++kx)
                            {
                                prevLayerDeltas3D[q][d + group * mInputChannels / mGroups][ox] += kernelsWeights3D[d + group * mInputChannels / mGroups][kernelIndex][kx] *
                                                                                                  deltasPadded2D[kernelIndex + group * mOutputChannels / mGroups][ox * mStride + kx * mDilation];
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
        auto inputs3D = inputs.reshape(yato::dims(batchSize, mInputChannels, mInputSize));
        auto gradWeights3D = gradWeights.reshape(yato::dims(mInputChannels, mOutputChannels / mGroups, mKernelSize));

        if (mNetworkParams.mCalculationMode == CalculationMode::DETERMINISTIC)
        {
            for (size_t i = 0; i < batchSize; ++i)
            {
                std::fill(mNetworkParams.mMemoryManager[mName / "TMP"].begin(), mNetworkParams.mMemoryManager[mName / "TMP"].end(), 0.0_dt);
                Common::addPadding1D(&deltas3D[i][0][0], &mNetworkParams.mMemoryManager[mName / "TMP"][0], mOutputChannels, mOutputSize, outputSizePadded);
                auto deltasPadded2D = mNetworkParams.mMemoryManager[mName / "TMP"].reshape(yato::dims(mOutputChannels, outputSizePadded));

                for (size_t group = 0; group < mGroups; ++group)
                {
                    for (size_t d = 0; d < mInputChannels / mGroups; ++d)
                    {
                        for (size_t kernelIndex = 0; kernelIndex < mOutputChannels / mGroups; ++kernelIndex)
                        {
                            for (size_t kx = 0; kx < mKernelSize; ++kx)
                            {
                                for (size_t ox = 0; ox < mInputSize; ++ox)
                                {
                                    gradWeights3D[d + group * mInputChannels / mGroups][kernelIndex][kx] +=
                                        deltasPadded2D[kernelIndex + group * mOutputChannels / mGroups][ox * mStride + kx * mDilation] * inputs3D[i][d + group * mInputChannels / mGroups][ox];
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
                Common::addPadding1D(&deltas3D[i][0][0], &mNetworkParams.mMemoryManager[mName / "TMP"][0], mOutputChannels, mOutputSize, outputSizePadded);
                auto deltasPadded2D = mNetworkParams.mMemoryManager[mName / "TMP"].reshape(yato::dims(mOutputChannels, outputSizePadded));

                for (size_t group = 0; group < mGroups; ++group)
                {
                    for (size_t d = 0; d < mInputChannels / mGroups; ++d)
                    {
                        for (size_t kernelIndex = 0; kernelIndex < mOutputChannels / mGroups; ++kernelIndex)
                        {
                            for (size_t kx = 0; kx < mKernelSize; ++kx)
                            {
                                for (size_t ox = 0; ox < mInputSize; ++ox)
                                {
                                    gradWeights3D[d + group * mInputChannels / mGroups][kernelIndex][kx] +=
                                        deltasPadded2D[kernelIndex + group * mOutputChannels / mGroups][ox * mStride + kx * mDilation] * inputs3D[i][d + group * mInputChannels / mGroups][ox];
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
            auto prefix = "TransposedConvolution1DLayer[" + mName + "::backwardCompute]: ";
            THROW(mTypeName, mName, "unexpected calculation mode");
        }

        if (mUseBias)
        {
            Tensor& gradBiases = mNetworkParams.mMemoryManager[mBiasesName.grad()];
            for (size_t i = 0; i < batchSize; ++i)
            {
                for (size_t kernelIndex = 0; kernelIndex < mOutputChannels; ++kernelIndex)
                {
                    gradBiases[kernelIndex] += std::accumulate(deltas3D[i][kernelIndex].begin(), deltas3D[i][kernelIndex].end(), 0.0_dt);
                }
            }
        }
    }
}
#else
void TransposedConvolution1DLayer::backwardComputeImpl()
{
    const size_t batchSize = mNetworkParams.mWorkflow.getBatchSize();

    const Tensor& deltas = mNetworkParams.mMemoryManager[mOutputName.grad()];
    auto deltas3D = deltas.reshape(yato::dims(batchSize, mOutputChannels, mOutputSize));

    auto finalKernelsWeights3D = mDilationEnabled ? mNetworkParams.mMemoryManager[mDilationTensor].reshape(yato::dims(mInputChannels, mOutputChannels / mGroups, mEffectiveReceptiveField))
                                                  : mNetworkParams.mMemoryManager[mWeightsName].reshape(yato::dims(mInputChannels, mOutputChannels / mGroups, mKernelSize));
    // if (mNetworkParams.isGradNeeded(mInputName))
    {
        Tensor& prevLayerDelta = mNetworkParams.mMemoryManager[mPrevLayerDeltaName];

        auto prevLayerDeltas3D = prevLayerDelta.reshape(yato::dims(batchSize, mInputChannels, mInputSize));

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
            Common::im2col(&deltas3D[q][0][0], mOutputSize, 1u, mOutputChannels, mEffectiveReceptiveField, 1u, mStride, 1u, mPadding, 0, &im2ColBack[0]);

            for (size_t group = 0; group < mGroups; ++group)
            {
                Common::gemm(CblasNoTrans,
                             CblasNoTrans,
                             mInputChannels / mGroups,
                             mInputSize,
                             mEffectiveReceptiveField * mOutputChannels / mGroups,
                             1.0_dt,
                             &finalKernelsWeights3D[group * mInputChannels / mGroups][0][0],
                             &im2ColBack[0] + group * mOutputChannels / mGroups * mEffectiveReceptiveField * mInputSize,
                             1.0_dt,
                             &prevLayerDeltas3D[q][group * mInputChannels / mGroups][0]);
            }
        }
    }

    if (!mFrozen)
    {
        const Tensor& inputs = mNetworkParams.mMemoryManager[mInputName];
        auto inputs3D = inputs.reshape(yato::dims(batchSize, mInputChannels, mInputSize));

        Tensor im2colMatrix(batchSize * mInputSize * mOutputChannels * mEffectiveReceptiveField);
        auto im2colMatrix2D = im2colMatrix.reshape(yato::dims(batchSize, mInputSize * mOutputChannels * mEffectiveReceptiveField));

        if (mDilationEnabled)
        {
            std::fill(mNetworkParams.mMemoryManager[mDilationTensor].begin(), mNetworkParams.mMemoryManager[mDilationTensor].end(), 0.0_dt);
        }

        Tensor& gradWeights = mNetworkParams.mMemoryManager[mWeightsName.grad()];
        auto gradWeights3D = mDilationEnabled ? mNetworkParams.mMemoryManager[mDilationTensor].reshape(yato::dims(mInputChannels, mOutputChannels / mGroups, mEffectiveReceptiveField))
                                              : gradWeights.reshape(yato::dims(mInputChannels, mOutputChannels / mGroups, mKernelSize));

        if (mNetworkParams.mCalculationMode == CalculationMode::DETERMINISTIC)
        {
            for (size_t q = 0; q < batchSize; ++q)
            {
                Common::im2col(&deltas3D[q][0][0], mOutputSize, 1u, mOutputChannels, mEffectiveReceptiveField, 1u, mStride, 1u, mPadding, 0, &im2colMatrix2D[q][0]);

                for (size_t group = 0; group < mGroups; ++group)
                {
                    Common::gemm(CblasNoTrans,
                                 CblasTrans,
                                 mInputChannels / mGroups,
                                 mEffectiveReceptiveField * mOutputChannels / mGroups,
                                 mInputSize,
                                 1.0_dt,
                                 &inputs3D[q][group * mInputChannels / mGroups][0],
                                 &im2colMatrix2D[q][0] + group * mInputSize * mOutputChannels * mEffectiveReceptiveField / mGroups,
                                 1.0_dt,
                                 &gradWeights3D[group * mInputChannels / mGroups][0][0]);
                }
            }
        }
#if defined(_OPENMP)
        else if (mNetworkParams.mCalculationMode == CalculationMode::FAST)
        {
#pragma omp parallel for
            for (size_t q = 0; q < batchSize; ++q)
            {
                Common::im2col(&deltas3D[q][0][0], mOutputSize, 1u, mOutputChannels, mEffectiveReceptiveField, 1u, mStride, 1u, mPadding, 0, &im2colMatrix2D[q][0]);

#pragma omp critical
                for (size_t group = 0; group < mGroups; ++group)
                {
                    Common::gemm(CblasNoTrans,
                                 CblasTrans,
                                 mInputChannels / mGroups,
                                 mEffectiveReceptiveField * mOutputChannels / mGroups,
                                 mInputSize,
                                 1.0_dt,
                                 &inputs3D[q][group * mInputChannels / mGroups][0],
                                 &im2colMatrix2D[q][0] + group * mInputSize * mOutputChannels * mEffectiveReceptiveField / mGroups,
                                 1.0_dt,
                                 &gradWeights3D[group * mInputChannels / mGroups][0][0]);
                }
            }
        }
#endif
        else
        {
            auto prefix = "TransposedConvolution1DLayer[" + mName + "::backwardCompute]: ";
            THROW(mTypeName, mName, "unexpected calculation mode");
        }

        if (mDilationEnabled)
        {
            auto realGradWeights3D = gradWeights.reshape(yato::dims(mInputChannels, mOutputChannels / mGroups, mKernelSize));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t d = 0; d < mInputChannels; ++d)
            {
                for (size_t kernelIndex = 0; kernelIndex < mOutputChannels / mGroups; ++kernelIndex)
                {
                    for (size_t kx = 0; kx < mKernelSize; ++kx)
                    {
                        realGradWeights3D[d][kernelIndex][kx] += gradWeights3D[d][kernelIndex][kx * mDilation];
                    }
                }
            }
        }

        if (mUseBias)
        {
            Tensor& gradBiases = mNetworkParams.mMemoryManager[mBiasesName.grad()];
            for (size_t i = 0; i < batchSize; ++i)
            {
                for (size_t kernelIndex = 0; kernelIndex < mOutputChannels; ++kernelIndex)
                {
                    gradBiases[kernelIndex] += std::accumulate(deltas3D[i][kernelIndex].begin(), deltas3D[i][kernelIndex].end(), 0.0_dt);
                }
            }
        }
    }
}
#endif

}