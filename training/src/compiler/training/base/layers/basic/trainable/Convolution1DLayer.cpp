// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Convolution1DLayer.h"

#include <functional>

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifdef _MSC_VER
#pragma warning(disable : 4938)
#endif

#include "impl/Convolution1DLayerCPU.h"

namespace raul
{

Convolution1DLayer::Convolution1DLayer(const Name& name, const Convolution1DParams& params, NetworkParameters& networkParameters)
    : TrainableLayer(name, "Convolution1D", params, networkParameters)
    , mOutputChannels(params.kernelsCount)
    , mKernelSize(params.kernelSize)
    , mStride(params.stride)
    , mPadding(params.padding)
    , mDilation(params.dilation)
    , mGroups(params.groups)
    , mUseBias(params.useBias)
    , mQuantizeWeights(params.quantizeWeights)
    , mDilationEnabled(false)
    , mTFStyle(params.tfStyle)
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";

    DECLARE_IMPL(Convolution1DLayer, Convolution1DLayerCPU<MemoryManager>, Convolution1DLayerCPU<MemoryManagerFP16>)

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

    if (mNetworkParams.mWorkflow.getHeight(mInputName) > 1 && mNetworkParams.mWorkflow.getDepth(mInputName) > 1)
    {
        THROW(mTypeName, mName, "height and depth can't both be > 1");
    }

    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, Workflow::Usage::ForwardAndBackward, Workflow::Mode::Read);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mInputName, mInputName.grad(), DEC_BACK_WRIT_ZERO);

    // Depends on style
    if (!mTFStyle)
    {
        mInputSize = mNetworkParams.mWorkflow.getWidth(mInputName);
        mInputChannels = mNetworkParams.mWorkflow.getHeight(mInputName) * mNetworkParams.mWorkflow.getDepth(mInputName);
    }
    else
    {
        mInputChannels = mNetworkParams.mWorkflow.getWidth(mInputName);
        mInputSize = mNetworkParams.mWorkflow.getHeight(mInputName) * mNetworkParams.mWorkflow.getDepth(mInputName);
    }

    if (mInputChannels % mGroups != 0 || mOutputChannels % mGroups != 0)
    {
        THROW(mTypeName, mName, "bad number of groups");
    }

    mEffectiveReceptiveField = mDilation * (mKernelSize - 1) + 1;
    mOutputSize = (mInputSize + 2 * mPadding - mEffectiveReceptiveField) / mStride + 1;

    WShape outputShape;
    if (mNetworkParams.mWorkflow.getDepth(mInputName) > 1)
    {
        if (mTFStyle)
        {
            outputShape = { BS(), mOutputSize, 1u, mOutputChannels };
        }
        else
        {
            outputShape = { BS(), mOutputChannels, 1u, mOutputSize };
        }
    }
    else
    {
        if (mTFStyle)
        {
            outputShape = { BS(), 1u, mOutputSize, mOutputChannels };
        }
        else
        {
            outputShape = { BS(), 1u, mOutputChannels, mOutputSize };
        }
    }

    mNetworkParams.mWorkflow.tensorNeeded(mName, mOutputName, outputShape, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(mName, mOutputName, mOutputName.grad(), DEC_BACK_READ);

#if !defined(RAUL_NAIVE_CONV_FORWARD) && !defined(RAUL_NAIVE_CONV_BACKWARD)
    if (mDilation > 1)
    {
        mDilationEnabled = true;
        mDilationTensor = "Dilated" + mWeightsName;
        mNetworkParams.mWorkflow.tensorNeeded(mName, mDilationTensor, WShape{ 1u, mOutputChannels, mInputChannels / mGroups, mEffectiveReceptiveField }, DEC_FORW_WRIT);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mDilationTensor, DEC_BACK_READ);
    }
    else if (mTFStyle)
    {
        mTempWeights = "Temp" + mWeightsName;
        mNetworkParams.mWorkflow.tensorNeeded(mName, mTempWeights, WShape{ 1u, mOutputChannels, mInputChannels / mGroups, mKernelSize }, DEC_FORW_WRIT);
        mNetworkParams.mWorkflow.copyDeclaration(mName, mTempWeights, DEC_BACK_READ);

        mNetworkParams.mWorkflow.copyDeclaration(mName, mTempWeights, mTempWeights.grad(), DEC_BACK_WRIT_ZERO);
    }
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

        mNetworkParams.mWorkflow.tensorNeeded(mName, im2ColF, WShape{ 1u, 1u, 1u, mOutputSize * mInputChannels * mEffectiveReceptiveField }, DEC_FORW_WRIT);
        mNetworkParams.mWorkflow.copyDeclaration(mName, im2ColF, im2ColB, DEC_BACK_WRIT);

        Name tempWeightsGrag = mName / "TempWeightsGrag" / Conversions::toString(i);
        mTempWeightsGrag.push_back(tempWeightsGrag);

        mNetworkParams.mWorkflow.tensorNeeded(mName, tempWeightsGrag, WShape{ 1u, mOutputChannels, mInputChannels / mGroups, mEffectiveReceptiveField }, DEC_BACK_WRIT_ZERO);
    }
    mTmpIm2ColName = mName / "TmpIm2Col";
    mNetworkParams.mWorkflow.tensorNeeded(mName, mTmpIm2ColName, WShape{ BS(), 1u, 1u, mOutputSize * mInputChannels * mEffectiveReceptiveField }, DEC_BACK_WRIT_ZERO);

#if defined(RAUL_NAIVE_CONV_BACKWARD)
    size_t inputSizePadded = mInputSize + 2 * mPadding;
    mTmpForBackwardName = mName / "TmpForBackward";
    mNetworkParams.mWorkflow.tensorNeeded(mName, mTmpForBackwardName, WShape{ 1u, 1u, 1u, mInputChannels * inputSizePadded }, DEC_BACK_WRIT_ZERO);
#endif

    // Weights can be stored as [1, kernel_size, input_channels, output_channels] (tf-style)
    // or as [1, output_channels, input_channels, kernel_size] (torch-style)
    if (mTFStyle)
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, mWeightsName, WShape{ 1u, mKernelSize, mInputChannels / mGroups, mOutputChannels }, DEC_TRAINABLE);
    }
    else
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, mWeightsName, WShape{ 1u, mOutputChannels, mInputChannels / mGroups, mKernelSize }, DEC_TRAINABLE);
    }

    if (mUseBias)
    {
        mNetworkParams.mWorkflow.tensorNeeded(mName, mBiasesName, WShape{ 1u, mOutputChannels, 1u, 1u }, DEC_TRAINABLE);
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
void Convolution1DLayer::forwardComputeImpl(NetworkMode)
{
    const size_t batchSize = mNetworkParams.mWorkflow.getBatchSize();

    Tensor& output = mNetworkParams.mMemoryManager[mOutputName];
    const Tensor& inputs = mNetworkParams.mMemoryManager[mInputName];
    const Tensor& weights = mNetworkParams.mMemoryManager[mWeightsName];

    Common::conv1d(&inputs[0],
                   &output[0],
                   &weights[0],
                   mUseBias ? &mNetworkParams.mMemoryManager[mBiasesName][0] : nullptr,
                   batchSize,
                   mInputSize,
                   mInputChannels,
                   mOutputSize,
                   mOutputChannels,
                   mKernelSize,
                   mPadding,
                   mStride,
                   mDilation,
                   mGroups,
                   mTFStyle);
}
#else
void Convolution1DLayer::forwardComputeImpl(NetworkMode mode)
{
    mImpl->forwardComputeImpl(mode);
}
#endif

#ifdef RAUL_NAIVE_CONV_BACKWARD
void Convolution1DLayer::backwardComputeImpl()
{
    const size_t batchSize = mNetworkParams.mWorkflow.getBatchSize();

    const Tensor& deltas = mNetworkParams.mMemoryManager[mOutputName.grad()];
    auto deltas3D = mTFStyle ? deltas.reshape(yato::dims(batchSize, mOutputSize, mOutputChannels)) : deltas.reshape(yato::dims(batchSize, mOutputChannels, mOutputSize));
    const Tensor& weights = mNetworkParams.mMemoryManager[mWeightsName];
    auto kernelsWeights3D =
        mTFStyle ? weights.reshape(yato::dims(mKernelSize, mInputChannels / mGroups, mOutputChannels)) : weights.reshape(yato::dims(mOutputChannels, mInputChannels / mGroups, mKernelSize));

    size_t inputSizePadded = mInputSize + 2 * mPadding;
    // prevDelta
    // if (mNetworkParams.isGradNeeded(mInputName))
    {
        Tensor& prevDeltaTmp = mNetworkParams.mMemoryManager[mTmpForBackwardName];

        auto prevDeltaTmp2D = mTFStyle ? prevDeltaTmp.reshape(yato::dims(inputSizePadded, mInputChannels)) : prevDeltaTmp.reshape(yato::dims(mInputChannels, inputSizePadded));
        Tensor& prevLayerDelta = mNetworkParams.mMemoryManager[mInputName.grad()];
        auto prevDeltas3D = mTFStyle ? prevLayerDelta.reshape(yato::dims(batchSize, mInputSize, mInputChannels)) : prevLayerDelta.reshape(yato::dims(batchSize, mInputChannels, mInputSize));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < batchSize; ++i)
        {
            std::fill(prevDeltaTmp.begin(), prevDeltaTmp.end(), 0.0_dt);

            for (size_t group = 0; group < mGroups; ++group)
            {
                for (size_t d = 0; d < mInputChannels / mGroups; ++d)
                {
                    for (size_t kernelIndex = 0; kernelIndex < mOutputChannels / mGroups; ++kernelIndex)
                    {
                        for (size_t ox = 0; ox < mOutputSize; ++ox)
                        {
                            for (size_t kx = 0; kx < mKernelSize; ++kx)
                            {
                                if (mTFStyle)
                                {
                                    prevDeltaTmp2D[ox * mStride + kx * mDilation][d + group * mInputChannels / mGroups] +=
                                        deltas3D[i][ox][kernelIndex + group * mOutputChannels / mGroups] * kernelsWeights3D[kx][d][kernelIndex + group * mOutputChannels / mGroups];
                                }
                                else
                                {
                                    prevDeltaTmp2D[d + group * mInputChannels / mGroups][ox * mStride + kx * mDilation] +=
                                        deltas3D[i][kernelIndex + group * mOutputChannels / mGroups][ox] * kernelsWeights3D[kernelIndex + group * mOutputChannels / mGroups][d][kx];
                                }
                            }
                        }
                    }
                }
            }

            Common::removePadding1D(&prevDeltaTmp2D[0][0], &prevDeltas3D[i][0][0], mInputChannels, inputSizePadded, mInputSize, mTFStyle, false);
        }
    }
    if (!mFrozen)
    {
        const Tensor& inputs = mNetworkParams.mMemoryManager[mInputName];
        Tensor& gradWeights = mNetworkParams.mMemoryManager[mWeightsName.grad()];

        auto inputs3D = mTFStyle ? inputs.reshape(yato::dims(batchSize, mInputSize, mInputChannels)) : inputs.reshape(yato::dims(batchSize, mInputChannels, mInputSize));
        auto gradWeights3D = mTFStyle ? gradWeights.reshape(yato::dims(mKernelSize, mInputChannels / mGroups, mOutputChannels))
                                      : gradWeights.reshape(yato::dims(mOutputChannels, mInputChannels / mGroups, mKernelSize));

        // gradients weights
        for (size_t i = 0; i < batchSize; ++i)
        {
            Tensor& inputPadded = mNetworkParams.mMemoryManager[mTmpForBackwardName];
            inputPadded = 0_dt;
            Common::addPadding1D(&inputs3D[i][0][0], inputPadded.data(), mInputChannels, mInputSize, inputSizePadded, mTFStyle);

            auto inputPadded2D = mTFStyle ? inputPadded.reshape(yato::dims(inputSizePadded, mInputChannels)) : inputPadded.reshape(yato::dims(mInputChannels, inputSizePadded));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t group = 0; group < mGroups; ++group)
            {
                for (size_t d = 0; d < mInputChannels / mGroups; ++d)
                {
                    for (size_t kernelIndex = 0; kernelIndex < mOutputChannels / mGroups; ++kernelIndex)
                    {
                        for (size_t kx = 0; kx < mKernelSize; ++kx)
                        {
                            for (size_t ox = 0; ox < mOutputSize; ++ox)
                            {
                                if (mTFStyle)
                                {
                                    gradWeights3D[kx][d][kernelIndex + group * mOutputChannels / mGroups] +=
                                        deltas3D[i][ox][kernelIndex + group * mOutputChannels / mGroups] * inputPadded2D[ox * mStride + kx * mDilation][d + group * mInputChannels / mGroups];
                                }
                                else
                                {
                                    gradWeights3D[kernelIndex + group * mOutputChannels / mGroups][d][kx] +=
                                        deltas3D[i][kernelIndex + group * mOutputChannels / mGroups][ox] * inputPadded2D[d + group * mInputChannels / mGroups][ox * mStride + kx * mDilation];
                                }
                            }
                        }
                    }
                }
            }
        }
        // gradients biases
        if (mUseBias)
        {
            Tensor& gradBiases = mNetworkParams.mMemoryManager[mBiasesName.grad()];

            for (size_t kernelIndex = 0; kernelIndex < mOutputChannels; ++kernelIndex)
            {
                auto gradBias = 0.0_dt;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : gradBias)
#endif
                for (size_t i = 0; i < batchSize; ++i)
                {
                    for (size_t ow = 0; ow < mOutputSize; ++ow)
                    {
                        if (mTFStyle)
                        {
                            gradBias += deltas3D[i][ow][kernelIndex];
                        }
                        else
                        {
                            gradBias += deltas3D[i][kernelIndex][ow];
                        }
                    }
                }
                gradBiases[kernelIndex] += gradBias;
            }
        }
    }
}
#else
void Convolution1DLayer::backwardComputeImpl()
{
    mImpl->backwardComputeImpl();
}
#endif

} // namespace raul