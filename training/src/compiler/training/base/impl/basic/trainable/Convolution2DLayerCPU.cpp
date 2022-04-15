// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Convolution2DLayerCPU.h"
#include <training/base/layers/basic/trainable/Convolution2DLayer.h>

#include <training/base/impl/ImplFactory.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::Convolution2DLayer, raul::Convolution2DLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::Convolution2DLayer, raul::Convolution2DLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
void Convolution2DLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    Workflow& work = mLayer.getNetworkParams().mWorkflow;

    const size_t batchSize = work.getBatchSize();

    auto& output = work.getMemoryManager<MM>()[mLayer.getOutputName()];

    const auto& inputs = work.getMemoryManager<MM>()[mLayer.getInputName()];

    auto& weights = work.getMemoryManager<MM>()[mLayer.getWeightsName()];

    if (mLayer.isQuantizeWeights())
    {
        work.getMemoryManager<MM>()[mLayer.getWeightsBackup()] = TORANGE_MM(weights);
        mLayer.getNetworkParams().mQuantizerPtr->quantize(weights.begin(), weights.end());
        mLayer.getNetworkParams().mQuantizerPtr->dequantize(weights.begin(), weights.end());
    }

#ifdef RAUL_NAIVE_CONV_FORWARD
    Common::conv2d(&inputs[0],
                   &output[0],
                   &weights[0],
                   mLayer.isUseBias() ? &work.getMemoryManager<MM>()[mLayer.getBiasesName()][0] : nullptr,
                   batchSize,
                   mLayer.getInputWidth(),
                   mLayer.getInputHeight(),
                   mLayer.getInputDepth(),
                   mLayer.getOutputWidth(),
                   mLayer.getOutputHeight(),
                   mLayer.getKernelsCount(),
                   mLayer.getKernelWidth(),
                   mLayer.getKernelHeight(),
                   mLayer.getPaddingW(),
                   mLayer.getPaddingH(),
                   mLayer.getStrideW(),
                   mLayer.getStrideH(),
                   mLayer.getDilationW(),
                   mLayer.getDilationH(),
                   mLayer.getGroups());
#else
    auto inputs3D = inputs.reshape(yato::dims(batchSize, mLayer.getInputDepth(), mLayer.getInputHeight() * mLayer.getInputWidth()));
    auto outputs3D = output.reshape(yato::dims(batchSize, mLayer.getKernelsCount(), mLayer.getOutputHeight() * mLayer.getOutputWidth()));

    // Fill dilated weights if needed
    if (mLayer.isDilationEnabled())
    {
        auto& dilationWeights = work.getMemoryManager<MM>()[mLayer.getDilationTensor()];

        auto kernelsWeights4D = weights.reshape(yato::dims(mLayer.getKernelsCount(), mLayer.getInputDepth() / mLayer.getGroups(), mLayer.getKernelHeight(), mLayer.getKernelWidth()));
        auto dilatedKernelsWeights4D =
            dilationWeights.reshape(yato::dims(mLayer.getKernelsCount(), mLayer.getInputDepth() / mLayer.getGroups(), mLayer.getEffectiveReceptiveFieldH(), mLayer.getEffectiveReceptiveFieldW()));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t kernelIndex = 0; kernelIndex < mLayer.getKernelsCount(); ++kernelIndex)
        {
            for (size_t d = 0; d < mLayer.getInputDepth() / mLayer.getGroups(); ++d)
            {
                for (size_t ky = 0; ky < mLayer.getKernelHeight(); ++ky)
                {
                    for (size_t kx = 0; kx < mLayer.getKernelWidth(); ++kx)
                    {
                        dilatedKernelsWeights4D[kernelIndex][d][ky * mLayer.getDilationH()][kx * mLayer.getDilationW()] = kernelsWeights4D[kernelIndex][d][ky][kx];
                    }
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

        auto& im2ColFor = work.getMemoryManager<MM>()[mLayer.getIm2ColForward()[index]];

        Common::im2col(&inputs3D[q][0][0],
                       mLayer.getInputWidth(),
                       mLayer.getInputHeight(),
                       mLayer.getInputDepth(),
                       mLayer.getEffectiveReceptiveFieldW(),
                       mLayer.getEffectiveReceptiveFieldH(),
                       mLayer.getStrideW(),
                       mLayer.getStrideH(),
                       mLayer.getPaddingW(),
                       mLayer.getPaddingH(),
                       &im2ColFor[0]);

        auto& wT = mLayer.isDilationEnabled() ? work.getMemoryManager<MM>()[mLayer.getDilationTensor()] : weights;

        for (size_t group = 0; group < mLayer.getGroups(); ++group)
        {
            Common::gemm(CblasNoTrans,
                         CblasNoTrans,
                         mLayer.getKernelsCount() / mLayer.getGroups(),
                         mLayer.getOutputWidth() * mLayer.getOutputHeight(),
                         mLayer.getEffectiveReceptiveFieldW() * mLayer.getEffectiveReceptiveFieldH() * mLayer.getInputDepth() / mLayer.getGroups(),
                         1.0_dt,
                         &wT[0] + group * mLayer.getKernelsCount() / mLayer.getGroups() * mLayer.getEffectiveReceptiveFieldW() * mLayer.getEffectiveReceptiveFieldH() * mLayer.getInputDepth() /
                                      mLayer.getGroups(),
                         &im2ColFor[0] + group * mLayer.getInputDepth() / mLayer.getGroups() * mLayer.getEffectiveReceptiveFieldW() * mLayer.getEffectiveReceptiveFieldH() * mLayer.getOutputWidth() *
                                             mLayer.getOutputHeight(),
                         0.0_dt,
                         &outputs3D[q][group * mLayer.getKernelsCount() / mLayer.getGroups()][0]);
        }
    }

    if (mLayer.isUseBias())
    {
        const auto& biases = work.getMemoryManager<MM>()[mLayer.getBiasesName()];

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < batchSize; ++q)
        {
            for (size_t kernelIndex = 0; kernelIndex < mLayer.getKernelsCount(); ++kernelIndex)
            {
                const auto bias = biases[kernelIndex];
                std::transform(
                    outputs3D[q][kernelIndex].begin(), outputs3D[q][kernelIndex].end(), outputs3D[q][kernelIndex].begin(), [bias](typename MM::type& val) -> typename MM::type { return val + bias; });
            }
        }
    }
#endif

    if (mLayer.isQuantizeWeights())
    {
        weights = TORANGE_MM(work.getMemoryManager<MM>()[mLayer.getWeightsBackup()]);
    }
}

template<typename MM>
void Convolution2DLayerCPU<MM>::backwardComputeImpl()
{
    Workflow& work = mLayer.getNetworkParams().mWorkflow;

    const size_t batchSize = work.getBatchSize();

    auto& deltas = work.getMemoryManager<MM>()[mLayer.getOutputName().grad()];

    const auto& weights = mLayer.isDilationEnabled() ? work.getMemoryManager<MM>()[mLayer.getDilationTensor()] : work.getMemoryManager<MM>()[mLayer.getWeightsName()];

#ifdef RAUL_NAIVE_CONV_BACKWARD
    auto deltas4D = deltas.reshape(yato::dims(batchSize, mLayer.getKernelsCount(), mLayer.getOutputHeight(), mLayer.getOutputWidth()));

    auto kernelsWeights4D = weights.reshape(yato::dims(mLayer.getKernelsCount(), mLayer.getInputDepth() / mLayer.getGroups(), mLayer.getKernelHeight(), mLayer.getKernelWidth()));

    size_t inputWidthPadded = mLayer.getInputWidth() + 2 * mLayer.getPaddingW();
    size_t inputHeightPadded = mLayer.getInputHeight() + 2 * mLayer.getPaddingH();

    auto& prevDeltaTmp = work.getMemoryManager<MM>()[mLayer.getTmpTensorName()];
    auto prevDeltaTmp3D = prevDeltaTmp.reshape(yato::dims(mLayer.getInputDepth(), inputHeightPadded, inputWidthPadded));

    // prevDelta
    ////if (mLayer.getNetworkParams().isGradNeeded(mLayer.getInputName()))
    {
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.getInputName().grad()];

        auto prevDeltas3D = prevLayerDelta.reshape(yato::dims(batchSize, mLayer.getInputDepth(), mLayer.getInputHeight() * mLayer.getInputWidth()));

        for (size_t i = 0; i < batchSize; ++i)
        {
            std::fill(prevDeltaTmp.begin(), prevDeltaTmp.end(), TOMMTYPE(0.0_dt));

            for (size_t group = 0; group < mLayer.getGroups(); ++group)
            {
                for (size_t d = 0; d < mLayer.getInputDepth() / mLayer.getGroups(); ++d)
                {
                    for (size_t kernelIndex = 0; kernelIndex < mLayer.getKernelsCount() / mLayer.getGroups(); ++kernelIndex)
                    {
                        for (size_t oy = 0; oy < mLayer.getOutputHeight(); ++oy)
                        {
                            for (size_t ox = 0; ox < mLayer.getOutputWidth(); ++ox)
                            {
                                for (size_t ky = 0; ky < mLayer.getKernelHeight(); ++ky)
                                {
                                    for (size_t kx = 0; kx < mLayer.getKernelWidth(); ++kx)
                                    {
                                        prevDeltaTmp3D[d + group * mLayer.getInputDepth() / mLayer.getGroups()][oy * mLayer.getStrideH() + ky * mLayer.getDilationH()]
                                                      [ox * mLayer.getStrideW() + kx * mLayer.getDilationW()] +=
                                            deltas4D[i][kernelIndex + group * mLayer.getKernelsCount() / mLayer.getGroups()][oy][ox] *
                                            kernelsWeights4D[kernelIndex + group * mLayer.getKernelsCount() / mLayer.getGroups()][d][ky][kx];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Common::removePadding2D(
                &prevDeltaTmp3D[0][0][0], &prevDeltas3D[i][0][0], mLayer.getInputDepth(), inputWidthPadded, inputHeightPadded, mLayer.getInputWidth(), mLayer.getInputHeight(), false);
        }
    }

    if (!mLayer.isFrozen())
    {
        auto& inputs = work.getMemoryManager<MM>()[mLayer.getInputName()];

        auto& gradWeights = work.getMemoryManager<MM>()[mLayer.getWeightsName().grad()];

        auto inputs3D = inputs.reshape(yato::dims(batchSize, mLayer.getInputDepth(), mLayer.getInputHeight() * mLayer.getInputWidth()));
        auto gradWeights4D = gradWeights.reshape(yato::dims(mLayer.getKernelsCount(), mLayer.getInputDepth() / mLayer.getGroups(), mLayer.getKernelHeight(), mLayer.getKernelWidth()));

        // gradients weights
        for (size_t i = 0; i < batchSize; ++i)
        {
            auto& inputPadded = work.getMemoryManager<MM>()[mLayer.getTmpTensorName()];
            inputPadded = TOMMTYPE(0_dt);

            Common::addPadding2D(&inputs3D[i][0][0], inputPadded.data(), mLayer.getInputDepth(), mLayer.getInputWidth(), mLayer.getInputHeight(), inputWidthPadded, inputHeightPadded);

            auto inputPadded3D = inputPadded.reshape(yato::dims(mLayer.getInputDepth(), inputHeightPadded, inputWidthPadded));

            for (size_t group = 0; group < mLayer.getGroups(); ++group)
            {
                for (size_t d = 0; d < mLayer.getInputDepth() / mLayer.getGroups(); ++d)
                {
                    for (size_t kernelIndex = 0; kernelIndex < mLayer.getKernelsCount() / mLayer.getGroups(); ++kernelIndex)
                    {
                        for (size_t ky = 0; ky < mLayer.getKernelHeight(); ++ky)
                        {
                            for (size_t kx = 0; kx < mLayer.getKernelWidth(); ++kx)
                            {
                                for (size_t oy = 0; oy < mLayer.getOutputHeight(); ++oy)
                                {
                                    for (size_t ox = 0; ox < mLayer.getOutputWidth(); ++ox)
                                    {
                                        gradWeights4D[kernelIndex + group * mLayer.getKernelsCount() / mLayer.getGroups()][d][ky][kx] +=
                                            deltas4D[i][kernelIndex + group * mLayer.getKernelsCount() / mLayer.getGroups()][oy][ox] *
                                            inputPadded3D[d + group * mLayer.getInputDepth() / mLayer.getGroups()][oy * mLayer.getStrideH() + ky * mLayer.getDilationH()]
                                                         [ox * mLayer.getStrideW() + kx * mLayer.getDilationW()];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // gradients biases
        if (!mLayer.isFrozen() && mLayer.isUseBias())
        {
            auto& gradBiases = work.getMemoryManager<MM>()[mLayer.getBiasesName().grad()];

            for (size_t kernelIndex = 0; kernelIndex < mLayer.getKernelsCount(); ++kernelIndex)
            {
                for (size_t i = 0; i < batchSize; ++i)
                {
                    for (size_t oh = 0; oh < mLayer.getOutputHeight(); ++oh)
                    {
                        for (size_t ow = 0; ow < mLayer.getOutputWidth(); ++ow)
                        {
                            gradBiases[kernelIndex] += deltas4D[i][kernelIndex][oh][ow];
                        }
                    }
                }
            }
        }
    }
#else
    auto deltas3D = deltas.reshape(yato::dims(batchSize, mLayer.getKernelsCount(), mLayer.getOutputHeight() * mLayer.getOutputWidth()));

    // prevDelta
    ////if (mLayer.getNetworkParams().isGradNeeded(mInputName))
    {
        auto& prevLayerDelta = work.getMemoryManager<MM>()[mLayer.getInputName().grad()];
        auto prevDeltas3D = prevLayerDelta.reshape(yato::dims(batchSize, mLayer.getInputDepth(), mLayer.getInputHeight() * mLayer.getInputWidth()));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < batchSize; ++i)
        {
            size_t index = 0;
#if defined(_OPENMP)
            index = omp_get_thread_num();
#endif
            auto& im2ColBack = work.getMemoryManager<MM>()[mLayer.getIm2ColBackward()[index]];
            for (size_t group = 0; group < mLayer.getGroups(); ++group)
            {
                Common::gemm(CblasTrans,
                             CblasNoTrans,
                             mLayer.getEffectiveReceptiveFieldW() * mLayer.getEffectiveReceptiveFieldH() * mLayer.getInputDepth() / mLayer.getGroups(),
                             mLayer.getOutputWidth() * mLayer.getOutputHeight(),
                             mLayer.getKernelsCount() / mLayer.getGroups(),
                             1.0_dt,
                             &weights[0] + group * mLayer.getKernelsCount() / mLayer.getGroups() * mLayer.getEffectiveReceptiveFieldW() * mLayer.getEffectiveReceptiveFieldH() *
                                               mLayer.getInputDepth() / mLayer.getGroups(),
                             &deltas3D[i][0][0],
                             0.0_dt,
                             &im2ColBack[0] + group * mLayer.getEffectiveReceptiveFieldW() * mLayer.getEffectiveReceptiveFieldH() * mLayer.getInputDepth() * mLayer.getOutputWidth() *
                                                  mLayer.getOutputHeight() / mLayer.getGroups());
            }

            Common::col2im(&im2ColBack[0],
                           mLayer.getInputWidth(),
                           mLayer.getInputHeight(),
                           mLayer.getInputDepth(),
                           mLayer.getEffectiveReceptiveFieldW(),
                           mLayer.getEffectiveReceptiveFieldH(),
                           mLayer.getStrideW(),
                           mLayer.getStrideH(),
                           mLayer.getPaddingW(),
                           mLayer.getPaddingH(),
                           &prevDeltas3D[i][0][0],
                           false,
                           false);
        }
    }

    // gradients weights
    if (!mLayer.isFrozen())
    {
        auto& inputs = work.getMemoryManager<MM>()[mLayer.getInputName()];

        auto inputs3D = inputs.reshape(yato::dims(batchSize, mLayer.getInputDepth(), mLayer.getInputHeight() * mLayer.getInputWidth()));

        auto& gradWeights = work.getMemoryManager<MM>()[mLayer.getWeightsName().grad()];

        if (mLayer.isDilationEnabled())
        {
            work.getMemoryManager<MM>()[mLayer.getDilationTensor()] = TOMMTYPE(0);
        }

        if (mLayer.getNetworkParams().mCalculationMode == CalculationMode::DETERMINISTIC)
        {
            auto& im2ColBack = work.getMemoryManager<MM>()[mLayer.getIm2ColBackward()[0]];

            auto& tG = mLayer.isDilationEnabled() ? work.getMemoryManager<MM>()[mLayer.getDilationTensor()] : gradWeights;

            for (size_t q = 0; q < batchSize; ++q)
            {
                Common::im2col(&inputs3D[q][0][0],
                               mLayer.getInputWidth(),
                               mLayer.getInputHeight(),
                               mLayer.getInputDepth(),
                               mLayer.getEffectiveReceptiveFieldW(),
                               mLayer.getEffectiveReceptiveFieldH(),
                               mLayer.getStrideW(),
                               mLayer.getStrideH(),
                               mLayer.getPaddingW(),
                               mLayer.getPaddingH(),
                               &im2ColBack[0]);
                for (size_t group = 0; group < mLayer.getGroups(); ++group)
                {
                    Common::gemm(CblasNoTrans,
                                 CblasTrans,
                                 mLayer.getKernelsCount() / mLayer.getGroups(),
                                 mLayer.getEffectiveReceptiveFieldW() * mLayer.getEffectiveReceptiveFieldH() * mLayer.getInputDepth() / mLayer.getGroups(),
                                 mLayer.getOutputWidth() * mLayer.getOutputHeight(),
                                 1.0_dt,
                                 &deltas3D[q][group * mLayer.getKernelsCount() / mLayer.getGroups()][0],
                                 &im2ColBack[0] + group * mLayer.getEffectiveReceptiveFieldW() * mLayer.getEffectiveReceptiveFieldH() * mLayer.getInputDepth() * mLayer.getOutputWidth() *
                                                      mLayer.getOutputHeight() / mLayer.getGroups(),
                                 1.0_dt,
                                 &tG[0] + group * mLayer.getKernelsCount() / mLayer.getGroups() * mLayer.getEffectiveReceptiveFieldW() * mLayer.getEffectiveReceptiveFieldH() * mLayer.getInputDepth() /
                                              mLayer.getGroups());
                }
            }
        }
#if defined(_OPENMP)
        else if (mLayer.getNetworkParams().mCalculationMode == CalculationMode::FAST)
        {
            auto& tG = mLayer.isDilationEnabled() ? work.getMemoryManager<MM>()[mLayer.getDilationTensor()] : gradWeights;

#pragma omp parallel for
            for (size_t q = 0; q < batchSize; ++q)
            {
                size_t index = omp_get_thread_num();

                auto& im2ColBack = work.getMemoryManager<MM>()[mLayer.getIm2ColBackward()[index]];

                Common::im2col(&inputs3D[q][0][0],
                               mLayer.getInputWidth(),
                               mLayer.getInputHeight(),
                               mLayer.getInputDepth(),
                               mLayer.getEffectiveReceptiveFieldW(),
                               mLayer.getEffectiveReceptiveFieldH(),
                               mLayer.getStrideW(),
                               mLayer.getStrideH(),
                               mLayer.getPaddingW(),
                               mLayer.getPaddingH(),
                               &im2ColBack[0]);
#pragma omp critical
                for (size_t group = 0; group < mLayer.getGroups(); ++group)
                {
                    Common::gemm(CblasNoTrans,
                                 CblasTrans,
                                 mLayer.getKernelsCount() / mLayer.getGroups(),
                                 mLayer.getEffectiveReceptiveFieldW() * mLayer.getEffectiveReceptiveFieldH() * mLayer.getInputDepth() / mLayer.getGroups(),
                                 mLayer.getOutputWidth() * mLayer.getOutputHeight(),
                                 1.0_dt,
                                 &deltas3D[q][group * mLayer.getKernelsCount() / mLayer.getGroups()][0],
                                 &im2ColBack[0] + group * mLayer.getEffectiveReceptiveFieldW() * mLayer.getEffectiveReceptiveFieldH() * mLayer.getInputDepth() * mLayer.getOutputWidth() *
                                                      mLayer.getOutputHeight() / mLayer.getGroups(),
                                 1.0_dt,
                                 &tG[0] + group * mLayer.getKernelsCount() / mLayer.getGroups() * mLayer.getEffectiveReceptiveFieldW() * mLayer.getEffectiveReceptiveFieldH() * mLayer.getInputDepth() /
                                              mLayer.getGroups());
                }
            }
        }
#endif
        else
        {
            THROW("Convolution2DLayer", mLayer.getName(), "unexpected calculation mode");
        }

        if (mLayer.isDilationEnabled())
        {
            const auto& dilationWeightsGrad = work.getMemoryManager<MM>()[mLayer.getDilationTensor()];
            auto gradWeights4D = gradWeights.reshape(yato::dims(mLayer.getKernelsCount(), mLayer.getInputDepth() / mLayer.getGroups(), mLayer.getKernelHeight(), mLayer.getKernelWidth()));
            const auto dilatedGradWeights4D = dilationWeightsGrad.reshape(
                yato::dims(mLayer.getKernelsCount(), mLayer.getInputDepth() / mLayer.getGroups(), mLayer.getEffectiveReceptiveFieldH(), mLayer.getEffectiveReceptiveFieldW()));
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t kernelIndex = 0; kernelIndex < mLayer.getKernelsCount(); ++kernelIndex)
            {
                for (size_t d = 0; d < mLayer.getInputDepth() / mLayer.getGroups(); ++d)
                {
                    for (size_t ky = 0; ky < mLayer.getKernelHeight(); ++ky)
                    {
                        for (size_t kx = 0; kx < mLayer.getKernelWidth(); ++kx)
                        {
                            gradWeights4D[kernelIndex][d][ky][kx] += dilatedGradWeights4D[kernelIndex][d][ky * mLayer.getDilationH()][kx * mLayer.getDilationW()];
                        }
                    }
                }
            }
        }
    }

    // gradients biases
    if (!mLayer.isFrozen() && mLayer.isUseBias())
    {
        auto& gradBiases = work.getMemoryManager<MM>()[mLayer.getBiasesName().grad()];
        for (size_t i = 0; i < batchSize; ++i)
        {
            for (size_t kernelIndex = 0; kernelIndex < mLayer.getKernelsCount(); ++kernelIndex)
            {
                gradBiases[kernelIndex] += std::accumulate(deltas3D[i][kernelIndex].begin(), deltas3D[i][kernelIndex].end(), TOMMTYPE(0), std::plus<typename MM::type>());
            }
        }
    }
#endif
}

template class Convolution2DLayerCPU<MemoryManager>;
template class Convolution2DLayerCPU<MemoryManagerFP16>;

} // namespace raul