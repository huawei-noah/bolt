// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "MatMulLayerCPU.h"
#include "../MatMulLayer.h"

#include <training/base/impl/ImplFactory.h>
namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::MatMulLayer, raul::MatMulLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::MatMulLayer, raul::MatMulLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
MatMulLayerCPU<MM>::MatMulLayerCPU(MatMulLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void MatMulLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const size_t batchSize = work.getBatchSize();

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];

    std::fill(output.begin(), output.end(), TOMMTYPE(0.0_dt));

    const auto& input1 = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
    const auto& input2 = work.getMemoryManager<MM>()[mLayer.mInputs[1]];

    size_t size = mLayer.mDepth * batchSize;

    auto input1_2D = input1.reshape(yato::dims(size, input1.getWidth() * input1.getHeight()));
    auto input2_2D = input2.reshape(yato::dims(size, input2.getWidth() * input2.getHeight()));
    auto output_2D = output.reshape(yato::dims(size, output.getWidth() * output.getHeight()));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t q = 0; q < size; ++q)
    {
        Common::gemm(CblasNoTrans,
                     CblasNoTrans,
                     input1.getHeight(),
                     input2.getWidth(),
                     input1.getWidth(),
                     mLayer.mCoeff,
                     &input1_2D[q][0],
                     &input2_2D[q][0],
                     0.0_dt,
                     &output_2D[q][0]);
    }
}

template<typename MM>
void MatMulLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const size_t batchSize = work.getBatchSize();

    const auto& deltas = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];
    const auto& input1 = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
    const auto& input2 = work.getMemoryManager<MM>()[mLayer.mInputs[1]];
    size_t size = mLayer.mDepth * batchSize;

    auto deltas2D = deltas.reshape(yato::dims(size, input1.getHeight() * input2.getWidth()));

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[0].grad()))
    {
        auto input2_2D = input2.reshape(yato::dims(size, input2.getHeight() * input2.getWidth()));
        auto grad1_2D = work.getMemoryManager<MM>()[mLayer.mInputs[0].grad()].reshape(yato::dims(size, input1.getHeight() * input1.getWidth()));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < size; ++q)
        {
            Common::gemm(CblasNoTrans,
                         CblasTrans,
                         deltas.getHeight(),
                         input2.getHeight(),
                         deltas.getWidth(),
                         mLayer.mCoeff,
                         &deltas2D[q][0],
                         &input2_2D[q][0],
                         1.0_dt,
                         &grad1_2D[q][0]);
        }
    }

    // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[1].grad()))
    {
        auto input1_2D = input1.reshape(yato::dims(size, input1.getHeight() * input1.getWidth()));
        auto grad2_2D = work.getMemoryManager<MM>()[mLayer.mInputs[1].grad()].reshape(yato::dims(size, input2.getHeight() * input2.getWidth()));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < size; ++q)
        {
            Common::gemm(CblasTrans,
                         CblasNoTrans,
                         input1.getWidth(),
                         deltas.getWidth(),
                         input1.getHeight(),
                         mLayer.mCoeff,
                         &input1_2D[q][0],
                         &deltas2D[q][0],
                         1.0_dt,
                         &grad2_2D[q][0]);
        }
    }
}

template class MatMulLayerCPU<MemoryManager>;
template class MatMulLayerCPU<MemoryManagerFP16>;

} // namespace raul