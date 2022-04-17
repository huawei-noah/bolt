// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ElementWiseDivLayerCPU.h"
#include "../ElementWiseDivLayer.h"

#include <training/base/impl/ImplFactory.h>
namespace
{
bool reg1 = raul::TheImplFactory::Instance().regCPUFP32<raul::ElementWiseDivLayer, raul::ElementWiseDivLayerCPU<raul::MemoryManager>>();
bool reg2 = raul::TheImplFactory::Instance().regCPUFP16<raul::ElementWiseDivLayer, raul::ElementWiseDivLayerCPU<raul::MemoryManagerFP16>>();
} // anonymous namespace

namespace raul
{

template<typename MM>
ElementWiseDivLayerCPU<MM>::ElementWiseDivLayerCPU(ElementWiseDivLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void ElementWiseDivLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    mLayer.determineBroadcastFlags();

    if (!mLayer.mBroadcast && (mLayer.mBroadcastQuery[0] || mLayer.mBroadcastQuery[1]))
    {
        THROW(mLayer.mTypeName, mLayer.mName, "input size mismatch");
    }

    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];

    // Copy dividend to the output
    if (mLayer.mBroadcastQuery[0])
    {
        auto dividend_viewer = work.getMemoryManager<MM>()[mLayer.mInputs[0]].getBroadcastedViewer(output.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < output.size(); ++q)
        {
            output[q] = dividend_viewer[q];
        }
    }
    else
    {
        output = TORANGE_MM(work.getMemoryManager<MM>()[mLayer.mInputs[0]]);
    }

    const auto& divisor = work.getMemoryManager<MM>()[mLayer.mInputs[1]];

    if (mLayer.mBroadcastQuery[1])
    {
        auto divisor_viewer = divisor.getBroadcastedViewer(output.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif

        for (size_t i = 0; i < output.size(); ++i)
        {
            output[i] /= divisor_viewer[i];
        }
    }
    else
    {
        std::transform(output.begin(), output.end(), divisor.begin(), output.begin(), std::divides<typename MM::type>());
    }
}

template<typename MM>
void ElementWiseDivLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto dividend_name = mLayer.mInputs[0];
    const auto divisor_name = mLayer.mInputs[1];

    const auto& delta = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];

    // if (mLayer.mNetworkParams.isGradNeeded(dividend_name))
    {
        auto& dividend_nabla_tensor = work.getMemoryManager<MM>()[dividend_name.grad()];
        const auto& divisor_factor_tensor = work.getMemoryManager<MM>()[divisor_name];

        if (mLayer.mBroadcastQuery[0])
        {
            auto dividend_nabla = dividend_nabla_tensor.getBroadcastedViewer(delta.getShape());
            if (mLayer.mBroadcastQuery[1])
            {
                const auto divisor_factor = divisor_factor_tensor.getBroadcastedViewer(delta.getShape());
                for (size_t q = 0; q < delta.size(); ++q)
                {
                    dividend_nabla[q] += (TOMMTYPE(1.0_dt) / divisor_factor[q]) * delta[q];
                }
            }
            else
            {
                for (size_t q = 0; q < delta.size(); ++q)
                {
                    dividend_nabla[q] += (TOMMTYPE(1.0_dt) / divisor_factor_tensor[q]) * delta[q];
                }
            }
        }
        else
        {
            if (mLayer.mBroadcastQuery[1])
            {
                const auto divisor_factor = divisor_factor_tensor.getBroadcastedViewer(delta.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t q = 0; q < delta.size(); ++q)
                {
                    dividend_nabla_tensor[q] += (TOMMTYPE(1.0_dt) / divisor_factor[q]) * delta[q];
                }
            }
            else
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t q = 0; q < delta.size(); ++q)
                {
                    dividend_nabla_tensor[q] += (TOMMTYPE(1.0_dt) / divisor_factor_tensor[q]) * delta[q];
                }
            }
        }
    }

    // if (mLayer.mNetworkParams.isGradNeeded(divisor_name))
    {
        auto& divisor_nabla_tensor = work.getMemoryManager<MM>()[divisor_name.grad()];
        const auto& divisor_factor_tensor = work.getMemoryManager<MM>()[divisor_name];
        const auto& dividend_factor_tensor = work.getMemoryManager<MM>()[dividend_name];

        if (mLayer.mBroadcastQuery[1])
        {
            auto divisor_nabla = divisor_nabla_tensor.getBroadcastedViewer(delta.getShape());
            const auto divisor_factor = divisor_factor_tensor.getBroadcastedViewer(delta.getShape());
            if (mLayer.mBroadcastQuery[0])
            {
                const auto dividend_factor = dividend_factor_tensor.getBroadcastedViewer(delta.getShape());
                for (size_t q = 0; q < delta.size(); ++q)
                {
                    divisor_nabla[q] += (dividend_factor[q] / divisor_factor[q] / divisor_factor[q] * TOMMTYPE(-1.0_dt)) * delta[q];
                }
            }
            else
            {
                for (size_t q = 0; q < delta.size(); ++q)
                {
                    divisor_nabla[q] += (dividend_factor_tensor[q] / divisor_factor[q] / divisor_factor[q] * TOMMTYPE(-1.0_dt)) * delta[q];
                }
            }
        }
        else
        {
            if (mLayer.mBroadcastQuery[0])
            {
                const auto dividend_factor = dividend_factor_tensor.getBroadcastedViewer(delta.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t q = 0; q < delta.size(); ++q)
                {
                    divisor_nabla_tensor[q] += (dividend_factor[q] / divisor_factor_tensor[q] / divisor_factor_tensor[q] * TOMMTYPE(-1.0_dt)) * delta[q];
                }
            }
            else
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t q = 0; q < delta.size(); ++q)
                {
                    divisor_nabla_tensor[q] += (dividend_factor_tensor[q] / divisor_factor_tensor[q] / divisor_factor_tensor[q] * TOMMTYPE(-1.0_dt)) * delta[q];
                }
            }
        }
    }
}

template class ElementWiseDivLayerCPU<MemoryManager>;
template class ElementWiseDivLayerCPU<MemoryManagerFP16>;

} // namespace raul
