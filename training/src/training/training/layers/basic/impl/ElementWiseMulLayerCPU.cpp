// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ElementWiseMulLayerCPU.h"
#include "../ElementWiseMulLayer.h"

//#define ELEMENTWISE_MUL_OPTIMIZED

namespace raul
{

template<typename MM>
ElementWiseMulLayerCPU<MM>::ElementWiseMulLayerCPU(ElementWiseMulLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void ElementWiseMulLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    mLayer.determineBroadcastFlags();

    if (!mLayer.mBroadcast && std::any_of(mLayer.mBroadcastQuery.begin(), mLayer.mBroadcastQuery.end(), [](const auto& needToBroadcast) { return needToBroadcast; }))
    {
        THROW(mLayer.mTypeName, mLayer.mName, "input size mismatch");
    }

    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& memoryManager = work.getMemoryManager<MM>();

    auto& output = memoryManager[mLayer.mOutputs[0]];

    // Copy the first factor to the output
    if (!mLayer.mBroadcastQuery[0])
    {
        output = TORANGE_MM(memoryManager[mLayer.mInputs[0]]);
    }
    else
    {
        auto input_viewer = memoryManager[mLayer.mInputs[0]].getBroadcastedViewer(output.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < output.size(); ++i)
        {
            output[i] = input_viewer[i];
        }
    }

    // Multiply other factors
    for (size_t q = 1; q < mLayer.mInputs.size(); ++q)
    {
        const auto& input = memoryManager[mLayer.mInputs[q]];

        if (mLayer.mBroadcastQuery[q])
        {
            auto input_viewer = input.getBroadcastedViewer(output.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t i = 0; i < output.size(); ++i)
            {
                output[i] *= input_viewer[i];
            }
        }
        else
        {
            std::transform(input.begin(), input.end(), output.begin(), output.begin(), std::multiplies<typename MM::type>());
        }
    }
}

template<typename MM>
void ElementWiseMulLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& memoryManager = work.getMemoryManager<MM>();

    const auto& delta = memoryManager[mLayer.mOutputs[0].grad()];
    const auto& output = memoryManager[mLayer.mOutputs[0]];

#ifdef ELEMENTWISE_MUL_OPTIMIZED
    const auto epsilon = TOMMTYPE(1e-12_dt);

    if (output.size() == 0)
    {
        THROW("ElementWiseMulLayer", mLayer.mName, "zero output size");
    }

    for (size_t q = 0; q < mLayer.mInputs.size(); ++q)
    {
        // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[q]))
        {
            const auto& input = memoryManager[mLayer.mInputs[q]];
            auto& inputNabla = memoryManager[mLayer.mInputs[q].grad()];
            if (!mLayer.mBroadcastQuery[q])
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t i = 0; i < inputNabla.size(); ++i)
                {
                    inputNabla[i] += output[i] / (input[i] + epsilon) * delta[i];
                }
            }
            else
            {
                const auto inputBroadcasted = input.getBroadcastedViewer(output.getShape());
                auto inputNablaBroadcasted = inputNabla.getBroadcastedViewer(output.getShape());

                for (size_t j = 0; q < output.size(); ++j)
                {

                    // for f=f(x,y,z)
                    // df = y*z dx + x*z dy + x*y dz
                    // So, the coefficient can be calculated as
                    // output (x*y*z) divided on the current value.
                    // NB: potential loss of precision
                    const auto coefficient_q = output[q] / (inputBroadcasted[q] + epsilon);
                    inputNablaBroadcasted[q] += coefficient_q * delta[q];
                }
            }
        }
    }
#else

    for (size_t i = 0; i < mLayer.mInputs.size(); ++i)
    {
        // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[i]))
        {
            const auto size = delta.size();

            typename MM::tensor coefficient_q_vector(size);
            std::fill(coefficient_q_vector.begin(), coefficient_q_vector.end(), TOMMTYPE(1.0_dt));

            for (size_t j = 0; j < mLayer.mInputs.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }
                const auto input_factor_name = mLayer.mInputs[j];
                auto& input_factor_tensor = memoryManager[input_factor_name];

                if (input_factor_tensor.empty())
                {
                    THROW("ElementWiseMulLayer", mLayer.mName, "zero tensor size (" + input_factor_name + ")");
                }

                if (mLayer.mBroadcastQuery[j])
                {
                    const auto input_factor = input_factor_tensor.getBroadcastedViewer(output.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                    for (size_t q = 0; q < size; ++q)
                    {
                        coefficient_q_vector[q] *= input_factor[q];
                    }
                }
                else
                {
                    coefficient_q_vector *= TORANGE_MM(input_factor_tensor);
                }
            }

            const auto input_name = mLayer.mInputs[i];
            auto& inputNabla = memoryManager[input_name.grad()];

            if (mLayer.mBroadcastQuery[i])
            {
                auto in_nabla = inputNabla.getBroadcastedViewer(output.getShape());

                for (size_t q = 0; q < size; ++q)
                {
                    in_nabla[q] += coefficient_q_vector[q] * delta[q];
                }
            }
            else
            {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                for (size_t q = 0; q < size; ++q)
                {
                    inputNabla[q] += coefficient_q_vector[q] * delta[q];
                }
            }
        }
    }
#endif
}

template class ElementWiseMulLayerCPU<MemoryManager>;
template class ElementWiseMulLayerCPU<MemoryManagerFP16>;

} // namespace raul
