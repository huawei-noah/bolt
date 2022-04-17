// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "SelectLayerCPU.h"
#include "../SelectLayer.h"

namespace raul
{

template<typename MM>
SelectLayerCPU<MM>::SelectLayerCPU(SelectLayer& layer)
    : mLayer(layer)
{
}

template<typename MM>
void SelectLayerCPU<MM>::forwardComputeImpl(NetworkMode)
{
    mLayer.determineBroadcastFlags();

    if (!mLayer.mBroadcast && std::any_of(mLayer.mBroadcastQuery.begin(), mLayer.mBroadcastQuery.end(), [](const auto& needToBroadcast) { return needToBroadcast; }))
    {
        THROW("SelectLayer", mLayer.mName, "input size mismatch");
    }

    auto& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputs[0]];

    const auto& cond = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
    const auto& x = work.getMemoryManager<MM>()[mLayer.mInputs[1]];
    const auto& y = work.getMemoryManager<MM>()[mLayer.mInputs[2]];

    if (!mLayer.mBroadcast || (!mLayer.mBroadcastQuery[0] && !mLayer.mBroadcastQuery[1] && !mLayer.mBroadcastQuery[2]))
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < output.size(); ++q)
        {
            output[q] = static_cast<bool>(cond[q]) ? x[q] : y[q];
        }
    }
    else
    {
        if (!mLayer.mBroadcastQuery[0] && !mLayer.mBroadcastQuery[1])
        {
            const auto y_viewer = y.getBroadcastedViewer(output.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < output.size(); ++q)
            {
                output[q] = static_cast<bool>(cond[q]) ? x[q] : y_viewer[q];
            }
        }
        else if (!mLayer.mBroadcastQuery[1] && !mLayer.mBroadcastQuery[2])
        {
            const auto cond_viewer = cond.getBroadcastedViewer(output.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < output.size(); ++q)
            {
                output[q] = static_cast<bool>(cond_viewer[q]) ? x[q] : y[q];
            }
        }
        else if (!mLayer.mBroadcastQuery[2] && !mLayer.mBroadcastQuery[0])
        {
            const auto x_viewer = x.getBroadcastedViewer(output.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < output.size(); ++q)
            {
                output[q] = static_cast<bool>(cond[q]) ? x_viewer[q] : y[q];
            }
        }
        else
        {
            const auto cond_viewer = cond.getBroadcastedViewer(output.getShape());
            const auto x_viewer = x.getBroadcastedViewer(output.getShape());
            const auto y_viewer = y.getBroadcastedViewer(output.getShape());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < output.size(); ++q)
            {
                output[q] = static_cast<bool>(cond_viewer[q]) ? x_viewer[q] : y_viewer[q];
            }
        }
    }
}

template<typename MM>
void SelectLayerCPU<MM>::backwardComputeImpl()
{
    auto& work = mLayer.mNetworkParams.mWorkflow;

    const auto& delta = work.getMemoryManager<MM>()[mLayer.mOutputs[0].grad()];

    if (mLayer.mBroadcastQuery[0])
    {
        const auto condition = work.getMemoryManager<MM>()[mLayer.mInputs[0]].getBroadcastedViewer(delta.getShape());
        for (size_t q = 1; q < mLayer.mInputs.size(); ++q)
        {
            // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[q]))
            {
                auto& in_nabla_tensor = work.getMemoryManager<MM>()[mLayer.mInputs[q].grad()];
                if (mLayer.mBroadcastQuery[q])
                {
                    auto in_nabla = in_nabla_tensor.getBroadcastedViewer(delta.getShape());
                    for (size_t i = 0; i < in_nabla.size(); ++i)
                    {
                        auto cond = q == 1 ? static_cast<bool>(condition[i]) : !static_cast<bool>(condition[i]);
                        in_nabla[i] += (cond) ? delta[i] : 0.0_hf;
                    }
                }
                else
                {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                    for (size_t i = 0; i < delta.size(); ++i)
                    {
                        auto cond = q == 1 ? static_cast<bool>(condition[i]) : !static_cast<bool>(condition[i]);
                        in_nabla_tensor[i] += (cond) ? delta[i] : 0.0_hf;
                    }
                }
            }
        }
    }
    else
    {
        const auto condition = work.getMemoryManager<MM>()[mLayer.mInputs[0]];
        for (size_t q = 1; q < mLayer.mInputs.size(); ++q)
        {
            // if (mLayer.mNetworkParams.isGradNeeded(mLayer.mInputs[q]))
            {
                auto& in_nabla_tensor = work.getMemoryManager<MM>()[mLayer.mInputs[q].grad()];
                if (mLayer.mBroadcastQuery[q])
                {
                    auto in_nabla = in_nabla_tensor.getBroadcastedViewer(delta.getShape());
                    for (size_t i = 0; i < in_nabla.size(); ++i)
                    {
                        auto cond = q == 1 ? static_cast<bool>(condition[i]) : !static_cast<bool>(condition[i]);
                        in_nabla[i] += (cond) ? delta[i] : 0.0_hf;
                    }
                }
                else
                {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
                    for (size_t i = 0; i < delta.size(); ++i)
                    {
                        auto cond = q == 1 ? static_cast<bool>(condition[i]) : !static_cast<bool>(condition[i]);
                        in_nabla_tensor[i] += (cond) ? delta[i] : 0.0_hf;
                    }
                }
            }
        }
    }
}

template class SelectLayerCPU<MemoryManager>;
template class SelectLayerCPU<MemoryManagerFP16>;

} // namespace raul
