// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "WorkflowActions.h"

namespace raul
{

void Backward::rescaleGrads(Names& grads)
{
    auto& memoryManagerFP32 = mWork.getMemoryManager<MemoryManager>();
    auto& memoryManagerFP16 = mWork.getMemoryManager<MemoryManagerFP16>();

    auto maxScale = 0.0_dt;
    for (auto& gradName : grads)
    {
        if (memoryManagerFP32.tensorExists(gradName))
        {
            auto grad = memoryManagerFP32[gradName];
            auto scale = grad.getScale();
            if (scale && std::abs(*scale) > std::abs(maxScale)) maxScale = *scale;
        }

        if (memoryManagerFP16.tensorExists(gradName))
        {
            auto grad = memoryManagerFP16[gradName];
            auto scale = grad.getScale();
            if (scale && std::abs(static_cast<dtype>(*scale)) > std::abs(maxScale)) maxScale = static_cast<dtype>(*scale);
        }
    }

    for (auto& gradName : grads)
    {
        if (memoryManagerFP32.tensorExists(gradName))
        {
            auto grad = memoryManagerFP32[gradName];
            auto scale = grad.getScale();
            if (scale)
            {
                auto unified_scale = maxScale / *scale;
                grad.scale(unified_scale);
            }
            else
            {
                grad.scale(maxScale);
            }
        }

        if (memoryManagerFP16.tensorExists(gradName))
        {
            auto grad = memoryManagerFP16[gradName];
            auto scale = grad.getScale();
            if (scale)
            {
                auto unified_scale = maxScale / *scale;
                grad.scale(static_cast<half>(unified_scale));
            }
            else
            {
                grad.scale(static_cast<half>(maxScale));
            }
        }
    }
}

void Backward::applyScale(const Names& grads, dtype scale)
{
    auto& memoryManagerFP32 = mWork.getMemoryManager<MemoryManager>();
    auto& memoryManagerFP16 = mWork.getMemoryManager<MemoryManagerFP16>();

    for (auto& gradName : grads)
    {
        if (memoryManagerFP32.tensorExists(gradName))
        {
            memoryManagerFP32[gradName].resetScale(scale);
        }
        else if (memoryManagerFP16.tensorExists(gradName))
        {
            memoryManagerFP16[gradName].resetScale(scale);
        }
    }
}

void Backward::scaleGrads()
{
    try
    {
        auto& memoryManagerFP32 = mWork.getMemoryManager<MemoryManager>();
        auto& memoryManagerFP16 = mWork.getMemoryManager<MemoryManagerFP16>();

        auto outputsGrads = mLayer->getOutputs();
        std::transform(outputsGrads.cbegin(), outputsGrads.cend(), outputsGrads.begin(), [](auto& x) { return x.grad(); });

        if (outputsGrads.size() > 1)
        {
            rescaleGrads(outputsGrads);
        }

        for (auto& gradName : outputsGrads)
        {
            if (memoryManagerFP32.tensorExists(gradName))
            {
                mScaling->scale(memoryManagerFP32[gradName]);
            }
            else if (memoryManagerFP16.tensorExists(gradName))
            {
                mScaling->scale(memoryManagerFP16[gradName]);
            }
        }
    }
    catch (...)
    {
        THROW_NONAME("Backward", "Cannot scale gradients");
    }
}

void Backward::propagateScale()
{
    try
    {
        auto outputsGrads = mLayer->getOutputs();
        auto inputsGrads = mLayer->getInputs();
        auto paramsGrads = mWork.getLayerParameterNames(mLayer->getName());
        std::transform(outputsGrads.cbegin(), outputsGrads.cend(), outputsGrads.begin(), [](auto& x) { return x.grad(); });
        std::transform(inputsGrads.cbegin(), inputsGrads.cend(), inputsGrads.begin(), [](auto& x) { return x.grad(); });
        std::transform(paramsGrads.cbegin(), paramsGrads.cend(), paramsGrads.begin(), [](auto& x) { return x.grad(); });

        auto& memoryManagerFP32 = mWork.getMemoryManager<MemoryManager>();
        auto& memoryManagerFP16 = mWork.getMemoryManager<MemoryManagerFP16>();

        std::optional<dtype> scale = std::nullopt;

        if (!outputsGrads.empty())
        {
            const auto deltaName = outputsGrads[0];
            if (memoryManagerFP32.tensorExists(deltaName))
            {
                scale = memoryManagerFP32[deltaName].getScale();
            }
            else if (memoryManagerFP16.tensorExists(deltaName))
            {
                scale = memoryManagerFP16[deltaName].getScale();
            }
        }

        if (scale)
        {
            applyScale(inputsGrads, *scale);
            applyScale(paramsGrads, *scale);
        }
    }
    catch (...)
    {
        THROW_NONAME("Backward", "Cannot propagate scale");
    }
}

} // namespace raul