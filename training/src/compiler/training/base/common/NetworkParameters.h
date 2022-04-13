// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef NETWORK_PARAMETERS_H
#define NETWORK_PARAMETERS_H

#include <training/base/common/MemoryManager.h>
#include <training/base/common/quantization/IQuantizer.h>

namespace raul
{

class BasicLayer;
class Workflow;

struct NetworkParameters
{

    enum class CallbackPlace : int
    {
        Before_Forward = 0,
        After_Forward = 1,
        Before_Backward = 2,
        After_Backward = 3
    };

    using CpuCallback = std::function<void(raul::BasicLayer*, raul::MemoryManager&, raul::NetworkParameters::CallbackPlace)>;
    using CpuFP16Callback = std::function<void(raul::BasicLayer*, raul::MemoryManagerFP16&, raul::NetworkParameters::CallbackPlace)>;

    NetworkParameters(
        MemoryManager& memoryManager,
        MemoryManagerFP16& memoryManagerFP16,
        Workflow& workflow,
        size_t lossReductionCoefficient,
        CompressionMode compressionMode,
        CalculationMode calculationMode,
        quantization::IQuantizer* quantizer = nullptr,
        CpuCallback callback = [](BasicLayer*, MemoryManager&, NetworkParameters::CallbackPlace) {},
        CpuFP16Callback callbackCPUFP16 = [](BasicLayer*, MemoryManagerFP16&, NetworkParameters::CallbackPlace) {})
        : mMemoryManager(memoryManager)
        , mMemoryManagerFP16(memoryManagerFP16)
        , mWorkflow(workflow)
        , mLossReductionCoefficient(lossReductionCoefficient)
        , mCompressionMode(compressionMode)
        , mCalculationMode(calculationMode)
        , mQuantizerPtr(quantizer)
        , mCallback(callback)
        , mCallbackFP16(callbackCPUFP16)
    {
    }

    void setCallback(CpuCallback callback) { mCallback = callback; }
    void setCallback(CpuFP16Callback callback) { mCallbackFP16 = callback; }

    MemoryManager& mMemoryManager;
    MemoryManagerFP16& mMemoryManagerFP16;
    Workflow& mWorkflow;
    size_t mLossReductionCoefficient;
    const CompressionMode mCompressionMode;
    const CalculationMode mCalculationMode;
    quantization::IQuantizer* mQuantizerPtr;

    CpuCallback mCallback;
    CpuFP16Callback mCallbackFP16;
};

NetworkParameters::CpuCallback CallbackHelper(std::optional<std::function<void(raul::BasicLayer*, raul::MemoryManager&)>> beforeForward,
                                              std::optional<std::function<void(raul::BasicLayer*, raul::MemoryManager&)>> afterForward,
                                              std::optional<std::function<void(raul::BasicLayer*, raul::MemoryManager&)>> beforeBackward,
                                              std::optional<std::function<void(raul::BasicLayer*, raul::MemoryManager&)>> afterBackward);

NetworkParameters::CpuFP16Callback CallbackHelperFP16(std::optional<std::function<void(raul::BasicLayer*, raul::MemoryManagerFP16&)>> beforeForward,
                                                      std::optional<std::function<void(raul::BasicLayer*, raul::MemoryManagerFP16&)>> afterForward,
                                                      std::optional<std::function<void(raul::BasicLayer*, raul::MemoryManagerFP16&)>> beforeBackward,
                                                      std::optional<std::function<void(raul::BasicLayer*, raul::MemoryManagerFP16&)>> afterBackward);

} // raul namespace

#endif