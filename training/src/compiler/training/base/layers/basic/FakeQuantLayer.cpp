// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "FakeQuantLayer.h"

#include <algorithm>

namespace raul
{

FakeQuantLayer::FakeQuantLayer(const Name& name, const FakeQuantParams& params, NetworkParameters& networkParameters)
    : BasicLayer(name, "FakeQuant", params, networkParameters)
    , mQuantizationMode(params.mQuantizationMode)
{
    auto prefix = mTypeName + "[" + mName + "::ctor]: ";
    if (!mNetworkParams.mQuantizerPtr)
    {
        THROW(mTypeName, mName, "quantizer is not defined");
    }

    if (mInputs.size() != mOutputs.size())
    {
        THROW(mTypeName,
              mName,
              "number of input tensors do not match the numbers of output ones (expected: " + Conversions::toString(mInputs.size()) + ", got: " + Conversions::toString(mOutputs.size()) + ")");
    }

    for (const auto& input_name : mInputs)
    {
        mNetworkParams.mWorkflow.copyDeclaration(mName, input_name, raul::Workflow::Usage::ForwardAndBackward, raul::Workflow::Mode::Read);

        mNetworkParams.mWorkflow.copyDeclaration(mName, input_name, input_name.grad(), DEC_BACK_WRIT_ZERO);
    }

    for (size_t i = 0; i < mOutputs.size(); ++i)
    {
        const auto output_name = mOutputs[i];
        const auto input_name = mInputs[i];

        mNetworkParams.mWorkflow.copyDeclaration(mName, input_name, output_name, DEC_FORW_WRIT);

        mNetworkParams.mWorkflow.copyDeclaration(mName, output_name, output_name.grad(), DEC_BACK_READ);
    }
}

void FakeQuantLayer::forwardComputeImpl(NetworkMode)
{

    for (size_t i = 0; i < mOutputs.size(); ++i)
    {
        const auto output_name = mOutputs[i];
        const auto input_name = mInputs[i];

        const Tensor& input = mNetworkParams.mMemoryManager[input_name];
        Tensor& output = mNetworkParams.mMemoryManager[output_name];
        output = TORANGE(input);
        switch (mQuantizationMode)
        {
            case QuantizationMode::over_full_tensor:
                mNetworkParams.mQuantizerPtr->quantize(output.begin(), output.end());
                mNetworkParams.mQuantizerPtr->dequantize(output.begin(), output.end());
                break;
            case QuantizationMode::over_batch:
            {
                const auto batch_size = mNetworkParams.mWorkflow.getBatchSize();
                const auto size = output.size() / batch_size;
                auto output2D = output.reshape(yato::dims(batch_size, size));
                for (size_t j = 0U; j < batch_size; ++j)
                {
                    mNetworkParams.mQuantizerPtr->quantize(output2D[j].begin(), output2D[j].end());
                    mNetworkParams.mQuantizerPtr->dequantize(output2D[j].begin(), output2D[j].end());
                }
            }
            break;
            default:
                THROW("FakeQuant", mName, "Quantization mode is not implemented");
        }
    }
}

void FakeQuantLayer::backwardComputeImpl()
{
    for (size_t i = 0; i < mOutputs.size(); ++i)
    {
        const auto output_name = mOutputs[i];
        const auto input_name = mInputs[i];

        const Tensor& input = mNetworkParams.mMemoryManager[input_name];
        const Tensor& delta = mNetworkParams.mMemoryManager[output_name.grad()];
        auto& in_nabla = mNetworkParams.mMemoryManager[input_name.grad()];
        switch (mQuantizationMode)
        {
            case QuantizationMode::over_full_tensor:
                mNetworkParams.mQuantizerPtr->backpropagate(input.cbegin(), input.cend(), delta.cbegin(), in_nabla.begin());
                break;
            case QuantizationMode::over_batch:
            {
                const auto batch_size = mNetworkParams.mWorkflow.getBatchSize();
                const auto size = input.size() / batch_size;
                auto input2D = input.reshape(yato::dims(batch_size, size));
                auto delta2D = delta.reshape(yato::dims(batch_size, size));
                auto in_nabla2D = in_nabla.reshape(yato::dims(batch_size, size));

                for (size_t j = 0U; j < batch_size; ++j)
                {
                    mNetworkParams.mQuantizerPtr->backpropagate(input2D[j].cbegin(), input2D[j].cend(), delta2D[j].cbegin(), in_nabla2D[j].begin());
                }
            }
            break;
            default:
                THROW("FakeQuant", mName, "Quantization mode is not implemented");
        }
    }
}

} // namespace raul