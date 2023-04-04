// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <training/base/layers/basic/PaddingLayer.h>

namespace raul
{

PaddingLayer::PaddingLayer(const Name& name, const PaddingLayerParams& layerParameters, NetworkParameters& networkParameters)
    : BasicLayer(name, "Padding", layerParameters, networkParameters)
    , calculationStrategy(CalculationStrategy::define(layerParameters, networkParameters))
    , mFillingValue(layerParameters.mFillingValue)
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

    mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[0], raul::Workflow::Usage::Forward, raul::Workflow::Mode::Read);

    mNetworkParams.mWorkflow.copyDeclaration(name, mInputs[0], mInputs[0].grad(), DEC_BACK_WRIT_ZERO);

    const size_t inputDepth = networkParameters.mWorkflow.getDepth(mInputs[0]);
    const size_t mInputHeight = networkParameters.mWorkflow.getHeight(mInputs[0]);
    const size_t mInputWidth = networkParameters.mWorkflow.getWidth(mInputs[0]);

    size_t output_depth = inputDepth;
    size_t output_height = mInputHeight + layerParameters.mTopPadding + layerParameters.mBottomPadding;
    size_t output_width = mInputWidth + layerParameters.mLeftPadding + layerParameters.mRightPadding;

    mNetworkParams.mWorkflow.tensorNeeded(name, mOutputs[0], raul::WShape{ raul::BS(), output_depth, output_height, output_width }, DEC_FORW_WRIT);
    mNetworkParams.mWorkflow.copyDeclaration(name, mOutputs[0], mOutputs[0].grad(), DEC_BACK_READ);
}

void PaddingLayer::forwardComputeImpl(NetworkMode /*mode*/)
{
    if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU)
    {
        const Tensor& input = mNetworkParams.mMemoryManager[mInputs[0]];
        Tensor& output = mNetworkParams.mMemoryManager[mOutputs[0]];

        auto input_4d_view = input.get4DView();
        auto output_4d_view = output.get4DView();
        for (size_t bIdx = 0, batch_size = output.getBatchSize(); bIdx < batch_size; ++bIdx)
        {
            for (size_t dIdx = 0, depth_size = output.getDepth(); dIdx < depth_size; ++dIdx)
            {
                for (size_t hIdx = 0, height_size = output.getHeight(); hIdx < height_size; ++hIdx)
                {
                    for (size_t wIdx = 0, width_size = output.getWidth(); wIdx < width_size; ++wIdx)
                    {
                        CalculationStrategy::Coordinates outputElementPosition{ bIdx, dIdx, hIdx, wIdx };
                        if (calculationStrategy->isNeedToGetFillingValueFromInput(outputElementPosition))
                        {
                            auto pos = calculationStrategy->getFillingValuePositionInInput(outputElementPosition);
                            output_4d_view[bIdx][dIdx][hIdx][wIdx] = input_4d_view[pos.BatchIdx][pos.DepthIdx][pos.HeightIdx][pos.WidthIdx];
                        }
                        else
                        {
                            output_4d_view[bIdx][dIdx][hIdx][wIdx] = mFillingValue;
                        }
                    }
                }
            }
        }
    }
    else
    {
        THROW(mTypeName, mName, "unsupported execution target");
    }
}

void PaddingLayer::backwardComputeImpl()
{
    if (mNetworkParams.mWorkflow.getExecutionTarget() == ExecutionTarget::CPU)
    {
        const Tensor& delta = mNetworkParams.mMemoryManager[mOutputs[0].grad()];
        auto& in_nabla = mNetworkParams.mMemoryManager[mInputs[0].grad()];
        auto in_nabla_4d_view = in_nabla.get4DView();
        auto delta_4d_view = delta.get4DView();
        for (size_t bIdx = 0, batch_size = delta.getBatchSize(); bIdx < batch_size; ++bIdx)
        {
            for (size_t dIdx = 0, depth_size = delta.getDepth(); dIdx < depth_size; ++dIdx)
            {
                for (size_t hIdx = 0, height_size = delta.getHeight(); hIdx < height_size; ++hIdx)
                {
                    for (size_t wIdx = 0, width_size = delta.getWidth(); wIdx < width_size; ++wIdx)
                    {
                        CalculationStrategy::Coordinates deltaElementPosition{ bIdx, dIdx, hIdx, wIdx };
                        if (calculationStrategy->isElementAffectToDerivative(deltaElementPosition))
                        {
                            auto pos = calculationStrategy->getPositionInDerivativeForUpdate(deltaElementPosition);
                            in_nabla_4d_view[pos.BatchIdx][pos.DepthIdx][pos.HeightIdx][pos.WidthIdx] += delta_4d_view[bIdx][dIdx][hIdx][wIdx];
                        }
                    }
                }
            }
        }
    }
    else
    {
        THROW(mTypeName, mName, "unsupported execution target");
    }
}

class ConstantPaddingCalculation : public PaddingLayer::CalculationStrategy
{
  public:
    ConstantPaddingCalculation(const PaddingLayerParams& layerParameters, const NetworkParameters& networkParameters)
        : CalculationStrategy(layerParameters, networkParameters)
    {
    }

    bool isNeedToGetFillingValueFromInput(const Coordinates& outputElementPosition) const final
    {
        if (isElementAddedByPadding(outputElementPosition))
        {
            return false;
        }

        return true;
    }
    Coordinates getFillingValuePositionInInputAccordingFillingMode(const Coordinates& /* outputElementPosition */) const final
    {
        throw std::logic_error("PaddingLayer[PaddingLayer] Constant padding must fill padded elements by constant");
    }

    bool isElementAffectToDerivative(const Coordinates& deltaElementPosition) const final
    {
        if (isElementAddedByPadding(deltaElementPosition))
        {
            return false;
        }

        return true;
    }

    Coordinates getPositionInDerivativeForUpdateAccordingFillingMode(const Coordinates& /* deltaElementPosition */) const final
    {
        throw std::logic_error("PaddingLayer[PaddingLayer] padded values in delta must not affect to derivative in constant padding");
    }
 ~ConstantPaddingCalculation(){}
};

class ReflectionPaddingCalculation : public PaddingLayer::CalculationStrategy
{
  public:
    ReflectionPaddingCalculation(const PaddingLayerParams& layerParameters, const NetworkParameters& networkParameters)
        : CalculationStrategy(layerParameters, networkParameters)
    {
        if (layerParameters.mTopPadding >= mInputHeight || layerParameters.mBottomPadding >= mInputHeight || layerParameters.mLeftPadding >= mInputWidth ||
            layerParameters.mRightPadding >= mInputWidth)
        {
            THROW_NONAME("PaddingLayer",
                         "inaccessible padding size in reflection mode. "
                         "Max value for top padding - " +
                             std::to_string(mInputHeight - 1) +
                             ", "
                             "Max value for bottom padding - " +
                             std::to_string(mInputHeight - 1) +
                             ", "
                             "Max value for left padding - " +
                             std::to_string(mInputWidth - 1) +
                             ", "
                             "Max value for right padding - " +
                             std::to_string(mInputWidth - 1));
        }
    }

    bool isNeedToGetFillingValueFromInput(const Coordinates& /* outputElementPosition */) const final { return true; }

    Coordinates getFillingValuePositionInInputAccordingFillingMode(const Coordinates& outputElementPosition) const final
    {
        size_t heightIdxInInput = calculatePositionUsing(mInputHeight, mTopPadding, mInputHeight - 1, outputElementPosition.HeightIdx);
        size_t widthIdxInInput = calculatePositionUsing(mInputWidth, mLeftPadding, mInputWidth - 1, outputElementPosition.WidthIdx);
        return { outputElementPosition.BatchIdx, outputElementPosition.DepthIdx, heightIdxInInput, widthIdxInInput };
    }

    bool isElementAffectToDerivative(const Coordinates& /* deltaElementPosition */) const final { return true; }

    Coordinates getPositionInDerivativeForUpdateAccordingFillingMode(const Coordinates& deltaElementPosition) const final
    {
        size_t heightIdxInDerivative = calculatePositionUsing(mInputHeight, mTopPadding, mInputHeight - 1, deltaElementPosition.HeightIdx);
        size_t widthIdxInDerivative = calculatePositionUsing(mInputWidth, mLeftPadding, mInputWidth - 1, deltaElementPosition.WidthIdx);
        return { deltaElementPosition.BatchIdx, deltaElementPosition.DepthIdx, heightIdxInDerivative, widthIdxInDerivative };
    }

  private:
    static size_t calculatePositionUsing(size_t inputDimension, size_t paddingBeforeInput, size_t maxPaddingAfterInput, size_t outputPosition)
    {
        if (outputPosition < paddingBeforeInput)
        {
            return paddingBeforeInput - outputPosition;
        }
        else if (outputPosition - paddingBeforeInput < inputDimension)
        {
            return outputPosition - paddingBeforeInput;
        }
        else
        {
            return inputDimension - 1 - (outputPosition - paddingBeforeInput - maxPaddingAfterInput);
        }
    }
};

class ReplicationPaddingCalculation : public PaddingLayer::CalculationStrategy
{
  public:
    ReplicationPaddingCalculation(const PaddingLayerParams& layerParameters, const NetworkParameters& networkParameters)
        : CalculationStrategy(layerParameters, networkParameters)
    {
    }
    ~ReplicationPaddingCalculation(){}
    bool isNeedToGetFillingValueFromInput(const Coordinates& /* outputElementPosition */) const final { return true; }

    Coordinates getFillingValuePositionInInputAccordingFillingMode(const Coordinates& outputElementPosition) const final
    {
        size_t heightIdxInInput = calculatePositionUsing(mInputHeight, mTopPadding, outputElementPosition.HeightIdx);
        size_t widthIdxInInput = calculatePositionUsing(mInputWidth, mLeftPadding, outputElementPosition.WidthIdx);
        return { outputElementPosition.BatchIdx, outputElementPosition.DepthIdx, heightIdxInInput, widthIdxInInput };
    }

    bool isElementAffectToDerivative(const Coordinates& /* deltaElementPosition */) const final { return true; }

    Coordinates getPositionInDerivativeForUpdateAccordingFillingMode(const Coordinates& deltaElementPosition) const final
    {
        size_t heightIdxInDerivative = calculatePositionUsing(mInputHeight, mTopPadding, deltaElementPosition.HeightIdx);
        size_t widthIdxInDerivative = calculatePositionUsing(mInputWidth, mLeftPadding, deltaElementPosition.WidthIdx);
        return { deltaElementPosition.BatchIdx, deltaElementPosition.DepthIdx, heightIdxInDerivative, widthIdxInDerivative };
    }
 
  private:
    static size_t calculatePositionUsing(size_t inputDimension, size_t paddingBeforeInput, size_t outputPosition)
    {
        if (outputPosition < paddingBeforeInput)
        {
            return 0;
        }
        else if (outputPosition - paddingBeforeInput < inputDimension)
        {
            return outputPosition - paddingBeforeInput;
        }
        else
        {
            return inputDimension - 1;
        }
    }
};

PaddingLayer::CalculationStrategy::CalculationStrategy(const PaddingLayerParams& layerParameters, const NetworkParameters& networkParameters)
    : mTopPadding(layerParameters.mTopPadding)
    , mLeftPadding(layerParameters.mLeftPadding)
{
    (void)networkParameters;
    if (layerParameters.getInputs().size() != 1)
    {
        THROW_NONAME("PaddingLayer", "wrong number of input names");
    }
    if (layerParameters.getInputs()[0].empty())
    {
        THROW_NONAME("PaddingLayer", "empty input name");
    }

    mInputHeight = networkParameters.mWorkflow.getHeight(layerParameters.getInputs()[0]);
    mInputWidth = networkParameters.mWorkflow.getWidth(layerParameters.getInputs()[0]);
}

PaddingLayer::CalculationStrategy::Coordinates PaddingLayer::CalculationStrategy::getFillingValuePositionInInput(const Coordinates& outputElementPosition) const
{
    if (isElementAddedByPadding(outputElementPosition))
    {
        return getFillingValuePositionInInputAccordingFillingMode(outputElementPosition);
    }
    else
    {
        return getPositionOfInputElementMappedToOutputElementWith(outputElementPosition);
    }
}

PaddingLayer::CalculationStrategy::Coordinates PaddingLayer::CalculationStrategy::getPositionInDerivativeForUpdate(const Coordinates& deltaElementPosition) const
{
    if (isElementAddedByPadding(deltaElementPosition))
    {
        return getPositionInDerivativeForUpdateAccordingFillingMode(deltaElementPosition);
    }
    else
    {
        return getPositionOfInputElementMappedToOutputElementWith(deltaElementPosition);
    }
}

PaddingLayer::CalculationStrategy::Coordinates PaddingLayer::CalculationStrategy::getPositionOfInputElementMappedToOutputElementWith(const Coordinates& elementPosition) const
{
    return { elementPosition.BatchIdx, elementPosition.DepthIdx, elementPosition.HeightIdx - mTopPadding, elementPosition.WidthIdx - mLeftPadding };
}

bool PaddingLayer::CalculationStrategy::isElementAddedByPadding(const Coordinates& elementPosition) const
{
    return (elementPosition.HeightIdx < mTopPadding || elementPosition.HeightIdx >= mTopPadding + mInputHeight) ||
           (elementPosition.WidthIdx < mLeftPadding || elementPosition.WidthIdx >= mLeftPadding + mInputWidth);
}

std::unique_ptr<PaddingLayer::CalculationStrategy> PaddingLayer::CalculationStrategy::define(const PaddingLayerParams& layerParameters, const NetworkParameters& networkParameters)
{
    switch (layerParameters.mFillingMode)
    {
        case PaddingLayerParams::USE_FILLING_VALUE:
        {
            return std::make_unique<ConstantPaddingCalculation>(layerParameters, networkParameters);
        }
        case PaddingLayerParams::REFLECTION:
        {
            return std::make_unique<ReflectionPaddingCalculation>(layerParameters, networkParameters);
        }
        case PaddingLayerParams::REPLICATION:
        {
            return std::make_unique<ReplicationPaddingCalculation>(layerParameters, networkParameters);
        }
        default:
        {
            THROW_NONAME("PaddingLayer", "unknown type of padding");
        }
    }
}

} // ! namespace raul
