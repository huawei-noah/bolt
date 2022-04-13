// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef PADDING_LAYER_H
#define PADDING_LAYER_H

#include "training/base/layers/BasicLayer.h"

namespace raul
{

/**
 * @brief Padding Layer
 *
 * The layer adds paddings to input tensors in a given manner.
 */
class PaddingLayer : public BasicLayer
{
  public:
    class CalculationStrategy
    {
      public:
        struct Coordinates
        {
            const size_t BatchIdx;
            const size_t DepthIdx;
            const size_t HeightIdx;
            const size_t WidthIdx;

            Coordinates(size_t batchIdx, size_t depthIdx, size_t heightIdx, size_t widthIdx)
                : BatchIdx(batchIdx)
                , DepthIdx(depthIdx)
                , HeightIdx(heightIdx)
                , WidthIdx(widthIdx)
            {
            }
        };

      public:
        static std::unique_ptr<CalculationStrategy> define(const PaddingLayerParams& layerParameters, const NetworkParameters& networkParameters);

        virtual bool isNeedToGetFillingValueFromInput(const Coordinates& outputElementPosition) const = 0;
        Coordinates getFillingValuePositionInInput(const Coordinates& outputElementPosition) const;
        virtual bool isElementAffectToDerivative(const Coordinates& deltaElementPosition) const = 0;
        Coordinates getPositionInDerivativeForUpdate(const Coordinates& deltaElementPosition) const;

        virtual ~CalculationStrategy() = default;

      protected:
        CalculationStrategy(const PaddingLayerParams& layerParameters, const NetworkParameters& networkParameters);

        bool isElementAddedByPadding(const Coordinates& elementPosition) const;
        Coordinates getPositionOfInputElementMappedToOutputElementWith(const Coordinates& elementPosition) const;

        virtual Coordinates getFillingValuePositionInInputAccordingFillingMode(const Coordinates& outputElementPosition) const = 0;
        virtual Coordinates getPositionInDerivativeForUpdateAccordingFillingMode(const Coordinates& deltaElementPosition) const = 0;

      protected:
        size_t mInputHeight = 0;
        size_t mInputWidth = 0;
        uint32_t mTopPadding;
        uint32_t mLeftPadding;
    };

  public:
    PaddingLayer(const Name& name, const PaddingLayerParams& layerParameters, NetworkParameters& networkParameters);

    PaddingLayer(PaddingLayer&&) = default;
    PaddingLayer(const PaddingLayer&) = delete;
    PaddingLayer& operator=(const PaddingLayer&) = delete;

    void forwardComputeImpl(NetworkMode /*mode*/) final;
    void backwardComputeImpl() final;

  private:
    std::unique_ptr<CalculationStrategy> calculationStrategy;
    dtype mFillingValue;
};

} // ! namespace raul

#endif // ! PADDING_LAYER_H