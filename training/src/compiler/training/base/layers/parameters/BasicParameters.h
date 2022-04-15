// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef BASIC_LAYER_PARAMETERS_H
#define BASIC_LAYER_PARAMETERS_H

#include <string>
#include <utility>
#include <vector>

#include <training/base/common/MemoryManager.h>
#include <training/system/Types.h>

namespace raul
{

/**
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param weights vector of names of weight/bias tensors (for weights sharing)
 */

struct BasicParams
{
    BasicParams() = delete;

    BasicParams(Names i, Names o, bool compressOut = true)
        : inputs(std::move(i))
        , outputs(std::move(o))
        , compressOutput(compressOut)
    {
    }

    BasicParams(Names i, Names o, Names w, bool compressOut = true)
        : inputs(std::move(i))
        , outputs(std::move(o))
        , sharedWeights(std::move(w))
        , compressOutput(compressOut)
    {
    }

    BasicParams(Names i, Names o, Name s, bool compressOut = true)
        : inputs(std::move(i))
        , outputs(std::move(o))
        , sharedLayer(std::move(s))
        , compressOutput(compressOut)
    {
    }

    BasicParams(const BasicParams&) = default;

    virtual ~BasicParams() = default;

    virtual void print(std::ostream&) const {}

    [[nodiscard]] virtual Names& getInputs() { return inputs; }

    [[nodiscard]] virtual Names& getOutputs() { return outputs; }

    [[nodiscard]] virtual Names& getSharedWeights() { return sharedWeights; }

    [[nodiscard]] virtual const Names& getInputs() const { return inputs; }

    [[nodiscard]] virtual const Names& getOutputs() const { return outputs; }

    [[nodiscard]] virtual const Names& getSharedWeights() const { return sharedWeights; }

    [[nodiscard]] virtual const Name& getSharedLayer() const { return sharedLayer; }
    [[nodiscard]] virtual Name& getSharedLayer() { return sharedLayer; }

    [[nodiscard]] virtual bool isCompressOutput() const { return compressOutput; }

  private:
    friend class Workflow; // to override layerExecutionTarget

    Names inputs;
    Names outputs;
    Names sharedWeights;
    Name sharedLayer;
    bool compressOutput; // switch output tensor compression, globally overridden buy compression mode in NetworkParams
};

/**
 * @param input name of input tensor
 * @param output name of output tensor
 * @param paramBatch new batch size
 * @param paramDepth new depth
 * @param paramHeight new height
 * @param paramWidth new width
 */

struct ViewParams : public BasicParams
{
    ViewParams() = delete;
    ViewParams(const Name& input, const Name& output, int paramDepth = -1, int paramHeight = -1, int paramWidth = -1)
        : BasicParams(Names(1, input), Names(1, output))
        , depth(paramDepth)
        , height(paramHeight)
        , width(paramWidth)
    {
    }

    ViewParams(const Name& input, const Name& output, const Name& weight, int paramDepth = -1, int paramHeight = -1, int paramWidth = -1)
        : BasicParams(Names(1, input), Names(1, output), Names(1, weight))
        , depth(paramDepth)
        , height(paramHeight)
        , width(paramWidth)
    {
    }

    int depth;
    int height;
    int width;

    void print(std::ostream&) const override;
};

/**
 * @param input name of input tensor
 * @param output name of output tensor
 * @param smoothing amount of smoothing in [0, 1]
 */
struct LabelSmoothingParams : public BasicParams
{
    LabelSmoothingParams() = delete;
    LabelSmoothingParams(const Name& input, const Name& output, float smoothingAmount, int paddingClassIdx = -1)
        : BasicParams(Names(1, input), Names(1, output))
        , smoothing(smoothingAmount)
        , paddingClass(paddingClassIdx)
    {
    }

    float smoothing = 0.f;
    int paddingClass = -1;

    void print(std::ostream& stream) const override;
};

/**
 * @param input name of input tensor
 * @param output name of output tensor
 * @param paramDim1 first dimension to swap
 * @param paramDim2 second dimension to swap
 */
struct TransposingParams : public BasicParams
{
    TransposingParams() = delete;
    TransposingParams(const Name& input, const Name& output, const std::string& paramDim1, const std::string& paramDim2);
    TransposingParams(const Name& input, const Name& output, Dimension paramDim1 = Dimension::Default, Dimension paramDim2 = Dimension::Default);

    Dimension dim1;
    Dimension dim2;

    void print(std::ostream& stream) const override;
};

/**
 * @param inputs names of input tensors
 * @param output name of output tensor
 * @param scale multiplier for result tensor
 */
struct MatMulParams : public BasicParams
{
    MatMulParams() = delete;
    MatMulParams(const Names& inputs, const Name& output, float scaleValue = 1.f);

    float scale = 1.f;

    void print(std::ostream& stream) const override;
};

/**
 * @param input name of input tensor
 * @param output name of output tensor
 * @param modelSize the size of each input and output vector, must be even
 * @param maxLength maximum number of vectors in batch
 * @param durationEncoding inputs are durations instead of embeddings
 * @param maxMelLengths maximum mel length (used when durationEncoding==true)
 */
struct PositionalEncodingParams : public BasicParams
{
    PositionalEncodingParams() = delete;
    PositionalEncodingParams(const Name& input, const Name& output, size_t modelSize, size_t maxLength = 5000, bool useDurations = false, size_t maxMelLength = 200);

    size_t modelSize = 2;
    size_t maxLength = 5000;
    bool durationEncoding = false;
    size_t maxMelLength = 200;

    void print(std::ostream& stream) const override;
};

/**
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param fillValue value to fill with
 */
struct FillParams : public BasicParams
{
    FillParams() = delete;
    FillParams(const Names& inputs, const Names& outputs, float paramfillValue)
        : BasicParams(inputs, outputs)
        , fillValue(paramfillValue)
    {
    }

    float fillValue;

    void print(std::ostream& stream) const override;
};

/**
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param fillValue value to fill with
 */
struct MaskedFillParams : public FillParams
{
    MaskedFillParams() = delete;
    MaskedFillParams(const Names& inputs, const Name& output, float paramfillValue, bool paramInverted = false)
        : FillParams(inputs, Names(1, output), paramfillValue)
        , inverted(paramInverted)
    {
    }

    bool inverted;

    void print(std::ostream& stream) const override;
};

struct BasicParamsWithDim : public BasicParams
{
    BasicParamsWithDim() = delete;

    BasicParamsWithDim(const Names& inputs, const Names& outputs, const std::string& paramDim = "default");

    BasicParamsWithDim(const Names& inputs, const Names& outputs, Dimension paramDim);

    Dimension dim = Dimension::Default;

    void print(std::ostream& stream) const override;
};

/**
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param kernelW/H the size of the window to take a max over
 * @param strideW/H the stride of the window
 * @param paddingW/H implicit zero padding to be added on both sides
 */
struct Pool2DParams : public BasicParams
{
    Pool2DParams() = delete;
    Pool2DParams(const Names& inputs, const Names& outputs, size_t kernelW, size_t kernelH, size_t strideWidth, size_t strideHeight, size_t paddingWidth, size_t paddingHeight);
    Pool2DParams(const Names& inputs, const Names& outputs, size_t kernelSize, size_t stride, size_t padding = 0);

    size_t kernelWidth = 0, kernelHeight = 0;
    size_t strideW = 0, strideH = 0;
    size_t paddingW = 0, paddingH = 0;

    void print(std::ostream& stream) const override;
};

struct LossParams : public BasicParams
{
    enum class Reduction : int
    {
        None = 0,
        Sum = 1,
        Mean = 2,                     // divide sum of losses by number of elements in input
        Batch_Mean = 3,               // divide sum of losses by batch size
        Sum_Over_Weights = 4,         // divide sum of losses by sum of weights
        Sum_Over_Nonzero_Weights = 5, // divide sum of losses by number of non-zero weights
        Custom_Mean = 6,              // divide by value from parameters of the network multiplied by total input size
        Custom_Batch_Mean = 7         // divide by value from parameters of the network
    };

    LossParams() = delete;

    LossParams(const Names& inputs, const Names& outputs, const std::string& reduction_type = "mean", bool isFinal = true);

    LossParams(const Names& inputs, const Names& outputs, Reduction reduction_type, bool isFinal = true);

    Reduction reduction = Reduction::Mean; // to conform to torch default behaviour
    bool mIsFinal;

    void print(std::ostream& stream) const override;
};

/** Parameters for hard swish activaction function
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param m3PointVal limit at -3 point (left, right, middle), default: left
 * @param p3PointVal limit at 3 point (left, right, middle), default: right
 */
struct HSwishActivationParams : public BasicParams
{
    HSwishActivationParams() = delete;

    HSwishActivationParams(const Names& inputs, const Names& outputs)
        : HSwishActivationParams(inputs, outputs, Limit::Left, Limit::Right)
    {
    }

    HSwishActivationParams(const Names& inputs, const Names& outputs, const Limit m3PointVal, const Limit p3PointVal)
        : BasicParams(inputs, outputs)
        , m3PointVal(m3PointVal)
        , p3PointVal(p3PointVal)
    {
    }

    void print(std::ostream& stream) const override;

    Limit m3PointVal = Limit::Left;
    Limit p3PointVal = Limit::Right;
};

/** Parameters for hard sigmoid activaction function
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param m3PointVal limit at -3 point (left, right, middle), default: left
 * @param p3PointVal limit at 3 point (left, right, middle), default: right
 */
struct HSigmoidActivationParams : public BasicParams
{
    HSigmoidActivationParams() = delete;

    HSigmoidActivationParams(const Names& inputs, const Names& outputs)
        : HSigmoidActivationParams(inputs, outputs, Limit::Left, Limit::Right)
    {
    }

    HSigmoidActivationParams(const Names& inputs, const Names& outputs, const Limit m3PointVal, const Limit p3PointVal)
        : BasicParams(inputs, outputs)
        , m3PointVal(m3PointVal)
        , p3PointVal(p3PointVal)
    {
    }

    void print(std::ostream& stream) const override;

    Limit m3PointVal = Limit::Left;
    Limit p3PointVal = Limit::Right;
};

/** Parameters for element-wize layers
 * @param inputs vector of names of input tensors
 * @param output name of output tensor
 * @param broadcast if true the layer does not throw exception when sizes do not match and tries to broadcast tensors
 *
 * @note Broadcast can fail too
 * @see Tensor broadcast implementation
 */
struct ElementWiseLayerParams : public BasicParams
{
    ElementWiseLayerParams() = delete;

    ElementWiseLayerParams(const Names& inputs, const Name& output, const bool broadcast = true)
        : BasicParams(inputs, Names(1, output))
        , mBroadcast(broadcast)
    {
    }

    ElementWiseLayerParams(const BasicParams& params, const bool broadcast = true)
        : BasicParams(params)
        , mBroadcast(broadcast)
    {
    }

    void print(std::ostream& stream) const override;

    bool mBroadcast;
};

struct PaddingLayerParams : public BasicParams
{
    enum FillingMode
    {
        USE_FILLING_VALUE,
        REFLECTION,
        REPLICATION
    };

    uint32_t mTopPadding;
    uint32_t mBottomPadding;
    uint32_t mLeftPadding;
    uint32_t mRightPadding;
    dtype mFillingValue;
    FillingMode mFillingMode;

    PaddingLayerParams(const Names& inputs, const Names& outputs, uint32_t padding, dtype fillingValue)
        : BasicParams(inputs, outputs)
        , mTopPadding(padding)
        , mBottomPadding(padding)
        , mLeftPadding(padding)
        , mRightPadding(padding)
        , mFillingValue(fillingValue)
        , mFillingMode(USE_FILLING_VALUE)
    {
    }

    PaddingLayerParams(const Names& inputs, const Names& outputs, uint32_t padding, FillingMode filling_mode)
        : BasicParams(inputs, outputs)
        , mTopPadding(padding)
        , mBottomPadding(padding)
        , mLeftPadding(padding)
        , mRightPadding(padding)
        , mFillingValue(0._dt)
        , mFillingMode(filling_mode)
    {
    }

    PaddingLayerParams(const Names& inputs, const Names& outputs, uint32_t topPadding, uint32_t bottomPadding, uint32_t leftPadding, uint32_t rightPadding, dtype fillingValue)
        : BasicParams(inputs, outputs)
        , mTopPadding(topPadding)
        , mBottomPadding(bottomPadding)
        , mLeftPadding(leftPadding)
        , mRightPadding(rightPadding)
        , mFillingValue(fillingValue)
        , mFillingMode(USE_FILLING_VALUE)
    {
    }

    PaddingLayerParams(const Names& inputs, const Names& outputs, uint32_t topPadding, uint32_t bottomPadding, uint32_t leftPadding, uint32_t rightPadding, FillingMode fillingMode)
        : BasicParams(inputs, outputs)
        , mTopPadding(topPadding)
        , mBottomPadding(bottomPadding)
        , mLeftPadding(leftPadding)
        , mRightPadding(rightPadding)
        , mFillingValue(0._dt)
        , mFillingMode(fillingMode)
    {
    }

    PaddingLayerParams(const Names& inputs, const Names& outputs, uint32_t topPadding, uint32_t bottomPadding, uint32_t leftPadding, uint32_t rightPadding, dtype fillingValue, FillingMode fillingMode)
        : BasicParams(inputs, outputs)
        , mTopPadding(topPadding)
        , mBottomPadding(bottomPadding)
        , mLeftPadding(leftPadding)
        , mRightPadding(rightPadding)
        , mFillingValue(fillingValue)
        , mFillingMode(fillingMode)
    {
    }

    void print(std::ostream& stream) const final;
};

/// @warning (ck): please add a new parameter class into separate files!

} // raul namespace

#endif
