// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "BasicParameters.h"

#include <map>

namespace raul
{

void LabelSmoothingParams::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    stream << "smoothing: " << smoothing << ", padding_class_idx: ";
    if (paddingClass >= 0)
        stream << paddingClass;
    else
        stream << "none";
}

void ViewParams::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    stream << "shape: [" << /*batch << ", " << */ depth << ", " << height << ", " << width << "]";
}

TransposingParams::TransposingParams(const raul::Name& input, const raul::Name& output, const std::string& paramDim1, const std::string& paramDim2)
    : BasicParams(Names(1, input), Names(1, output))
{
    std::map<std::string, Dimension> dmap{ { "width", Dimension::Width }, { "height", Dimension::Height }, { "depth", Dimension::Depth } /*, { "batch", Dimension::Batch } */ };
    if (dmap.find(paramDim1) != dmap.end())
        dim1 = dmap[paramDim1];
    else
        THROW_NONAME("TransposingParams", "Unknown dimension: " + paramDim1);
    if (dmap.find(paramDim2) != dmap.end())
        dim2 = dmap[paramDim2];
    else
        THROW_NONAME("TransposingParams", "Unknown dimension: " + paramDim2);
}

TransposingParams::TransposingParams(const raul::Name& input, const raul::Name& output, Dimension paramDim1, Dimension paramDim2)
    : BasicParams(Names(1, input), Names(1, output))
    , dim1(paramDim1)
    , dim2(paramDim2)
{
}

void TransposingParams::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    std::map<Dimension, std::string> dmap{
        { Dimension::Default, "default" }, { Dimension::Width, "width" }, { Dimension::Height, "height" }, { Dimension::Depth, "depth" } /*, { Dimension::Batch, "batch" }*/
    };
    stream << "dim1: " << dmap[dim1] << ", dim2: " << dmap[dim2];
}

MatMulParams::MatMulParams(const Names& inputs, const raul::Name& output, float scaleValue)
    : BasicParams(inputs, Names(1, output))
    , scale(scaleValue)
{
}

void MatMulParams::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    stream << "scale: " << scale;
}

PositionalEncodingParams::PositionalEncodingParams(const raul::Name& input, const raul::Name& output, size_t modelSize, size_t maxLength, bool useDurations, size_t maxMelLength)
    : BasicParams(Names(1, input), Names(1, output))
    , modelSize(modelSize)
    , maxLength(maxLength)
    , durationEncoding(useDurations)
    , maxMelLength(maxMelLength)
{
}

void PositionalEncodingParams::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    stream << "model_size: " << modelSize << ", max_length: " << maxLength << ", durations: " << (durationEncoding ? "true" : "false");
    if (durationEncoding)
    {
        stream << ", max_mel_length: " << maxMelLength;
    }
}

void FillParams::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    stream << "fill_value: " << fillValue;
}

void MaskedFillParams::print(std::ostream& stream) const
{
    FillParams::print(stream);
    stream << ", inverted: " << (inverted ? "yes" : "no");
}

Pool2DParams::Pool2DParams(const Names& inputs, const Names& outputs, size_t kernelW, size_t kernelH, size_t strideWidth, size_t strideHeight, size_t paddingWidth, size_t paddingHeight)
    : BasicParams(inputs, outputs)
    , kernelWidth(kernelW)
    , kernelHeight(kernelH)
    , strideW(strideWidth)
    , strideH(strideHeight)
    , paddingW(paddingWidth)
    , paddingH(paddingHeight)
{
}

Pool2DParams::Pool2DParams(const Names& inputs, const Names& outputs, size_t kernelSize, size_t stride, size_t padding)
    : BasicParams(inputs, outputs)
    , kernelWidth(kernelSize)
    , kernelHeight(kernelSize)
    , strideW(stride)
    , strideH(stride)
    , paddingW(padding)
    , paddingH(padding)
{
}

void Pool2DParams::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    stream << "kernel: [" << kernelWidth << " x " << kernelHeight;
    stream << "], stride: [" << strideW << ", " << strideH;
    stream << "], padding: [" << paddingW << ", " << paddingH << "]";
}

LossParams::LossParams(const Names& inputs, const Names& outputs, const std::string& reduction_type, bool isFinal)
    : BasicParams(inputs, outputs)
{
    std::string r = reduction_type;
    if (r == "none")
        reduction = LossParams::Reduction::None;
    else if (r == "mean")
        reduction = LossParams::Reduction::Mean;
    else if (r == "batch_mean")
        reduction = LossParams::Reduction::Batch_Mean;
    else if (r == "sum")
        reduction = LossParams::Reduction::Sum;
    else if (r == "sum_over_weights")
        reduction = LossParams::Reduction::Sum_Over_Weights;
    else if (r == "sum_over_nonzero_weights")
        reduction = LossParams::Reduction::Sum_Over_Nonzero_Weights;
    else if (r == "custom_mean")
        reduction = LossParams::Reduction::Custom_Mean;
    else if (r == "custom_batch_mean")
        reduction = LossParams::Reduction::Custom_Batch_Mean;
    else
        THROW_NONAME("TransposingParams", "Unknown reduction type: " + r);

    mIsFinal = isFinal;
}

LossParams::LossParams(const Names& inputs, const Names& outputs, LossParams::Reduction reduction_type, bool isFinal)
    : BasicParams(inputs, outputs)
    , reduction(reduction_type)
    , mIsFinal(isFinal)
{
}

void LossParams::print(std::ostream& stream) const
{
    using namespace std::string_literals;
    BasicParams::print(stream);
    std::string ss = std::map<LossParams::Reduction, std::string>{ { LossParams::Reduction::None, "none" }, { LossParams::Reduction::Sum, "sum" }, { LossParams::Reduction::Mean, "mean" } }[reduction];
    stream << "reduction: " << ss << ", final: " << (mIsFinal ? "true"s : "false"s);
}

BasicParamsWithDim::BasicParamsWithDim(const Names& inputs, const Names& outputs, const std::string& paramDim)
    : BasicParams(inputs, outputs)
{
    std::string r = paramDim;
    if (r == "width")
        dim = Dimension::Width;
    else if (r == "height")
        dim = Dimension::Height;
    else if (r == "depth")
        dim = Dimension::Depth;
    else if (r == "batch")
        dim = Dimension::Batch;
    else if (r == "default")
        dim = Dimension::Default;
    else
        THROW_NONAME("TransposingParams", "Unknown dimension: " + r);
}

BasicParamsWithDim::BasicParamsWithDim(const Names& inputs, const Names& outputs, Dimension paramDim)
    : BasicParams(inputs, outputs)
    , dim(paramDim)
{
}

void BasicParamsWithDim::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    std::string s = std::map<Dimension, std::string>{
        { Dimension::Default, "default" }, { Dimension::Width, "width" }, { Dimension::Height, "height" }, { Dimension::Depth, "depth" }, { Dimension::Batch, "batch" }
    }[dim];
    stream << "dim: " << s;
}

void HSwishActivationParams::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    const auto to_str = [](const auto& x) {
        std::string s = std::map<Limit, std::string>{ { Limit::Left, "Left" }, { Limit::Middle, "Middle" }, { Limit::Right, "Right" } }[x];
        return s;
    };
    stream << "limit at -3: " << to_str(m3PointVal) << ", limit at 3: " << to_str(p3PointVal);
}

void HSigmoidActivationParams::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    const auto to_str = [](const auto& x) {
        std::string s = std::map<Limit, std::string>{ { Limit::Left, "Left" }, { Limit::Middle, "Middle" }, { Limit::Right, "Right" } }[x];
        return s;
    };
    stream << "limit at -3: " << to_str(m3PointVal) << ", limit at 3: " << to_str(p3PointVal);
}

void ElementWiseLayerParams::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    stream << (mBroadcast ? "broadcast enabled" : "broadcast disabled");
}

void PaddingLayerParams::print(std::ostream& stream) const
{
    BasicParams::print(stream);
    stream << "padding top:   " << mTopPadding << "padding back:  " << mBottomPadding << "padding left:  " << mLeftPadding << "padding right: " << mRightPadding;
    stream << "filling mode: ";
    if (mFillingMode == USE_FILLING_VALUE)
    {
        stream << "use given filling value";
        stream << "filling value: " << mFillingValue;
    }
    else
    {
        stream << "reflection";
    }
}

} // namespace raul
