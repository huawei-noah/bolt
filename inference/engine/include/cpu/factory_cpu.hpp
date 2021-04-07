// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _FACTORY_CPU_H
#define _FACTORY_CPU_H

#include "factory.hpp"
#include "attention.hpp"
#include "jump.hpp"
#include "cpu/resize_cpu.hpp"
#include "cpu/pooling_cpu.hpp"
#include "cpu/convolution_cpu.hpp"
#include "cpu/deconvolution_cpu.hpp"
#include "cpu/eltwise_cpu.hpp"
#include "cpu/softmax_cpu.hpp"
#include "cpu/activation_cpu.hpp"
#include "cpu/fully_connected_cpu.hpp"
#include "cpu/scale_cpu.hpp"
#include "cpu/concat_cpu.hpp"
#include "cpu/clip_cpu.hpp"
#include "cpu/squeeze_cpu.hpp"
#include "cpu/reshape_cpu.hpp"
#include "cpu/embedding_cpu.hpp"
#include "cpu/layer_norm_cpu.hpp"
#include "cpu/matmul_cpu.hpp"
#include "cpu/power_cpu.hpp"
#include "cpu/transpose_cpu.hpp"
#include "cpu/slice_cpu.hpp"
#include "cpu/shared_weight_cpu.hpp"
#include "cpu/repeat_cpu.hpp"
#include "cpu/copy_cpu.hpp"
#include "cpu/check_cpu.hpp"
#include "cpu/preallocated_memory_cpu.hpp"
#include "cpu/argmax_cpu.hpp"
#include "cpu/unsqueeze_cpu.hpp"
#include "cpu/rnncell_cpu.hpp"
#include "cpu/rnn_cpu.hpp"
#include "cpu/padding_cpu.hpp"
#include "attention_mask.hpp"
#include "relative_position_embedding.hpp"
#include "relative_shift.hpp"
#include "detection_output.hpp"
#include "prior_box.hpp"
#include "yolov3_detection_output.hpp"
#include "cpu/channel_resize_cpu.hpp"
#include "cpu/l2normalization_cpu.hpp"
#include "cpu/tile_cpu.hpp"
#include "cpu/prelu_cpu.hpp"
#include "cpu/tfslice_cpu.hpp"
#include "cpu/splice_cpu.hpp"
#include "cpu/shape_cpu.hpp"
#include "cpu/reduction_cpu.hpp"
#include "cpu/where_cpu.hpp"
#include "cpu/tdnn_convolution_cpu.hpp"
#include "cpu/tdnn_fully_connected_cpu.hpp"
#include "cpu/batch_norm_cpu.hpp"
#include "cpu/cast_cpu.hpp"
#include "cpu/equal_cpu.hpp"

class FactoryCPU : public Factory {
public:
    std::shared_ptr<Operator> createConvolution(DataType dt,
        ConvolutionParamSpec p,
        ActivationParamSpec dwActivationParamSpec,
        ActivationParamSpec pwActivationParamSpec) override
    {
        auto cep =
            (Convolution *)(new ConvolutionCPU(dt, p, dwActivationParamSpec, pwActivationParamSpec));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createDeconvolution(
        DataType dt, ConvolutionParamSpec p, ActivationParamSpec activationDesc) override
    {
        auto cep = new DeconvolutionCPU(dt, p, activationDesc);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createPooling(PoolingParamSpec p) override
    {
        auto cep = (Pooling *)(new PoolingCPU(p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createFullyConnected(
        DataType dt, FullyConnectedParamSpec p, U32 numInput) override
    {
        auto cep = (FullyConnected *)(new FullyConnectedCPU(dt, p, numInput));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createSoftmax(DataType dt, SoftmaxParamSpec p) override
    {
        auto cep = new SoftmaxCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createConcat(ConcatParamSpec p) override
    {
        auto cep = (Concat *)(new ConcatCPU(p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createActivation(ActivationParamSpec activeDesc) override
    {
        auto cep = (Activation *)new ActivationCPU(activeDesc);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createEltwise(EltwiseParamSpec eltwiseDesc) override
    {
        auto cep = (Eltwise *)new EltwiseCPU(eltwiseDesc);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createScale(DataType dt, ScaleParamSpec p, int numChannels) override
    {
        auto cep = (Scale *)(new ScaleCPU(dt, p, numChannels));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createRNN(DataType dt, RNNParamSpec p) override
    {
        auto cep = (RNNCell *)new RNNCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createRNNCell(DataType dt, RNNParamSpec p) override
    {
        auto cep = (RNNCell *)new RNNCellCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createEmbedding(DataType dt, EmbedParamSpec p) override
    {
        auto cep = (Embedding *)(new EmbeddingCPU(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createPower(DataType dt, PowerParamSpec p) override
    {
        auto cep = (Power *)(new PowerCPU(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createMatMul(DataType dt, MatMulParamSpec p) override
    {
        auto cep = (MatMul *)(new MatMulCPU(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createLayerNorm(DataType dt, U32 weightNum) override
    {
        auto cep = (LayerNorm *)(new LayerNormCPU(dt, weightNum));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createReshape(DataType dt, ReshapeParamSpec p) override
    {
        auto cep = (Reshape *)(new ReshapeCPU(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createResize(DataType paramDT, ResizeParamSpec p) override
    {
        auto cep = (Resize *)(new ResizeCPU(paramDT, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createSlice(DataType dt, SliceParamSpec p) override
    {
        auto cep = (Slice *)(new SliceCPU(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createTranspose(DataType dt, TransposeParamSpec p) override
    {
        auto cep = (Transpose *)new TransposeCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createAttention(DataType dt, AttentionParamSpec p) override
    {
        auto cep = new Attention(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createClip(DataType dt, ClipParamSpec p) override
    {
        auto cep = (Clip *)(new ClipCPU(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createSqueeze(DataType dt, SqueezeParamSpec p) override
    {
        auto cep = (Squeeze *)(new SqueezeCPU(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createUnsqueeze(DataType dt, UnsqueezeParamSpec p) override
    {
        auto cep = (Unsqueeze *)new UnsqueezeCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createReduction(DataType dt, ReductionParamSpec p) override
    {
        auto cep = new ReductionCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createArgMax(DataType dt, ArgMaxParamSpec p) override
    {
        auto cep = (ArgMax *)new ArgMaxCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createCopy(DataType dt, CopyParamSpec p) override
    {
        auto cep = (Copy *)new CopyCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createCheck(DataType dt, CheckParamSpec p) override
    {
        auto cep = (Check *)new CheckCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createRepeat(
        DataType dt, RepeatParamSpec p, I32 jumpOperatorIndex, I32 currentOperatorIndex) override
    {
        auto cep = (Repeat *)new RepeatCPU(dt, p, jumpOperatorIndex, currentOperatorIndex);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createBilateralSliceApply(BilateralSliceApplyParamSpec p) override
    {
        OP_UNSUP(1, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createPreAllocatedMemory(DataType dt, TensorDesc desc) override
    {
        auto cep = (PreAllocatedMemory *)new PreAllocatedMemoryCPU(dt, desc);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createSharedWeight(DataType dt,
        TensorDesc desc,
        std::string outputTensorName,
        std::map<std::string, std::shared_ptr<Tensor>> *tensorMapPtr) override
    {
        auto cep = (SharedWeight *)new SharedWeightCPU(dt, desc, outputTensorName, tensorMapPtr);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createJump(
        DataType dt, I32 jumpOperatorIndex, I32 currentOperatorIndex) override
    {
        auto cep = new Jump(dt, jumpOperatorIndex, currentOperatorIndex);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createSpace2Depth(DataType dt) override
    {
        OP_UNSUP(1, dt);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createDepth2Space(DataType dt, Depth2SpaceParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createPReLU(DataType dt) override
    {
        auto cep = new PReLUCPU(dt);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createAttentionMask(DataType dt, AttentionMaskParamSpec p) override
    {
        auto cep = new AttentionMask(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createRelativePositionEmbedding(DataType dt, EmbedParamSpec p) override
    {
        auto cep = new RelativePositionEmbedding(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createRelativeShift(DataType dt, RelativeShiftParamSpec p) override
    {
        auto cep = new RelativeShift(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createPadding(DataType dt, PadParamSpec p) override
    {
        auto cep = (Padding *)(new PaddingCPU(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createPriorBox(DataType dt, PriorBoxParamSpec p) override
    {
        auto cep = new PriorBox(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createDetectionOutput(DataType dt, DetectionOutputParamSpec p) override
    {
        auto cep = new DetectionOutput(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createYolov3DetectionOutput(
        DataType dt, Yolov3DetectionOutputParamSpec p) override
    {
        auto cep = new Yolov3DetectionOutput(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createChannelResize(DataType dt, ChannelResizeParamSpec p) override
    {
        auto cep = new ChannelResizeCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createL2Normalization(DataType dt) override
    {
        auto cep = new L2NormalizationCPU(dt);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createTile(DataType dt, TileParamSpec p) override
    {
        auto cep = new TileCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createTfSlice(DataType dt, TfSliceParamSpec p) override
    {
        auto cep = new TfSliceCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createSplice(DataType dt, SpliceParamSpec p) override
    {
        auto cep = new SpliceCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createShape() override
    {
        auto cep = new ShapeCPU();
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createWhere(DataType dt) override
    {
        auto cep = new WhereCPU(dt);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createTdnn(DataType dt, TdnnParamSpec p) override
    {
        //auto cep = new TdnnConvolutionCPU(dt, p);
        auto cep = new TdnnFullyConnectedCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createBatchNorm(DataType dt, BatchNormParamSpec p) override
    {
        auto cep = new BatchNormCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createTopK(DataType dt, TopKParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        //auto cep = new TopKCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createCast(DataType dt, CastParamSpec p) override
    {
        auto cep = new CastCPU(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createEqual(DataType dt) override
    {
        auto cep = new EqualCPU(dt);
        return std::shared_ptr<Operator>(cep);
    }
};
#endif  // _FACTORY_CPU_H
