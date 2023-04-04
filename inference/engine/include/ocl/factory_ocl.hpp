// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _FACTORY_OCL_H
#define _FACTORY_OCL_H
#include "factory.hpp"
#include "ocl/resize_ocl.hpp"
#include "ocl/channel_resize_ocl.hpp"
#include "ocl/deconvolution_ocl.hpp"
#include "ocl/bilateral_slice_apply_ocl.hpp"
#include "ocl/pooling_ocl.hpp"
#include "ocl/convolution_ocl.hpp"
#include "ocl/eltwise_ocl.hpp"
#include "ocl/softmax_ocl.hpp"
#include "ocl/activation_ocl.hpp"
#include "ocl/fully_connected_ocl.hpp"
#include "ocl/scale_ocl.hpp"
#include "ocl/concat_ocl.hpp"
#include "ocl/clip_ocl.hpp"
#include "ocl/squeeze_ocl.hpp"
#include "ocl/reshape_ocl.hpp"
#include "ocl/space2depth_ocl.hpp"
#include "ocl/depth2space_ocl.hpp"
#include "ocl/embedding_ocl.hpp"
#include "ocl/layer_norm_ocl.hpp"
#include "ocl/matmul_ocl.hpp"
#include "ocl/power_ocl.hpp"
#include "ocl/transpose_ocl.hpp"
#include "ocl/slice_ocl.hpp"
#include "ocl/shared_weight_ocl.hpp"
#include "ocl/repeat_ocl.hpp"
#include "ocl/copy_ocl.hpp"
#include "ocl/check_ocl.hpp"
#include "ocl/preallocated_memory_ocl.hpp"
#include "ocl/argmax_ocl.hpp"
#include "ocl/unsqueeze_ocl.hpp"
#include "ocl/rnn_ocl.hpp"
#include "ocl/rnncell_ocl.hpp"
#include "ocl/padding_ocl.hpp"
#include "ocl/prelu_ocl.hpp"
#include "ocl/reduction_ocl.hpp"
#include "ocl/topk_ocl.hpp"
#include "ocl/tfslice_ocl.hpp"
#include "ocl/cast_ocl.hpp"
#include "ocl/expand_ocl.hpp"
#include "ocl/tile_ocl.hpp"
#include "ocl/roialign_ocl.hpp"
#include "ocl/generate_proposals_ocl.hpp"
#include "ocl/gather_ocl.hpp"
#include "ocl/prior_box_ocl.hpp"
#include "ocl/detection_output_ocl.hpp"
#include "ocl/flatten_ocl.hpp"
#include "ocl/convert_color_ocl.hpp"
#include "ocl/lut_preprocess_ocl.hpp"
#include "ocl/lut_ocl.hpp"
#include "ocl/instance_norm_ocl.hpp"

class FactoryOCL : public Factory {
public:
    std::shared_ptr<Operator> createConvolution(DataType dt,
        ConvolutionParamSpec p,
        ActivationParamSpec dwActivationParamSpec,
        ActivationParamSpec pwActivationParamSpec) override
    {
        auto cep =
            (Convolution *)(new ConvolutionOCL(dt, p, dwActivationParamSpec, pwActivationParamSpec));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createDeconvolution(
        DataType dt, ConvolutionParamSpec p, ActivationParamSpec activationDesc) override
    {
        auto cep = new DeconvolutionOCL(dt, p, activationDesc);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createPooling(PoolingParamSpec p) override
    {
        auto cep = (Pooling *)(new PoolingOCL(p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createFullyConnected(
        DataType dt, FullyConnectedParamSpec p, U32 numInput) override
    {
        auto cep = (FullyConnectedOCL *)(new FullyConnectedOCL(dt, p, numInput));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createSoftmax(DataType dt, SoftmaxParamSpec p) override
    {
        auto cep = new SoftmaxOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createConcat(ConcatParamSpec p) override
    {
        auto cep = (Concat *)(new ConcatOCL(p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createActivation(ActivationParamSpec activeDesc) override
    {
        auto cep = (Activation *)new ActivationOCL(activeDesc);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createEltwise(EltwiseParamSpec eltwiseDesc) override
    {
        auto cep = (Eltwise *)new EltwiseOCL(eltwiseDesc);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createScale(DataType dt, ScaleParamSpec p, int numChannels) override
    {
        auto cep = (Scale *)(new ScaleOCL(dt, p, numChannels));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createPReLU(DataType dt) override
    {
        auto cep = (PReLU *)(new PReLUOCL(dt));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createRNN(DataType dt, RNNParamSpec p) override
    {
        auto cep = (RNNCell *)(new RNNOCL(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createRNNCell(DataType dt, RNNParamSpec p) override
    {
        auto cep = (RNNCell *)(new RNNCellOCL(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createEmbedding(DataType dt, EmbedParamSpec p) override
    {
        auto cep = (Embedding *)new EmbeddingOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createPower(DataType dt, PowerParamSpec p) override
    {
        auto cep = (Power *)new PowerOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createMatMul(DataType dt, MatMulParamSpec p) override
    {
        auto cep = (MatMul *)(new MatMulOCL(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createLayerNorm(
        DataType dt, LayerNormParamSpec p, U32 weightNum) override
    {
        auto cep = (LayerNorm *)new LayerNormOCL(dt, p, weightNum);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createReshape(DataType dt, ReshapeParamSpec p) override
    {
        auto cep = (Reshape *)(new ReshapeOCL(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createResize(DataType dt, ResizeParamSpec p) override
    {
        auto cep = (Resize *)(new ResizeOCL(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createSlice(DataType dt, SliceParamSpec p) override
    {
        auto cep = (Slice *)(new SliceOCL(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createTranspose(DataType dt, TransposeParamSpec p) override
    {
        auto cep = (Transpose *)(new TransposeOCL(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createAttention(DataType dt, AttentionParamSpec p) override
    {
        // auto cep = new AttentionOCL(dt, numHeads, fromSequenceLength, toSequenceLength);
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createClip(DataType dt, ClipParamSpec p) override
    {
        auto cep = (Clip *)(new ClipOCL(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createSqueeze(DataType dt, SqueezeParamSpec p) override
    {
        auto cep = (Squeeze *)(new SqueezeOCL(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createUnsqueeze(DataType dt, UnsqueezeParamSpec p) override
    {
        auto cep = (Unsqueeze *)new UnsqueezeOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createReduction(DataType dt, ReductionParamSpec p) override
    {
        auto cep = new ReductionOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createArgMax(DataType dt, ArgMaxParamSpec p) override
    {
        auto cep = (ArgMax *)new ArgMaxOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createCopy(DataType dt, CopyParamSpec p) override
    {
        auto cep = (Copy *)new CopyOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createCheck(DataType dt, CheckParamSpec p) override
    {
        auto cep = (Check *)new CheckOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createRepeat(
        DataType dt, RepeatParamSpec p, I32 jumpOperatorIndex, I32 currentOperatorIndex) override
    {
        auto cep = (Repeat *)new RepeatOCL(dt, p, jumpOperatorIndex, currentOperatorIndex);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createBilateralSliceApply(BilateralSliceApplyParamSpec p) override
    {
        auto cep = (BilateralSliceApply *)(new BilateralSliceApplyOCL(p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createPreAllocatedMemory(PreAllocatedMemoryParamSpec p) override
    {
        auto cep = (PreAllocatedMemory *)new PreAllocatedMemoryOCL(p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createSharedWeight(DataType dt,
        TensorDesc desc,
        std::string outputTensorName,
        std::map<std::string, std::shared_ptr<Tensor>> *tensorMapPtr) override
    {
        auto cep = (SharedWeight *)new SharedWeightOCL(dt, desc, outputTensorName, tensorMapPtr);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createJump(
        DataType dt, I32 jumpOperatorIndex, I32 currentOperatorIndex) override
    {
        OP_UNSUP(3, dt, jumpOperatorIndex, currentOperatorIndex);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createSpace2Depth(DataType dt, Space2DepthParamSpec p) override
    {
        auto cep = (Space2Depth *)(new Space2DepthOCL(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createDepth2Space(DataType dt, Depth2SpaceParamSpec p) override
    {
        auto cep = (Depth2Space *)(new Depth2SpaceOCL(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createAttentionMask(DataType dt, AttentionMaskParamSpec p) override
    {
        OP_UNSUP(2, dt, p)
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createRelativePositionEmbedding(DataType dt, EmbedParamSpec p) override
    {
        OP_UNSUP(2, dt, p)
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createRelativeShift(DataType dt, RelativeShiftParamSpec p) override
    {
        OP_UNSUP(2, dt, p)
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createPadding(DataType dt, PadParamSpec p) override
    {
        auto cep = (Padding *)(new PaddingOCL(dt, p));
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createPriorBox(DataType dt, PriorBoxParamSpec p) override
    {
        auto cep = new PriorBoxOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createDetectionOutput(DataType dt, DetectionOutputParamSpec p) override
    {
        auto cep = new DetectionOutputOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createYolov3DetectionOutput(
        DataType dt, Yolov3DetectionOutputParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createChannelResize(DataType dt, ChannelResizeParamSpec p) override
    {
        auto cep = new ChannelResizeOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createL2Norm(DataType dt) override
    {
        OP_UNSUP(1, dt);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createTile(DataType dt, TileParamSpec p) override
    {
        auto cep = (Tile *)new TileOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createTfSlice(DataType dt, TfSliceParamSpec p) override
    {
        auto cep = (TopK *)new TfSliceOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createSplice(DataType dt, SpliceParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createShape() override
    {
        OP_UNSUP(0);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createWhere(DataType dt) override
    {
        OP_UNSUP(1, dt);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createTdnn(DataType dt, TdnnParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createBatchNorm(DataType dt, BatchNormParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createTopK(DataType dt, TopKParamSpec p) override
    {
        auto cep = (TopK *)new TopKOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createCast(DataType dt, CastParamSpec p) override
    {
        auto cep = (Cast *)new CastOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createInstanceNorm(DataType dt, InstanceNormParamSpec p) override
    {
        auto cep = (InstanceNorm *)new InstanceNormOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createExpand(DataType dt, ExpandParamSpec p) override
    {
        auto cep = (Expand *)new ExpandOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createScatter(DataType dt, ScatterParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createGather(DataType dt, GatherParamSpec p) override
    {
        auto cep = new GatherOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createLogSoftmax(DataType dt, SoftmaxParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createSelect(DataType dt) override
    {
        OP_UNSUP(1, dt);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createRoIAlign(DataType dt, RoIAlignParamSpec p) override
    {
        auto cep = (RoIAlign *)new RoIAlignOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createGenerateProposals(
        DataType dt, GenerateProposalsParamSpec p) override
    {
        auto cep = (GenerateProposals *)new GenerateProposalsOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createGAT(DataType dt, GATParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }

#ifdef _USE_INT8
    std::shared_ptr<Operator> createQuantizeLinear(DataType dt, QuantizeLinearParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }
#endif

    std::shared_ptr<Operator> createGridSample(DataType dt, GridSampleParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createOneHot(DataType dt, OneHotParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createCum(DataType dt, CumParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createNonMaxSuppression(
        DataType dt, NonMaxSuppressionParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createConstantOfShape(DataType dt, ConstantOfShapeParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createNonZero(DataType dt) override
    {
        OP_UNSUP(1, dt);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createRange(DataType dt, RangeParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createEinsum(DataType dt, EinsumParamSpec p) override
    {
        OP_UNSUP(2, dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createUnPooling(PoolingParamSpec p) override
    {
        OP_UNSUP(1, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createRandom(RandomParamSpec p) override
    {
        OP_UNSUP(1, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createFlatten(FlattenParamSpec p) override
    {
        auto cep = new FlattenOCL(p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createConvertColor(DataType dt, ConvertColorParamSpec p) override
    {
        auto cep = new ConvertColorOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createLutPreprocess(DataType dt) override
    {
        auto cep = new LutPreprocessOCL(dt);
        return std::shared_ptr<Operator>(cep);
    }

    std::shared_ptr<Operator> createLut(DataType dt, LutParamSpec p) override
    {
        auto cep = new LutOCL(dt, p);
        return std::shared_ptr<Operator>(cep);
    }
};
#endif  // _FACTORY_OCL_H
