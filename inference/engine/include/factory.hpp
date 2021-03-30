// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _FACTORY_H
#define _FACTORY_H

#include "operator.hpp"
#include "tensor_computing.h"

#define NOT_SUPPORT       \
    Operator *cep = NULL; \
    CHECK_STATUS(NOT_SUPPORTED);
#define NOT_USE0()
#define NOT_USE1(a1) \
    {                \
        UNUSED(a1);  \
    }
#define NOT_USE2(a1, a2)          \
    {                             \
        NOT_USE1(a1) NOT_USE1(a2) \
    }
#define NOT_USE3(a1, a2, a3)          \
    {                                 \
        NOT_USE2(a1, a2) NOT_USE1(a3) \
    }
#define NOT_USE4(a1, a2, a3, a4)          \
    {                                     \
        NOT_USE2(a1, a2) NOT_USE2(a3, a4) \
    }
#define NOT_USE5(a1, a2, a3, a4, a5)          \
    {                                         \
        NOT_USE4(a1, a2, a3, a4) NOT_USE1(a5) \
    }
#define NOT_USE6(a1, a2, a3, a4, a5, a6)          \
    {                                             \
        NOT_USE4(a1, a2, a3, a4) NOT_USE2(a5, a6) \
    }
#define NOT_USE8(a1, a2, a3, a4, a5, a6, a7, a8)          \
    {                                                     \
        NOT_USE4(a1, a2, a3, a4) NOT_USE4(a5, a6, a7, a8) \
    }
#define NOT_USE10(a1, a2, a3, a4, a5, a6, a7, a8, a9, aa)         \
    {                                                             \
        NOT_USE8(a1, a2, a3, a4, a5, a6, a7, a8) NOT_USE2(a9, aa) \
    }
#define NOT_USE16(a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, ag)         \
    {                                                                                     \
        NOT_USE8(a1, a2, a3, a4, a5, a6, a7, a8) NOT_USE8(a9, aa, ab, ac, ad, ae, af, ag) \
    }
#define OP_UNSUP(num, ...) NOT_USE##num(__VA_ARGS__) NOT_SUPPORT

class Factory {
public:
    virtual ~Factory()
    {}

    virtual std::shared_ptr<Operator> createConvolution(DataType dt,
        ConvolutionParamSpec p,
        ActivationParamSpec dwActivationParamSpec,
        ActivationParamSpec pwActivationParamSpec) = 0;

    virtual std::shared_ptr<Operator> createDeconvolution(
        DataType dt, ConvolutionParamSpec p, ActivationParamSpec activationDesc) = 0;

    virtual std::shared_ptr<Operator> createPooling(PoolingParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createFullyConnected(
        DataType dt, FullyConnectedParamSpec p, U32 numInput) = 0;

    virtual std::shared_ptr<Operator> createSoftmax(DataType dt, SoftmaxParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createConcat(ConcatParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createActivation(ActivationParamSpec activationDesc) = 0;

    virtual std::shared_ptr<Operator> createEltwise(EltwiseParamSpec eltwiseDesc) = 0;

    virtual std::shared_ptr<Operator> createScale(
        DataType dt, ScaleParamSpec p, int numChannels) = 0;

    virtual std::shared_ptr<Operator> createRNN(DataType dt, RNNParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createRNNCell(DataType dt, RNNParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createEmbedding(DataType dt, EmbedParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createPower(DataType dt, PowerParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createMatMul(DataType dt, MatMulParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createLayerNorm(DataType dt, U32 weightNum) = 0;

    virtual std::shared_ptr<Operator> createReshape(DataType dt, ReshapeParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createResize(DataType paramDT, ResizeParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createSlice(DataType dt, SliceParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createTranspose(DataType dt, TransposeParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createAttention(DataType dt, AttentionParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createClip(DataType dt, ClipParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createSqueeze(DataType dt, SqueezeParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createUnsqueeze(DataType dt, UnsqueezeParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createReduction(DataType dt, ReductionParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createArgMax(DataType dt, ArgMaxParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createCopy(DataType dt, CopyParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createCheck(DataType dt, CheckParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createRepeat(
        DataType dt, RepeatParamSpec p, I32 jumpOperatorIndex, I32 currentOperatorIndex) = 0;

    virtual std::shared_ptr<Operator> createBilateralSliceApply(BilateralSliceApplyParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createPreAllocatedMemory(DataType dt, TensorDesc desc) = 0;

    virtual std::shared_ptr<Operator> createSharedWeight(DataType dt,
        TensorDesc desc,
        std::string outputTensorName,
        std::map<std::string, std::shared_ptr<Tensor>> *tensorMapPtr) = 0;

    virtual std::shared_ptr<Operator> createJump(
        DataType dt, I32 jumpOperatorIndex, I32 currentOperatorIndex) = 0;

    virtual std::shared_ptr<Operator> createSpace2Depth(DataType dt) = 0;

    virtual std::shared_ptr<Operator> createDepth2Space(DataType dt, Depth2SpaceParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createAttentionMask(DataType dt, AttentionMaskParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createRelativePositionEmbedding(
        DataType dt, EmbedParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createRelativeShift(DataType dt, RelativeShiftParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createPadding(DataType dt, PadParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createPReLU(DataType dt) = 0;

    virtual std::shared_ptr<Operator> createPriorBox(DataType dt, PriorBoxParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createDetectionOutput(
        DataType dt, DetectionOutputParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createYolov3DetectionOutput(
        DataType dt, Yolov3DetectionOutputParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createChannelResize(DataType dt, ChannelResizeParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createL2Normalization(DataType dt) = 0;

    virtual std::shared_ptr<Operator> createTile(DataType dt, TileParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createTfSlice(DataType dt, TfSliceParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createSplice(DataType dt, SpliceParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createShape() = 0;

    virtual std::shared_ptr<Operator> createWhere(DataType dt) = 0;

    virtual std::shared_ptr<Operator> createTdnn(DataType dt, TdnnParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createBatchNorm(DataType dt, BatchNormParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createTopK(DataType dt, TopKParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createCast(DataType dt, CastParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createEqual(DataType dt) = 0;

    std::shared_ptr<Operator> createOperators(OperatorSpec curOps,
        DataType dt,
        std::map<std::string, U32> &operatorIndexMap,
        std::map<std::string, std::shared_ptr<Tensor>> *tensorMapPtr,
        std::vector<std::string> &inputTensorsName,
        std::vector<std::string> &outputTensorsName,
        std::set<std::string> *weightOpOutputNames)
    {
        OperatorType opType = curOps.type;
        DataType dtNoQ = (dt == DT_F16_8Q) ? DT_F16 : dt;
        std::string opName = curOps.name;
        std::shared_ptr<Operator> op;
        auto curPs = curOps.ps;
        switch (opType) {
            case OT_Conv: {
                ActivationParamSpec dwActiveDesc;
                ActivationParamSpec pwActiveDesc;
                dwActiveDesc.mode = curPs.conv_spec.dw_activation_type;
                pwActiveDesc.mode = curPs.conv_spec.pw_activation_type;
                dwActiveDesc.value[0] = 0;
                pwActiveDesc.value[0] = 0;
                op = createConvolution(dt, curPs.conv_spec, dwActiveDesc, pwActiveDesc);
                break;
            }
            case OT_Deconvolution: {
                ActivationParamSpec activeDesc;
                activeDesc.mode = curPs.conv_spec.pw_activation_type;
                activeDesc.value[0] = 0;
                op = createDeconvolution(dtNoQ, curPs.conv_spec, activeDesc);
                break;
            }
            case OT_FC: {
                op = createFullyConnected(dt, curPs.fc_spec, 0);
                break;
            }
            case OT_Pooling: {
                op = createPooling(curPs.pooling_spec);
                break;
            }
            case OT_Softmax: {
                op = createSoftmax(dtNoQ, curPs.softmax_spec);
                break;
            }
            case OT_Relu: {
                ActivationParamSpec activationDesc;
                activationDesc.mode = ACTIVATION_RELU;
                activationDesc.value[0] = curOps.ps.relu_spec.neg_slope;
                op = createActivation(activationDesc);
                break;
            }
            case OT_Relu6: {
                ActivationParamSpec activationDesc;
                activationDesc.mode = ACTIVATION_RELU6;
                op = createActivation(activationDesc);
                break;
            }
            case OT_HSwish: {
                ActivationParamSpec activationDesc;
                activationDesc.mode = ACTIVATION_H_SWISH;
                op = createActivation(activationDesc);
                break;
            }
            case OT_HSwishNoDiv: {
                ActivationParamSpec activationDesc;
                activationDesc.mode = ACTIVATION_H_SWISH_NODIV;
                op = createActivation(activationDesc);
                break;
            }
            case OT_Sigmoid: {
                ActivationParamSpec activationDesc;
                activationDesc.mode = ACTIVATION_SIGMOID;
                op = createActivation(activationDesc);
                break;
            }
            case OT_HSigmoid: {
                ActivationParamSpec activationDesc;
                activationDesc.mode = ACTIVATION_H_SIGMOID;
                op = createActivation(activationDesc);
                break;
            }
            case OT_Gelu: {
                ActivationParamSpec activationDesc;
                activationDesc.mode = ACTIVATION_GELU;
                op = createActivation(activationDesc);
                break;
            }
            case OT_TanH: {
                ActivationParamSpec activationDesc;
                activationDesc.mode = ACTIVATION_TANH;
                op = createActivation(activationDesc);
                break;
            }
            case OT_Mish: {
                ActivationParamSpec activationDesc;
                activationDesc.mode = ACTIVATION_MISH;
                op = createActivation(activationDesc);
                break;
            }
            case OT_Greater: {
                ActivationParamSpec activationDesc;
                activationDesc.mode = ACTIVATION_GREATER;
                op = createActivation(activationDesc);
                break;
            }
            case OT_Concat: {
                op = createConcat(curPs.concat_spec);
                break;
            }
            case OT_Eltwise: {
                op = createEltwise(curOps.ps.eltwise_spec);
                break;
            }
            case OT_Embedding: {
                op = createEmbedding(dtNoQ, curPs.embed_spec);
                break;
            }
            case OT_MatMul: {
                op = createMatMul(dt, curPs.matmul_spec);
                break;
            }
            case OT_Power: {
                op = createPower(dt, curPs.power_spec);
                break;
            }
            case OT_Scale: {
                op = createScale(dtNoQ, curPs.scale_spec, 0);
                break;
            }
            case OT_LayerNorm: {
                op = createLayerNorm(dt, 0);
                break;
            }
            case OT_Reshape: {
                op = createReshape(dt, curPs.reshape_spec);
                break;
            }
            case OT_Resize: {
                if (curPs.resize_spec.num_sizes > 0) {
                    op = createResize(DT_U32, curPs.resize_spec);
                } else {
                    CHECK_REQUIREMENT(curPs.resize_spec.num_scales == 4);
                    op = createResize(DT_F32, curPs.resize_spec);
                }
                break;
            }
            case OT_Slice: {
                op = createSlice(dt, curPs.slice_spec);
                break;
            }
            case OT_Transpose: {
                op = createTranspose(dt, curPs.transpose_spec);
                break;
            }
            case OT_Attention: {
                op = createAttention(dtNoQ, curPs.attention_spec);
                break;
            }
            case OT_Clip: {
                op = createClip(dtNoQ, curPs.clip_spec);
                break;
            }
            case OT_RNN: {
                if (curPs.rnn_spec.steps >= 0) {
                    op = createRNN(dt, curPs.rnn_spec);
                } else {
                    op = createRNNCell(dt, curPs.rnn_spec);
                }
                break;
            }
            case OT_Squeeze: {
                op = createSqueeze(dtNoQ, curPs.squeeze_spec);
                break;
            }
            case OT_Unsqueeze: {
                op = createUnsqueeze(dtNoQ, curPs.unsqueeze_spec);
                break;
            }
            case OT_Reduction: {
                op = createReduction(dtNoQ, curPs.reduction_spec);
                break;
            }
            case OT_ArgMax: {
                op = createArgMax(dtNoQ, curPs.argmax_spec);
                break;
            }
            case OT_PreAllocatedMemory: {
                PreAllocatedMemoryParamSpec curPreAllocatedMemoryParamSpec =
                    curOps.ps.preallocated_memory_spec;
                TensorDesc desc = curPreAllocatedMemoryParamSpec.desc;
                op = createPreAllocatedMemory(dtNoQ, desc);
                break;
            }
            case OT_SharedWeight: {
                SharedWeightParamSpec curSharedWeightParamSpec = curOps.ps.shared_weight_spec;
                TensorDesc desc = curSharedWeightParamSpec.desc;
                op = createSharedWeight(dtNoQ, desc, outputTensorsName[0], tensorMapPtr);
                weightOpOutputNames->insert(outputTensorsName[0]);
                break;
            }
            case OT_Repeat: {
                op = createRepeat(dtNoQ, curPs.repeat_spec, operatorIndexMap[inputTensorsName[0]],
                    operatorIndexMap[opName]);
                break;
            }
            case OT_Check: {
                op = createCheck(dtNoQ, curPs.check_spec);
                break;
            }
            case OT_Copy: {
                op = createCopy(dtNoQ, curPs.copy_spec);
                break;
            }
            case OT_BilateralSliceApply: {
                op = createBilateralSliceApply(curPs.bilateral_slice_apply_spec);
                break;
            }
            case OT_Jump: {
                op = createJump(
                    dtNoQ, operatorIndexMap[inputTensorsName[0]], operatorIndexMap[opName]);
                break;
            }
            case OT_Space2Depth: {
                op = createSpace2Depth(dt);
                break;
            }
            case OT_Depth2Space: {
                op = createDepth2Space(dt, curPs.depth2space_spec);
                break;
            }
            case OT_AttentionMask: {
                op = createAttentionMask(dt, curPs.attention_mask_spec);
                break;
            }
            case OT_RelativePositionEmbedding: {
                op = createRelativePositionEmbedding(dtNoQ, curPs.embed_spec);
                break;
            }
            case OT_RelativeShift: {
                op = createRelativeShift(dt, curPs.relative_shift_spec);
                break;
            }
            case OT_Pad: {
                op = createPadding(dt, curPs.pad_spec);
                break;
            }
            case OT_PriorBox: {
                op = createPriorBox(dt, curPs.prior_box_spec);
                break;
            }
            case OT_DetectionOutput: {
                op = createDetectionOutput(dt, curPs.detection_output_spec);
                break;
            }
            case OT_Yolov3DetectionOutput: {
                op = createYolov3DetectionOutput(dt, curPs.yolov3_detection_output_spec);
                break;
            }
            case OT_ChannelResize: {
                op = createChannelResize(dt, curPs.channel_resize_spec);
                break;
            }
            case OT_L2Normalization: {
                op = createL2Normalization(dt);
                break;
            }
            case OT_PRelu: {
                op = createPReLU(dt);
                break;
            }
            case OT_Tile: {
                op = createTile(dt, curPs.tile_spec);
                break;
            }
            case OT_TfSlice: {
                op = createTfSlice(dt, curPs.tfslice_spec);
                break;
            }
            case OT_Splice: {
                op = createSplice(dt, curPs.splice_spec);
                break;
            }
            case OT_Shape: {
                op = createShape();
                break;
            }
            case OT_Where: {
                op = createWhere(dt);
                break;
            }
            case OT_SoftPlus: {
                ActivationParamSpec activationDesc;
                activationDesc.mode = ACTIVATION_SOFTPLUS;
                op = createActivation(activationDesc);
                break;
            }
            case OT_Exp: {
                ActivationParamSpec activationDesc;
                activationDesc.mode = ACTIVATION_EXP;
                op = createActivation(activationDesc);
                break;
            }
            case OT_Tdnn: {
                op = createTdnn(dt, curPs.tdnn_spec);
                break;
            }
            case OT_BatchNorm: {
                op = createBatchNorm(dt, curPs.bn_spec);
                break;
            }
            case OT_TopK: {
                op = createTopK(dt, curPs.topk_spec);
                break;
            }
            case OT_Abs: {
                ActivationParamSpec activationDesc;
                activationDesc.mode = ACTIVATION_ABS;
                op = createActivation(activationDesc);
                break;
            }
            case OT_Cast: {
                op = createCast(dt, curPs.cast_spec);
                break;
            }
            case OT_Equal: {
                op = createEqual(dt);
                break;
            }
            case OT_Sign: {
                ActivationParamSpec activationDesc;
                activationDesc.mode = ACTIVATION_SIGN;
                op = createActivation(activationDesc);
                break;
            }
            default: {
                UNI_ERROR_LOG("unsupported layer %s\n", OperatorTypeName()[opType]);
                break;
            }
        }
        return op;
    }
};

#endif  // _FACTORY_H
