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

#define NOT_SUPPORT       \
    Operator *cep = NULL; \
    UNI_ERROR_LOG("not support to create operator in %s.\n", __FUNCTION__);
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

    virtual std::shared_ptr<Operator> createPooling(PoolingParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createActivation(ActivationParamSpec activationDesc) = 0;

    virtual std::shared_ptr<Operator> createEltwise(EltwiseParamSpec eltwiseDesc) = 0;

    virtual std::shared_ptr<Operator> createChannelResize(DataType dt, ChannelResizeParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createFullyConnected(
        DataType dt, FullyConnectedParamSpec p, U32 numInput) = 0;

    virtual std::shared_ptr<Operator> createReshape(DataType dt, ReshapeParamSpec p) = 0;

#ifdef _USE_INT8
    virtual std::shared_ptr<Operator> createQuantizeLinear(
        DataType dt, QuantizeLinearParamSpec p) = 0;
#endif
#ifndef _USE_LITE
    virtual std::shared_ptr<Operator> createConcat(ConcatParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createFlatten(FlattenParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createSoftmax(DataType dt, SoftmaxParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createLogSoftmax(DataType dt, SoftmaxParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createDeconvolution(
        DataType dt, ConvolutionParamSpec p, ActivationParamSpec activationDesc) = 0;

    virtual std::shared_ptr<Operator> createScale(
        DataType dt, ScaleParamSpec p, int numChannels) = 0;

    virtual std::shared_ptr<Operator> createRNN(DataType dt, RNNParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createRNNCell(DataType dt, RNNParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createEmbedding(DataType dt, EmbedParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createPower(DataType dt, PowerParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createMatMul(DataType dt, MatMulParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createLayerNorm(
        DataType dt, LayerNormParamSpec p, U32 weightNum) = 0;

    virtual std::shared_ptr<Operator> createResize(DataType dt, ResizeParamSpec p) = 0;

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

    virtual std::shared_ptr<Operator> createPreAllocatedMemory(PreAllocatedMemoryParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createSharedWeight(DataType dt,
        TensorDesc desc,
        std::string outputTensorName,
        std::map<std::string, std::shared_ptr<Tensor>> *tensorMapPtr) = 0;

    virtual std::shared_ptr<Operator> createJump(
        DataType dt, I32 jumpOperatorIndex, I32 currentOperatorIndex) = 0;

    virtual std::shared_ptr<Operator> createSpace2Depth(DataType dt, Space2DepthParamSpec p) = 0;

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

    virtual std::shared_ptr<Operator> createL2Norm(DataType dt) = 0;

    virtual std::shared_ptr<Operator> createTile(DataType dt, TileParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createTfSlice(DataType dt, TfSliceParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createSplice(DataType dt, SpliceParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createShape() = 0;

    virtual std::shared_ptr<Operator> createWhere(DataType dt) = 0;

    virtual std::shared_ptr<Operator> createTdnn(DataType dt, TdnnParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createBatchNorm(DataType dt, BatchNormParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createTopK(DataType dt, TopKParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createCast(DataType dt, CastParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createExpand(DataType dt, ExpandParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createScatter(DataType dt, ScatterParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createGather(DataType dt, GatherParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createSelect(DataType dt) = 0;

    virtual std::shared_ptr<Operator> createInstanceNorm(DataType dt, InstanceNormParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createRoIAlign(DataType dt, RoIAlignParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createGenerateProposals(
        DataType dt, GenerateProposalsParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createGAT(DataType dt, GATParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createGridSample(DataType dt, GridSampleParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createOneHot(DataType dt, OneHotParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createCum(DataType dt, CumParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createNonMaxSuppression(
        DataType dt, NonMaxSuppressionParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createConstantOfShape(
        DataType dt, ConstantOfShapeParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createNonZero(DataType dt) = 0;

    virtual std::shared_ptr<Operator> createRange(DataType dt, RangeParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createEinsum(DataType dt, EinsumParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createUnPooling(PoolingParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createRandom(RandomParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createConvertColor(DataType dt, ConvertColorParamSpec p) = 0;

    virtual std::shared_ptr<Operator> createLutPreprocess(DataType dt) = 0;

    virtual std::shared_ptr<Operator> createLut(DataType dt, LutParamSpec p) = 0;
#endif

    std::shared_ptr<Operator> createOperators(OperatorSpec op,
        DataType dt,
        std::map<std::string, U32> &operatorIndexMap,
        std::map<std::string, std::shared_ptr<Tensor>> *tensorMapPtr,
        std::vector<std::string> &inputTensorsName,
        std::vector<std::string> &outputTensorsName,
        std::set<std::string> *weightOpOutputNames)
    {
        if (dt == DT_F32 || dt == DT_F32_8Q) {
#ifndef _USE_FP32
            UNI_ERROR_LOG("this library not support to inference float32/int8+float32, please "
                          "recompile with --fp32=on.\n");
#endif
        }
        if (dt == DT_F16 || dt == DT_F16_8Q) {
#if !(defined(_USE_FP16) || defined(_USE_GPU))
            UNI_ERROR_LOG("this library not support to inference float16/int8+float16, please "
                          "recompile with --fp16=on. Only Armv8.2+ cpu and gpu support.\n");
#endif
        }
        if (isQuantMixDataType(dt)) {
#ifndef _USE_INT8
            UNI_ERROR_LOG("this library not support to inference int8, please recompile with "
                          "--int8=on. Only Armv7+ and x86 AVX512/AVX512-VNNI cpu support.\n");
#endif
        }
        std::string name = op.name;
        OperatorType type = op.type;
        static std::map<OperatorType, ActivationMode> activations = {{OT_Relu6, ACTIVATION_RELU6},
            {OT_HSwish, ACTIVATION_H_SWISH}, {OT_HSwishNoDiv, ACTIVATION_H_SWISH_NODIV},
            {OT_Sigmoid, ACTIVATION_SIGMOID}, {OT_HSigmoid, ACTIVATION_H_SIGMOID},
            {OT_Gelu, ACTIVATION_GELU}, {OT_TanH, ACTIVATION_TANH}, {OT_Mish, ACTIVATION_MISH},
            {OT_Greater, ACTIVATION_GREATER}, {OT_Exp, ACTIVATION_EXP},
            {OT_Softplus, ACTIVATION_SOFTPLUS}, {OT_Abs, ACTIVATION_ABS}, {OT_Sign, ACTIVATION_SIGN},
            {OT_Not, ACTIVATION_NOT}, {OT_Log, ACTIVATION_LOG}, {OT_Neg, ACTIVATION_NEG},
            {OT_Round, ACTIVATION_ROUND}, {OT_Floor, ACTIVATION_FLOOR}, {OT_Ceil, ACTIVATION_CEIL},
            {OT_Swish, ACTIVATION_SWISH}, {OT_Reciprocal, ACTIVATION_RECIPROCAL},
            {OT_Sin, ACTIVATION_SIN}, {OT_Cos, ACTIVATION_COS}};
        if (activations.find(type) != activations.end()) {
            ActivationParamSpec param;
            param.mode = activations[type];
            return createActivation(param);
        }
        auto ps = op.ps;
        std::shared_ptr<Operator> ret;
        DataType dtNoQ = noQuantDataType(dt);
        switch (type) {
            case OT_Conv: {
                ActivationParamSpec dwActiveDesc;
                ActivationParamSpec pwActiveDesc;
                dwActiveDesc.mode = ps.conv_spec.dw_activation_type;
                pwActiveDesc.mode = ps.conv_spec.pw_activation_type;
                dwActiveDesc.value[0] = 0;
                pwActiveDesc.value[0] = 0;
                ret = createConvolution(dt, ps.conv_spec, dwActiveDesc, pwActiveDesc);
                break;
            }
            case OT_Pooling: {
                ret = createPooling(ps.pooling_spec);
                break;
            }
            case OT_Relu: {
                ActivationParamSpec param;
                param.mode = ACTIVATION_RELU;
                param.value[0] = ps.relu_spec.neg_slope;
                ret = createActivation(param);
                break;
            }
            case OT_Elu: {
                ActivationParamSpec param;
                param.mode = ACTIVATION_ELU;
                param.value[0] = ps.relu_spec.neg_slope;
                ret = createActivation(param);
                break;
            }
            case OT_Eltwise: {
                ret = createEltwise(ps.eltwise_spec);
                break;
            }
            case OT_ChannelResize: {
                ret = createChannelResize(dt, ps.channel_resize_spec);
                break;
            }
            case OT_FC: {
                ret = createFullyConnected(dt, ps.fc_spec, 0);
                break;
            }
            case OT_Reshape: {
                ret = createReshape(dt, ps.reshape_spec);
                break;
            }
#ifdef _USE_INT8
            case OT_QuantizeLinear: {
                ret = createQuantizeLinear(dt, ps.quant_spec);
                break;
            }
#endif
#ifndef _USE_LITE
            case OT_Concat: {
                ret = createConcat(ps.concat_spec);
                break;
            }
            case OT_Flatten: {
                ret = createFlatten(ps.flatten_spec);
                break;
            }
            case OT_Deconvolution: {
                ActivationParamSpec param;
                param.mode = ps.conv_spec.pw_activation_type;
                param.value[0] = 0;
                ret = createDeconvolution(dt, ps.conv_spec, param);
                break;
            }
            case OT_Softmax: {
                ret = createSoftmax(dtNoQ, ps.softmax_spec);
                break;
            }
            case OT_LogSoftmax: {
                ret = createLogSoftmax(dtNoQ, ps.softmax_spec);
                break;
            }
            case OT_Embedding: {
                ret = createEmbedding(dtNoQ, ps.embed_spec);
                break;
            }
            case OT_MatMul: {
                ret = createMatMul(dt, ps.matmul_spec);
                break;
            }
            case OT_Power: {
                ret = createPower(dt, ps.power_spec);
                break;
            }
            case OT_Scale: {
                ret = createScale(dtNoQ, ps.scale_spec, 0);
                break;
            }
            case OT_LayerNorm: {
                ret = createLayerNorm(dt, ps.ln_spec, 0);
                break;
            }
            case OT_Resize: {
                ret = createResize(dt, ps.resize_spec);
                break;
            }
            case OT_Slice: {
                ret = createSlice(dt, ps.slice_spec);
                break;
            }
            case OT_Transpose: {
                ret = createTranspose(dt, ps.transpose_spec);
                break;
            }
            case OT_Attention: {
                ret = createAttention(dtNoQ, ps.attention_spec);
                break;
            }
            case OT_Clip: {
                ret = createClip(dtNoQ, ps.clip_spec);
                break;
            }
            case OT_RNN: {
                if (ps.rnn_spec.steps >= 0) {
                    ret = createRNN(dt, ps.rnn_spec);
                } else {
                    ret = createRNNCell(dt, ps.rnn_spec);
                }
                break;
            }
            case OT_Squeeze: {
                ret = createSqueeze(dtNoQ, ps.squeeze_spec);
                break;
            }
            case OT_Unsqueeze: {
                ret = createUnsqueeze(dtNoQ, ps.unsqueeze_spec);
                break;
            }
            case OT_Reduction: {
                ret = createReduction(dtNoQ, ps.reduction_spec);
                break;
            }
            case OT_ArgMax: {
                ret = createArgMax(dtNoQ, ps.argmax_spec);
                break;
            }
            case OT_PreAllocatedMemory: {
                ret = createPreAllocatedMemory(ps.preallocated_memory_spec);
                break;
            }
            case OT_SharedWeight: {
                SharedWeightParamSpec param = ps.shared_weight_spec;
                TensorDesc desc = param.desc;
                ret = createSharedWeight(dtNoQ, desc, outputTensorsName[0], tensorMapPtr);
                weightOpOutputNames->insert(outputTensorsName[0]);
                break;
            }
            case OT_Repeat: {
                ret = createRepeat(dtNoQ, ps.repeat_spec, operatorIndexMap[inputTensorsName[0]],
                    operatorIndexMap[name]);
                break;
            }
            case OT_Check: {
                ret = createCheck(dtNoQ, ps.check_spec);
                break;
            }
            case OT_Copy: {
                ret = createCopy(dtNoQ, ps.copy_spec);
                break;
            }
            case OT_BilateralSliceApply: {
                ret = createBilateralSliceApply(ps.bilateral_slice_apply_spec);
                break;
            }
            case OT_Jump: {
                ret = createJump(
                    dtNoQ, operatorIndexMap[inputTensorsName[0]], operatorIndexMap[name]);
                break;
            }
            case OT_Space2Depth: {
                ret = createSpace2Depth(dt, ps.space2depth_spec);
                break;
            }
            case OT_Depth2Space: {
                ret = createDepth2Space(dt, ps.depth2space_spec);
                break;
            }
            case OT_AttentionMask: {
                ret = createAttentionMask(dt, ps.attention_mask_spec);
                break;
            }
            case OT_RelativePositionEmbedding: {
                ret = createRelativePositionEmbedding(dtNoQ, ps.embed_spec);
                break;
            }
            case OT_RelativeShift: {
                ret = createRelativeShift(dt, ps.relative_shift_spec);
                break;
            }
            case OT_Pad: {
                ret = createPadding(dt, ps.pad_spec);
                break;
            }
            case OT_PriorBox: {
                ret = createPriorBox(dt, ps.prior_box_spec);
                break;
            }
            case OT_DetectionOutput: {
                ret = createDetectionOutput(dt, ps.detection_output_spec);
                break;
            }
            case OT_Yolov3DetectionOutput: {
                ret = createYolov3DetectionOutput(dt, ps.yolov3_detection_output_spec);
                break;
            }
            case OT_L2Norm: {
                ret = createL2Norm(dt);
                break;
            }
            case OT_PRelu: {
                ret = createPReLU(dt);
                break;
            }
            case OT_Tile: {
                ret = createTile(dt, ps.tile_spec);
                break;
            }
            case OT_TfSlice: {
                ret = createTfSlice(dt, ps.tfslice_spec);
                break;
            }
            case OT_Splice: {
                ret = createSplice(dt, ps.splice_spec);
                break;
            }
            case OT_Shape: {
                ret = createShape();
                break;
            }
            case OT_Where: {
                ret = createWhere(dt);
                break;
            }
            case OT_Tdnn: {
                ret = createTdnn(dt, ps.tdnn_spec);
                break;
            }
            case OT_BatchNorm: {
                ret = createBatchNorm(dt, ps.bn_spec);
                break;
            }
            case OT_TopK: {
                ret = createTopK(dt, ps.topk_spec);
                break;
            }
            case OT_Cast: {
                ret = createCast(dt, ps.cast_spec);
                break;
            }
            case OT_InstanceNorm: {
                ret = createInstanceNorm(dt, ps.in_spec);
                break;
            }
            case OT_Expand: {
                ret = createExpand(dt, ps.expand_spec);
                break;
            }
            case OT_Scatter: {
                ret = createScatter(dt, ps.scatter_spec);
                break;
            }
            case OT_Gather: {
                ret = createGather(dt, ps.gather_spec);
                break;
            }
            case OT_Select: {
                ret = createSelect(dt);
                break;
            }
            case OT_GAT: {
                ret = createGAT(dt, ps.gat_spec);
                break;
            }
            case OT_RoIAlign: {
                ret = createRoIAlign(dt, ps.roialign_spec);
                break;
            }
            case OT_GenerateProposals: {
                ret = createGenerateProposals(dt, ps.generate_proposals_spec);
                break;
            }
            case OT_GridSample: {
                ret = createGridSample(dt, ps.grid_sample_spec);
                break;
            }
            case OT_OneHot: {
                ret = createOneHot(dt, ps.onehot_spec);
                break;
            }
            case OT_Cum: {
                ret = createCum(dt, ps.cum_spec);
                break;
            }
            case OT_NonMaxSuppression: {
                ret = createNonMaxSuppression(dt, ps.non_max_suppression_spec);
                break;
            }
            case OT_ConstantOfShape: {
                ret = createConstantOfShape(dt, ps.constant_of_shape_spec);
                break;
            }
            case OT_NonZero: {
                ret = createNonZero(dt);
                break;
            }
            case OT_Range: {
                ret = createRange(dt, ps.range_spec);
                break;
            }
            case OT_Einsum: {
                ret = createEinsum(dt, ps.einsum_spec);
                break;
            }
            case OT_UnPooling: {
                ret = createUnPooling(ps.pooling_spec);
                break;
            }
            case OT_Random: {
                ret = createRandom(ps.random_spec);
                break;
            }
            case OT_ConvertColor: {
                ret = createConvertColor(dt, ps.convert_color_spec);
                break;
            }
            case OT_LutPreprocess: {
                ret = createLutPreprocess(dt);
                break;
            }
            case OT_Lut: {
                ret = createLut(dt, ps.lut_spec);
                break;
            }
#endif
            default: {
                UNI_ERROR_LOG(
                    "can not create layer %s type:%s.\n", name.c_str(), OperatorTypeName()[type]);
                break;
            }
        }
        return ret;
    }
};

#endif  // _FACTORY_H
