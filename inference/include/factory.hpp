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
#define NOT_SUPPORT Operator* cep = NULL;CHECK_STATUS(NOT_SUPPORTED);
#define NOT_USE1(a1)                             {UNUSED(a1);}
#define NOT_USE2(a1, a2)                         {NOT_USE1(a1)             NOT_USE1(a2)}
#define NOT_USE3(a1, a2, a3)                     {NOT_USE2(a1, a2)         NOT_USE1(a3)}
#define NOT_USE4(a1, a2, a3, a4)                 {NOT_USE2(a1, a2)         NOT_USE2(a3, a4)}
#define NOT_USE5(a1, a2, a3, a4, a5)             {NOT_USE4(a1, a2, a3, a4) NOT_USE1(a5)}
#define NOT_USE6(a1, a2, a3, a4, a5, a6)         {NOT_USE4(a1, a2, a3, a4) NOT_USE2(a5, a6)}
#define NOT_USE8(a1, a2, a3, a4, a5, a6, a7, a8) {NOT_USE4(a1, a2, a3, a4) NOT_USE4(a5, a6, a7, a8)}
#define NOT_USE16(a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, ag) {NOT_USE8(a1, a2, a3, a4, a5, a6, a7, a8) NOT_USE8(a9, aa, ab, ac, ad, ae, af, ag)}
#define OP_UNSUP(num,...) NOT_USE##num(__VA_ARGS__) NOT_SUPPORT

class Factory {
public:
    virtual ~Factory(){};
    virtual std::shared_ptr<Operator> createConvolution(DataType dt, U32 nf, U32 ksizeH, U32 ksizeW, U32 kstrideH, U32 kstrideW,
        U32 kpaddingT, U32 kpaddingB, U32 kpaddingL, U32 kpaddingR,
        ActivationDesc dwActivationDesc, ActivationDesc pwActivationDesc,
        ConvolutionMode convolutionType, U32 group, U32 dilateH, U32 dilateW) = 0;

    virtual std::shared_ptr<Operator> createDeconvolution(DataType dt, U32 nf, U32 ksizeH, U32 ksizeW, U32 kstrideH, U32 kstrideW,
        U32 kpaddingT, U32 kpaddingB, U32 kpaddingL, U32 kpaddingR,
        ActivationDesc dwActivationDesc, ActivationDesc pwActivationDesc,
        ConvolutionMode convolutionType, U32 group, U32 dilateH, U32 dilateW) = 0;

    virtual std::shared_ptr<Operator> createPooling(PoolingMode mode, U32 ksH, U32 ksW, U32 strideH, U32 strideW,
        U32 paddingT, U32 paddingB, U32 paddingL, U32 paddingR, RoundMode rm) = 0;

    virtual std::shared_ptr<Operator> createFullyConnected(DataType dt, U32 numInput, U32 numOutput,
        U32 numSlice, I32* slicePoint) = 0;

    virtual std::shared_ptr<Operator> createSoftmax(DataType dt, int axis) = 0;

    virtual std::shared_ptr<Operator> createConcat(int axis) = 0;

    virtual std::shared_ptr<Operator> createActivation(ActivationDesc activationDesc) = 0;

    virtual std::shared_ptr<Operator> createEltwise(EltwiseMode eltMode, I32 coeffSize, F32* coeffValues) = 0;

    virtual std::shared_ptr<Operator> createScale(DataType dt, int axis, int numChannels, int numSource) = 0;

    virtual std::shared_ptr<Operator> createLSTM(DataType dt, U32 numOutput, U32 numProjection, F32 zoneoutCell, F32 zoneoutOutput, bool biDirection) = 0;

    virtual std::shared_ptr<Operator> createLSTMCell(DataType dt, U32 numOutput, U32 numProjection, F32 zoneoutCell, F32 zoneoutOutput, bool biDirection) = 0;

    virtual std::shared_ptr<Operator> createLSTM(DataType dt, U32 numOutput, U32 numProjection, F32 zoneoutCell, F32 zoneoutOutput, I32 steps) = 0;

    virtual std::shared_ptr<Operator> createEmbedding(DataType dt, U32 inputDim, U32 numOutput, bool transpose) = 0;

    virtual std::shared_ptr<Operator> createMultiply(DataType dt, F32 scale, F32 bias) = 0;

    virtual std::shared_ptr<Operator> createMatMul(DataType dt, bool transposeA, bool transposeB) = 0;

    virtual std::shared_ptr<Operator> createLayerNorm(DataType dt, U32 weightNum) = 0;

    virtual std::shared_ptr<Operator> createReshape(DataType dt, I32* shapeDims, I32 shapeSize, I32 axis, I32 numAxes) = 0;

    virtual std::shared_ptr<Operator> createResize(DataType paramDT, void* paramPtr) = 0;

    virtual std::shared_ptr<Operator> createSlice(DataType dt, I32 axis, I32* slicePoints, U32 sliceSize) = 0;

    virtual std::shared_ptr<Operator> createTranspose(DataType dt, U32* transDims, U32 transSize) = 0;

    virtual std::shared_ptr<Operator> createAttention(DataType dt, U32 numHeads, U32 fromSequenceLength, U32 toSequenceLength) = 0;

    virtual std::shared_ptr<Operator> createClip(DataType dt, F32 clipMinScalar, F32 clipMaxScalar) = 0;

    virtual std::shared_ptr<Operator> createSqueeze(DataType dt, I32 axis, I32 *dims, I32 dimSize) = 0;

    virtual std::shared_ptr<Operator> createUnsqueeze(DataType dt, I32 axis, I32 *dims, I32 dimSize) = 0;

    virtual std::shared_ptr<Operator> createReduction(DataType dt, I32 axis, bool keepDim, ReductionMode reductionMode, float coeff) = 0;

    virtual std::shared_ptr<Operator> createArgMax(DataType dt, I32 axis) = 0;

    virtual std::shared_ptr<Operator> createCopy(DataType dt, I32 *srcDims, I32 *dstDims, I32 length) = 0;

    virtual std::shared_ptr<Operator> createCheck(DataType dt, CheckMode checkMode) = 0;

    virtual std::shared_ptr<Operator> createRepeat(DataType dt, I32 loops, I32 axis,
        I32 jumpOperatorIndex, I32 currentOperatorIndex) = 0;

    virtual std::shared_ptr<Operator> createBilateralSliceApply(U32 coefficiency_len,
        bool has_offset, BilateralSliceApplyMode mode) = 0;

    virtual std::shared_ptr<Operator> createPreAllocatedMemory(DataType dt, TensorDesc desc) = 0;

    virtual std::shared_ptr<Operator> createSharedWeight(DataType dt, TensorDesc desc) = 0;

    virtual std::shared_ptr<Operator> createJump(DataType dt, I32 jumpOperatorIndex, I32 currentOperatorIndex) = 0;

    virtual std::shared_ptr<Operator> createSpace2Depth(DataType dt) = 0;

    virtual std::shared_ptr<Operator> createDepth2Space(DataType dt) = 0;

    virtual std::shared_ptr<Operator> createAttentionMask(DataType dt, I32 attentionLength,
        bool sameLength, float mask) = 0;

    virtual std::shared_ptr<Operator> createRelativePositionEmbedding(DataType dt, U32 inputDim,
        U32 numOutput, bool transpose, I32 axis) = 0;

    virtual std::shared_ptr<Operator> createRelativeShift(DataType dt, I32 axis,
        I32 shiftLength) = 0;

    virtual std::shared_ptr<Operator> createPadding(DataType dt, PadDesc padDesc) = 0;

    virtual std::shared_ptr<Operator> createPriorBox(DataType dt, PriorBoxDesc priorboxDesc) = 0;

    virtual std::shared_ptr<Operator> createDetectionOutput(DataType dt, DetectionOutputDesc detectionoutputDesc) = 0;

    std::shared_ptr<Operator> createOperators(OperatorSpec curOps, DataType dt, HashMap<std::string, U32> operatorIndexMap, Vec<std::string> inputTensorsName) {
        OperatorType opType = curOps.type;
        DataType dtNoQ = (dt == DT_F16_8Q) ? DT_F16 : dt;
        std::string opName = curOps.name;
        std::shared_ptr<Operator> op;
        switch (opType) {
            case OT_Conv: {
                ConvolutionParamSpec curConvParamSpec = curOps.ps.conv_spec;
                U32 nf = curConvParamSpec.num_outputs;
                U32 ksizeH = curConvParamSpec.kernel_size_h;
                U32 ksizeW = curConvParamSpec.kernel_size_w;
                ConvolutionMode curConvolutionType = curConvParamSpec.convolution_type;
                U32 group = curConvParamSpec.group;
                U32 dilateH = curConvParamSpec.dilatedRate_h;
                U32 dilateW = curConvParamSpec.dilatedRate_w;
                U32 kstrideH = curConvParamSpec.stride_h;
                U32 kstrideW = curConvParamSpec.stride_w;
                U32 paddingT = curConvParamSpec.padding_top;
                U32 paddingB = curConvParamSpec.padding_bottom;
                U32 paddingL = curConvParamSpec.padding_left;
                U32 paddingR = curConvParamSpec.padding_right;
                ActivationDesc dwActiveDesc;
                ActivationDesc pwActiveDesc;
                dwActiveDesc.mode = curConvParamSpec.dw_activation_type;
                pwActiveDesc.mode = curConvParamSpec.pw_activation_type;
                dwActiveDesc.value[0] = 0;
                pwActiveDesc.value[0] = 0;
                op = createConvolution(dt, nf, ksizeH, ksizeW, kstrideH, kstrideW,
                     paddingT, paddingB, paddingL, paddingR,
                     dwActiveDesc, pwActiveDesc, curConvolutionType, group, dilateH, dilateW);
                break;
            }
            case OT_Deconvolution: {
                ConvolutionParamSpec curConvParamSpec = curOps.ps.conv_spec;
                U32 nf = curConvParamSpec.num_outputs;
                U32 ksizeH = curConvParamSpec.kernel_size_h;
                U32 ksizeW = curConvParamSpec.kernel_size_w;
                ConvolutionMode curConvolutionType = curConvParamSpec.convolution_type;
                U32 group = curConvParamSpec.group;
                U32 dilateH = curConvParamSpec.dilatedRate_h;
                U32 dilateW = curConvParamSpec.dilatedRate_w;
                U32 kstrideH = curConvParamSpec.stride_h;
                U32 kstrideW = curConvParamSpec.stride_w;
                U32 paddingT = curConvParamSpec.padding_top;
                U32 paddingB = curConvParamSpec.padding_bottom;
                U32 paddingL = curConvParamSpec.padding_left;
                U32 paddingR = curConvParamSpec.padding_right;
                ActivationDesc dwActiveDesc;
                ActivationDesc pwActiveDesc;
                dwActiveDesc.mode = curConvParamSpec.dw_activation_type;
                pwActiveDesc.mode = curConvParamSpec.pw_activation_type;
                dwActiveDesc.value[0] = 0;
                pwActiveDesc.value[0] = 0;
                op = createDeconvolution(dtNoQ, nf, ksizeH, ksizeW, kstrideH, kstrideW,
                     paddingT, paddingB, paddingL, paddingR,
                     dwActiveDesc, pwActiveDesc, curConvolutionType, group, dilateH, dilateW);
                break;
            }
            case OT_FC: {
                FullyConnectedParamSpec curFcParamSpec = curOps.ps.fc_spec;
                I32 curNumOutput = curFcParamSpec.num_outputs;
                I32 curNumSlice = curFcParamSpec.num_slices;
                op = createFullyConnected(dt, 0, curNumOutput, curNumSlice, curFcParamSpec.slice_point);
                break;
            }
            case OT_Pooling: {
                PoolingParamSpec curPoolingParamSpec = curOps.ps.pooling_spec;
                PoolingMode mode = curPoolingParamSpec.mode;
                U32 ksH = curPoolingParamSpec.kernel_size_h;
                U32 ksW = curPoolingParamSpec.kernel_size_w;
                U32 strideH = curPoolingParamSpec.stride_h;
                U32 strideW = curPoolingParamSpec.stride_w;
                U32 paddingT = curPoolingParamSpec.padding_top;
                U32 paddingB = curPoolingParamSpec.padding_bottom;
                U32 paddingL = curPoolingParamSpec.padding_left;
                U32 paddingR = curPoolingParamSpec.padding_right;
                RoundMode rm = curPoolingParamSpec.rm;
                op = createPooling(mode, ksH, ksW, strideH, strideW, paddingT, paddingB, paddingL, paddingR, rm);
                break;
            }
            case OT_Softmax: {
                SoftmaxParamSpec curSoftmaxParamSpec = curOps.ps.softmax_spec;
                I32 axis = curSoftmaxParamSpec.axis;
                op = createSoftmax(dtNoQ, axis);
                break;
            }
            case OT_Relu: {
                ActivationDesc activationDesc;
                activationDesc.mode = ACTIVATION_RELU;
                activationDesc.value[0] = curOps.ps.relu_spec.neg_slope;
                op = createActivation(activationDesc);
                break;
            }
            case OT_Relu6: {
                ActivationDesc activationDesc;
                activationDesc.mode = ACTIVATION_RELU6;
                op = createActivation(activationDesc);
                break;
            }
            case OT_HSwish: {
                ActivationDesc activationDesc;
                activationDesc.mode = ACTIVATION_H_SWISH;
                op = createActivation(activationDesc);
                break;
            }
            case OT_Sigmoid: {
                ActivationDesc activationDesc;
                activationDesc.mode = ACTIVATION_SIGMOID;
                op = createActivation(activationDesc);
                break;
            }
            case OT_HSigmoid: {
                ActivationDesc activationDesc;
                activationDesc.mode = ACTIVATION_H_SIGMOID;
                op = createActivation(activationDesc);
                break;
            }
            case OT_Gelu: {
                ActivationDesc activationDesc;
                activationDesc.mode = ACTIVATION_GELU;
                op = createActivation(activationDesc);
                break;
            }
            case OT_TanH: {
                ActivationDesc activationDesc;
                activationDesc.mode = ACTIVATION_TANH;
                op = createActivation(activationDesc);
                break;
            }
            case OT_Concat: {
                ConcatParamSpec curConcatParamSpec = curOps.ps.concat_spec;
                I32 axis = curConcatParamSpec.axis;
                op = createConcat(axis);
                break;
            }
            case OT_Eltwise: {
                EltwiseParamSpec curEltwiseParamSpec = curOps.ps.eltwise_spec;
                EltwiseMode curEltMode = curEltwiseParamSpec.elt_mode;
                EltwiseSumSpec curEltSumSpec = curEltwiseParamSpec.elt_sum_spec;
                op = createEltwise(curEltMode, curEltSumSpec.coeff_size, curEltSumSpec.coeff_values);
                break;
            }
            case OT_Embedding: {
                EmbedParamSpec curEmbedParamSpec = curOps.ps.embed_spec;
                U32 curInputDim = curEmbedParamSpec.input_dim;
                U32 curNumOutput = curEmbedParamSpec.num_output;
                bool curTranspose = curEmbedParamSpec.transpose;
                op = createEmbedding(dtNoQ, curInputDim, curNumOutput, curTranspose);
                break;
            }
            case OT_MatMul: {
                MatMulParamSpec curMatMulParamSpec = curOps.ps.matmul_spec;
                bool transposeA = curMatMulParamSpec.transpose_a;
                bool transposeB = curMatMulParamSpec.transpose_b;
                op = createMatMul(dt, transposeA, transposeB);
                break;
            }
            case OT_Multiply: {
                MultiplyParamSpec curMultiplyParamSpec = curOps.ps.multiply_spec;
                F32 scale = curMultiplyParamSpec.scale;
                F32 bias = curMultiplyParamSpec.bias;
                op = createMultiply(dt, scale, bias);
                break;
            }
            case OT_Scale: {
                ScaleParamSpec curScaleParamSpec = curOps.ps.scale_spec;
                I32 num = curScaleParamSpec.num_concat;
                I32 axis = curScaleParamSpec.axis;
                op = createScale(dtNoQ, axis, 0, num);
                break;
            }
            case OT_LayerNorm: {
                op = createLayerNorm(dt, 0);
                break;
            }
            case OT_Reshape: {
                ReshapeParamSpec curReshapeParamSpec = curOps.ps.reshape_spec;
                I32* curShapeDims = curReshapeParamSpec.shape_dims;
                I32 curShapeSize = curReshapeParamSpec.shape_size;
                I32 curAxis = curReshapeParamSpec.axis;
                I32 curNumAxes = curReshapeParamSpec.num_axes;
                op = createReshape(dt, curShapeDims, curShapeSize, curAxis, curNumAxes);
                break;
            }
            case OT_Upsample: {
                UpsampleParamSpec curUpsampleParamSpec = curOps.ps.upsample_spec;
                F32* paramPtr = curUpsampleParamSpec.scale;
                op = createResize(DT_F32, paramPtr);
                break;
            }
            case OT_Interp: {
                InterpParamSpec curInterpParamSpec = curOps.ps.interp_spec;
                U32 size[2];
                size[0] = curInterpParamSpec.height;
                size[1] = curInterpParamSpec.width;
                op = createResize(DT_U32, size);
                break;
            }
            case OT_Slice: {
                SliceParamSpec curSliceParamSpec = curOps.ps.slice_spec;
                I32 curAxis = curSliceParamSpec.axis;
                I32* curSlicePoints = curSliceParamSpec.slice_points;
                I32 curSliceSize = curSliceParamSpec.slice_size;
                op = createSlice(dt, curAxis, curSlicePoints, curSliceSize);
                break;
            }
            case OT_Transpose: {
                TransposeParamSpec curTransposeSpec = curOps.ps.transpose_spec;
                U32* curTransDimsPtr = curTransposeSpec.trans_dims;
                U32 curTransSize = curTransposeSpec.trans_size;
                op = createTranspose(dt, curTransDimsPtr, curTransSize);
                break;
            }
            case OT_Attention: {
                AttentionParamSpec curAttentionSpec = curOps.ps.attention_spec;
                U32 numHeads = curAttentionSpec.num_heads;
                U32 fromSequenceLength = curAttentionSpec.from_sequence_length;
                U32 toSequenceLength = curAttentionSpec.to_sequence_length;
                op = createAttention(dtNoQ, numHeads, fromSequenceLength, toSequenceLength);
                break;
            }
            case OT_Clip: {
                ClipParamSpec curClipSpec = curOps.ps.clip_spec;
                F32 curClipMinScalar = curClipSpec.min;
                F32 curClipMaxScalar = curClipSpec.max;
                op = createClip(dtNoQ, curClipMinScalar, curClipMaxScalar);
                break;
            }
            case OT_LSTM: {
                LSTMParamSpec curLSTMParamSpec = curOps.ps.lstm_spec;
                U32 numOutput = curLSTMParamSpec.num_output;
                U32 numProjection = curLSTMParamSpec.num_projection;
                F32 zoneoutCell = curLSTMParamSpec.zoneout_cell;
                F32 zoneoutOutput = curLSTMParamSpec.zoneout_output;
                I32 steps = curLSTMParamSpec.steps;
                op = createLSTM(dtNoQ, numOutput, numProjection, zoneoutCell, zoneoutOutput, steps);
                break;
            }
            case OT_Squeeze: {
                SqueezeParamSpec curSqueezeParamSpec = curOps.ps.squeeze_spec;
                I32 axis = curSqueezeParamSpec.axis;
                I32 *squeezeAxes = curSqueezeParamSpec.squeeze_axes;
                I32 numAxes = curSqueezeParamSpec.axes_num;
                op = createSqueeze(dtNoQ, axis, squeezeAxes, numAxes);
                break;
            }
            case OT_Unsqueeze: {
                UnsqueezeParamSpec curUnsqueezeParamSpec = curOps.ps.unsqueeze_spec;
                I32 axis = curUnsqueezeParamSpec.axis;
                I32 *unsqueezeAxes = curUnsqueezeParamSpec.unsqueeze_axes;
                I32 numAxes = curUnsqueezeParamSpec.axes_num;
                op = createUnsqueeze(dtNoQ, axis, unsqueezeAxes, numAxes);
                break;
            }
            case OT_Reduction: {
                ReductionParamSpec curReductionParamSpec = curOps.ps.reduction_spec;
                I32 axis = curReductionParamSpec.axis;
                bool keepDim = curReductionParamSpec.keep_dim;
                ReductionMode reductionMode = curReductionParamSpec.reduction_mode;
                float coeff = curReductionParamSpec.coeff;
                op = createReduction(dtNoQ, axis, keepDim, reductionMode, coeff);
                break;
            }
            case OT_ArgMax: {
                ArgMaxParamSpec curArgMaxParamSpec = curOps.ps.argmax_spec;
                I32 axis = curArgMaxParamSpec.axis;
                op = createArgMax(dtNoQ, axis);
                break;
            }
            case OT_PreAllocatedMemory: {
                PreAllocatedMemoryParamSpec curPreAllocatedMemoryParamSpec = curOps.ps.preallocated_memory_spec;
                TensorDesc desc= curPreAllocatedMemoryParamSpec.desc;
                op = createPreAllocatedMemory(dtNoQ, desc);
                break;
            }
            case OT_SharedWeight: {
                SharedWeightParamSpec curSharedWeightParamSpec = curOps.ps.shared_weight_spec;
                TensorDesc desc = curSharedWeightParamSpec.desc;
                op = createSharedWeight(dtNoQ, desc);
                break;
            }
            case OT_Repeat: {
                RepeatParamSpec curRepeatParamSpec = curOps.ps.repeat_spec;
                I32 loops = curRepeatParamSpec.loops;
                I32 axis = curRepeatParamSpec.axis;
                op = createRepeat(dtNoQ, loops, axis, operatorIndexMap[inputTensorsName[0]], operatorIndexMap[opName]);
                break;
            }
            case OT_Check: {
                CheckParamSpec curCheckParamSpec = curOps.ps.check_spec;
                CheckMode checkMode = curCheckParamSpec.check_mode;
                op = createCheck(dtNoQ, checkMode);
                break;
            }
            case OT_Copy: {
                CopyParamSpec curCopyParamSpec = curOps.ps.copy_spec;
                I32 *srcDims = curCopyParamSpec.src_dims;
                I32 *dstDims = curCopyParamSpec.dst_dims;
                I32 length = curCopyParamSpec.length;
                op = createCopy(dtNoQ, srcDims, dstDims, length);
                break;
            }
            case OT_BilateralSliceApply: {
                BilateralSliceApplyParamSpec curBilateralSliceApplyParamSpec = curOps.ps.bilateral_slice_apply_spec;
                U32 coefficient_len = curBilateralSliceApplyParamSpec.coefficient_len;
                bool has_offset = curBilateralSliceApplyParamSpec.has_offset;
                BilateralSliceApplyMode mode = curBilateralSliceApplyParamSpec.mode;
                op = createBilateralSliceApply(coefficient_len, has_offset, mode);
                break;
            }
            case OT_Jump: {
                op = createJump(dtNoQ, operatorIndexMap[inputTensorsName[0]], operatorIndexMap[opName]);
                break;
            }
            case OT_Space2Depth: {
                op = createSpace2Depth(dt);
                break;
            }
            case OT_Depth2Space: {
                op = createDepth2Space(dt);
                break;
            }
            case OT_AttentionMask: {
                AttentionMaskParamSpec curAttentionMaskParamSpec = curOps.ps.attention_mask_spec;
                I32 attention_length = curAttentionMaskParamSpec.attention_length;
                bool same_length = curAttentionMaskParamSpec.same_length;
                float mask = curAttentionMaskParamSpec.mask;
                op = createAttentionMask(dt, attention_length, same_length, mask);
                break;
            }
            case OT_RelativePositionEmbedding: {
                RelativePositionEmbedParamSpec curRelativePositionEmbedParamSpec = curOps.ps.relative_position_embed_spec;
                U32 curInputDim = curRelativePositionEmbedParamSpec.input_dim;
                U32 curNumOutput = curRelativePositionEmbedParamSpec.num_output;
                bool curTranspose = curRelativePositionEmbedParamSpec.transpose;
                I32 axis = curRelativePositionEmbedParamSpec.axis;
                op = createRelativePositionEmbedding(dtNoQ, curInputDim, curNumOutput, curTranspose, axis);
                break;
            }
            case OT_RelativeShift: {
                RelativeShiftParamSpec curRelativeShiftParamSpec = curOps.ps.relative_shift_spec;
                I32 axis = curRelativeShiftParamSpec.axis;
                I32 shift_length = curRelativeShiftParamSpec.shift_length;
                op = createRelativeShift(dt, axis, shift_length);
                break;
            }
            case OT_Pad: {
                PadParamSpec curPadParamSpec = curOps.ps.pad_spec;
                PadDesc padDesc;
                padDesc.top = curPadParamSpec.top;
                padDesc.bottom = curPadParamSpec.bottom;
                padDesc.left = curPadParamSpec.left;
                padDesc.right = curPadParamSpec.right;
                padDesc.constant_value = curPadParamSpec.constant_value;
                padDesc.pad_mode = curPadParamSpec.pad_mode;
                op = createPadding(dt, padDesc);
                break;
            }
            case OT_PriorBox: {
                PriorBoxParamSpec curPriorBoxParamSpec = curOps.ps.prior_box_spec;
                PriorBoxDesc priorboxDesc;
                for(int i = 0; i < 2; i++ ){
                    if(curPriorBoxParamSpec.min_sizes[i] == 0)
                        break;
                    priorboxDesc.min_sizes.push_back(curPriorBoxParamSpec.min_sizes[i]);
                }
                for(int i = 0; i < 2; i++ ){
                    if(curPriorBoxParamSpec.max_sizes[i] == 0)
                        break;
                    priorboxDesc.max_sizes.push_back(curPriorBoxParamSpec.max_sizes[i]);
                }
                for(int i = 0; i < 2; i++ ){
                    if(curPriorBoxParamSpec.aspect_ratios[i] == 0)
                        break;
                    priorboxDesc.aspect_ratios.push_back(curPriorBoxParamSpec.aspect_ratios[i]);
                }
                priorboxDesc.flip = curPriorBoxParamSpec.flip;
                priorboxDesc.clip = curPriorBoxParamSpec.clip;
                for(int i = 0; i < 4; i++){
                    priorboxDesc.variances[i] = curPriorBoxParamSpec.variances[i];
                }
                priorboxDesc.image_h = curPriorBoxParamSpec.image_h; 
                priorboxDesc.image_w = curPriorBoxParamSpec.image_w; 
                priorboxDesc.step_h = curPriorBoxParamSpec.step_h;
                priorboxDesc.step_w = curPriorBoxParamSpec.step_w;
                priorboxDesc.offset = curPriorBoxParamSpec.offset;
                op = createPriorBox(dt, priorboxDesc);
                break;
            }
            case OT_DetectionOutput: {
                DetectionOutputParamSpec curDetectionoutputParamSpec = curOps.ps.detection_output_spec;
                DetectionOutputDesc detectionoutputDesc;
                detectionoutputDesc.num_class = curDetectionoutputParamSpec.num_class;
                detectionoutputDesc.nms_threshold = curDetectionoutputParamSpec.nms_threshold;
                detectionoutputDesc.nms_top_k = curDetectionoutputParamSpec.nms_top_k;
                detectionoutputDesc.keep_top_k = curDetectionoutputParamSpec.keep_top_k;
                detectionoutputDesc.confidence_threshold = curDetectionoutputParamSpec.confidence_threshold;
                op = createDetectionOutput(dt, detectionoutputDesc);
                break;
            }            
            default: {
                std::cerr << "[ERROR] unsupported layer " << OperatorTypeName()[opType] << std::endl;
                exit(1);
                break;
            }
        }
        return op;
    }
};

#endif //_FACTORY_H
