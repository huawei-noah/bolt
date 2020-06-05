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
#include "operator.hpp"
/*
#include "deconvolution.hpp"
#include "activation.hpp"
#include "scale.hpp"
#include "lstm.hpp"
#include "lstmcell.hpp"
#include "resize.hpp"
#include "attention.hpp"
#include "squeeze.hpp"
#include "unsqueeze.hpp"
#include "reduction.hpp"
#include "argmax.hpp"
#include "check.hpp"
#include "repeat.hpp"
#include "preallocated_memory.hpp"
#include "shared_weight.hpp"
#include "copy.hpp"
*/
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
#include "ocl/multiply_ocl.hpp"
#include "ocl/transpose_ocl.hpp"
#include "ocl/slice_ocl.hpp"

class FactoryOCL: public Factory {
public:
    virtual std::shared_ptr<Operator> createConvolution(DataType dt, U32 nf, U32 ksizeH, U32 ksizeW, U32 kstrideH, U32 kstrideW,
        U32 kpaddingT, U32 kpaddingB, U32 kpaddingL, U32 kpaddingR,
        ActivationDesc dwActivationDesc, ActivationDesc pwActivationDesc,
        ConvolutionMode convolutionType, U32 group, U32 dilateH, U32 dilateW) override {
        auto cep = (Convolution*)(new ConvolutionOCL(dt, nf, ksizeH, ksizeW, kstrideH, kstrideW,
                                                    kpaddingT, kpaddingB, kpaddingL, kpaddingR,
                                                    dwActivationDesc, pwActivationDesc,
                                                    convolutionType, group, dilateH, dilateW));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createDeconvolution(DataType dt, U32 nf, U32 ksizeH, U32 ksizeW, U32 kstrideH, U32 kstrideW,
        U32 kpaddingT, U32 kpaddingB, U32 kpaddingL, U32 kpaddingR,
        ActivationDesc dwActivationDesc, ActivationDesc pwActivationDesc,
        ConvolutionMode convolutionType, U32 group, U32 dilateH, U32 dilateW) override {
        /*
        //auto cep = new DeconvolutionOCL(dt, nf, ksizeH, ksizeW, kstrideH, kstrideW,
                                                    kpaddingT, kpaddingB, kpaddingL, kpaddingR,
                                                    dwActivationDesc, pwActivationDesc,
                                                    convolutionType, group, dilateH, dilateW);
        */
        OP_UNSUP(16, dt, nf, ksizeH, ksizeW, kstrideH, kstrideW,
            kpaddingT, kpaddingB, kpaddingL, kpaddingR,
            dwActivationDesc, pwActivationDesc,
            convolutionType, group, dilateH, dilateW);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createPooling(PoolingMode mode, U32 ksH, U32 ksW, U32 strideH, U32 strideW,
        U32 paddingT, U32 paddingB, U32 paddingL, U32 paddingR, RoundMode rm) override {
        auto cep = (Pooling*)(new PoolingOCL(mode, ksH, ksW, strideH, strideW, paddingT, paddingB, paddingL, paddingR, rm));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createFullyConnected(DataType dt, U32 numInput, U32 numOutput,
        U32 numSlice, I32* slicePoint) override {
        auto cep = (FullyConnectedOCL*)(new FullyConnectedOCL(dt,
            numInput, numOutput, numSlice, slicePoint));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createSoftmax(DataType dt, int axis) override {
        auto cep = new SoftmaxOCL(dt, axis);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createConcat(int axis) override {
        auto cep = (Concat*)(new ConcatOCL(axis));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createActivation(ActivationDesc activeDesc) override {
        auto cep = (Activation*) new ActivationOCL(activeDesc);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createEltwise(EltwiseMode eltMode, I32 coeffSize, F32* coeffValues) override {
        auto cep = (Eltwise*)new EltwiseOCL(eltMode, coeffSize, coeffValues);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createScale(DataType dt, int axis, int numChannels, int numSource) override {
        auto cep = (Scale*)(new ScaleOCL(dt, axis, numChannels, numSource));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createLSTM(DataType dt, U32 numOutput, U32 numProjection,
        F32 zoneoutCell, F32 zoneoutOutput, bool biDirection) override
    {
        OP_UNSUP(6, dt, numOutput, numProjection, zoneoutCell, zoneoutOutput, biDirection);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createLSTMCell(DataType dt, U32 numOutput, U32 numProjection,
        F32 zoneoutCell, F32 zoneoutOutput, bool biDirection) override
    {
        OP_UNSUP(6, dt, numOutput, numProjection, zoneoutCell, zoneoutOutput, biDirection);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createLSTM(DataType dt, U32 numOutput, U32 numProjection,
        F32 zoneoutCell, F32 zoneoutOutput, I32 steps) override {
        if (steps == -2)
            return FactoryOCL::createLSTM(dt, numOutput, numProjection, zoneoutCell, zoneoutOutput, true);
        if (steps >= 0)
            return FactoryOCL::createLSTM(dt, numOutput, numProjection, zoneoutCell, zoneoutOutput, false);
        else
            return FactoryOCL::createLSTMCell(dt, numOutput, numProjection, zoneoutCell, zoneoutOutput, false);
    }

    virtual std::shared_ptr<Operator> createEmbedding(DataType dt, U32 inputDim, U32 numOutput, bool transpose) override {   
        auto cep = (Embedding*)new EmbeddingOCL(dt, inputDim, numOutput, transpose);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createMultiply(DataType dt, F32 scale, F32 bias) override {
        auto cep = (Multiply*)new MultiplyOCL(dt, scale, bias);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createMatMul(DataType dt, bool transposeA, bool transposeB) override {
        auto cep = (MatMul*)(new MatMulOCL(dt, transposeA, transposeB));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createLayerNorm(DataType dt, U32 weightNum) override {
        auto cep = (LayerNorm*) new LayerNormOCL(dt, weightNum);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createReshape(DataType dt, I32* shapeDims, I32 shapeSize, I32 axis, I32 numAxes) override {
        auto cep = (Reshape*)(new ReshapeOCL(dt, shapeDims, shapeSize, axis, numAxes));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createResize(DataType paramDT, void* paramPtr) override {
        //auto cep = new Resize(paramDT, paramPtr);
        OP_UNSUP(2, paramDT, paramPtr);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createSlice(DataType dt, I32 axis, I32* slicePoints, U32 sliceSize) override {
        auto cep = (Slice*)(new SliceOCL(dt, axis, slicePoints, sliceSize));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createTranspose(DataType dt, U32* transDims, U32 transSize) override {
        auto cep = (Transpose*)(new TransposeOCL(dt, transDims, transSize));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createAttention(DataType dt, U32 numHeads, U32 fromSequenceLength,
        U32 toSequenceLength) override {
        //auto cep = new AttentionOCL(dt, numHeads, fromSequenceLength, toSequenceLength);
        OP_UNSUP(4, dt, numHeads, fromSequenceLength, toSequenceLength);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createClip(DataType dt, F32 clipMinScalar, F32 clipMaxScalar) override {
        auto cep = (Clip*)(new ClipOCL(dt, clipMinScalar, clipMaxScalar));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createSqueeze(DataType dt, I32 axis, I32 *dims, I32 dimSize) override {
        auto cep = (Squeeze*)(new SqueezeOCL(dt, axis, dims, dimSize));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createUnsqueeze(DataType dt, I32 axis, I32 *dims, I32 dimSize)
    override {
        //auto cep = new UnsqueezeOCL(dt, axis, dims, dimSize);
        OP_UNSUP(4, dt, axis, dims, dimSize);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createReduction(DataType dt, I32 axis, bool keepDim, ReductionMode reductionMode, float coeff) override {
        OP_UNSUP(5, dt, axis, keepDim, reductionMode, coeff);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createArgMax(DataType dt, I32 axis) override {
        //auto cep = new ArgMaxOCL(dt, axis);
        OP_UNSUP(2, dt, axis);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createCopy(DataType dt, I32 *srcDims, I32 *dstDims, I32 length) override {
        //auto cep = new CopyOCL(dt, srcDims, dstDims, length);
        OP_UNSUP(4, dt, srcDims, dstDims, length);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createCheck(DataType dt, CheckMode checkMode) override {
        //auto cep = new CheckOCL(dt, checkMode);
        OP_UNSUP(2, dt, checkMode);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createRepeat(DataType dt, I32 loops, I32 axis,
        I32 jumpOperatorIndex, I32 currentOperatorIndex) override {
        //auto cep = new RepeatOCL(dt, loops, jumpOperatorIndex, currentOperatorIndex);
        OP_UNSUP(5, dt, loops, axis, jumpOperatorIndex, currentOperatorIndex);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createBilateralSliceApply(U32 coefficiency_len, bool has_offset,
        BilateralSliceApplyMode mode) override {
        auto cep = (BilateralSliceApply*)(new BilateralSliceApplyOCL(coefficiency_len, has_offset, mode));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createPreAllocatedMemory(DataType dt, TensorDesc desc) override {
        //auto cep = new PreAllocatedMemoryOCL(dt, desc);
        OP_UNSUP(2, dt, desc)
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createSharedWeight(DataType dt, TensorDesc desc) override {
        //auto cep = new SharedWeightOCL(dt, desc);
        OP_UNSUP(2, dt, desc)
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createJump(DataType dt, I32 jumpOperatorIndex, I32 currentOperatorIndex) override {
        OP_UNSUP(3, dt, jumpOperatorIndex, currentOperatorIndex);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createSpace2Depth(DataType dt) override {
        auto cep = (Space2Depth*)(new Space2DepthOCL(dt));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createDepth2Space(DataType dt) override {
        auto cep = (Space2Depth*)(new Depth2SpaceOCL(dt));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createAttentionMask(DataType dt, I32 attentionLength,
        bool sameLength, float mask) override {
        OP_UNSUP(4, dt, attentionLength, sameLength, mask)
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createRelativePositionEmbedding(DataType dt, U32 inputDim,
        U32 numOutput, bool transpose, I32 axis) override {
        OP_UNSUP(5, dt, inputDim, numOutput, transpose, axis)
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createRelativeShift(DataType dt, I32 axis,
        I32 shiftLength) override {
        OP_UNSUP(3, dt, axis, shiftLength)
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createPadding(DataType dt, PadDesc padDesc) override {
        OP_UNSUP(2, dt, padDesc)
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createPriorBox(DataType dt, PriorBoxDesc priorboxDesc) override {
        OP_UNSUP(2, dt, priorboxDesc);
        return std::shared_ptr<Operator>(cep);
    }
    virtual std::shared_ptr<Operator> createDetectionOutput(DataType dt, DetectionOutputDesc detectionoutputDesc) override {
        OP_UNSUP(2, dt, detectionoutputDesc);
        return std::shared_ptr<Operator>(cep);
    }
};

#endif //_FACTORY_OCL_H
