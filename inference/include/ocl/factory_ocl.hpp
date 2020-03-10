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
#include <optional>
#include "operator.hpp"
/*
#include "deconvolution.hpp"
#include "activation.hpp"
#include "scale.hpp"
#include "embedding.hpp"
#include "lstm.hpp"
#include "lstmcell.hpp"
#include "matmul.hpp"
#include "multiply.hpp"
#include "layer_norm.hpp"
#include "reshape.hpp"
#include "resize.hpp"
#include "slice.hpp"
#include "transpose.hpp"
#include "attention.hpp"
#include "clip.hpp"
#include "squeeze.hpp"
#include "unsqueeze.hpp"
#include "axis_mean.hpp"
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
#include "ocl/fully_connected_eltwise_ocl.hpp"
#include "ocl/scale_ocl.hpp"
#include "ocl/concat_ocl.hpp"

class FactoryOCL: public Factory {
public:
    virtual std::shared_ptr<Operator> createConvolution(DataType dt, U32 nf, U32 ksizeH, U32 ksizeW, U32 kstrideH, U32 kstrideW,
        U32 kpaddingT, U32 kpaddingB, U32 kpaddingL, U32 kpaddingR,
        ActivationMode dwActiveMode, ActivationMode pwActiveMode,
        ConvolutionMode convolutionType, U32 group, U32 dilateH, U32 dilateW) override {
        auto cep = (Convolution*)(new ConvolutionOCL(dt, nf, ksizeH, ksizeW, kstrideH, kstrideW,
                                                    kpaddingT, kpaddingB, kpaddingL, kpaddingR,
                                                    dwActiveMode, pwActiveMode,
                                                    convolutionType, group, dilateH, dilateW));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createDeconvolution(DataType dt, U32 nf, U32 ksizeH, U32 ksizeW, U32 kstrideH, U32 kstrideW,
        U32 kpaddingT, U32 kpaddingB, U32 kpaddingL, U32 kpaddingR,
        ActivationMode dwActiveMode, ActivationMode pwActiveMode,
        ConvolutionMode convolutionType, U32 group, U32 dilateH, U32 dilateW) override {
        /*
        //auto cep = new DeconvolutionOCL(dt, nf, ksizeH, ksizeW, kstrideH, kstrideW,
                                                    kpaddingT, kpaddingB, kpaddingL, kpaddingR,
                                                    dwActiveMode, pwActiveMode,
                                                    convolutionType, group, dilateH, dilateW);
        */
        
        OP_UNSUP(16, dt, nf, ksizeH, ksizeW, kstrideH, kstrideW,
            kpaddingT, kpaddingB, kpaddingL, kpaddingR,
            dwActiveMode, pwActiveMode,
            convolutionType, group, dilateH, dilateW);
        
        
        
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createPooling(PoolingMode mode, U32 ksH, U32 ksW, U32 strideH, U32 strideW,
        U32 paddingT, U32 paddingB, U32 paddingL, U32 paddingR, RoundMode rm) override {
        auto cep = (Pooling*)(new PoolingOCL(mode, ksH, ksW, strideH, strideW, paddingT, paddingB, paddingL, paddingR, rm));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createFullyConnectedEltwise(DataType dt, U32 numInput, U32 numOutput, std::optional<EltwiseType> eltwiseType) override {
        auto cep = (FullyConnectedEltwiseOCL*)(new FullyConnectedEltwiseOCL(dt, numInput, numOutput, eltwiseType));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createFullyConnectedEltwise(DataType dt, U32 numInput, U32 numOutput) override {
        std::optional<EltwiseType> etNull;
        return FactoryOCL::createFullyConnectedEltwise(dt, numInput, numOutput, etNull);
    }

    virtual std::shared_ptr<Operator> createSoftmax(DataType dt) override {
        auto cep = new SoftmaxOCL(dt);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createConcat(U32 concatDim) override {
        auto cep = (Concat*)(new ConcatOCL(concatDim));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createActivation(ActivationMode activeMode) override {
        auto cep = (Activation*) new ActivationOCL(activeMode);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createEltwise(EltwiseMode eltMode, I32 coeffSize, F32* coeffValues) override {
        auto cep = (Eltwise*)new EltwiseOCL(eltMode, coeffSize, coeffValues);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createScale(DataType dt, int numChannels, int numSource) override {
        auto cep = (Scale*)(new ScaleOCL(dt, numChannels, numSource));
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createLSTM(DataType dt, U32 numOutput, std::optional<EltwiseType> eltwiseType) override {
        //auto cep = new LSTMOCL(dt, numOutput, eltwiseType);
        OP_UNSUP(3, dt, numOutput, eltwiseType);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createLSTM(DataType dt, U32 numOutput) override {
        std::optional<EltwiseType> etNull;
        return FactoryOCL::createLSTM(dt, numOutput, etNull);
    }

    virtual std::shared_ptr<Operator> createLSTMCell(DataType dt, U32 numOutput, std::optional<EltwiseType> eltwiseType) override {
        //auto cep = new LSTMCellOCL(dt, numOutput, eltwiseType);
        OP_UNSUP(3, dt, numOutput, eltwiseType);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createLSTMCell(DataType dt, U32 numOutput) override {
        std::optional<EltwiseType> etNull;
        return FactoryOCL::createLSTMCell(dt, numOutput, etNull);
    }

    virtual std::shared_ptr<Operator> createLSTM(DataType dt, U32 numOutput, I32 steps) override {
        std::optional<EltwiseType> etNull;
        if (steps >= 0)
            return FactoryOCL::createLSTM(dt, numOutput, etNull);
        else
            return FactoryOCL::createLSTMCell(dt, numOutput, etNull);
    }

    virtual std::shared_ptr<Operator> createEmbedding(DataType dt, U32 inputDim, U32 numOutput, bool transpose) override {   
        //auto cep = new EmbeddingOCL(dt, inputDim, numOutput, transpose);
        OP_UNSUP(4, dt, inputDim, numOutput, transpose);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createMultiply(DataType dt, F32 scale, F32 bias) override {
        //auto cep = new MultiplyOCL(dt, scale, bias);
        OP_UNSUP(3, dt, scale, bias);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createMatMul(DataType dt, bool transposeA, bool transposeB) override {
        //auto cep = new MatMulOCL(dt, transposeA, transposeB);
        OP_UNSUP(3, dt, transposeA, transposeB);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createLayerNorm(DataType dt, U32 weightNum) override {
        //auto cep = new LayerNormOCL(dt, weightNum);
        OP_UNSUP(2, dt, weightNum);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createReshape(DataType dt, I32* shapeDims, I32 shapeSize, I32 axis, I32 numAxes) override {
        //auto cep = new ReshapeOCL(dt, shapeDims, shapeSize, axis, numAxes);
        OP_UNSUP(5, dt, shapeDims, shapeSize, axis, numAxes);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createResize(DataType paramDT, void* paramPtr) override {
        //auto cep = new Resize(paramDT, paramPtr);
        OP_UNSUP(2, paramDT, paramPtr);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createSlice(DataType dt, U32 axis, U32* slicePoints, U32 sliceSize) override {
        //auto cep = new SliceOCL(dt, axis, slicePoints, sliceSize);
        OP_UNSUP(4, dt, axis, slicePoints, sliceSize);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createTranspose(DataType dt, U32* transDims, U32 transSize) override {
        //auto cep = new TransposeOCL(dt, transDims, transSize);
        OP_UNSUP(3, dt, transDims, transSize);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createAttention(DataType dt, U32 numHeads, U32 fromSequenceLength, U32 toSequenceLength) override {
        //auto cep = new AttentionOCL(dt, numHeads, fromSequenceLength, toSequenceLength);
        OP_UNSUP(4, dt, numHeads, fromSequenceLength, toSequenceLength);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createClip(DataType dt, F32 clipMinScalar, F32 clipMaxScalar) override {
        //auto cep = new ClipOCL(dt, clipMinScalar, clipMaxScalar);
        OP_UNSUP(3, dt, clipMinScalar, clipMaxScalar);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createSqueeze(DataType dt, I32 axis, I32 *dims, I32 dimSize) override {
        //auto cep = new SqueezeOCL(dt, axis, dims, dimSize);
        OP_UNSUP(4, dt, axis, dims, dimSize)
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createUnsqueeze(DataType dt, I32 axis, I32 *dims, I32 dimSize)
    override {
        //auto cep = new UnsqueezeOCL(dt, axis, dims, dimSize);
        OP_UNSUP(4, dt, axis, dims, dimSize);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createAxisMean(DataType dt, I32 axis) override {
        //auto cep = new AxisMeanOCL(dt, axis);
        OP_UNSUP(2, dt, axis);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createArgMax(DataType dt, I32 axis) override {
        //auto cep = new ArgMaxOCL(dt, axis);
        OP_UNSUP(2, dt, axis);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createCopy(DataType dt, U32 *srcDims, U32 *dstDims, U32 length) override {
        //auto cep = new CopyOCL(dt, srcDims, dstDims, length);
        OP_UNSUP(4, dt, srcDims, dstDims, length);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createCheck(DataType dt, CheckMode checkMode) override {
        //auto cep = new CheckOCL(dt, checkMode);
        OP_UNSUP(2, dt, checkMode);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createRepeat(DataType dt, I32 loops, I32 jumpOperatorIndex, I32 currentOperatorIndex) override {
        //auto cep = new RepeatOCL(dt, loops, jumpOperatorIndex, currentOperatorIndex);
        OP_UNSUP(4, dt, loops, jumpOperatorIndex, currentOperatorIndex);
        return std::shared_ptr<Operator>(cep);
    }

    virtual std::shared_ptr<Operator> createBilateralSliceApply(U32 coefficiency_len, bool has_offset, BilateralSliceApplyMode mode) override {
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

};

#endif //_FACTORY_OCL_H
