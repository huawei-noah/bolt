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
#include <optional>
#define NOT_SUPPORT Operator* cep = NULL;CHECK_STATUS(NOT_SUPPORTED);
#define NOT_USE1(a1)                             {UNUSED(a1);}
#define NOT_USE2(a1, a2)                         {NOT_USE1(a1)             NOT_USE1(a2)}
#define NOT_USE3(a1, a2, a3)                     {NOT_USE2(a1, a2)         NOT_USE1(a3)}
#define NOT_USE4(a1, a2, a3, a4)                 {NOT_USE2(a1, a2)         NOT_USE2(a3, a4)}
#define NOT_USE5(a1, a2, a3, a4, a5)             {NOT_USE4(a1, a2, a3, a4) NOT_USE1(a5)}
#define NOT_USE8(a1, a2, a3, a4, a5, a6, a7, a8) {NOT_USE4(a1, a2, a3, a4) NOT_USE4(a5, a6, a7, a8)}
#define NOT_USE16(a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, ag) {NOT_USE8(a1, a2, a3, a4, a5, a6, a7, a8) NOT_USE8(a9, aa, ab, ac, ad, ae, af, ag)}
#define OP_UNSUP(num,...) NOT_USE##num(__VA_ARGS__) NOT_SUPPORT

class Factory {
public:
    virtual ~Factory(){};
    virtual std::shared_ptr<Operator> createConvolution(DataType dt, U32 nf, U32 ksizeH, U32 ksizeW, U32 kstrideH, U32 kstrideW,
        U32 kpaddingT, U32 kpaddingB, U32 kpaddingL, U32 kpaddingR,
        ActivationMode dwActiveMode, ActivationMode pwActiveMode,
        ConvolutionMode convolutionType, U32 group, U32 dilateH, U32 dilateW) = 0;

    virtual std::shared_ptr<Operator> createDeconvolution(DataType dt, U32 nf, U32 ksizeH, U32 ksizeW, U32 kstrideH, U32 kstrideW,
        U32 kpaddingT, U32 kpaddingB, U32 kpaddingL, U32 kpaddingR,
        ActivationMode dwActiveMode, ActivationMode pwActiveMode,
        ConvolutionMode convolutionType, U32 group, U32 dilateH, U32 dilateW) = 0;

    virtual std::shared_ptr<Operator> createPooling(PoolingMode mode, U32 ksH, U32 ksW, U32 strideH, U32 strideW,
                                                        U32 paddingT, U32 paddingB, U32 paddingL, U32 paddingR, RoundMode rm) = 0;

    virtual std::shared_ptr<Operator> createFullyConnectedEltwise(DataType dt, U32 numInput, U32 numOutput, std::optional<EltwiseType> eltwiseType) = 0;

    virtual std::shared_ptr<Operator> createFullyConnectedEltwise(DataType dt, U32 numInput, U32 numOutput) = 0;

    virtual std::shared_ptr<Operator> createSoftmax(DataType dt) = 0;

    virtual std::shared_ptr<Operator> createConcat(U32 concatDim) = 0;

    virtual std::shared_ptr<Operator> createActivation(ActivationMode activeMode) = 0;

    virtual std::shared_ptr<Operator> createEltwise(EltwiseMode eltMode, I32 coeffSize, F32* coeffValues) = 0;

    virtual std::shared_ptr<Operator> createScale(DataType dt, int numChannels, int numSource) = 0;

    virtual std::shared_ptr<Operator> createLSTM(DataType dt, U32 numOutput, std::optional<EltwiseType> eltwiseType) = 0;

    virtual std::shared_ptr<Operator> createLSTM(DataType dt, U32 numOutput) = 0;

    virtual std::shared_ptr<Operator> createLSTMCell(DataType dt, U32 numOutput, std::optional<EltwiseType> eltwiseType) = 0;

    virtual std::shared_ptr<Operator> createLSTMCell(DataType dt, U32 numOutput) = 0;

    virtual std::shared_ptr<Operator> createLSTM(DataType dt, U32 numOutput, I32 steps) = 0;

    virtual std::shared_ptr<Operator> createEmbedding(DataType dt, U32 inputDim, U32 numOutput, bool transpose) = 0;

    virtual std::shared_ptr<Operator> createMultiply(DataType dt, F32 scale, F32 bias) = 0;

    virtual std::shared_ptr<Operator> createMatMul(DataType dt, bool transposeA, bool transposeB) = 0;

    virtual std::shared_ptr<Operator> createLayerNorm(DataType dt, U32 weightNum) = 0;

    virtual std::shared_ptr<Operator> createReshape(DataType dt, I32* shapeDims, I32 shapeSize, I32 axis, I32 numAxes) = 0;

    virtual std::shared_ptr<Operator> createResize(DataType paramDT, void* paramPtr) = 0;

    virtual std::shared_ptr<Operator> createSlice(DataType dt, U32 axis, U32* slicePoints, U32 sliceSize) = 0;

    virtual std::shared_ptr<Operator> createTranspose(DataType dt, U32* transDims, U32 transSize) = 0;

    virtual std::shared_ptr<Operator> createAttention(DataType dt, U32 numHeads, U32 fromSequenceLength, U32 toSequenceLength) = 0;

    virtual std::shared_ptr<Operator> createClip(DataType dt, F32 clipMinScalar, F32 clipMaxScalar) = 0;

    virtual std::shared_ptr<Operator> createSqueeze(DataType dt, I32 axis, I32 *dims, I32 dimSize) = 0;

    virtual std::shared_ptr<Operator> createUnsqueeze(DataType dt, I32 axis, I32 *dims, I32 dimSize) = 0;

    virtual std::shared_ptr<Operator> createAxisMean(DataType dt, I32 axis) = 0;

    virtual std::shared_ptr<Operator> createArgMax(DataType dt, I32 axis) = 0;

    virtual std::shared_ptr<Operator> createCopy(DataType dt, U32 *srcDims, U32 *dstDims, U32 length) = 0;

    virtual std::shared_ptr<Operator> createCheck(DataType dt, CheckMode checkMode) = 0;

    virtual std::shared_ptr<Operator> createRepeat(DataType dt, I32 loops, I32 jumpOperatorIndex, I32 currentOperatorIndex) = 0;

    virtual std::shared_ptr<Operator> createBilateralSliceApply(U32 coefficiency_len, bool has_offset, BilateralSliceApplyMode mode) = 0;

    virtual std::shared_ptr<Operator> createPreAllocatedMemory(DataType dt, TensorDesc desc) = 0;

    virtual std::shared_ptr<Operator> createSharedWeight(DataType dt, TensorDesc desc) = 0;
};

#endif //_FACTORY_H
