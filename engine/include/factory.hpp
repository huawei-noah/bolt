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
#include "operator.hpp"
#include "convolution.hpp"
#include "pooling.hpp"
#include "fully_connected_eltwise.hpp"
#include "softmax.hpp"
#include "concat.hpp"
#include "activation.hpp"
#include "eltwise.hpp"
#include "scale.hpp"
#include "embedding.hpp"
#include "lstm.hpp"
#include "matmul.hpp"
#include "multiply.hpp"
#include "layer_norm.hpp"
#include "reshape.hpp"
#include "slice.hpp"
#include "transpose.hpp"
#include "attention.hpp"
#include "clip.hpp"
#include "squeeze.hpp"

class Factory {
public:
    //TODO remove these static
    // in engine, create a Factory object
    template<Arch A>
    static std::shared_ptr<Operator<A>> createConvolution(DataType dt, U32 nf, U32 ksize, U32 kstride, U32 kpadding, 
        std::optional<EltwiseType> et, std::optional<PoolingMode> pm, U32 psize, U32 pstride, U32 ppadding, 
        ActivationMode dwActiveMode, ActivationMode pwActiveMode,
        ConvolutionMode convolutionType, U32 group, U32 dilation)
    {
        auto cep = new Convolution<A>(dt, nf, ksize, kstride, kpadding, et,
                                                    pm, psize, pstride, ppadding,
                                                    dwActiveMode, pwActiveMode,
                                                    convolutionType, group, dilation);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template<Arch A>
    static std::shared_ptr<Operator<A>> createConvolution(DataType dt, U32 nf, U32 ksize, U32 kstride, U32 kpadding,
        std::optional<EltwiseType> et, ActivationMode dwActiveMode, ActivationMode pwActiveMode,
        ConvolutionMode convolutionType, U32 group, U32 dilation)
    {
        std::optional<PoolingMode> pmNull;
        return Factory::createConvolution<A>(dt, nf, ksize, kstride, kpadding, et,
                                                           pmNull, 1, 1, 0,
                                                           dwActiveMode, pwActiveMode,
                                                           convolutionType, group, dilation);
    }

    template<Arch A>
    static std::shared_ptr<Operator<A>> createConvolution(DataType dt, U32 nf, U32 ksize, U32 kstride, U32 kpadding,
        ActivationMode dwActiveMode, ActivationMode pwActiveMode,
        ConvolutionMode convolutionType, U32 group, U32 dilation)
    {
        std::optional<EltwiseType> etNull;
        return Factory::createConvolution<A>(dt, nf, ksize, kstride, kpadding,
                                                           etNull, dwActiveMode, pwActiveMode,
                                                           convolutionType, group, dilation);
    }

    template<Arch A>
    static std::shared_ptr<Operator<A>> createConvolution(DataType dt, U32 nf, U32 ksize, U32 kpadding,
        ActivationMode dwActiveMode, ActivationMode pwActiveMode,
        ConvolutionMode convolutionType, U32 group, U32 dilation)
    {
        return Factory::createConvolution<A>(dt, nf, ksize, 1, kpadding,
                                                           dwActiveMode, pwActiveMode,
                                                           convolutionType, group, dilation);
    }

    template<Arch A>
    static std::shared_ptr<Operator<A>> createPooling(PoolingMode mode, U32 ks, U32 stride, U32 padding, RoundMode rm)
    {
        auto cep = new Pooling<A>(mode, ks, stride, padding, rm);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template<Arch A>
    static std::shared_ptr<Operator<A>> createFullyConnectedEltwise(DataType dt, U32 numOutput, std::optional<EltwiseType> eltwiseType)
    {
        auto cep = new FullyConnectedEltwise<A>(dt, numOutput, eltwiseType);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template<Arch A>
    static std::shared_ptr<Operator<A>> createFullyConnectedEltwise(DataType dt, U32 numOutput)
    {
        std::optional<EltwiseType> etNull;
        return Factory::createFullyConnectedEltwise<A>(dt, numOutput, etNull);
    }

    template<Arch A>
    static std::shared_ptr<Operator<A>> createSoftmax(DataType dt)
    {
        auto cep = new Softmax<A>(dt);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template<Arch A>
    static std::shared_ptr<Operator<A>> createConcat(U32 concatDim)
    {
        auto cep = new Concat<A>(concatDim);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template<Arch A>
    static std::shared_ptr<Operator<A>> createActivation(ActivationMode activeMode, OperatorType opType)
    {
        auto cep = new Activation<A>(activeMode, opType);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template<Arch A>
    static std::shared_ptr<Operator<A>> createEltwise(EltwiseMode eltMode, I32 coeffSize, F32* coeffValues)
    {
        auto cep = new Eltwise<A>(eltMode, coeffSize, coeffValues);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template<Arch A>
    static std::shared_ptr<Operator<A>> createScale(DataType dt)
    {
        auto cep = new Scale<A>(dt);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template <Arch A>
    static std::shared_ptr<Operator<A>> createLstm(DataType dt, U32 numOutput, std::optional<EltwiseType> eltwiseType)
    {
        auto cep = new Lstm<A>(dt, numOutput, eltwiseType);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template<Arch A>
    static std::shared_ptr<Operator<A>> createLstm(DataType dt, U32 numOutput)
    {
        std::optional<EltwiseType> etNull;
        return Factory::createLstm<A>(dt, numOutput, etNull);
    }

    template <Arch A>
    static std::shared_ptr<Operator<A>> createEmbedding(DataType dt, U32 inputDim, U32 numOutput)
    {   
        auto cep = new Embedding<A>(dt, inputDim, numOutput);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template <Arch A>
    static std::shared_ptr<Operator<A>> createMultiply(DataType dt, F16 scale)
    {
        auto cep = new Multiply<A>(dt, scale);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template <Arch A>
    static std::shared_ptr<Operator<A>> createMatMul(DataType dt)
    {
        auto cep = new MatMul<A>(dt);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template <Arch A>
    static std::shared_ptr<Operator<A>> createLayerNorm(DataType dt)
    {
        auto cep = new LayerNorm<A>(dt);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template <Arch A>
    static std::shared_ptr<Operator<A>> createReshape(DataType dt, I32* shapeDims, I32 shapeSize, I32 axis, I32 numAxes)
    {
        auto cep = new Reshape<A>(dt, shapeDims, shapeSize, axis, numAxes);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template <Arch A>
    static std::shared_ptr<Operator<A>> createSlice(DataType dt, U32 axis, U32* slicePoints, U32 sliceSize)
    {
        auto cep = new Slice<A>(dt, axis, slicePoints, sliceSize);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template <Arch A>
    static std::shared_ptr<Operator<A>> createTranspose(DataType dt, U32* transDims, U32 transSize)
    {
        auto cep = new Transpose<A>(dt, transDims, transSize);
        return std::shared_ptr<Operator<A>>(cep);
    }


    template <Arch A>
    static std::shared_ptr<Operator<A>> createAttention(DataType dt, int numAttention)
    {
        auto cep = new Attention<A>(dt, numAttention);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template <Arch A>
    static std::shared_ptr<Operator<A>> createClip(DataType dt, F16 clipMinScalar, F16 clipMaxScalar)
    {
        auto cep = new Clip<A>(dt, clipMinScalar, clipMaxScalar);
        return std::shared_ptr<Operator<A>>(cep);
    }

    template <Arch A>
    static std::shared_ptr<Operator<A>> createSqueeze(DataType dt)
    {
        auto cep = new Squeeze<A>(dt);
        return std::shared_ptr<Operator<A>>(cep);
    }



    /**
     * @param mf
     */
    //ModelAccess* get_model_access_instance(ModelFormat mf);

    /**
     * @param os
     */
    //template<typename T> std::shared_ptr<Operator<T>> get_operator_instance(const OperatorSpec* os);

    /**
     * @param ms
     */
    //template<typename T> void get_model_instance(const ModelSpec* ms, Model<T>* model);

    //template<typename T> tuple<Tensor<T>, Tensor<T>> get_tensor_from(const WeightSpec ws);

};

#endif //_FACTORY_H
