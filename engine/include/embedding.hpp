// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


/**
 * Project deploy
 */


#ifndef _EMBEDDING_H
#define _EMBEDDING_H
#include <optional>
#include "weight_operator.hpp"
#include "tensor_computing.h"


template<Arch A>
class Embedding: public WeightOperator<A> {
public:
    Embedding(DataType dt, U32 inputDim, U32 numOutput)
    {
        this->dt = dt;
        this->inputDim = inputDim;
        this->numOutput = numOutput;
        this->set_op_type(OT_Embedding);
    }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor =  this->inputTensors[0];
        Tensor weightTensor = this->weightTensors[0];
        Tensor outputTensor = this->outputTensors[0];

        U32* inputPtr = (U32*)(inputTensor.get_val().get());
        U8* weightPtr = weightTensor.get_val().get();
        U8* outputPtr = outputTensor.get_val().get();

        TensorDesc inputDesc = inputTensor.get_desc();
        U32 len = tensorNumElements(inputDesc);
        U32 wordEmbeddingBytes = bytesOf(this->dt) * this->numOutput;
        for (U32 i = 0; i < len; i++) {
            U32 word_index = inputPtr[i];
            U8* src  = weightPtr + word_index * wordEmbeddingBytes;
            U8* dest = outputPtr;
            memcpy(dest, src, wordEmbeddingBytes);
            outputPtr += wordEmbeddingBytes;
        }

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc inDim = inDims[0];
        DataType dt;
        DataFormat df;
        U32 batch, step;
    	CHECK_REQUIREMENT(tensorIs2d(inDim));
    	CHECK_STATUS_WITH_RETURN(tensor2dfGet(inDim, &dt, &df, &batch, &step));

        (*outDims)[0] = tensor3df(this->dt, DF_MTK, batch, step, this->numOutput);
        return SUCCESS;
    }

    EE init_weight_bias_from_model(U8** modelPtr)
    {
        TensorDesc weightDesc = tensor2df(this->dt, DF_NORMAL, this->inputDim, this->numOutput);
        U32 weightBytes = tensorNumBytes(weightDesc);

        U8* modelWeightPtr = nullptr;
        if (modelPtr != nullptr) {
            modelWeightPtr = (U8*)operator new(weightBytes);
            memcpy(modelWeightPtr, *modelPtr, weightBytes);
            *modelPtr += weightBytes;
        }
        else {
            auto curOpWs = this->get_weightspec_ptr();
            modelWeightPtr = curOpWs.weight;
        }

        std::shared_ptr<U8> weightVal(modelWeightPtr);
        Tensor weightTensor = Tensor(weightDesc, weightVal);
        this->weightTensors.push_back(weightTensor);

        return SUCCESS;

    }

public:
    U32 inputDim;
    U32 numOutput;
};

#endif //_EMBEDDING__H
