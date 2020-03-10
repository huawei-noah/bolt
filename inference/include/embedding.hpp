// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _EMBEDDING_H
#define _EMBEDDING_H
#include "weight_operator.hpp"
#include "tensor_computing.h"

class Embedding: public WeightOperator {
public:
    Embedding(DataType dt, U32 inputDim, U32 numOutput, bool transpose)
    {
        this->dt = dt;
        this->inputDim = inputDim;
        this->numOutput = numOutput;
        this->transpose = transpose;
    }

    OperatorType get_op_type() override
    {
        return OT_Embedding;
    }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor =  this->inputTensors[0];
        Tensor weightTensor;
        if (this->weightTensors.size() > 0)
            weightTensor = this->weightTensors[0];
        else
            weightTensor = this->inputTensors[1];
        Tensor outputTensor = this->outputTensors[0];

        U32* inputPtr = (U32*)(inputTensor.get_val());
        U8* weightPtr = weightTensor.get_val();
        U8* outputPtr = outputTensor.get_val();

        TensorDesc inputDesc = inputTensor.get_desc();
        U32 len = tensorNumElements(inputDesc);
        U32 elementBytes = bytesOf(this->dt);
        U32 wordEmbeddingBytes = elementBytes * this->numOutput;
        U32 transposeStride = elementBytes * this->inputDim;
        for (U32 i = 0; i < len; i++) {
            U32 wordIndex = inputPtr[i];
            U8* dest = outputPtr;
            if (transpose) {
                U8* src = weightPtr + wordIndex * elementBytes;
                for (U32 j = 0; j < this->numOutput; j++) {
                    memcpy(dest, src, elementBytes);
                    src += transposeStride;
                    dest += elementBytes;
                }
            } else {
                U8* src = weightPtr + wordIndex * wordEmbeddingBytes;
                memcpy(dest, src, wordEmbeddingBytes);
            }
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
    	CHECK_STATUS(tensor2dfGet(inDim, &dt, &df, &batch, &step));

        (*outDims)[0] = tensor3df(this->dt, DF_MTK, batch, step, this->numOutput);
        return SUCCESS;
    }

    EE init_weight_bias_from_model(U8** modelPtr)
    {
        TensorDesc weightDesc;
        if (transpose)
            weightDesc = tensor2df(this->dt, DF_TRANSPOSE, this->numOutput, this->inputDim);
        else
            weightDesc = tensor2df(this->dt, DF_NORMAL, this->inputDim, this->numOutput);
        U32 weightBytes = tensorNumBytes(weightDesc);

        std::shared_ptr<Tensor> modelWeightTensor(new Tensor());
        modelWeightTensor->set_desc(weightDesc);

        bool set_ptr = false;
        if(modelPtr != nullptr){
            modelWeightTensor->alloc();
            memcpy((U8*)modelWeightTensor->get_val(), *modelPtr, weightBytes);
            *modelPtr += weightBytes;
            set_ptr = true;
        } else {
            auto curOpWs = this->get_weightspec_ptr();
            if (curOpWs.weight != nullptr) {
                modelWeightTensor->set_val(curOpWs.weight);
                set_ptr = true;
            }
        }
        if(set_ptr) this->weightTensors.push_back(*modelWeightTensor.get());
        return SUCCESS;
    }

private:
    U32 inputDim;
    U32 numOutput;
    bool transpose;
};

#endif //_EMBEDDING__H
