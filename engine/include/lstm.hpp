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
#ifndef _LSTM_H
#define _LSTM_H

#include <optional>
#include "weight_operator.hpp"
#include "tensor_computing.h"


// weight matrix are 8 seperate matrixes: h_I, h_F, h_O, h_G, x_I, x_F, x_O, x_G
// and each one is K = num_cols
template<Arch A>
class Lstm: public WeightOperator<A> {
public:
    Lstm(DataType dt, U32 numOutput, std::optional<EltwiseType> eltwiseType)
    {
        this->dt = dt;
	    this->numOutput = numOutput;
        this->eltwiseType = eltwiseType;
        this->set_op_type(OT_LSTM);
        this->hasBias = false;
    }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor =  this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();

        Tensor weightTensor = this->weightTensors[0];
        TensorDesc weightDesc = weightTensor.get_desc();

        LSTMDesc lstmDesc ;
        lstmDesc.num_output = this->numOutput;
    
        Tensor biasTensor = this->biasTensors[0];
        TensorDesc biasDesc = biasTensor.get_desc();
        
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        //NOTE: no clean tmp and output
        CHECK_STATUS(lstm(inputDesc, inputTensor.get_val().get(),
                          weightDesc, weightTensor.get_val().get(),
                          lstmDesc,
                          biasDesc, biasTensor.get_val().get(),
                          this->lenOfTemp, this->temp.get(),
                          outputDesc, outputTensor.get_val().get(), A));
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc inDim = inDims[0];

        DataType dt;
        DataFormat df;
        U32 iB, inT, iX;
        CHECK_STATUS_WITH_RETURN(tensor3dGet(inDim, &dt, &df, &iB, &inT, &iX));
        this->xDim = iX;
        this->filterRow = 4 * this->numOutput;
        this->filterCol = this->numOutput + iX;
        TensorDesc filter_dim = tensor2df(this->dt, DF_8NK, this->filterRow, this->filterCol);
        LSTMDesc lstmDesc;
        lstmDesc.num_output = this->numOutput;
      	U32 outBytes = 0;
        CHECK_STATUS_WITH_RETURN(lstm_infer_output_size(inDim, filter_dim, lstmDesc, &((*outDims)[0]), &outBytes));
        return SUCCESS;
    }

    U32 infer_tmp_memory_size()
    {
        TensorDesc inputDesc = (this->inputTensors[0]).desc;
        TensorDesc filterDesc = (this->weightTensors[0]).desc;
        TensorDesc outputDesc = (this->outputTensors[0]).desc;
        LSTMDesc lstmDesc ;
        lstmDesc.num_output = this->numOutput;
        U32 bytes = 0;
        CHECK_STATUS(lstm_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, lstmDesc, &bytes, A));
        return bytes;
    }

    U32 infer_wtm_memory_size(TensorDesc fileDesc)
    {
        U32 byte = 0;
        CHECK_STATUS(lstm_transform_filter_bytes(fileDesc, &byte, A));
        return byte;
    }
	
    EE transform_filter()
    {
        Tensor weightTensor = this->weightTensors[0];
        TensorDesc weightDesc = weightTensor.get_desc();
        U8* weightPtr = weightTensor.get_val().get();
    
        auto wtmBytes = this->infer_wtm_memory_size(weightDesc);
        std::shared_ptr<U8> wtmPtr((U8*) operator new(wtmBytes));
        this->set_wtm_memory(wtmBytes, wtmPtr);
    
        TensorDesc wtmDesc;
        CHECK_STATUS_WITH_RETURN(lstm_transform_filter(weightDesc, weightPtr, &wtmDesc, this->get_wtm().get(), this->xDim, this->numOutput, A));
        Tensor wtmTensor = Tensor(wtmDesc, this->get_wtm());
        this->weightTensors[0] = wtmTensor;
        return SUCCESS;
    }

    EE init_weight_bias_from_model(U8** modelPtr)
    {
        TensorDesc weightDesc = tensor2df(this->dt, DF_8NK, this->filterRow, this->filterCol);
        TensorDesc biasDesc = tensor1d(this->dt, 4 * this->numOutput);
        U32 weightBytes = bytesOf(this->dt) * 4 * this->numOutput * (this->numOutput + this->xDim);

        U8* modelWeightPtr = nullptr;
        U8* modelBiasPtr = nullptr;
        if (modelPtr != nullptr) {
            modelWeightPtr = (U8 *)operator new(weightBytes);
            memcpy(modelWeightPtr, *modelPtr, weightBytes);
            *modelPtr += weightBytes;
            if (this->hasBias) {
                modelBiasPtr = (*modelPtr);
                *modelPtr += tensorNumBytes(biasDesc);
            }
        }
        else {
            auto curOpWs = this->get_weightspec_ptr();
            modelWeightPtr = curOpWs.weight;
            modelBiasPtr = curOpWs.vec;
        }

        std::shared_ptr<U8> weightVal(modelWeightPtr);
        Tensor weightTensor = Tensor(weightDesc, weightVal);
        this->weightTensors.push_back(weightTensor);

        // bias
        std::shared_ptr<U8> biasVal;
        Tensor biasTensor = Tensor(biasDesc, biasVal);
        biasTensor.alloc();
        U8* biasPtr = biasTensor.get_val().get();
        if (this->hasBias) {
            memcpy(biasPtr, modelBiasPtr, tensorNumBytes(biasDesc));
        } else {
            memset(biasPtr, 0, tensorNumBytes(biasDesc));
        }
        this->biasTensors.push_back(biasTensor);

        return SUCCESS;
    }

public:
    std::optional<EltwiseType> eltwiseType;
    U32 numOutput;
    U32 filterRow;
    U32 filterCol;
    U32 xDim;
};

#endif //_LSTM_H
