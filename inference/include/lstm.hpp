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

#include "lstmcell.hpp"
#include "tensor_computing.h"

class LSTM: public LSTMCell {
public:
    LSTM(DataType dt, U32 numOutput, std::optional<EltwiseType> eltwiseType)
        :LSTMCell(dt, numOutput, eltwiseType) {}

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor =  this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();

        Tensor weightTensor = this->weightTensors[0];
        TensorDesc weightDesc = weightTensor.get_desc();

        Tensor biasTensor = this->biasTensors[0];
        TensorDesc biasDesc = biasTensor.get_desc();
        
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        //NOTE: no clean tmp and output
        CHECK_STATUS(lstm(inputDesc, inputTensor.get_val(),
                          weightDesc, weightTensor.get_val(),
                          biasDesc, biasTensor.get_val(),
                          this->lenOfTemp, this->temp.get(),
                          this->lstmDesc,
                          outputDesc, outputTensor.get_val(), this->schedule));
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc inDim = inDims[0];

        DataType dt;
        DataFormat df;
        U32 iB, inT, iX;
        CHECK_STATUS(tensor3dGet(inDim, &dt, &df, &iB, &inT, &iX));
        this->xDim = iX;
        this->filterRow = 4 * this->lstmDesc.numOutput;
        this->filterCol = this->lstmDesc.numOutput + iX;
        TensorDesc filter_dim = tensor2df(this->dt, DF_8NK, this->filterRow, this->filterCol);
      	U32 outBytes = 0;
        CHECK_STATUS(lstm_infer_output_size(inDim, filter_dim, this->lstmDesc, &((*outDims)[0]), &outBytes));
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        TensorDesc inputDesc = (this->inputTensors[0]).desc;
        TensorDesc filterDesc = (this->weightTensors[0]).desc;
        TensorDesc outputDesc = (this->outputTensors[0]).desc;
        U32 bytes = 0;
        CHECK_STATUS(lstm_infer_forward_tmp_bytes(inputDesc, filterDesc, outputDesc, this->lstmDesc, &bytes, this->schedule));
        return bytes;
    }
};

#endif //_LSTM_H
