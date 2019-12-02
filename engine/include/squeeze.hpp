// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _SQUEEZE_H
#define _SQUEEZE_H

#include "operator.hpp"
#include "tensor_computing.h"
#include "tensor_desc.h"
#include "model_tools.h"

template <Arch A>
class Squeeze: public Operator<A>
{
public:
    /**
    @param mode
    */
    Squeeze(DataType dt)
    {
        this->dt = dt;
        this->set_op_type(OT_Squeeze);
    }

    void run() override
    {
        // the input ptr assign to the output ptr
        UTIL_TIME_TIC(__CLASS_FUNCTION__)

        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        Tensor outputTensor = this->outputTensors[0];
        auto inPtr = inputTensor.get_val().get();
        auto outPtr = outputTensor.get_val().get();
        if(inPtr != outPtr) {
            memcpy(outPtr, inPtr, tensorNumBytes(inputDesc));
        }

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override
    {
        // temp deal with desc there, todo: move to tc
        auto outDimsPtr = &((*outDims)[0]);
        outDimsPtr->dt = inDims[0].dt;
        outDimsPtr->df = inDims[0].df;
        int oneDimNum = 0;
        int copyIndex = 0;
        for (int i = 0; i < (int)inDims[0].nDims; i++) {
            if (inDims[0].dims[i] == 1) {
                oneDimNum++;
                continue;
            }else{
                outDimsPtr->dims[copyIndex] = inDims[0].dims[i];
                copyIndex++;
            }
        }
        outDimsPtr->nDims = inDims[0].nDims - oneDimNum;
        // temp deal with desc there, todo: move to tc

        return SUCCESS;
    }
};

#endif //_SQUEEZE_H


