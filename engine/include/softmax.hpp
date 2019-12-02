// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _SOFTMAX_HPP_
#define _SOFTMAX_HPP_

#include "tensor_computing.h"
#include "error.h"

template<Arch A>
class Softmax : public Operator<A> {
public:
    explicit Softmax(DataType dt)
    {
        this->dt = dt;
        this->set_op_type(OT_Softmax);
    }

    TensorDesc reshape(TensorDesc inputDesc) {
        TensorDesc reshapeDesc = inputDesc;
        int i = 0, j = 0;
        for (; i < (int)inputDesc.nDims; i++) {
            if (inputDesc.dims[i] != 1)
                break;
        }
        for (; i < (int)inputDesc.nDims; i++) {
            reshapeDesc.dims[j++] = inputDesc.dims[i];
        }
        reshapeDesc.nDims = j;
        return reshapeDesc;
    }
    
    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        F16 *inputPtr = (F16*)inputTensor.get_val().get();
        
        TensorDesc reshapeDesc = reshape(outputDesc);
        outputDesc = reshapeDesc;

        if (DT_I8 == inputDesc.dt) {
            F16 inputScale = inputTensor.get_scale();
            INT8 *inQ = (INT8*)inputPtr;

            U32 numData = tensorNumElements(inputDesc);
            std::shared_ptr<F16> inBuf((F16*) operator new(numData*bytesOf(DT_F16)));
            
            inputPtr = inBuf.get();
            for (U32 i=0; i<numData; i++) {
                inputPtr[i] = inQ[i] / inputScale;
            }
            inputDesc = reshapeDesc;
            CHECK_STATUS(softmax(inputDesc, inputPtr, outputDesc, outputTensor.get_val().get(), A));
        } else {
            inputDesc = reshapeDesc;
            CHECK_STATUS(softmax(inputDesc, inputPtr, outputDesc, outputTensor.get_val().get(), A));
        }
        
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        CHECK_STATUS_WITH_RETURN(softmax_infer_output_size(inDims[0], &((*outDims)[0])));
        (*outDims)[0].dt = DT_F16;
        return SUCCESS;
    }
};

#endif
