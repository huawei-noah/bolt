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


#ifndef _ELTWISE_H
#define _ELTWISE_H

#include "operator.hpp"

template<Arch A>
class Eltwise: public Operator<A> {
public:
    Eltwise(EltwiseMode eltMode, I32 coeffSize, F32* coeffValues) 
    {
        this->eltMode = eltMode;
        this->coeffSize = coeffSize;
        this->coeffValues = coeffValues;
        this->set_op_type(OT_Eltwise);
    }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)

        Vec<TensorDesc> inputDesc;
        Vec<void*> inputPtr;
        for (Tensor tensorIn: this->inputTensors) {
            inputDesc.push_back(tensorIn.get_desc());
            inputPtr.push_back((void*)tensorIn.get_val().get());
        }
        auto outputDesc = this->outputTensors[0].get_desc();
        auto outputPtr = this->outputTensors[0].get_val().get();

        if (inputDesc.size() == 2 && tensorNumElements(inputDesc[0]) != tensorNumElements(inputDesc[1])) {
            CHECK_STATUS(scale(this->inputTensors[1].get_val().get(), nullptr, outputDesc, this->inputTensors[0].get_val().get(), A));
            memcpy(outputPtr, this->inputTensors[0].get_val().get(), tensorNumBytes(outputDesc));
        } else {
            CHECK_STATUS(eltwise(inputDesc, inputPtr, outputDesc, outputPtr, this->eltMode, A));
        }

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override
    {
        CHECK_STATUS_WITH_RETURN(eltwise_infer_output_size(inDims, &((*outDims)[0])));
        return SUCCESS;
    }

private:
    EltwiseMode eltMode;
    I32 coeffSize;
    F32* coeffValues;

};

#endif //_ELTWISE_H
