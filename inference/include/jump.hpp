// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _JUMP_H
#define _JUMP_H

#include "operator.hpp"

class Jump: public Operator
{
public:
    /**
    @param mode
    */
    Jump(DataType dt, I32 jumpOperatorIndex, I32 currentOperatorIndex)
    {
        this->dt = dt;
        this->jumpOperatorIndex = jumpOperatorIndex;
        this->nextOperatorIndex = currentOperatorIndex + 1;
    }

    OperatorType get_op_type() override
    {
        return OT_Jump;
    }

    void run() override
    { }

    int get_next_operator_index() override
    {
        // check status
        if (this->inputTensors.size() > 1) {
            Tensor inputTensor = this->inputTensors[1];
            TensorDesc inputDesc = inputTensor.get_desc();
            I32 *ptr = (I32 *)(inputTensor.get_val());
            U32 length = tensorNumElements(inputDesc);
            for (U32 i = 0; i < length; i++) {
                if (ptr[i]) {
                    return this->jumpOperatorIndex;
                }
            }
        }
        return this->nextOperatorIndex;
    }
    EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override
    {
        UNUSED(inDims);

        (*outDims)[0].dt = this->dt;
        (*outDims)[0].nDims = 0;
        return SUCCESS;
    }

private:
    int jumpOperatorIndex;
    int nextOperatorIndex;
};

#endif //_JUMP_H
