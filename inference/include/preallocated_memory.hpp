// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _PREALLOCATED_MEMORY_H
#define _PREALLOCATED_MEMORY_H

#include "operator.hpp"

class PreAllocatedMemory: public Operator
{
public:
    /**
    @param mode
    */
    PreAllocatedMemory(DataType dt, TensorDesc desc)
    {
        this->dt = dt;
        this->desc = desc;
    }

    OperatorType get_op_type() override
    {
        return OT_PreAllocatedMemory;
    }

    void run() override {
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();
        memset(outputTensor.get_val(), 0, tensorNumBytes(outputDesc));
    }

    EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override
    {
        if (inDims.size() > 0)
            CHECK_STATUS(NOT_MATCH);

        (*outDims)[0] = this->desc;
        return SUCCESS;
    }

private:
    TensorDesc desc;
};

#endif //_PREALLOCATED_MEMORY_H
