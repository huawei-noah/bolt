// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _REDUCTION_H
#define _REDUCTION_H

#include "operator.hpp"
#include "tensor_computing.h"

class Reduction: public Operator
{
public:
    /**
    @param mode
    */
    Reduction(DataType dt, I32 axis, bool keepDim, ReductionMode reductionMode, float coeff)
    {
        this->dt = dt;
        this->axis = axis;
        this->keepDim = keepDim;
        this->reductionMode = reductionMode;
        this->coeff = coeff;
    }

    OperatorType get_op_type() override
    {
        return OT_Reduction;
    }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor =  this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();
        TensorDesc maskDesc;
        U8 *mask;
        if (this->inputTensors.size() > 1) {
            maskDesc = this->inputTensors[1].get_desc();;
            mask = this->inputTensors[1].get_val();
        } else {
            maskDesc.nDims = 0;
	    mask = nullptr;
        }

        CHECK_STATUS(reduction(inputDesc, inputTensor.get_val(),
                          maskDesc, mask,
                          this->axis,
                          this->reductionMode,
                          this->coeff,
                          outputDesc, outputTensor.get_val(), this->schedule));

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc maskDesc;
        if (inDims.size() > 1)
            maskDesc = inDims[1];
        else
            maskDesc.nDims = 0;
        CHECK_STATUS(reduction_infer_output_size(inDims[0], maskDesc, this->axis, this->keepDim, &((*outDims)[0])));
        return SUCCESS;
    }

private:
    I32 axis;
    bool keepDim;
    ReductionMode reductionMode;
    float coeff;
};

#endif
