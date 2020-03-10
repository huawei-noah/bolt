// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _SLICE_H
#define _SLICE_H

#include "operator.hpp"
#include "tensor_computing.h"

class Slice: public Operator {
public:
    Slice(DataType dt, U32 axis, U32* slicePointsPtr, U32 sliceSize)
    {
        this->dt = dt;
        this->axis = axis;
        this->slicePoints = Vec<U32>(sliceSize);
        memcpy(this->slicePoints.data(), slicePointsPtr, sizeof(U32) * sliceSize);
    }

    OperatorType get_op_type() override
    {
        return OT_Slice;
    }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        
        Vec<Tensor> outputTensors = this->get_output_tensors();
        Vec<TensorDesc> outputTensorDescs;
        Vec<void*> outputPtrs;
        for (U32 i = 0; i < outputTensors.size(); i++) {
            outputTensorDescs.push_back(outputTensors[i].get_desc());
            outputPtrs.push_back(outputTensors[i].get_val());
        }

        CHECK_STATUS(slice(inputDesc, inputTensor.get_val(), outputTensorDescs, &outputPtrs, this->schedule));
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc in_dim = inDims[0];
        CHECK_STATUS(slice_infer_output_size(in_dim, outDims, this->axis, this->slicePoints.data()));
        return SUCCESS;
    }

private:
    Vec<U32> slicePoints;
    U32 axis;
};

#endif //_SLICE_H
