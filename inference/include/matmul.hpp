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
#ifndef _MATMUL_H
#define _MATMUL_H

#include "operator.hpp"
#include "tensor_computing.h"

class MatMul: public Operator {
public:
    MatMul(DataType dt, bool transposeA, bool transposeB)
    {
        this->dt = dt;
        this->transposeA = transposeA;
        this->transposeB = transposeB;
    }

    OperatorType get_op_type() override
    {
        return OT_MatMul;
    }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensorA =  this->inputTensors[0];
        TensorDesc inputDescA = inputTensorA.get_desc();
        Tensor inputTensorB =  this->inputTensors[1];
        TensorDesc inputDescB = inputTensorB.get_desc();
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        CHECK_STATUS(matmul(inputDescA, this->transposeA, inputTensorA.get_val(),
                            inputDescB, this->transposeB, inputTensorB.get_val(),
                            this->temp.get(), this->lenOfTemp,
                            outputDesc, outputTensor.get_val(), this->schedule));
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc inDimA = inDims[0];
        TensorDesc inDimB = inDims[1];
        CHECK_STATUS(matmul_infer_output_size(inDimA, this->transposeA, inDimB, this->transposeB, &((*outDims)[0])));
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        TensorDesc inputDescA = (this->inputTensors[0]).desc;
        TensorDesc inputDescB = (this->inputTensors[1]).desc;
        U32 bytes = 0;
        CHECK_STATUS(matmul_infer_forward_tmp_bytes(inputDescA, this->transposeA, inputDescB, this->transposeB, &bytes, this->schedule));
        return bytes;
    }

private:
    bool transposeA;
    bool transposeB;
};

#endif //_MATMUL_H
