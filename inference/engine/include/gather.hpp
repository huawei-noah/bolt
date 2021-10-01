// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _GATHER_H
#define _GATHER_H

#include "weight_operator.hpp"

class Gather : public WeightOperator {
public:
    Gather(DataType dt, GatherParamSpec p)
    {
        this->dt = dt;
        this->p = p;
    }

    OperatorType get_type() override
    {
        return OT_Gather;
    }

    Tensor *get_data_tensor_ptr(std::vector<Tensor *> inTensors, Tensor *tmpTensor)
    {
        if (this->p.data_desc.nDims > 0) {
            tmpTensor->resize(this->p.data_desc);
            return tmpTensor;
        }
        CHECK_REQUIREMENT(0 < inTensors.size());
        return inTensors[0];
    }

    Tensor *get_index_tensor_ptr(std::vector<Tensor *> inTensors, Tensor *tmpTensor)
    {
        if (this->p.index_desc.nDims > 0) {
            tmpTensor->resize(this->p.index_desc);
            return tmpTensor;
        }
        U32 inputCount = (this->p.data_desc.nDims > 0) ? 0 : 1;
        CHECK_REQUIREMENT(inputCount < inTensors.size());
        return inTensors[inputCount];
    }

    Tensor get_data_tensor()
    {
        Tensor dataTensor;
        if (this->p.data_desc.nDims > 0) {
            CHECK_REQUIREMENT(0 < this->weightTensors.size());
            dataTensor = this->weightTensors[0];
        } else {
            CHECK_REQUIREMENT(0 < this->inputTensors.size());
            dataTensor = this->inputTensors[0];
        }
        return dataTensor;
    }

    Tensor get_index_tensor()
    {
        U32 inputCount = 0;
        if (this->p.data_desc.nDims == 0) {
            inputCount++;
        }
        Tensor indexTensor;
        if (this->p.index_desc.nDims > 0) {
            CHECK_REQUIREMENT(0 < this->biasTensors.size());
            indexTensor = this->biasTensors[0];
        } else {
            CHECK_REQUIREMENT(inputCount < this->inputTensors.size());
            indexTensor = this->inputTensors[inputCount];
        }
        return indexTensor;
    }

protected:
    GatherParamSpec p;
};

#endif  // _GATHER_H
