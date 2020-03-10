// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _SOFTMAX_H
#define _SOFTMAX_H

#include "operator.hpp"
#include "tensor_computing.h"

class Softmax : public Operator {
public:
    explicit Softmax(DataType dt)
    {
        this->dt = dt;
    }

    OperatorType get_op_type() override
    {
        return OT_Softmax;
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
};
#endif //_SOFTMAX_H
