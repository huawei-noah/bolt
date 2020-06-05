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
    Slice(DataType dt, I32 axis, I32* slicePointsPtr, I32 sliceSize)
    {
        this->dt = dt;
        this->axis = axis;
        this->slicePoints = Vec<I32>(sliceSize);
        memcpy(this->slicePoints.data(), slicePointsPtr, sizeof(I32) * sliceSize);
    }

    OperatorType get_op_type() override
    {
        return OT_Slice;
    }

protected:
    Vec<I32> slicePoints;
    I32 axis;
};

#endif //_SLICE_H
