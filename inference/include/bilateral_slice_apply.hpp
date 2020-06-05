// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _BILATERAL_SLICE_APPLY_H
#define _BILATERAL_SLICE_APPLY_H
#include <math.h>
#include "operator.hpp"
#include "tensor_computing.h"
#include "tensor_desc.h"
#include "op_type.h"

class BilateralSliceApply: public Operator{
public:

/**
 * @param coefficient_len
 * @param has_offset
 */
    BilateralSliceApply(U32 coefficient_len, bool has_offset, BilateralSliceApplyMode mode)
    {
        this->coefficient_len = coefficient_len;
        this->has_offset = has_offset;
        this->mode = mode;
    }
    virtual ~BilateralSliceApply(){};

    OperatorType get_op_type() override
    {
        return OT_BilateralSliceApply;
    }

    BilateralSliceApplyDesc create_BilateralSliceApplyDesc(U32 coefficient_len, bool has_offset, BilateralSliceApplyMode mode)
    {
        BilateralSliceApplyDesc bilateralSliceApplyDesc;
        bilateralSliceApplyDesc.coefficient_len = coefficient_len;
        bilateralSliceApplyDesc.has_offset = has_offset;
        bilateralSliceApplyDesc.mode = mode;
        return bilateralSliceApplyDesc;
    }

    void set_coefficient_len(U32 coefficient_len) {    
        this->coefficient_len = coefficient_len;
    }

    void set_has_offset(bool has_offset) {
        this->has_offset = has_offset;
    }

    void set_mode(BilateralSliceApplyMode mode) {
        this->mode = mode;
    }

protected:
    U32  coefficient_len;
    bool has_offset;
    BilateralSliceApplyMode mode;
};

#endif //_BILATERAL_SLICE_APPLY_H
