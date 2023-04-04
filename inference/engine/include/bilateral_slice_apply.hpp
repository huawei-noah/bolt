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

#include "operator.hpp"

class BilateralSliceApply : public Operator {
public:
    BilateralSliceApply(BilateralSliceApplyParamSpec p)
    {
        this->p = p;
    }
    virtual ~BilateralSliceApply()
    {}

    OperatorType get_type() override
    {
        return OT_BilateralSliceApply;
    }

    void update_guide(std::vector<Tensor *> inTensors)
    {
        TensorDesc guideDesc;
        if (inTensors.size() > 2) {
            guideDesc = inTensors[2]->get_desc();
        } else {
            guideDesc = inTensors[0]->get_desc();
            int axis = -1;
            if (guideDesc.dims[0] == 3) {
                axis = 0;
            }
            if (guideDesc.dims[guideDesc.nDims - 2] == 3) {
                axis = guideDesc.nDims - 2;
            }
            if (axis == -1) {
                UNI_ERROR_LOG("can not infer guide tensor from input(%s).\n",
                    tensorDesc2Str(guideDesc).c_str());
            }
            guideDesc.dims[axis] = 1;
            guideDesc.dt = inTensors[1]->get_desc().dt;
        }
        this->guideTensor.resize(guideDesc);
    }

protected:
    BilateralSliceApplyParamSpec p;
    Tensor guideTensor;
};

#endif  // _BILATERAL_SLICE_APPLY_H
