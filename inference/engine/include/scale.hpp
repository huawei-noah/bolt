// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _SCALE_H
#define _SCALE_H

#include "weight_operator.hpp"

class Scale : public WeightOperator {
public:
    Scale(DataType dt, ScaleParamSpec p, int numChannels)
    {
        this->dt = dt;
        this->p = p;
        this->numChannels = numChannels;
        this->dataID = 0;
    }

    OperatorType get_type() override
    {
        return OT_Scale;
    }

    bool can_input_output_the_same() override
    {
        return true;
    }

    U32 find_target_axis_len(std::vector<Tensor *> inTensors)
    {
        auto curOpWs = this->get_weightspec();
        U32 weightNum = 0;
        U32 vecNum = 0;
        if (0 != curOpWs.bytes_of_weight) {
            weightNum = curOpWs.bytes_of_weight / UNI_MAX(1, bytesOf(curOpWs.mdt));
        } else if (0 != curOpWs.bytes_of_vec) {
            vecNum = curOpWs.bytes_of_vec / UNI_MAX(1, bytesOf(curOpWs.mdt));
        }
        if (weightNum > 0 && vecNum > 0 && weightNum != vecNum) {
            CHECK_STATUS(NOT_MATCH);
        }
        this->numChannels = (weightNum) ? weightNum : vecNum;
        if (weightNum == 0 && vecNum == 0) {
            if (inTensors.size() == 1) {
                CHECK_STATUS(NOT_MATCH);
            }
            TensorDesc desc = inTensors[1 - dataID]->get_desc();
            this->numChannels = tensorNumElements(desc);
        }

        TensorDesc inputDesc = inTensors[dataID]->get_desc();
        U32 axisLen = this->numChannels;
        I32 axis = p.axis;
        U32 nDims = inputDesc.nDims;
        axis = (nDims + axis) % nDims;
        axis = nDims - 1 - axis;
        if (axisLen != inputDesc.dims[axis]) {
            for (U32 i = 0; i < nDims; i++) {
                if (inputDesc.dims[nDims - 1 - i] == axisLen) {
                    p.axis = i;
                }
            }
        }
        return axisLen;
    }

protected:
    ScaleParamSpec p;
    U32 numChannels;
    int dataID;
};

#endif  // _SCALE_H
