// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _GENERATE_PROPOSALS_H
#define _GENERATE_PROPOSALS_H

#include "weight_operator.hpp"

class GenerateProposals : public WeightOperator {
public:
    GenerateProposals(DataType dt, GenerateProposalsParamSpec p)
    {
        this->dt = dt;
        this->p = p;
    }

    OperatorType get_type() override
    {
        return OT_GenerateProposals;
    }

    void init_generate_proposals_info(std::vector<Tensor *> inTensors)
    {
        bool findId = false;
        this->anchorBlockDim = 4;
        U32 tensorNum = inTensors.size();
        CHECK_REQUIREMENT(tensorNum == 3);
        for (U32 i = 0; i < tensorNum; i++) {
            U32 j = (i + 1) % tensorNum;
            TensorDesc iDesc = inTensors[i]->get_desc();
            TensorDesc jDesc = inTensors[j]->get_desc();
            if (iDesc.nDims == 4 && jDesc.nDims == 4 && iDesc.dims[0] == jDesc.dims[0] &&
                iDesc.dims[1] == jDesc.dims[1]) {
                if (iDesc.dims[2] == jDesc.dims[2] * this->anchorBlockDim) {
                    this->deltaTensorId = i;
                    this->logitTensorId = j;
                    this->imgInfoTensorId = (j + 1) % tensorNum;
                    this->anchorNum = jDesc.dims[2];
                    findId = true;
                    break;
                } else if (iDesc.dims[2] * this->anchorBlockDim == jDesc.dims[2]) {
                    this->deltaTensorId = j;
                    this->logitTensorId = i;
                    this->imgInfoTensorId = (j + 1) % tensorNum;
                    this->anchorNum = iDesc.dims[2];
                    findId = true;
                    break;
                }
            }
        }
        CHECK_REQUIREMENT(findId);
    }

protected:
    GenerateProposalsParamSpec p;
    U8 deltaTensorId;
    U8 logitTensorId;
    U8 imgInfoTensorId;
    U32 anchorNum;
    U32 anchorBlockDim;
};

#endif  // _GENERATE_PROPOSALS_H
