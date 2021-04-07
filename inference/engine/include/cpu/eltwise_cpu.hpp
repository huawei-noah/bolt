// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _ELTWISE_CPU_H
#define _ELTWISE_CPU_H

#include "eltwise.hpp"

class EltwiseCPU : public Eltwise {
public:
    EltwiseCPU(EltwiseParamSpec eltwiseDesc) : Eltwise(eltwiseDesc)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<EltwiseCPU> mem =
            std::shared_ptr<EltwiseCPU>(new EltwiseCPU(this->eltwiseDesc));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        std::vector<TensorDesc> inputDesc;
        for (auto p : this->inputTensors) {
            inputDesc.push_back(p.get_desc());
        }
        if (this->eltwiseDesc.elt_mode == ELTWISE_PROD && inputDesc.size() == 2 &&
            inputDesc[0].nDims > 1 && inputDesc[1].nDims > 1 &&
            inputDesc[0].dims[inputDesc[0].nDims - 2] == inputDesc[1].dims[inputDesc[1].nDims - 2] &&
            (inputDesc[1].nDims == 2 || (inputDesc[1].nDims == 3 && inputDesc[1].dims[0] == 1) ||
                (inputDesc[1].nDims == 4 && inputDesc[1].dims[0] == 1 && inputDesc[1].dims[1] == 1)) &&
            tensorNumElements(inputDesc[0]) != tensorNumElements(inputDesc[1])) {
            Tensor inTensor = this->inputTensors[1];
            U8 *alpha = (U8 *)((CpuMemory *)(inTensor.get_memory()))->get_ptr();
            ScaleParamSpec scaleParam;
            scaleParam.axis = 1;
            CHECK_STATUS(scale(this->inputTensors[0], alpha, nullptr, scaleParam,
                this->outputTensors[0], &this->archInfo));
        } else {
            CHECK_STATUS(eltwise(this->inputTensors, this->eltwiseDesc, this->temp,
                this->outputTensors[0], &this->archInfo));
        }
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        std::vector<TensorDesc> inDims;
        for (auto p : inTensors) {
            inDims.push_back(p->get_desc());
        }
        if (this->eltwiseDesc.elt_mode == ELTWISE_PROD && inDims.size() == 2 &&
            (inDims[1].nDims == 2 || (inDims[1].nDims == 3 && inDims[1].dims[0] == 1) ||
                (inDims[1].nDims == 4 && inDims[1].dims[0] == 1 && inDims[1].dims[1] == 1)) &&
            tensorNumElements(inDims[0]) != tensorNumElements(inDims[1])) {
            CHECK_STATUS(scale_infer_output_size(inTensors[0], outTensors[0], &this->archInfo));
        } else {
            CHECK_STATUS(eltwise_infer_output_size(inTensors, outTensors[0], &this->archInfo));
        }
        return SUCCESS;
    }
};

#endif  // _ELTWISE_CPU_H
