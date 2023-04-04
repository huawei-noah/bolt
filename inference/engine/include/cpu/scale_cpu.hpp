// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _SCALE_CPU_H
#define _SCALE_CPU_H

#include "scale.hpp"

class ScaleCPU : public Scale {
public:
    ScaleCPU(DataType dt, ScaleParamSpec p, int numChannels) : Scale(dt, p, numChannels)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<ScaleCPU> mem =
            std::shared_ptr<ScaleCPU>(new ScaleCPU(this->dt, this->p, 0));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        int inputTensorNumber = this->inputTensors.size();
        Tensor inputTensor = this->inputTensors[this->dataID];
        Tensor outputTensor = this->outputTensors[0];

        void *alpha, *beta;
        if (inputTensorNumber == 1) {
            alpha = ((CpuMemory *)(this->weightTensors[0].get_memory()))->get_ptr();
            beta = ((CpuMemory *)(this->biasTensors[0].get_memory()))->get_ptr();
        } else {
            alpha = ((CpuMemory *)(this->inputTensors[1 - this->dataID].get_memory()))->get_ptr();
            beta = nullptr;
        }
        CHECK_STATUS(scale(inputTensor, alpha, beta, this->p, outputTensor, &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        if (inTensors.size() > 1 &&
            tensorNumElements(inTensors[1]->get_desc()) >
                tensorNumElements(inTensors[0]->get_desc())) {
            this->dataID = 1;
        }
        U32 axisLen = find_target_axis_len(inTensors);
        return scale_infer_output_size(
            inTensors[this->dataID], this->p, axisLen, outTensors[0], &this->archInfo);
    }

    EE infer_weight_desc() override
    {
        this->weightTensors = std::vector<Tensor>(1);
        this->weightTensors[0].resize(
            tensor1d(this->dt, this->ws.bytes_of_weight / UNI_MAX(1, bytesOf(this->ws.mdt))));
        this->biasTensors = std::vector<Tensor>(1);
        this->biasTensors[0].resize(
            tensor1d(this->dt, this->ws.bytes_of_vec / UNI_MAX(1, bytesOf(this->ws.mdt))));
        return SUCCESS;
    }
};

#endif  // _SCALE_CPU_H
