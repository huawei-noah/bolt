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
            std::shared_ptr<ScaleCPU>(new ScaleCPU(this->dt, this->p, this->numChannels));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        int inputTensorNumber = this->inputTensors.size();
        Tensor inputTensor = this->inputTensors[this->dataID];
        Tensor outputTensor = this->outputTensors[0];
        outputTensor.resize(inputTensor.get_desc());

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
        auto inDim = inTensors[0]->get_desc();
        auto curOpWs = this->get_weightspec();

        if (curOpWs.bytes_of_weight == bytesOf(curOpWs.mdt) ||
            curOpWs.bytes_of_vec == bytesOf(curOpWs.mdt)) {
            this->p.axis = 0;
        }

        if (!(curOpWs.bytes_of_weight == 0 && curOpWs.bytes_of_vec == 0)) {
            int maxBytes = curOpWs.bytes_of_weight > curOpWs.bytes_of_vec ? curOpWs.bytes_of_weight
                                                                          : curOpWs.bytes_of_vec;
            for (int i = 0; i < (int)(inDim.nDims); i++) {
                if (inDim.dims[inDim.nDims - 1 - i] == (maxBytes / bytesOf(curOpWs.mdt))) {
                    this->p.axis = i;
                    break;
                }
            }
        }

        I32 tmpAxis = (this->p.axis + inDim.nDims) % inDim.nDims;
        tmpAxis = inDim.nDims - 1 - tmpAxis;
        CHECK_REQUIREMENT(tmpAxis < (I32)inDim.nDims);
        U32 ic = inDim.dims[tmpAxis];

        if (0 != curOpWs.bytes_of_weight) {
            this->numChannels = curOpWs.bytes_of_weight / UNI_MAX(1, bytesOf(curOpWs.mdt));
        } else if (0 != curOpWs.bytes_of_vec) {
            this->numChannels = curOpWs.bytes_of_vec / UNI_MAX(1, bytesOf(curOpWs.mdt));
        } else {
            this->numChannels = 0;
        }

        if (curOpWs.bytes_of_weight == 0 && curOpWs.bytes_of_vec == tensorNumBytes(inDim)) {
            this->numChannels = 0;
        } else if (curOpWs.bytes_of_vec == 0 && curOpWs.bytes_of_weight == tensorNumBytes(inDim)) {
            this->numChannels = 0;
        }

        if (ic != numChannels && 0 != numChannels) {
            UNI_ERROR_LOG("ScaleCPU input channels (IC) do not match. Perhaps some channel padding "
                          "has been done earlier\n"
                          "          IC is now %u but should be %u\n",
                ic, numChannels);
            CHECK_STATUS(NOT_SUPPORTED);
            return NOT_SUPPORTED;
        } else {
            if (inTensors.size() > 1 &&
                tensorNumElements(inTensors[1]->get_desc()) > tensorNumElements(inDim)) {
                this->dataID = 1;
            }
        }

        CHECK_STATUS(
            scale_infer_output_size(inTensors[this->dataID], outTensors[0], &this->archInfo));
        return SUCCESS;
    }

    EE infer_weight_desc() override
    {
        auto curOpWs = this->get_weightspec();
        this->weightTensors = std::vector<Tensor>(1);
        this->weightTensors[0].resize(
            tensor1d(this->dt, curOpWs.bytes_of_weight / UNI_MAX(1, bytesOf(curOpWs.mdt))));
        this->biasTensors = std::vector<Tensor>(1);
        this->biasTensors[0].resize(
            tensor1d(this->dt, curOpWs.bytes_of_vec / UNI_MAX(1, bytesOf(curOpWs.mdt))));
        return SUCCESS;
    }
};

#endif  // _SCALE_CPU_H
