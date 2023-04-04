// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _GENERATE_PROPOSALS_OCL_H
#define _GENERATE_PROPOSALS_OCL_H

#include "generate_proposals.hpp"

class GenerateProposalsOCL : public GenerateProposals {
public:
    GenerateProposalsOCL(DataType dt, GenerateProposalsParamSpec p) : GenerateProposals(dt, p)
    {
        maliPara.handle = OCLContext::getInstance().handle.get();
        this->archInfo.archPara = &maliPara;
    }

    ~GenerateProposalsOCL()
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<GenerateProposalsOCL> mem =
            std::shared_ptr<GenerateProposalsOCL>(new GenerateProposalsOCL(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    EE infer_weight_desc() override
    {
        TensorDesc anchorDesc =
            tensor2df(this->dt, DF_NORMAL, this->anchorNum, this->anchorBlockDim);
        Tensor modelWeightTensor = Tensor(OCLMem);
        modelWeightTensor.resize(anchorDesc);
        this->weightTensors.push_back(modelWeightTensor);
        return SUCCESS;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        GenerateProposals::init_generate_proposals_info(inTensors);
        Tensor *deltaTensor = inTensors[this->deltaTensorId];
        Tensor *logitTensor = inTensors[this->logitTensorId];
        return generate_proposals_infer_output_size(
            deltaTensor, logitTensor, this->p, outTensors[0], &this->archInfo);
    }

    U32 infer_tmp_memory_size() override
    {
        Tensor deltaTensor = this->inputTensors[this->deltaTensorId];
        Tensor logitTensor = this->inputTensors[this->logitTensorId];
        U32 bytes[2] = {0};
        CHECK_STATUS(generate_proposals_infer_forward_tmp_bytes(
            deltaTensor, logitTensor, this->p, bytes, &this->archInfo));
        U32 gpuBytes = bytes[0];
        U32 cpuBytes = bytes[1];
        TensorDesc tmpDescCpu = tensor1d(DT_U8, cpuBytes);
        tmpTensorCpu.resize(tmpDescCpu);
        tmpTensorCpu.alloc();
        return gpuBytes;
    }

    virtual void run() override
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        Tensor deltaTensor = this->inputTensors[this->deltaTensorId];
        Tensor logitTensor = this->inputTensors[this->logitTensorId];
        Tensor imgInfoTensor = this->inputTensors[this->imgInfoTensorId];
        std::vector<Tensor> tmpTensors;
        tmpTensors.push_back(this->temp);
        tmpTensors.push_back(this->tmpTensorCpu);
        CHECK_STATUS(generate_proposals(deltaTensor, logitTensor, imgInfoTensor,
            this->weightTensors[0], this->p, tmpTensors, this->outputTensors[0], &this->archInfo));
    }

private:
    Tensor tmpTensorCpu;
    MaliPara maliPara;
};
#endif  // _GENERATE_PROPOSALS_OCL_H
