// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _SCALE_GPU_H
#define _SCALE_GPU_H

#include "scale.hpp"

class ScaleOCL : public Scale {
public:
    ScaleOCL(DataType dt, ScaleParamSpec p, int numChannels) : Scale(dt, p, numChannels)
    {
        setMALIArchInfo(
            &(this->archInfo), nullptr, &this->needSetKernelVec, &this->needSelectKernelLS);
    }

    ~ScaleOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<ScaleOCL> mem =
            std::shared_ptr<ScaleOCL>(new ScaleOCL(this->dt, this->p, this->numChannels));
        *mem = *this;
        return mem;
    }

    EE infer_weight_desc() override
    {
        auto curOpWs = this->get_weightspec();
        if (0 != curOpWs.bytes_of_weight) {
            this->numChannels = curOpWs.bytes_of_weight / UNI_MAX(1, bytesOf(curOpWs.mdt));
        } else if (0 != curOpWs.bytes_of_vec) {
            this->numChannels = curOpWs.bytes_of_vec / UNI_MAX(1, bytesOf(curOpWs.mdt));
        } else {
            this->numChannels = 0;
        }
        Tensor modelWeightTensor = Tensor(OCLMem);
        Tensor modelBiasTensor = Tensor(OCLMem);
        TensorDesc weightDesc = tensor1d(this->dt, this->numChannels);
        TensorDesc biasDesc = weightDesc;
        modelWeightTensor.resize(weightDesc);
        modelBiasTensor.resize(biasDesc);
        auto weightMem = (OclMemory *)modelWeightTensor.get_memory();
        auto vectorMem = (OclMemory *)modelBiasTensor.get_memory();

        U32 stride[3] = {(this->numChannels + 3) / 4 * 4, 1, 1};
        U32 offset[3] = {0, 0, 0};
        GCLMemType mt = GCL_MEM_BUF;
        MemFlags flags = CL_MEM_READ_WRITE;
        GCLMemDesc desc = gclmem_build_desc();
        CHECK_STATUS(gclmem_set_desc_padding(&desc, stride, offset, this->dt, DF_NCHW, mt, flags));
        weightMem->padding(desc);
        vectorMem->padding(desc);
        this->weightTensors.push_back(modelWeightTensor);
        this->biasTensors.push_back(modelBiasTensor);
        return SUCCESS;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        int inputNum = this->inputTensors.size();
        Tensor inputTensor = this->inputTensors[this->dataID];
        Tensor outputTensor = this->outputTensors[0];
        if (inputNum == 1 && weightTensors.size() == 0) {
            CHECK_STATUS(NOT_MATCH);
        }

        if (inputNum > 1) {
            U32 cNum = this->inputTensors[0].get_desc().dims[2];
            for (int i = 1; i < inputNum; i++) {
                if (cNum != this->inputTensors[i].get_desc().dims[2]) {
                    CHECK_STATUS(NOT_MATCH);
                }
            }
        }

        void *alpha, *beta;
        if (inputNum == 1) {
            alpha = ((OclMemory *)(this->weightTensors[0].get_memory()))->get_ptr();
            beta = ((OclMemory *)(this->biasTensors[0].get_memory()))->get_ptr();
        } else {
            alpha = ((OclMemory *)(this->inputTensors[1 - this->dataID].get_memory()))->get_ptr();
            beta = nullptr;
        }
        CHECK_STATUS(scale(inputTensor, alpha, beta, this->p, outputTensor, &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        if (inTensors.size() > 1) {
            U32 len0 = inTensors[0]->length();
            U32 len1 = inTensors[1]->length();
            if (len1 > len0) {
                this->dataID = 1;
            }
        }
        CHECK_STATUS(
            scale_infer_output_size(inTensors[this->dataID], outTensors[0], &this->archInfo));
        return SUCCESS;
    }

    REGISTER_OCL_OPERATOR_RUN
};

#endif  // _SCALE_GPU_H
