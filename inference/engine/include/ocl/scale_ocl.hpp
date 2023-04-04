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
        INIT_GPU_INFO(nullptr)
    }

    ~ScaleOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<ScaleOCL> mem =
            std::shared_ptr<ScaleOCL>(new ScaleOCL(this->dt, this->p, 0));
        *mem = *this;
        return mem;
    }

    EE infer_weight_desc() override
    {
        U32 weightNum = 0;
        U32 vecNum = 0;
        if (0 != this->ws.bytes_of_weight) {
            weightNum = this->ws.bytes_of_weight / UNI_MAX(1, bytesOf(this->ws.mdt));
        }
        if (0 != this->ws.bytes_of_vec) {
            vecNum = this->ws.bytes_of_vec / UNI_MAX(1, bytesOf(this->ws.mdt));
        }
        if (weightNum) {
            Tensor modelWeightTensor = Tensor(OCLMem);
            TensorDesc weightDesc = tensor1d(this->dt, weightNum);
            modelWeightTensor.resize(weightDesc);
            auto weightMem = (OclMemory *)modelWeightTensor.get_memory();
            U32 pr = (weightNum + 3) / 4 * 4 - weightNum;
            weightMem->padding(0, pr, 0, 0);
            this->weightTensors.push_back(modelWeightTensor);
        }
        if (vecNum) {
            Tensor modelBiasTensor = Tensor(OCLMem);
            TensorDesc biasDesc = tensor1d(this->dt, vecNum);
            modelBiasTensor.resize(biasDesc);
            auto biasMem = (OclMemory *)modelBiasTensor.get_memory();
            U32 pr = (vecNum + 3) / 4 * 4 - vecNum;
            biasMem->padding(0, pr, 0, 0);
            this->biasTensors.push_back(modelBiasTensor);
        }
        return SUCCESS;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        int inputNum = this->inputTensors.size();
        Tensor inputTensor = this->inputTensors[this->dataID];
        Tensor outputTensor = this->outputTensors[0];
        CHECK_REQUIREMENT(inputNum != 1 || weightTensors.size() != 0 || biasTensors.size() != 0);
        if (inputNum > 1) {
            U32 cNum = this->inputTensors[this->dataID].get_desc().dims[2];
            for (int i = 0; i < inputNum; i++) {
                if (i != this->dataID) {
                    auto mem = (OclMemory *)(this->inputTensors[i].get_memory());
                    GCLMemDesc desc = mem->get_desc();
                    if (desc.memFormat == DF_NCHW) {
                        if (desc.num == cNum && desc.offset[0] == 0 && desc.offset[1] == 0) {
                            continue;
                        }
                    } else if (desc.memFormat == DF_NCHWC4) {
                        if (desc.stride[0] == 1 && desc.stride[1] == 1 &&
                            desc.stride[2] * 4 == (cNum + 3) / 4 * 4 && desc.offset[0] == 0 &&
                            desc.offset[1] == 0) {
                            continue;
                        }
                    } else {
                        UNI_ERROR_LOG("gpu scale not support %s format input.\n",
                            DataFormatName()[desc.memFormat]);
                    }
                }
            }
        }

        void *alpha = nullptr;
        void *beta = nullptr;
        if (inputNum == 1) {
            if (weightTensors.size()) {
                alpha = ((OclMemory *)(this->weightTensors[0].get_memory()))->get_ptr();
            }
            if (biasTensors.size()) {
                beta = ((OclMemory *)(this->biasTensors[0].get_memory()))->get_ptr();
            }
        } else {
            alpha = ((OclMemory *)(this->inputTensors[1 - this->dataID].get_memory()))->get_ptr();
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
            auto alphaMem = (OclMemory *)inTensors[1 - this->dataID]->get_memory();
            U32 alphaLen = alphaMem->length();
            U32 pr = (alphaLen + 3) / 4 * 4 - alphaLen;
            alphaMem->padding(0, pr, 0, 0);
        }
        U32 axisLen = find_target_axis_len(inTensors);
        EE ret = scale_infer_output_size(
            inTensors[this->dataID], this->p, axisLen, outTensors[0], &this->archInfo);
        if (ret == SUCCESS && check_tensors_image(inTensors, this->dataID)) {
            ret = set_tensors_image(outTensors, inTensors.size());
        }
        return ret;
    }

    REGISTER_OCL_OPERATOR_RUN
};

#endif  // _SCALE_GPU_H
