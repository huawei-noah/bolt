// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _PRELU_OCL_H
#define _PRELU_OCL_H

#include "prelu.hpp"

class PReLUOCL : public PReLU {
public:
    PReLUOCL(DataType dt) : PReLU(dt)
    {
        setMALIArchInfo(
            &(this->archInfo), nullptr, &this->needSetKernelVec, &this->needSelectKernelLS);
    }

    ~PReLUOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<PReLUOCL> mem = std::shared_ptr<PReLUOCL>(new PReLUOCL(this->dt));
        *mem = *this;
        return mem;
    }

    EE infer_weight_desc() override
    {
        auto curOpWs = this->get_weightspec();
        U32 weightNum = 0;
        if (curOpWs.weight != nullptr) {
            weightNum = curOpWs.bytes_of_weight / UNI_MAX(1, bytesOf(curOpWs.mdt));
        }
        if (weightNum == 0) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        if (weightNum == 1) {
            this->preluDesc.propagate_down = true;
        } else {
            this->preluDesc.propagate_down = false;
        }
        Tensor modelWeightTensor = Tensor(OCLMem);
        auto weightMem = (OclMemory *)modelWeightTensor.get_memory();
        TensorDesc weightDesc = tensor1d(this->dt, weightNum);
        modelWeightTensor.resize(weightDesc);

        U32 stride[3] = {1, 1, 1};
        U32 offset[3] = {0, 0, 0};
        stride[0] = (weightNum > 1) ? (weightNum + 3) / 4 * 4 : 1;
        GCLMemType mt = GCL_MEM_BUF;
        MemFlags flags = CL_MEM_READ_WRITE;
        GCLMemDesc desc = gclmem_build_desc();
        CHECK_STATUS(gclmem_set_desc_padding(&desc, stride, offset, this->dt, DF_NCHW, mt, flags));
        weightMem->padding(desc);
        this->weightTensors.push_back(modelWeightTensor);
        return SUCCESS;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        CHECK_STATUS(prelu(this->inputTensors[0], this->weightTensors[0], this->preluDesc,
            this->outputTensors[0], &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        CHECK_STATUS(prelu_infer_output_size(inTensors[0], outTensors[0], &this->archInfo));
        return SUCCESS;
    }

    REGISTER_OCL_OPERATOR_RUN
};

#endif  // _PRELU_OCL_H
