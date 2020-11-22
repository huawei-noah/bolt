// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _REPEAT_OCL_H
#define _REPEAT_OCL_H

#include "repeat.hpp"

class RepeatOCL : public Repeat {
public:
    RepeatOCL(DataType dt, RepeatParamSpec p, I32 jumpOperatorIndex, I32 currentOperatorIndex)
        : Repeat(dt, p, jumpOperatorIndex, currentOperatorIndex)
    {
        setMALIArchInfo(
            &(this->archInfo), nullptr, &this->needSetKernelVec, &this->needSelectKernelLS);
    }

    ~RepeatOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<RepeatOCL> mem = std::shared_ptr<RepeatOCL>(
            new RepeatOCL(this->dt, this->p, this->jumpOperatorIndex, this->nextOperatorIndex - 1));
        *mem = *this;
        return mem;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
    }

    int get_next_operator_index() override
    {
        // check status
        if (this->inputTensors.size() > 1) {
            Tensor inputTensor = this->inputTensors[1];
            TensorDesc inputDesc = inputTensor.get_desc();
            GCLMem_t ptr = (GCLMem_t)(((OclMemory *)(inputTensor.get_memory()))->get_ptr());
            U32 length = tensorNumElements(inputDesc);
            DataFormat df = ptr->desc.memFormat;
            if (df != DF_NCHW) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
            U32 w_off, h_off;
            w_off = ptr->desc.offset[0];
            h_off = ptr->desc.offset[1];
            if (w_off != 0 || h_off != 0) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
            I32 *val = hostVal.get();
            CHECK_STATUS(gcl_trans_memory(OCLContext::getInstance().handle.get(), ptr, val, &length,
                DEVICE_BUF_TO_HOST, CL_TRUE));
            for (U32 i = 0; i < length; i++) {
                // end loop
                if (val[i]) {
                    this->iter = 0;
                    return this->nextOperatorIndex;
                }
            }
        }

        // check loop
        if (this->iter < this->p.loops) {
            this->iter++;
            return this->jumpOperatorIndex;
        } else {
            this->iter = 0;
            return this->nextOperatorIndex;
        }
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        this->iter = 0;
        if (this->p.axis >= 0) {
            int axisIndex = 0;
            if (inTensors.size() > 2) {
                axisIndex = 2;
            } else {
                UNI_ERROR_LOG("[ERROR] set to use axis feature of Repeat must meet input tensors "
                              ">= 3 requirement\n");
            }
            TensorDesc desc = inTensors[axisIndex]->get_desc();
            this->p.loops = desc.dims[desc.nDims - 1 - this->p.axis];
        }
        TensorDesc outDesc = outTensors[0]->get_desc();
        outDesc.dt = this->dt;
        outDesc.nDims = 0;
        outTensors[0]->resize(outDesc);
        auto inTensor = inTensors[1];
        TensorDesc inDesc = inTensor->get_desc();
        U32 length = tensorNumElements(inDesc);
        hostVal = std::shared_ptr<I32>(new I32(length));
        return SUCCESS;
    }

    REGISTER_OCL_OPERATOR_RUN

private:
    std::shared_ptr<I32> hostVal;
};

#endif  // _REPEAT_OCL_H
