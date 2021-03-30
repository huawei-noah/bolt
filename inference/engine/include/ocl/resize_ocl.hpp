// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _RESIZE_OCL_H
#define _RESIZE_OCL_H

#include "resize.hpp"
#include "image.h"

class ResizeOCL : public Resize {
public:
    ResizeOCL(DataType paramDT, ResizeParamSpec p) : Resize(paramDT, p)
    {
        setMALIArchInfo(
            &(this->archInfo), nullptr, &this->needSetKernelVec, &this->needSelectKernelLS);
    }

    ~ResizeOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<ResizeOCL> mem =
            std::shared_ptr<ResizeOCL>(new ResizeOCL(this->paramDT, this->p));
        *mem = *this;
        return mem;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        Tensor inputTensor = this->inputTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        CHECK_STATUS(resize(inputTensor, this->temp, outputTensor, this->p, &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        U32 bytes;
        switch (paramDT) {
            case DT_F32: {
                CHECK_REQUIREMENT(1 == this->p.scales[0] && 1 == this->p.scales[1]);
                CHECK_STATUS(resize_infer_output_size(inTensors[0], this->paramDT,
                    this->p.scales + 2, outTensors[0], &bytes, &this->archInfo));
                break;
            }
            case DT_U32: {
                CHECK_STATUS(resize_infer_output_size(inTensors[0], this->paramDT, this->p.sizes,
                    outTensors[0], &bytes, &this->archInfo));
                break;
            }
            default: {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        return SUCCESS;
    }

    REGISTER_OCL_OPERATOR_RUN
};

#endif  // _RESIZE_H
