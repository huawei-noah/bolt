// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _SPACE2DEPTH_OCL_H
#define _SPACE2DEPTH_OCL_H

#include "space2depth.hpp"

class Space2DepthOCL : public Space2Depth {
public:
    Space2DepthOCL(DataType dt) : Space2Depth(dt)
    {
        setMALIArchInfo(
            &(this->archInfo), nullptr, &this->needSetKernelVec, &this->needSelectKernelLS);
    }

    ~Space2DepthOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<Space2DepthOCL> mem =
            std::shared_ptr<Space2DepthOCL>(new Space2DepthOCL(this->dt));
        *mem = *this;
        return mem;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        CHECK_STATUS(space2depth(this->inputTensors[0], this->outputTensors[0], &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        CHECK_STATUS(space2depth_infer_output_size(inTensors[0], outTensors[0], &this->archInfo));
        return SUCCESS;
    }

    REGISTER_OCL_OPERATOR_RUN
};

#endif  // _SPACE2DEPTH_OCL_H
