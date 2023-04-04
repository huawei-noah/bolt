// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _DETECTION_OUTPUT_OCL_H
#define _DETECTION_OUTPUT_OCL_H

#include "detection_output.hpp"

class DetectionOutputOCL : public DetectionOutput {
public:
    DetectionOutputOCL(DataType dt, DetectionOutputParamSpec p) : DetectionOutput(dt, p)
    {
        INIT_GPU_INFO(nullptr)
    }

    ~DetectionOutputOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<DetectionOutputOCL> mem = std::shared_ptr<DetectionOutputOCL>(new DetectionOutputOCL(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        CHECK_STATUS(
            detectionoutput(this->inputTensors, this->p, this->outputTensors[0], &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        return detectionoutput_infer_output_size(inTensors, this->p, outTensors[0], &this->archInfo);
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes = 0;
        //CHECK_STATUS(detection_output_infer_forward_tmp_bytes(
        //    this->inputTensors[0], this->outputTensors[0], &bytes, &this->archInfo));
        return bytes;
    }

    REGISTER_OCL_OPERATOR_RUN
};

#endif  // _DETECTION_OUTPUT_OCL_H
