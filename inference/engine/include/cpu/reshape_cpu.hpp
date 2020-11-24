// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _RESHAPE_CPU_H
#define _RESHAPE_CPU_H

#include "reshape.hpp"

class ReshapeCPU : public Reshape {
public:
    ReshapeCPU(DataType dt, ReshapeParamSpec p) : Reshape(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<ReshapeCPU> mem =
            std::shared_ptr<ReshapeCPU>(new ReshapeCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        Tensor inputTensor = this->inputTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        Tensor tmpInputTensor = inputTensor;
        Tensor tmpOutputTensor = outputTensor;
        auto inputDesc = inputTensor.get_desc();
        auto outputDesc = outputTensor.get_desc();
        // if axis is 8, the mode of a model for reshape is tflite.
        if (this->p.axis == 8 && outputDesc.nDims == 4) {
            auto tmpOutputDesc = outputTensor.get_desc();
            tmpOutputDesc.df = DF_NHWC;
            tmpOutputTensor = this->temp;
            tmpOutputTensor.resize(tmpOutputDesc);
        }

        // if axis is 8, the mode of a model for reshape is tflite.
        // NCHW/NCHWC8 -> NHWC
        if (this->p.axis == 8 && inputDesc.nDims == 4) {
            auto inputDesc = inputTensor.get_desc();
            auto tmpInputDesc = inputDesc;
            tmpInputDesc.df = DF_NHWC;
            transformToNHWC(inputDesc, ((CpuMemory *)(inputTensor.get_memory()))->get_ptr(),
                tmpInputDesc, ((CpuMemory *)(this->temp.get_memory()))->get_ptr());
            tmpInputTensor = this->temp;
            tmpInputTensor.resize(tmpInputDesc);
        }

        CHECK_STATUS(reshape(tmpInputTensor, this->temp, tmpOutputTensor, &this->archInfo));
        // NHWC -> NCHW
        if (this->p.axis == 8 && outputDesc.nDims == 4) {
            auto outputDesc = outputTensor.get_desc();
            auto tmpOutputDesc = tmpOutputTensor.get_desc();
            void *tmpOutputPtr = ((CpuMemory *)(tmpOutputTensor.get_memory()))->get_ptr();
            transformToNCHW(tmpOutputDesc, tmpOutputPtr, outputDesc,
                ((CpuMemory *)(outputTensor.get_memory()))->get_ptr());
        }
        outputTensor.set_scale(inputTensor.get_scale());
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        CHECK_STATUS(
            reshape_infer_output_size(inTensors[0], this->p, outTensors[0], &this->archInfo));
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes = 0;
        CHECK_STATUS(reshape_infer_forward_tmp_bytes(
            this->inputTensors[0], this->outputTensors[0], &bytes, &this->archInfo));
        return bytes;
    }
};

#endif  // _RESHAPE_CPU_H
