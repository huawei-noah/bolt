// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _CHANNEL_RESIZE_OCL_H
#define _CHANNEL_RESIZE_OCL_H

#include "channel_resize.hpp"

class ChannelResizeOCL : public ChannelResize {
public:
    ChannelResizeOCL(DataType dt, ChannelResizeParamSpec p) : ChannelResize(dt, p)
    {
        setMALIArchInfo(
            &(this->archInfo), nullptr, &this->needSetKernelVec, &this->needSelectKernelLS);
    }

    ~ChannelResizeOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<ChannelResizeOCL> mem =
            std::shared_ptr<ChannelResizeOCL>(new ChannelResizeOCL(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();

        Tensor inputTensor = this->inputTensors[0];
        Tensor outputTensor = this->outputTensors[0];

        CHECK_STATUS(channel_resize(inputTensor, this->p, outputTensor, &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        TensorDesc inDesc = inTensors[0]->get_desc();
        int channelAxis = inDesc.nDims - 2;
        if ((int)inDesc.dims[channelAxis] != this->p.channel_before) {
            this->p.channel_before = inDesc.dims[channelAxis];
        }
        if (this->p.group == 0) {
            this->p.group = 1;
            this->p.channel_before = (int)inDesc.dims[channelAxis];
            this->p.channel_after = this->p.channel_before;
        }
        if (this->p.group != 1) {
            return NOT_SUPPORTED;
        }
        CHECK_STATUS(
            channel_resize_infer_output_size(inTensors[0], this->p, outTensors[0], &this->archInfo));
        return SUCCESS;
    }

    REGISTER_OCL_OPERATOR_RUN
};

#endif  // _CHANNEL_RESIZE_OCL_H
