// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _CHANNEL_RESIZE_CPU_H
#define _CHANNEL_RESIZE_CPU_H

#include "channel_resize.hpp"

class ChannelResizeCPU : public ChannelResize {
public:
    ChannelResizeCPU(DataType dt, ChannelResizeParamSpec p) : ChannelResize(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<ChannelResizeCPU> mem =
            std::shared_ptr<ChannelResizeCPU>(new ChannelResizeCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        U32 inputSize = tensorNumBytes(inputDesc);
        U8 *inputPtr = (U8 *)((CpuMemory *)(inputTensor.get_memory()))->get_ptr();
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();
        U32 outputSize = tensorNumBytes(outputDesc);
        U8 *outputPtr = (U8 *)((CpuMemory *)(outputTensor.get_memory()))->get_ptr();
        // don't need to span or cut
        if (!this->valid) {
            if (inputPtr != outputPtr) {
                CHECK_REQUIREMENT(inputSize == outputSize);
                memcpy(outputPtr, inputPtr, inputSize);
            }
        } else if (this->rearrange && DF_NCHWC8 == inputDesc.df && DF_NCHWC8 == outputDesc.df) {
            transformNCHWC8ToNCHWC8ByGroup(
                inputDesc, inputPtr, this->p.group, outputDesc, outputPtr);
        } else {
            U32 batch = inputDesc.dims[inputDesc.nDims - 1];
            U32 inputChannelGroupSize = this->p.channel_before / this->p.group;
            U32 inputTileSize = inputSize / (batch * this->p.group);
            U32 outputChannelGroupSize = this->p.channel_after / this->p.group;
            U32 outputTileSize = outputSize / (batch * this->p.group);
            int channelAxis = inputDesc.nDims - 2;
            TensorDesc tmpInputDesc = inputDesc;
            tmpInputDesc.dims[channelAxis] = inputChannelGroupSize;
            TensorDesc tmpOutputDesc = outputDesc;
            tmpOutputDesc.dims[channelAxis] = outputChannelGroupSize;
            for (int g = 0; g < this->p.group; g++) {
                if (this->p.channel_after > this->p.channel_before) {
                    transformNCHWToNCHWC8(tmpInputDesc, inputPtr, tmpOutputDesc, outputPtr);
                } else {
                    transformToNCHW(tmpInputDesc, inputPtr, tmpOutputDesc, outputPtr);
                }
                inputPtr += inputTileSize;
                outputPtr += outputTileSize;
            }
        }
#ifdef _USE_INT8
        outputTensor.set_scale(inputTensor.get_scale());
#endif
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        CHECK_REQUIREMENT(inTensors.size() > 0);
        auto inDesc = inTensors[0]->get_desc();
        CHECK_REQUIREMENT(inDesc.nDims > 0);
        // don't need to span
        if (this->p.channel_after > this->p.channel_before && inDesc.df == DF_NCHWC8) {
            this->valid = false;
        }
        // don't need to cut
        if (this->p.channel_after < this->p.channel_before && inDesc.df == DF_NCHW) {
            this->valid = false;
        }
        if (!this->valid) {
            outTensors[0]->resize(inDesc);
            return SUCCESS;
        }

        int channelAxis = inDesc.nDims - 2;
        // channel span or cut for OT_Resize
        if (this->p.group == 0) {
            this->p.group = 1;
            this->p.channel_before = (int)inDesc.dims[channelAxis];
            this->p.channel_after =
                (this->p.channel_before / 8 + ((this->p.channel_before % 8 == 0) ? 0 : 1)) * 8;
        } else {
            CHECK_REQUIREMENT((int)inDesc.dims[channelAxis] == this->p.channel_before);
        }

        inDesc.dims[channelAxis] = this->p.channel_after;
        DataFormat dataFormat;
        if (this->p.channel_after > this->p.channel_before ||
            (this->rearrange && this->p.channel_after % 8 == 0)) {
            dataFormat = DF_NCHWC8;
        } else {
            dataFormat = DF_NCHW;
        }
        inDesc.df = dataFormat;
        outTensors[0]->resize(inDesc);
        return SUCCESS;
    }
};

#endif  // _CHANNEL_RESIZE_CPU_H
