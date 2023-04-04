// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _BATCH_NORM_CPU_H
#define _BATCH_NORM_CPU_H

#include "batch_norm.hpp"

class BatchNormCPU : public BatchNorm {
public:
    BatchNormCPU(DataType dt, BatchNormParamSpec p) : BatchNorm(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<BatchNormCPU> mem =
            std::shared_ptr<BatchNormCPU>(new BatchNormCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        CHECK_STATUS(batch_norm(this->inputTensors[0], this->biasTensors[0], this->weightTensors[0],
            this->p, this->outputTensors[0], &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        outTensors[0]->resize(inTensors[0]->get_desc());
        return SUCCESS;
    }

    int get_channels_num()
    {
        int ret = 0;
        if (0 != this->ws.bytes_of_weight) {
            ret = this->ws.bytes_of_weight / UNI_MAX(1, bytesOf(this->ws.mdt));
        } else if (0 != this->ws.bytes_of_vec) {
            ret = this->ws.bytes_of_vec / UNI_MAX(1, bytesOf(this->ws.mdt));
        }
        return ret;
    }

    EE infer_weight_desc() override
    {
        int num = this->get_channels_num();
        this->weightTensors = std::vector<Tensor>(1);
        this->weightTensors[0].resize(tensor1d(this->dt, num));
        this->biasTensors = std::vector<Tensor>(1);
        this->biasTensors[0].resize(tensor1d(this->dt, num));
        return SUCCESS;
    }

    EE transform_filter() override
    {
        U32 bytes[2];
        CHECK_STATUS(batch_norm_transform_filter_bytes(
            this->biasTensors[0], this->weightTensors[0], this->p, bytes, &this->archInfo));
        Tensor alphaTensor = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, bytes[0]));
        Tensor betaTensor = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, bytes[1]));
        CHECK_STATUS(batch_norm_transform_filter(this->biasTensors[0], this->weightTensors[0],
            this->p, alphaTensor, betaTensor, &this->archInfo));
        this->biasTensors[0] = alphaTensor;
        this->weightTensors[0] = betaTensor;
        return SUCCESS;
    }
};

#endif  // _BATCH_NORM_CPU_H
