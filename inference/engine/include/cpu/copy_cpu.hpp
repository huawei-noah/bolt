// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _COPY_CPU_H
#define _COPY_CPU_H

#include "copy.hpp"

class CopyCPU : public Copy {
public:
    CopyCPU(DataType dt, CopyParamSpec p) : Copy(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<CopyCPU> mem = std::shared_ptr<CopyCPU>(new CopyCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        Tensor srcTensor = this->inputTensors[0];
        TensorDesc srcDesc = srcTensor.get_desc();
        Tensor dstTensor = this->inputTensors[1];
        TensorDesc dstDesc = dstTensor.get_desc();

        std::vector<void *> input;
        input.push_back(((CpuMemory *)(srcTensor.get_memory()))->get_ptr());
        input.push_back(((CpuMemory *)(dstTensor.get_memory()))->get_ptr());

        U32 batch = srcDesc.dims[srcDesc.nDims - 1];
        U32 copyLength = (this->p.length >= 0) ? this->p.length : tensorNumElements(srcDesc) / batch;
        U32 srcBatchStride = (this->p.src_dims[0] >= 0) ? this->p.src_dims[0]
                                                        : tensorNumElements(srcDesc) / batch;
        U32 srcStride = (this->p.src_dims[0] >= 0) ? this->p.src_dims[1]
                                                   : tensorNumElements(srcDesc) / batch;
        U32 dstBatchStride = (this->p.dst_dims[0] >= 0) ? this->p.dst_dims[0]
                                                        : tensorNumElements(dstDesc) / batch;
        U32 dstStride = (this->p.dst_dims[0] >= 0) ? this->p.dst_dims[1]
                                                   : tensorNumElements(dstDesc) / batch;
        for (U32 i = 0; i < batch; i++) {
            U32 srcBlockIndex = 0;
            if (this->inputTensors.size() > 2) {
                U32 *ptr = (U32 *)((CpuMemory *)(this->inputTensors[2].get_memory()))->get_ptr();
                srcBlockIndex = ptr[i];
            }
            U32 dstBlockIndex = 0;
            if (this->inputTensors.size() > 3) {
                U32 *ptr = (U32 *)((CpuMemory *)(this->inputTensors[3].get_memory()))->get_ptr();
                dstBlockIndex = ptr[i];
            }
            U32 srcIndex = i * srcBatchStride + srcBlockIndex * srcStride + this->p.src_dims[2];
            U32 dstIndex = i * dstBatchStride + dstBlockIndex * dstStride + this->p.dst_dims[2];
            CHECK_STATUS(
                copy(this->inputTensors, srcIndex, dstIndex, 0, 0, copyLength, &this->archInfo));
        }
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        UNUSED(inTensors);
        auto desc = outTensors[0]->get_desc();
        desc.dt = this->dt;
        desc.df = getTensorDefaultDataFormat(0);
        desc.nDims = 0;
        outTensors[0]->resize(desc);
        return SUCCESS;
    }
};

#endif  // _COPY_CPU_H
