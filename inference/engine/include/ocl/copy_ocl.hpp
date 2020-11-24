// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _COPY_OCL_H
#define _COPY_OCL_H

#include "copy.hpp"

class CopyOCL : public Copy {
public:
    CopyOCL(DataType dt, CopyParamSpec p) : Copy(dt, p)
    {
        setMALIArchInfo(
            &(this->archInfo), nullptr, &this->needSetKernelVec, &this->needSelectKernelLS);
    }

    ~CopyOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<CopyOCL> mem = std::shared_ptr<CopyOCL>(new CopyOCL(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        TensorDesc srcDesc = this->inputTensors[0].get_desc();
        TensorDesc dstDesc = this->inputTensors[1].get_desc();
        U32 batch = srcDesc.dims[srcDesc.nDims - 1];
        if (batch > 1) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        U32 copyLength = (this->p.length >= 0) ? this->p.length : tensorNumElements(srcDesc) / batch;
        U32 srcStride = (this->p.src_dims[0] >= 0) ? this->p.src_dims[1]
                                                   : tensorNumElements(srcDesc) / batch;
        U32 dstStride = (this->p.dst_dims[0] >= 0) ? this->p.dst_dims[1]
                                                   : tensorNumElements(dstDesc) / batch;
        U32 srcIndex = this->p.src_dims[2];
        U32 dstIndex = this->p.dst_dims[2];
        CHECK_STATUS(copy(this->inputTensors, srcIndex, dstIndex, srcStride, dstStride, copyLength,
            &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        CHECK_STATUS(copy_infer_output_size(inTensors, &this->archInfo));
        auto desc = outTensors[0]->get_desc();
        desc.dt = this->dt;
        desc.df = getTensorDefaultDataFormat(0);
        desc.nDims = 0;
        outTensors[0]->resize(desc);
        return SUCCESS;
    }

    REGISTER_OCL_OPERATOR_RUN
};

#endif  // _COPY_OCL_H
