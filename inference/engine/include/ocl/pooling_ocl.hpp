// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _POOLING_OCL_H
#define _POOLING_OCL_H

#include "pooling.hpp"

class PoolingOCL : public Pooling {
public:
    PoolingOCL(PoolingParamSpec p) : Pooling(p)
    {
        INIT_GPU_INFO(nullptr)
    }

    ~PoolingOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<PoolingOCL> mem = std::shared_ptr<PoolingOCL>(new PoolingOCL(this->p));
        *mem = *this;
        return mem;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        CHECK_STATUS(pooling(
            this->inputTensors[0], this->p, this->temp, this->outputTensors, &this->archInfo));
    }

    inline bool use_output_tensor_image(Tensor *inputTensor)
    {
        if (this->p.kernel_h == 0 || this->p.kernel_w == 0) {
            return false;
        }
        TensorDesc desc = inputTensor->get_desc();
        if (desc.dims[0] <= this->p.kernel_w || desc.dims[1] <= this->p.kernel_h) {
            return false;
        }
        return true;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        if (this->p.kernel_h == 0 && this->p.kernel_w == 0) {
            Pooling::set_stride(1, 1);
        }
        EE ret = pooling_infer_output_size(inTensors[0], this->p, outTensors[0], &this->archInfo);
        if (ret == SUCCESS && check_tensors_image(inTensors) &&
            use_output_tensor_image(inTensors[0])) {
            ret = set_tensors_image(outTensors, inTensors.size());
        }
        return ret;
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes = 0;
        CHECK_STATUS(pooling_infer_forward_tmp_bytes(
            this->inputTensors[0], this->outputTensors[0], &bytes, &this->archInfo));
        return bytes;
    }

    REGISTER_OCL_OPERATOR_RUN
};

#endif  // _POOLING_OCL_H
