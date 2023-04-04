// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _CONCAT_OCL_H
#define _CONCAT_OCL_H

#include "concat.hpp"

class ConcatOCL : public Concat {
public:
    ConcatOCL(ConcatParamSpec p) : Concat(p)
    {
        INIT_GPU_INFO(nullptr)
    }

    ~ConcatOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<ConcatOCL> mem = std::shared_ptr<ConcatOCL>(new ConcatOCL(this->p));
        *mem = *this;
        return mem;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        auto outputTensor = this->outputTensors[0];
        CHECK_STATUS(concat(this->inputTensors, this->p, this->temp, outputTensor, &this->archInfo));
    }

    inline bool use_output_tensor_image(std::vector<Tensor *> inTensors)
    {
        bool axisWAlign = true;
        for (auto &tensor : inTensors) {
            TensorDesc desc = tensor->get_desc();
            if (desc.df == DF_NCHWC4) {
                return true;
            }
            U32 concatDim = (p.axis + desc.nDims) % desc.nDims;
            if (concatDim == desc.nDims - 1) {
                if ((desc.dims[0] & 3) != 0) {
                    axisWAlign = false;
                }
            }
        }
        return axisWAlign;
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        EE ret = concat_infer_output_size(inTensors, this->p, outTensors[0], &this->archInfo);
        if (ret == SUCCESS && check_tensors_image(inTensors) && use_output_tensor_image(inTensors)) {
            ret = set_tensors_image(outTensors, inTensors.size());
        }
        return ret;
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes = 0;
        CHECK_STATUS(concat_infer_forward_tmp_bytes(
            this->inputTensors, this->outputTensors[0], &bytes, &this->archInfo));
        return bytes;
    }
    REGISTER_OCL_OPERATOR_RUN
};

#endif  // _CONCAT_OCL_H
