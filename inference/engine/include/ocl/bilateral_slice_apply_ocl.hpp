// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _BILATERAL_SLICE_APPLY_OCL_H
#define _BILATERAL_SLICE_APPLY_OCL_H

#include "bilateral_slice_apply.hpp"

class BilateralSliceApplyOCL : public BilateralSliceApply {
public:
    BilateralSliceApplyOCL(BilateralSliceApplyParamSpec p) : BilateralSliceApply(p)
    {
        this->guideTensor = Tensor(OCLMem);
        setMALIArchInfo(
            &(this->archInfo), nullptr, &this->needSetKernelVec, &this->needSelectKernelLS);
    }

    ~BilateralSliceApplyOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<BilateralSliceApplyOCL> mem =
            std::shared_ptr<BilateralSliceApplyOCL>(new BilateralSliceApplyOCL(this->p));
        *mem = *this;
        return mem;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        Tensor inputTensor = this->inputTensors[0];
        Tensor gridTensor = this->inputTensors[1];
        Tensor outputTensor = this->outputTensors[0];

        if (this->p.mode == BSliceApply_NULL) {
            this->guideTensor = this->inputTensors[2];
        }
        CHECK_STATUS(bilateral_slice_apply(
            inputTensor, guideTensor, gridTensor, p, this->temp, outputTensor, &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        auto inTensor = inTensors[0];
        auto gridTensor = inTensors[1];
        auto inDim = inTensor->get_desc();
        DataType dt;
        DataFormat df;
        U32 width;
        U32 height;
        U32 numChannels;
        U32 num;
        CHECK_STATUS(tensor4dGet(inDim, &dt, &df, &num, &numChannels, &height, &width));
        TensorDesc guideDesc = tensor4df(DT_F16, df, num, 1, height, width);
        this->guideTensor.resize(guideDesc);

        CHECK_STATUS(bilateral_slice_apply_infer_output_size(
            inTensor, &guideTensor, gridTensor, p, outTensors[0], &this->archInfo));
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes = 0;
        CHECK_STATUS(bilateral_slice_apply_infer_forward_tmp_bytes(this->inputTensors[0],
            this->guideTensor, this->inputTensors[1], p, &bytes, &this->archInfo));
        return bytes;
    }

    REGISTER_OCL_OPERATOR_RUN
private:
    Tensor guideTensor;
};

#endif  // _BILATERAL_SLICE_APPLY_OCL_H
