// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _RNN_OCL_H
#define _RNN_OCL_H

#include "ocl/rnncell_ocl.hpp"

class RNNOCL : public RNNCellOCL {
public:
    RNNOCL(DataType dt, RNNParamSpec p) : RNNCellOCL(dt, p)
    {}

    ~RNNOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<RNNOCL> mem = std::shared_ptr<RNNOCL>(new RNNOCL(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        Tensor inputTensor = this->inputTensors[0];
        Tensor outputTensor = this->outputTensors[0];

        // NOTE: no clean tmp and output
        CHECK_STATUS(rnn(inputTensor, this->weightTensors, this->biasTensors, this->p, this->temp,
            outputTensor, &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        return NOT_SUPPORTED;
        TensorDesc inDim = inTensors[0]->get_desc();

        DataType dt;
        DataFormat df;
        U32 iB, inT, iX;
        CHECK_STATUS(tensor3dGet(inDim, &dt, &df, &iB, &inT, &iX));
        this->xDim = iX;
        CHECK_STATUS(rnn_infer_output_size(inTensors, this->p, outTensors, &this->archInfo));
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes = 0;
        CHECK_STATUS(rnn_infer_forward_tmp_bytes(this->inputTensors[0], this->weightTensors[0],
            this->outputTensors[0], this->p, &bytes, &this->archInfo));
        return bytes;
    }

    REGISTER_OCL_OPERATOR_RUN
};

#endif  // _RNN_OCL_H
