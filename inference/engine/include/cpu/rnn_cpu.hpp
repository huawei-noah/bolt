// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _RNN_CPU_H
#define _RNN_CPU_H

#include "cpu/rnncell_cpu.hpp"

class RNNCPU : public RNNCellCPU {
public:
    RNNCPU(DataType dt, RNNParamSpec p) : RNNCellCPU(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<RNNCPU> mem = std::shared_ptr<RNNCPU>(new RNNCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        Tensor inputTensor = this->inputTensors[0];
        Tensor outputTensor = this->outputTensors[0];

        if (this->inputTensors.size() == 1) {
            I32 column = this->p.numProjection > 0 ? this->p.numProjection : this->p.numOutput;
            // bi-direction rnn has forward-states and backward-states
            I32 num = p.biDirection ? 2 : 1;
            memset(get_ptr_from_tensor(this->temp, this->archInfo.arch), 0,
                num * (this->p.numOutput + column) * bytesOf(this->inputTensors[0].get_desc().dt));
        } else if (this->inputTensors.size() == 2) {
            // do not support bi-direction
            memcpy(get_ptr_from_tensor(this->temp, this->archInfo.arch),
                get_ptr_from_tensor(this->inputTensors[1], this->archInfo.arch),
                tensorNumBytes(this->inputTensors[1].get_desc()));
        } else if (this->inputTensors.size() == 3) {
            // do not support bi-direction
            U8 *state = (U8 *)get_ptr_from_tensor(this->temp, this->archInfo.arch);
            U32 cStateBytes = tensorNumBytes(this->inputTensors[1].get_desc());
            memcpy(state, get_ptr_from_tensor(this->inputTensors[1], this->archInfo.arch),
                cStateBytes);
            memcpy(state + cStateBytes,
                get_ptr_from_tensor(this->inputTensors[2], this->archInfo.arch),
                tensorNumBytes(this->inputTensors[2].get_desc()));
        }

        // NOTE: no clean tmp and output
        CHECK_STATUS(rnn(inputTensor, this->weightTensors, this->biasTensors, this->p, this->temp,
            outputTensor, &this->archInfo));

        if (this->outputTensors.size() == 2) {
            memcpy(get_ptr_from_tensor(this->outputTensors[1], this->archInfo.arch),
                get_ptr_from_tensor(this->temp, this->archInfo.arch),
                tensorNumBytes(this->outputTensors[1].get_desc()));
        } else if (this->outputTensors.size() == 3) {
            U8 *state = (U8 *)get_ptr_from_tensor(this->temp, this->archInfo.arch);
            U32 cStateBytes = tensorNumBytes(this->outputTensors[1].get_desc());
            memcpy(get_ptr_from_tensor(this->outputTensors[1], this->archInfo.arch), state,
                cStateBytes);
            memcpy(get_ptr_from_tensor(this->outputTensors[2], this->archInfo.arch),
                state + cStateBytes, tensorNumBytes(this->outputTensors[2].get_desc()));
        }
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
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
};

#endif  // _RNN_CPU_H
