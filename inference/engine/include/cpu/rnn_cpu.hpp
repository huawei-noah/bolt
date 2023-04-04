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

        U8 *state = (U8 *)get_ptr_from_tensor(this->temp, this->archInfo.arch);
        TensorDesc desc = inputTensor.get_desc();
        int batch = desc.dims[desc.nDims - 1];
        I32 num = p.bi_direction ? 2 : 1;
        I32 column = this->p.num_projection > 0 ? this->p.num_projection : this->p.num_outputs;
        U32 ch_size = (this->p.num_outputs + column) * bytesOf(desc.dt);
#if defined(_USE_INT8)
        if (isQuantMixDataType(this->dt)) {
            TensorDesc inputDesc = inputTensor.get_desc();
            if (DT_I8 != inputDesc.dt && DT_U8_Q != inputDesc.dt && featureScale.size() > 0 &&
                featureScale[0][0] > 0) {
                this->scales.get()[0] = featureScale[0][0];
            }
        }
#endif
        if (this->inputTensors.size() == 1) {
            // bi-direction rnn has forward-states and backward-states
            UNI_MEMSET(state, 0, batch * num * ch_size);
        } else if (this->inputTensors.size() == 2) {
            if (num != 1) {
                UNI_ERROR_LOG("currently not support to set bi-direction RNN's h or c.\n");
            }
            UNI_MEMCPY(state, get_ptr_from_tensor(this->inputTensors[1], this->archInfo.arch),
                tensorNumBytes(this->inputTensors[1].get_desc()));
        } else if (this->inputTensors.size() == 3) {
            if (num != 1) {
                UNI_ERROR_LOG("currently not support to set bi-direction RNN's h or c.\n");
            }
            U8 *h = (U8 *)get_ptr_from_tensor(this->inputTensors[1], this->archInfo.arch);
            U8 *c = (U8 *)get_ptr_from_tensor(this->inputTensors[2], this->archInfo.arch);
            U32 c_size = column * bytesOf(desc.dt);
            U32 input_h_tile = tensorNumBytes(this->inputTensors[1].get_desc()) / batch;
            U32 input_c_tile = tensorNumBytes(this->inputTensors[2].get_desc()) / batch;
            for (int i = 0; i < batch; i++) {
                U8 *ptr = state + i * ch_size;
                UNI_MEMCPY(ptr, c + input_c_tile * i, input_c_tile);
                UNI_MEMCPY(ptr + c_size, h + input_h_tile * i, input_h_tile);
            }
        }

        std::vector<Tensor> tmpTensor(1, this->temp);
        CHECK_STATUS(rnn(this->inputTensors, this->weightTensors, this->biasTensors, this->p,
            tmpTensor, this->outputTensors, this->scales.get(), &this->archInfo));

        if (this->outputTensors.size() == 2) {
            UNI_MEMCPY(get_ptr_from_tensor(this->outputTensors[1], this->archInfo.arch), state,
                tensorNumBytes(this->outputTensors[1].get_desc()));
        } else if (this->outputTensors.size() == 3) {
            U8 *h = (U8 *)get_ptr_from_tensor(this->outputTensors[1], this->archInfo.arch);
            U8 *c = (U8 *)get_ptr_from_tensor(this->outputTensors[2], this->archInfo.arch);
            U32 c_size = column * bytesOf(desc.dt);
            U32 output_h_tile = tensorNumBytes(this->outputTensors[1].get_desc()) / batch / num;
            U32 output_c_tile = tensorNumBytes(this->outputTensors[2].get_desc()) / batch / num;
            for (int i = 0, k = 0; i < batch; i++) {
                for (int j = 0; j < num; j++, k++) {
                    U8 *ptr = state + k * ch_size;
                    UNI_MEMCPY(c + k * output_c_tile, ptr, output_c_tile);
                    UNI_MEMCPY(h + k * output_h_tile, ptr + c_size, output_h_tile);
                }
            }
        }
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        TensorDesc inputDesc = inTensors[0]->get_desc();
        CHECK_REQUIREMENT(inputDesc.nDims >= 3);
        this->xDim = inputDesc.dims[inputDesc.nDims - 3];
        for (U32 i = 0; i < inputDesc.nDims - 3; ++i) {
            xDim *= inputDesc.dims[i];
        }
        return rnn_infer_output_size(inTensors, this->p, outTensors, &this->archInfo);
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
