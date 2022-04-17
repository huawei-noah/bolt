// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _GATHER_CPU_H
#define _GATHER_CPU_H

#include "gather.hpp"

class GatherCPU : public Gather {
public:
    GatherCPU(DataType dt, GatherParamSpec p) : Gather(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<GatherCPU> mem =
            std::shared_ptr<GatherCPU>(new GatherCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        CHECK_STATUS(gather(get_data_tensor(), get_index_tensor(), this->p, this->temp,
            this->outputTensors[0], &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        Tensor tensor0, tensor1;
        if (is_shape(inTensors)) {
            if ((this->p.data_desc.nDims > 0 && this->weightTensors.size() == 0) ||
                (this->p.index_desc.nDims > 0 && this->biasTensors.size() == 0)) {
                CHECK_STATUS(this->init_weight_bias_from_model());
            }
            if (this->p.data_desc.nDims > 0) {
                this->p.data_desc = tensor_shape(this->weightTensors[0]);
            }
            if (this->p.index_desc.nDims > 0) {
                this->p.index_desc = tensor_shape(this->biasTensors[0]);
            }
        }
        Tensor *dataTensor = get_data_tensor_ptr(inTensors, &tensor0);
        Tensor *indexTensor = get_index_tensor_ptr(inTensors, &tensor1);
        CHECK_STATUS(gather_infer_output_size(
            dataTensor, indexTensor, this->p, outTensors[0], &this->archInfo));
        return SUCCESS;
    }

    EE infer_weight_desc() override
    {
        Tensor dataTensor, indexTensor;
        if (this->p.data_desc.nDims > 0 && this->weightTensors.size() == 0) {
            dataTensor.resize(this->p.data_desc);
            this->weightTensors.push_back(dataTensor);
        }
        if (this->p.index_desc.nDims > 0 && this->biasTensors.size() == 0) {
            indexTensor.resize(this->p.index_desc);
            this->biasTensors.push_back(indexTensor);
        }
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes = 0;
        CHECK_STATUS(gather_infer_forward_tmp_bytes(get_data_tensor(), get_index_tensor(), this->p,
            this->outputTensors[0], &bytes, &this->archInfo));
        return bytes;
    }
};
#endif  // _GATHER_CPU_H
