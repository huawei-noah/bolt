// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _SCATTER_CPU_H
#define _SCATTER_CPU_H

#include "scatter.hpp"

class ScatterCPU : public Scatter {
public:
    ScatterCPU(DataType dt, ScatterParamSpec p) : Scatter(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<ScatterCPU> mem =
            std::shared_ptr<ScatterCPU>(new ScatterCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    Tensor get_data_tensor()
    {
        Tensor dataTensor;
        if (this->p.data_desc.nDims > 0) {
            CHECK_REQUIREMENT(0 < this->weightTensors.size());
            dataTensor = this->weightTensors[0];
        } else {
            CHECK_REQUIREMENT(0 < this->inputTensors.size());
            dataTensor = this->inputTensors[0];
        }
        return dataTensor;
    }

    Tensor get_index_tensor()
    {
        U32 inputCount = 0;
        if (this->p.data_desc.nDims == 0) {
            inputCount++;
        }
        Tensor indexTensor;
        if (this->p.index_desc.nDims > 0) {
            CHECK_REQUIREMENT(0 < this->biasTensors.size());
            indexTensor = this->biasTensors[0];
        } else {
            CHECK_REQUIREMENT(inputCount < this->inputTensors.size());
            indexTensor = this->inputTensors[inputCount];
        }
        return indexTensor;
    }

    Tensor get_update_tensor()
    {
        U32 inputCount = 0, weightCount = 0;
        if (this->p.data_desc.nDims == 0) {
            inputCount++;
        } else {
            weightCount++;
        }
        if (this->p.index_desc.nDims == 0) {
            inputCount++;
        }
        Tensor updateTensor;
        if (this->p.update_desc.nDims > 0) {
            CHECK_REQUIREMENT(weightCount < this->weightTensors.size());
            updateTensor = this->weightTensors[weightCount++];
        } else {
            CHECK_REQUIREMENT(inputCount < this->inputTensors.size());
            updateTensor = this->inputTensors[inputCount];
        }
        return updateTensor;
    }

    void run() override
    {
        CHECK_STATUS(scatter(get_data_tensor(), get_index_tensor(), get_update_tensor(), this->p,
            this->temp, this->outputTensors[0], &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        Tensor tensor;
        Tensor *dataTensor;
        if (this->p.data_desc.nDims > 0) {
            tensor.resize(this->p.data_desc);
            dataTensor = &tensor;
        } else {
            dataTensor = inTensors[0];
        }
        CHECK_STATUS(scatter_infer_output_size(dataTensor, outTensors[0], &this->archInfo));
        return SUCCESS;
    }

    EE infer_weight_desc() override
    {
        Tensor dataTensor, indexTensor, updateTensor;
        if (this->p.data_desc.nDims > 0) {
            dataTensor.resize(this->p.data_desc);
            this->weightTensors.push_back(dataTensor);
        }
        if (this->p.update_desc.nDims > 0) {
            updateTensor.resize(this->p.update_desc);
            this->weightTensors.push_back(updateTensor);
        }
        if (this->p.index_desc.nDims > 0) {
            indexTensor.resize(this->p.index_desc);
            this->biasTensors.push_back(indexTensor);
        }
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        U32 bytes = 0;
        CHECK_STATUS(scatter_infer_forward_tmp_bytes(
            get_data_tensor(), get_update_tensor(), &bytes, &this->archInfo));
        return bytes;
    }
};

#endif  // _SCATTER_CPU_H
