// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _EMBEDDING_CPU_H
#define _EMBEDDING_CPU_H

#include "embedding.hpp"

class EmbeddingCPU : public Embedding {
public:
    EmbeddingCPU(DataType dt, EmbedParamSpec p) : Embedding(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<EmbeddingCPU> mem =
            std::shared_ptr<EmbeddingCPU>(new EmbeddingCPU(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    void run() override
    {
        Tensor weightTensor = (this->weightTensors.size()) ? this->weightTensors[0]
                                                           : this->inputTensors[1];
        CHECK_STATUS(embedding(
            this->inputTensors[0], weightTensor, this->p, this->outputTensors[0], &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        CHECK_STATUS(embedding_infer_output_size(
            inTensors[0], this->p, this->dt, outTensors[0], &this->archInfo));
        return SUCCESS;
    }

    EE init_weight_bias_from_model(std::shared_ptr<U8> *modelPtrShared) override
    {
        U8 *modelPtr = nullptr;
        if (modelPtrShared != nullptr) {
            modelPtr = (*modelPtrShared).get();
        }
        TensorDesc weightDesc;
        if (this->p.transpose) {
            weightDesc = tensor2df(this->dt, DF_TRANSPOSE, this->p.num_output, this->p.input_dim);
        } else {
            weightDesc = tensor2df(this->dt, DF_NORMAL, this->p.input_dim, this->p.num_output);
        }
        U32 weightBytes = tensorNumBytes(weightDesc);

        std::shared_ptr<Tensor> modelWeightTensor(new Tensor());
        modelWeightTensor->resize(weightDesc);

        bool set_ptr = false;
        modelWeightTensor->alloc();
        if (modelPtr != nullptr) {
            memcpy(
                ((CpuMemory *)(modelWeightTensor->get_memory()))->get_ptr(), modelPtr, weightBytes);
            *modelPtrShared = std::shared_ptr<U8>(*modelPtrShared, modelPtr + weightBytes);
            set_ptr = true;
        } else {
            auto curOpWs = this->get_weightspec();
            if (curOpWs.weight != nullptr) {
                memcpy(((CpuMemory *)(modelWeightTensor->get_memory()))->get_ptr(), curOpWs.weight,
                    weightBytes);
                set_ptr = true;
            }
        }
        if (set_ptr) {
            this->weightTensors.push_back(*modelWeightTensor.get());
        }
        return SUCCESS;
    }
};

#endif  // _EMBEDDING_CPU_H
