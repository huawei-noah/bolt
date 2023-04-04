// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _EMBEDDING_OCL_H
#define _EMBEDDING_OCL_H

#include "embedding.hpp"

class EmbeddingOCL : public Embedding {
public:
    EmbeddingOCL(DataType dt, EmbedParamSpec p) : Embedding(dt, p)
    {
        INIT_GPU_INFO(nullptr)
    }

    ~EmbeddingOCL(){DESTROY_OCL_KERNEL}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<EmbeddingOCL> mem =
            std::shared_ptr<EmbeddingOCL>(new EmbeddingOCL(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    inline void run_prepare()
    {
        OCLContext::getInstance().handle.get()->curOpName = this->get_name();
        Tensor inputTensor = this->inputTensors[0];
        Tensor weightTensor;
        if (this->weightTensors.size() > 0) {
            weightTensor = this->weightTensors[0];
        } else {
            weightTensor = this->inputTensors[1];
        }
        Tensor outputTensor = this->outputTensors[0];
        CHECK_STATUS(embedding(
            inputTensor, weightTensor, this->p, this->temp, outputTensor, &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        if (this->p.num_outputs <= 0) {
            CHECK_REQUIREMENT(inTensors.size() > 1);
            TensorDesc desc = inTensors[1]->get_desc();
            CHECK_REQUIREMENT(desc.nDims == 2);
            if (this->p.transpose) {
                this->p.num_inputs = desc.dims[0];
                this->p.num_outputs = desc.dims[1];
            } else {
                this->p.num_inputs = desc.dims[1];
                this->p.num_outputs = desc.dims[0];
            }
        }
        return embedding_infer_output_size(
            inTensors[0], this->p, this->dt, outTensors[0], &this->archInfo);
    }

    EE init_weight_bias_from_model(std::shared_ptr<U8> *modelPtr) override
    {
        if (modelPtr == nullptr && this->ws.weight == nullptr) {
            return SUCCESS;
        }
        TensorDesc weightDesc;
        if (this->p.transpose) {
            weightDesc = tensor2df(this->dt, DF_TRANSPOSE, this->p.num_outputs, this->p.num_inputs);
        } else {
            weightDesc = tensor2df(this->dt, DF_NORMAL, this->p.num_inputs, this->p.num_outputs);
        }
        Tensor modelWeightTensor = Tensor(OCLMem);
        modelWeightTensor.resize(weightDesc);

        CpuMemory weight_mem_src;
        std::shared_ptr<U8> weight_ptr;
        if (modelPtr) {
            weight_ptr = *modelPtr;
        } else {
            weight_ptr = std::shared_ptr<U8>(this->ws.weight, [](U8 *) {});
        }
        weight_mem_src.resize(weightDesc);
        weight_mem_src.set_shared_ptr(std::shared_ptr<U8>(weight_ptr));

        auto weightMem = (OclMemory *)modelWeightTensor.get_memory();
        weightMem->copy_from((Memory *)&weight_mem_src);
        this->weightTensors.push_back(modelWeightTensor);
        if (modelPtr) {
            *modelPtr =
                std::shared_ptr<U8>(*modelPtr, (*modelPtr).get() + tensorNumBytes(weightDesc));
        }
        return SUCCESS;
    }

    REGISTER_OCL_OPERATOR_RUN
};

#endif  // _EMBEDDING_OCL_H
