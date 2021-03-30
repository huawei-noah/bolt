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
        setMALIArchInfo(
            &(this->archInfo), nullptr, &this->needSetKernelVec, &this->needSelectKernelLS);
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
        CHECK_STATUS(embedding(inputTensor, weightTensor, this->p, outputTensor, &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        this->needSetKernelVec = true;
        if (this->p.num_output <= 0) {
            if (inTensors.size() <= 1) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
            auto mem = (OclMemory *)inTensors[1]->get_memory();
            GCLMemDesc desc = mem->get_desc();
            if (desc.nDims != 2) {
                CHECK_STATUS(NOT_MATCH);
            }
            if (this->p.transpose) {
                this->p.input_dim = desc.dims[0];
                this->p.num_output = desc.dims[1];
            } else {
                this->p.input_dim = desc.dims[1];
                this->p.num_output = desc.dims[0];
            }
            if (desc.byteSize == 0) {
                GCLMemType mt = GCL_MEM_BUF;
                MemFlags flags = CL_MEM_READ_WRITE;
                U32 stride[3] = {desc.dims[0], desc.dims[1], 1};
                U32 offset[3] = {0, 0, 0};
                CHECK_STATUS(
                    gclmem_set_desc_padding(&desc, stride, offset, this->dt, DF_NCHW, mt, flags));
                mem->padding(desc);
            }
        }
        CHECK_STATUS(embedding_infer_output_size(
            inTensors[0], this->p, this->dt, outTensors[0], &this->archInfo));
        return SUCCESS;
    }

    EE init_weight_bias_from_model(std::shared_ptr<U8> *modelPtr) override
    {
        auto curOpWs = this->get_weightspec();
        if (modelPtr == nullptr && curOpWs.weight == nullptr) {
            return SUCCESS;
        }
        TensorDesc weightDesc;
        if (this->p.transpose) {
            weightDesc = tensor2df(this->dt, DF_TRANSPOSE, this->p.num_output, this->p.input_dim);
        } else {
            weightDesc = tensor2df(this->dt, DF_NORMAL, this->p.input_dim, this->p.num_output);
        }
        Tensor modelWeightTensor = Tensor(OCLMem);
        auto weightMem = (OclMemory *)modelWeightTensor.get_memory();
        modelWeightTensor.resize(weightDesc);
        U32 stride[3] = {weightDesc.dims[0], weightDesc.dims[1], 1};
        U32 offset[3] = {0, 0, 0};
        GCLMemType mt = GCL_MEM_BUF;
        MemFlags flags = CL_MEM_READ_WRITE;
        GCLMemDesc desc = gclmem_build_desc();
        CHECK_STATUS(gclmem_set_desc_padding(&desc, stride, offset, this->dt, DF_NCHW, mt, flags));
        weightMem->padding(desc);

        CpuMemory weight_mem_src;
        std::shared_ptr<U8> weight_ptr;
        if (modelPtr) {
            weight_ptr = *modelPtr;
        } else {
            weight_ptr = std::shared_ptr<U8>(curOpWs.weight, [](U8 *) {});
        }
        weight_mem_src.resize(weightDesc);
        weight_mem_src.set_shared_ptr(std::shared_ptr<U8>(weight_ptr));
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
