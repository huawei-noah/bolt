// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _RELATIVE_POSITION_EMBEDDING_H
#define _RELATIVE_POSITION_EMBEDDING_H

#include "cpu/embedding_cpu.hpp"

class RelativePositionEmbedding : public EmbeddingCPU {
public:
    RelativePositionEmbedding(DataType dt, EmbedParamSpec p) : EmbeddingCPU(dt, p)
    {}

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<RelativePositionEmbedding> mem = std::shared_ptr<RelativePositionEmbedding>(
            new RelativePositionEmbedding(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    OperatorType get_type() override
    {
        return OT_RelativePositionEmbedding;
    }

    void run() override
    {
        Tensor inputTensor = this->inputTensors[0];
        Tensor weightTensor;
        if (this->weightTensors.size() > 0) {
            weightTensor = this->weightTensors[0];
        } else {
            weightTensor = this->inputTensors[1];
        }
        Tensor outputTensor = this->outputTensors[0];

        TensorDesc inputDesc = inputTensor.get_desc();
        U8 *weightPtr = (U8 *)((CpuMemory *)weightTensor.get_memory())->get_ptr();
        U8 *outputPtr = (U8 *)((CpuMemory *)outputTensor.get_memory())->get_ptr();

        I32 tmpAxis = (this->p.axis + inputDesc.nDims) % inputDesc.nDims;
        U32 batch = inputDesc.dims[inputDesc.nDims - 1];
        U32 length = inputDesc.dims[inputDesc.nDims - 1 - tmpAxis];
        for (U32 in = 0; in < batch; in++) {
            U8 *ptr = outputPtr + in * length * this->p.num_output * bytesOf(this->dt);
            if (length > this->p.input_dim) {
                U32 size = (length - this->p.input_dim) * this->p.num_output * bytesOf(this->dt);
                memset(ptr, 0, size);
                ptr += size;
            }
            U32 start = 0;
            U32 copyLength = this->p.input_dim;
            if (length < this->p.input_dim) {
                start = this->p.input_dim - length;
                copyLength = length;
            }
            if (this->p.transpose) {
                for (U32 i = 0; i < copyLength; i++) {
                    for (U32 j = 0; j < this->p.num_output; j++) {
                        memcpy(ptr,
                            weightPtr + (j * this->p.input_dim + start + i) * bytesOf(this->dt),
                            bytesOf(this->dt));
                    }
                }
            } else {
                memcpy(ptr, weightPtr + start * this->p.num_output * bytesOf(this->dt),
                    copyLength * this->p.num_output * bytesOf(this->dt));
            }
        }
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        TensorDesc inDim = inTensors[0]->get_desc();
        I32 tmpAxis = (this->p.axis + inDim.nDims) % inDim.nDims;
        U32 batch = inDim.dims[inDim.nDims - 1];
        U32 length = inDim.dims[inDim.nDims - 1 - tmpAxis];
        TensorDesc outDim = tensor3df(this->dt, DF_MTK, batch, length, this->p.num_output);
        outTensors[0]->resize(outDim);
        return SUCCESS;
    }
};

#endif  // _RELATIVE_POSITION_EMBEDDING_H
