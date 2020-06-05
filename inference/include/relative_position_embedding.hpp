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
#include "weight_operator.hpp"
#include "embedding.hpp"
#include "tensor_computing.h"

class RelativePositionEmbedding: public Embedding {
public:
    RelativePositionEmbedding(DataType dt, U32 inputDim, U32 numOutput, bool transpose, I32 axis)
        :Embedding(dt, inputDim, numOutput, transpose)
    {
        this->axis = axis;
    }

    OperatorType get_op_type() override
    {
        return OT_RelativePositionEmbedding;
    }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor =  this->inputTensors[0];
        Tensor weightTensor;
        if (this->weightTensors.size() > 0)
            weightTensor = this->weightTensors[0];
        else
            weightTensor = this->inputTensors[1];
        Tensor outputTensor = this->outputTensors[0];

        TensorDesc inputDesc = inputTensor.get_desc();
        U8* weightPtr = weightTensor.get_val();
        U8* outputPtr = outputTensor.get_val();

        I32 tmpAxis = (this->axis + inputDesc.nDims) % inputDesc.nDims;
        U32 batch = inputDesc.dims[inputDesc.nDims-1];
        U32 length = inputDesc.dims[inputDesc.nDims - 1 - tmpAxis];
        for (U32 in = 0; in < batch; in++) {
            U8* ptr = outputPtr + in * length * this->numOutput * bytesOf(this->dt);
            if (length > this->inputDim) {
                U32 size = (length - this->inputDim) * this->numOutput * bytesOf(this->dt);
                memset(ptr, 0, size);
                ptr += size;
            }
            U32 start = 0;
            U32 copyLength = this->inputDim;
            if (length < this->inputDim) {
                start = this->inputDim - length;
                copyLength = length;
            }
            if (transpose) {
                for (U32 i = 0; i < copyLength; i++) {
                    for (U32 j = 0; j < this->numOutput; j++) {
                        memcpy(ptr, weightPtr+(j*this->inputDim+start+i)*bytesOf(this->dt),
                            bytesOf(this->dt));
                    }
                }
            } else {
                memcpy(ptr, weightPtr+start*this->numOutput*bytesOf(this->dt),
                    copyLength*this->numOutput*bytesOf(this->dt));
            }
        }
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc inDim = inDims[0];
        I32 tmpAxis = (this->axis + inDim.nDims) % inDim.nDims;
        U32 batch = inDim.dims[inDim.nDims-1];
        U32 length = inDim.dims[inDim.nDims - 1 - tmpAxis];
        (*outDims)[0] = tensor3df(this->dt, DF_MTK, batch, length, this->numOutput);
        return SUCCESS;
    }
private:
    int axis;
};

#endif //_RELATIVE_POSITION_EMBEDDING_H
