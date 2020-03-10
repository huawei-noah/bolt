// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _COPY_H
#define _COPY_H

#include "operator.hpp"

class Copy: public Operator
{
public:
    /**
    @param mode
    */
    Copy(DataType dt, U32 *srcDimsPtr, U32 *dstDimsPtr, U32 len)
    {
        this->dt = dt;
        this->srcDims = Vec<U32>(3);
        memcpy(this->srcDims.data(), srcDimsPtr, 3 * sizeof(U32));
        this->dstDims = Vec<U32>(3);
        memcpy(this->dstDims.data(), dstDimsPtr, 3 * sizeof(U32));
        this->length = len;
    }

    OperatorType get_op_type() override
    {
        return OT_Copy;
    }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor srcTensor = this->inputTensors[0];
        TensorDesc srcDesc = srcTensor.get_desc();
        Tensor dstTensor = this->inputTensors[1];

        U32 batch = srcDesc.dims[srcDesc.nDims - 1];
        for (U32 i = 0; i < batch; i++) {
            U32 srcBlockIndex = 0;
            if (this->inputTensors.size() > 2)
                srcBlockIndex = ((U32 *)(this->inputTensors[2].get_val()))[i];
            U32 dstBlockIndex = 0;
            if (this->inputTensors.size() > 3)
                dstBlockIndex = ((U32 *)(this->inputTensors[3].get_val()))[i];
            U32 srcIndex = i * this->srcDims[0] + srcBlockIndex * this->srcDims[1] + this->srcDims[2];
            U32 dstIndex = i * this->dstDims[0] + dstBlockIndex * this->dstDims[1] + this->dstDims[2];
            memcpy((U8*)(dstTensor.get_val()) + bytesOf(srcDesc.dt) * dstIndex,
                   (U8*)(srcTensor.get_val()) + bytesOf(srcDesc.dt) * srcIndex,
                   this->length * bytesOf(srcDesc.dt));
        }

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override
    {
        UNUSED(inDims);

        (*outDims)[0].dt = this->dt;
        (*outDims)[0].df = getTensorDefaultDataFormat(0);
        (*outDims)[0].nDims = 0;
        return SUCCESS;
    }

private:
    Vec<U32> srcDims;
    Vec<U32> dstDims;
    U32 length;
};

#endif //_COPY_H
