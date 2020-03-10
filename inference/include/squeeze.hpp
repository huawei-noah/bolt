// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _SQUEEZE_H
#define _SQUEEZE_H

#include "operator.hpp"

class Squeeze: public Operator
{
public:
    /**
    @param mode
    */
    Squeeze(DataType dt, I32 axis, I32 *dims, I32 dimSize)
    {
        this->dt = dt;
        this->axis = axis;
        this->dims = Vec<I32>(dimSize);
        memcpy(this->dims.data(), dims, sizeof(I32) * dimSize);
    }

    OperatorType get_op_type() override
    {
        return OT_Squeeze;
    }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)

        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        Tensor outputTensor = this->outputTensors[0];
        U8* inPtr = inputTensor.get_val();
        U8* outPtr = outputTensor.get_val();
        if(inPtr != outPtr) {
            memcpy(outPtr, inPtr, tensorNumBytes(inputDesc));
        }

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override
    {
        auto outDimsPtr = &((*outDims)[0]);
        outDimsPtr->dt = inDims[0].dt;
        int axis = this->axis;
        if (axis < 0)
            axis += inDims[0].nDims;
        if (axis >= 0 && axis < (int)(inDims[0].nDims)) {
            axis = inDims[0].nDims - 1 - axis;
            for (int i = 0; i < axis; i++) {
                outDimsPtr->dims[i] = inDims[0].dims[i];
            }
            if (inDims[0].dims[axis] != 1) {
                CHECK_STATUS(NOT_MATCH);
            }
            for (int i = axis+1; i < (int)(inDims[0].nDims); i++) {
                outDimsPtr->dims[i-1] = inDims[0].dims[i];
            }
            outDimsPtr->nDims = inDims[0].nDims - 1;
        }
        else {
            for (U32 i = 0; i < inDims[0].nDims; i++)
                outDimsPtr->dims[i] = inDims[0].dims[i];
            for (U32 i = 0; i < this->dims.size(); i++) {
                outDimsPtr->dims[inDims[0].nDims - 1 - this->dims[i]] = 0;
            }
            U32 index = 0;
            for (U32 i = 0; i < inDims[0].nDims; i++) {
                if (outDimsPtr->dims[i] != 0)
                    outDimsPtr->dims[index++] = outDimsPtr->dims[i];
            }
            CHECK_REQUIREMENT(index + this->dims.size() == inDims[0].nDims);
            outDimsPtr->nDims = index;
        }
        outDimsPtr->df = getTensorDefaultDataFormat(outDimsPtr->nDims);
        return SUCCESS;
    }

private:
    I32 axis;
    Vec<I32> dims;
};

#endif //_SQUEEZE_H
