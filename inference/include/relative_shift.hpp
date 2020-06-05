// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _RELATIVE_SHIFT_H
#define _RELATIVE_SHIFT_H
#include "operator.hpp"

class RelativeShift: public Operator {
public:
    RelativeShift(DataType dt, I32 axis, I32 shiftLength)
    {
        this->dt = dt;;
        this->axis = axis;
        this->shiftLength = shiftLength;
    }

    OperatorType get_op_type() override
    {
        return OT_RelativeShift;
    }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor =  this->inputTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        U8* inputPtr = inputTensor.get_val();
        U8* outputPtr = outputTensor.get_val();

        TensorDesc inputDesc = inputTensor.get_desc();
        I32 tmpAxis = (this->axis + inputDesc.nDims) % inputDesc.nDims;
        tmpAxis = (I32)inputDesc.nDims - 1 - tmpAxis;
        U32 length = inputDesc.dims[tmpAxis];
        if (tmpAxis + 1 >= (I32)inputDesc.nDims) {
            memcpy(outputPtr, inputPtr, tensorNumBytes(inputDesc));
            return;
        }
        U32 loops = inputDesc.dims[tmpAxis+1];
        U32 innerLength = 1;
        U32 outerLength = 1;
        for (I32 i = 0; i < tmpAxis; i++) {
            innerLength *= inputDesc.dims[i];
        }
        for (U32 i = tmpAxis+2; i < inputDesc.nDims; i++) {
            outerLength *= inputDesc.dims[i];
        }
        U32 tileSize = innerLength * bytesOf(inputDesc.dt);
        U32 chunkSize = length * tileSize;
        U8* dstPtr = outputPtr;
        for (U32 i = 0; i < outerLength; i++) {
            U8* srcPtr = inputPtr + i * loops * chunkSize;
            U32 num = loops * length - (loops - shiftLength) * (shiftLength + length);
            U32 start = shiftLength * length - num;
            U32 srcIndex = start * tileSize;
            memcpy(dstPtr, srcPtr+srcIndex, num*tileSize);
            dstPtr += num * tileSize;
            srcIndex += num * tileSize;
            for (U32 j = shiftLength; j < loops; j++) {
                memset(dstPtr, 0, shiftLength*tileSize);
                dstPtr += shiftLength * tileSize;
                memcpy(dstPtr, srcPtr+srcIndex, chunkSize);
                dstPtr += chunkSize;
                srcIndex += chunkSize;
            }
        }
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        (*outDims)[0] = inDims[0];
        return SUCCESS;
    }
private:
    int axis;
    int shiftLength;
};

#endif //_RELATIVE_SHIFT_H
