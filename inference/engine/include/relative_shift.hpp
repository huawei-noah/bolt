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

class RelativeShift : public Operator {
public:
    RelativeShift(DataType dt, RelativeShiftParamSpec p)
    {
        this->dt = dt;
        this->p = p;
    }

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<RelativeShift> mem =
            std::shared_ptr<RelativeShift>(new RelativeShift(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    OperatorType get_type() override
    {
        return OT_RelativeShift;
    }

    void run() override
    {
        Tensor inputTensor = this->inputTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        U8 *inputPtr = (U8 *)((CpuMemory *)(inputTensor.get_memory()))->get_ptr();
        U8 *outputPtr = (U8 *)((CpuMemory *)(outputTensor.get_memory()))->get_ptr();

        TensorDesc inputDesc = inputTensor.get_desc();
        I32 tmpAxis = (this->p.axis + inputDesc.nDims) % inputDesc.nDims;
        tmpAxis = (I32)inputDesc.nDims - 1 - tmpAxis;
        U32 length = inputDesc.dims[tmpAxis];
        if (tmpAxis + 1 >= (I32)inputDesc.nDims) {
            U32 bytes = inputTensor.bytes();
            memcpy(outputPtr, inputPtr, bytes);
            return;
        }
        U32 loops = inputDesc.dims[tmpAxis + 1];
        U32 innerLength = 1;
        U32 outerLength = 1;
        for (I32 i = 0; i < tmpAxis; i++) {
            innerLength *= inputDesc.dims[i];
        }
        for (U32 i = tmpAxis + 2; i < inputDesc.nDims; i++) {
            outerLength *= inputDesc.dims[i];
        }
        U32 tileSize = innerLength * bytesOf(inputDesc.dt);
        U32 chunkSize = length * tileSize;
        U8 *dstPtr = outputPtr;
        for (U32 i = 0; i < outerLength; i++) {
            U8 *srcPtr = inputPtr + i * loops * chunkSize;
            U32 num =
                loops * length - (loops - this->p.shift_length) * (this->p.shift_length + length);
            U32 start = this->p.shift_length * length - num;
            U32 srcIndex = start * tileSize;
            memcpy(dstPtr, srcPtr + srcIndex, num * tileSize);
            dstPtr += num * tileSize;
            srcIndex += num * tileSize;
            for (U32 j = this->p.shift_length; j < loops; j++) {
                memset(dstPtr, 0, this->p.shift_length * tileSize);
                dstPtr += this->p.shift_length * tileSize;
                memcpy(dstPtr, srcPtr + srcIndex, chunkSize);
                dstPtr += chunkSize;
                srcIndex += chunkSize;
            }
        }
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        outTensors[0]->resize(inTensors[0]->get_desc());
        return SUCCESS;
    }

private:
    RelativeShiftParamSpec p;
};

#endif  // _RELATIVE_SHIFT_H
