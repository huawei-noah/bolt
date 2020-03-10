// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _RESIZE_H
#define _RESIZE_H

#include "operator.hpp"
#include "image.h"

class Resize: public Operator {
public:
    Resize(DataType paramDT, void* paramPtr)
    {
        switch (paramDT) {
            case DT_F32: {
                memcpy(this->scale, paramPtr, 4 * bytesOf(paramDT));
                break;
            }
            case DT_U32: {
                memcpy(this->size, paramPtr, 2 * bytesOf(paramDT));
                break;
            }
            default: {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        this->paramDT = paramDT;
    }

    OperatorType get_op_type() override
    {
        return OT_Resize;
    }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)

        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();

        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        CHECK_STATUS(resize(inputDesc, inputTensor.get_val(), outputDesc, outputTensor.get_val(), this->schedule));

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc inputDesc = inDims[0];
        ResizeDesc resizeDesc;
        resizeDesc.paramDT = this->paramDT;
        U32 bytes;
        switch (paramDT) {
            case DT_F32: {
                CHECK_REQUIREMENT(1 == scale[0] && 1 == scale[1]);
                CHECK_STATUS(resize_infer_output_size(inputDesc, resizeDesc, this->scale + 2, &((*outDims)[0]), &bytes));
                break;
            }
            case DT_U32: {
                CHECK_STATUS(resize_infer_output_size(inputDesc, resizeDesc, this->size, &((*outDims)[0]), &bytes));
                break;
            }
            default: {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        return SUCCESS;
    }

private:
    DataType paramDT;
    F32 scale[4];
    U32 size[2];
};

#endif //_RESIZE_H
