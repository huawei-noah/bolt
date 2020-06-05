// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _SOFTMAX_CPU_H
#define _SOFTMAX_CPU_H

#include "operator.hpp"
#include "tensor_computing.h"
#include "softmax.hpp"

class SoftmaxCPU : public Softmax {
public:

    SoftmaxCPU(DataType dt, int axis):
        Softmax(dt, axis) { }
    
    virtual void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        U8 *inputPtr = (U8*)inputTensor.get_val();
        
        if (DT_I8 == inputDesc.dt) {
#ifdef _USE_INT8
            F32 inputScale = inputTensor.get_scale();
            INT8 *inQ = (INT8*)inputPtr;
            U32 numData = tensorNumElements(inputDesc);
            F16* inD = (F16*)this->temp->get_val();
            dequantize_int8_to_fp16(numData, inQ, inputScale, inD);
            CHECK_STATUS(softmax(outputDesc, inD, this->axis, outputDesc, outputTensor.get_val(), this->schedule));
#endif
        } else {
            CHECK_STATUS(softmax(inputDesc, inputPtr, this->axis, outputDesc, outputTensor.get_val(), this->schedule));
        }
        
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        CHECK_STATUS(softmax_infer_output_size(inDims[0], &((*outDims)[0]), this->schedule));
        if (DT_I8 == (*outDims)[0].dt) {
            (*outDims)[0].dt = DT_F16;
            this->lenOfTemp = tensorNumBytes((*outDims)[0]);
        }
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        return this->lenOfTemp;
    }
};

#endif //SOFTMAX_CPU_H
