// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _TRANSPOSE_CPU_H
#define _TRANSPOSE_CPU_H

#include <algorithm>
#include "operator.hpp"
#include "tensor_computing.h"
#include "transpose.hpp"

class TransposeCPU: public Transpose {
public:
    TransposeCPU(DataType dt, U32* transDimsPtr, U32 transDimsSize) : Transpose(dt, transDimsPtr, transDimsSize) {}

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)

        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();

        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        if (DF_NCHWC8 == inputDesc.df) {
            inputDesc.nDims = 5;
            for (int i = 3; i >= 0; i--) {
                inputDesc.dims[i + 1] = inputDesc.dims[i];
            }
            inputDesc.dims[3] /= 8;
            inputDesc.dims[0] = 8;

            TensorDesc desc = outputDesc;
            desc.nDims = 5;
            U32 idx = 4;
            for (int i = 3; i >= 0; i--) {
                if (1 == transDims[3 - i]) {  // C
                    desc.dims[idx] = outputDesc.dims[i] / 8;
                    idx--;
                    desc.dims[idx] = 8;
                    idx--;
                } else {
                    desc.dims[idx] = outputDesc.dims[i];
                    idx--;
                }
            }
            outputDesc = desc;
        }

        CHECK_STATUS(transpose(inputDesc, inputTensor.get_val(), outputDesc, outputTensor.get_val(), this->transDims.data(), this->schedule));
        outputTensor.set_scale(inputTensor.get_scale());
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc> imDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc inputDesc = imDims[0];
        if (DF_NCHWC8 == inputDesc.df) {
            if (this->transDims.size() == 4) {
                auto ptr = std::find(this->transDims.begin(), this->transDims.end(), 1);
                this->transDims.insert(ptr+1, 4);
            }
        }
        CHECK_STATUS(transpose_infer_output_size(inputDesc, &((*outDims)[0]), this->transDims.data(), this->schedule));
        return SUCCESS;
    }
};

#endif //_TRANSPOSE_CPU_H
