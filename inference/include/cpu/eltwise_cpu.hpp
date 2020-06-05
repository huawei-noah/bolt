// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


/**
 * Project deploy
 */


#ifndef _ELTWISE_CPU_H
#define _ELTWISE_CPU_H

#include "operator.hpp"
#include "eltwise.hpp"

class EltwiseCPU: public Eltwise {
public:
    EltwiseCPU(EltwiseMode eltMode, I32 coeffSize, F32* coeffValues):Eltwise(eltMode, coeffSize, coeffValues){}

    virtual void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)

        Vec<TensorDesc> inputDesc;
        Vec<void*> inputPtr;
#ifdef _USE_INT8
        F16 *inD = (F16*)this->temp->get_val();
#endif
        for (Tensor tensorIn: this->inputTensors) {
            TensorDesc desc = tensorIn.get_desc();
            U8 *ptr = tensorIn.get_val();
#ifdef _USE_INT8
            if (DT_I8 == desc.dt) {
                INT8 *inQ = (INT8*)ptr;
                F32 inputScale = tensorIn.get_scale();
                dequantize_int8_to_fp16(tensorNumElements(desc), inQ, inputScale, inD);
                desc.dt = DT_F16;
                ptr = (U8*)inD;
                inD += tensorNumElements(desc);
            }
#endif
            inputDesc.push_back(desc);
            inputPtr.push_back((void*)ptr);
        }
        auto outputDesc = this->outputTensors[0].get_desc();
        auto outputPtr = this->outputTensors[0].get_val();

        if (this->eltMode == ELTWISE_PROD && inputDesc.size() == 2 &&
            (inputDesc[1].nDims == 2 || (inputDesc[1].nDims == 4 && inputDesc[1].dims[0] == 1 && inputDesc[1].dims[1] == 1)) &&
            tensorNumElements(inputDesc[0]) != tensorNumElements(inputDesc[1]) ) { 
                CHECK_STATUS(scale(this->inputTensors[0].get_desc(), this->inputTensors[0].get_val(),
                    1, this->inputTensors[1].get_val(), nullptr,
                    outputDesc, outputPtr, this->schedule));
        } else {
            CHECK_STATUS(eltwise(inputDesc, inputPtr, outputDesc, outputPtr, this->eltMode, this->schedule));
        }

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override
    {
        CHECK_STATUS(eltwise_infer_output_size(inDims, &((*outDims)[0]), this->schedule));
        if (DT_I8 == (*outDims)[0].dt) {
            (*outDims)[0].dt = DT_F16;
            this->lenOfTemp = 0;
            for (auto desc : inDims) {
                if (DT_I8 == desc.dt) {
                    this->lenOfTemp += tensorNumElements(desc) * bytesOf(DT_F16);
                }
            }
        }
        return SUCCESS;
    }

    U32 infer_tmp_memory_size() override
    {
        return this->lenOfTemp;
    }
};

#endif //_ELTWISE_CPU_H
